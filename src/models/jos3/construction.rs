//! Body-parameter validation, per-segment array construction, and physiological
//! constant construction (thermal conductance / thermal capacity) for the JOS3
//! thermoregulation model.
//!
//! Mirrors:
//! `pythermalcomfort/jos3_functions/construction.py`
//! (pythermalcomfort 4.4.0, 789 lines).
//!
//! # Ported
//!
//! - [`validate_body_parameters`] — Python: `validate_body_parameters`.
//! - [`to_array_body_parts_scalar`], [`to_array_body_parts_by_name`] — Python:
//!   `to_array_body_parts`. Python dispatches dynamically on the input's runtime type
//!   (`int | float`, `dict`, `list | np.ndarray`). Rust's type system makes that
//!   dispatch static, so the single Python function becomes explicitly-typed functions
//!   per case; the `list | np.ndarray` case needs no function of its own here, since
//!   [`super::jos3::PerBodyPart::BySegment`] already carries an owned `[f64; 17]` —
//!   a fixed-size array statically has the length Python's runtime check enforces, so
//!   there is nothing left to validate (see that variant's doc comment for the full
//!   reasoning).
//! - [`bsa_rate`] — Python: `bsa_rate`. Takes a [`BsaFormula`] directly rather than a
//!   `bsa_equation: &str`, so (unlike Python, which resolves the string against
//!   `BodySurfaceAreaEquations` inside `body_surface_area`) there is no "unrecognized
//!   equation name" runtime error to model.
//! - [`local_bsa`] — Python: `local_bsa`.
//! - [`weight_rate`] — Python: `weight_rate`.
//! - [`bfb_rate`] — Python: `bfb_rate`.
//! - [`conductance`] — Python: `conductance`.
//! - [`capacity`] — Python: `capacity`.
//!
//! # Not ported
//!
//! - `pass_values_to_jos3_body_parts`: converts a plain `[f64; 17]` (rounded) into a
//!   `JOS3BodyParts` dataclass, i.e. it belongs to the public-API return-type boundary
//!   (`classes_return.JOS3BodyParts`), which is being designed separately elsewhere in
//!   this port. Nothing in this module needs it.
//!
//! # BSA formula reuse
//!
//! `bsa_rate` (and everything built on it: `local_bsa`, `bfb_rate`, `conductance`,
//! `capacity`) calls Python's `pythermalcomfort.utilities.body_surface_area`, which this
//! crate already ports as [`crate::utilities::body_surface_area`] /
//! [`crate::utilities::BsaFormula`]. That existing implementation is reused as-is here
//! rather than duplicated.
//!
//! # `NUM_NODES` / body-segment-layer indices
//!
//! `conductance` and `capacity` build an 85-node matrix/vector whose layout (which
//! "layer" of which body segment maps to which flat index) is defined by
//! `pythermalcomfort/jos3_functions/matrix.py`'s `index_order()` (`IDICT`, `NUM_NODES`).
//! That module is ported as [`super::matrix`], which computes `IDICT`/`NUM_NODES` at
//! compile time from the same fixed 17-segment body plan Python's `index_order()` walks.
//! This module reuses those definitions directly ([`super::matrix::IDICT`],
//! [`super::matrix::NUM_NODES`]) rather than keeping its own copy.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::utilities::{
    BodySurfaceAreaInputs, BodySurfaceAreaOptions, BsaFormula, body_surface_area,
};
use crate::{Length, Mass};

use super::matrix::{IDICT, NUM_NODES};
use super::parameters::{BODY_PART_NAMES, NUM_BODY_PARTS, defaults};

// ---------------------------------------------------------------------------
// validate_body_parameters
// ---------------------------------------------------------------------------

/// A body parameter fell outside the range JOS3 was validated for.
/// Python: `validate_body_parameters` raises `ValueError` for each of these cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyParameterError {
    /// Height was not in `[0.5, 3.0]` meters.
    Height,
    /// Weight was not in `[20.0, 200.0]` kilograms.
    Weight,
    /// Age was not in `[5, 100]` years.
    Age,
    /// Body fat percentage was not in `[1, 90]` (1% to 90%).
    BodyFat,
}

impl core::fmt::Display for BodyParameterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BodyParameterError::Height => {
                write!(f, "Height must be in the range [0.5, 3.0] meters.")
            }
            BodyParameterError::Weight => {
                write!(f, "Weight must be in the range [20.0, 200.0] kilograms.")
            }
            BodyParameterError::Age => write!(f, "Age must be in the range [5, 100] years."),
            BodyParameterError::BodyFat => write!(
                f,
                "Body fat percentage must be in the range [1, 90] (1% to 90%)."
            ),
        }
    }
}

impl core::error::Error for BodyParameterError {}

/// Validate the body parameters: height, weight, age, and body fat percentage.
///
/// Python: `validate_body_parameters(height, weight, age, body_fat)`.
///
/// # Errors
///
/// Returns [`BodyParameterError`] for whichever parameter is out of range first,
/// checked in the order height, weight, age, body fat (matching Python's check order).
pub(crate) fn validate_body_parameters(
    height: f64,
    weight: f64,
    age: i32,
    body_fat: f64,
) -> Result<(), BodyParameterError> {
    if !(0.5..=3.0).contains(&height) {
        return Err(BodyParameterError::Height);
    }
    if !(20.0..=200.0).contains(&weight) {
        return Err(BodyParameterError::Weight);
    }
    if !(5..=100).contains(&age) {
        return Err(BodyParameterError::Age);
    }
    if !(1.0..=90.0).contains(&body_fat) {
        return Err(BodyParameterError::BodyFat);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// to_array_body_parts
// ---------------------------------------------------------------------------

/// A per-body-part input couldn't be turned into a `[f64; 17]`.
/// Python: `to_array_body_parts` raises `ValueError` for these cases. (Python's
/// "unsupported input type" case has no Rust equivalent: the type system already
/// prevents passing anything but a scalar, a fixed-size array, or a name/value list;
/// nor is there a "wrong length" case, for the same reason — see
/// [`super::jos3::PerBodyPart::BySegment`].)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPartsInputError {
    /// `to_array_body_parts_by_name` was missing a required body-part name.
    /// Python: dict subscript `inp[key]` raises `KeyError` for a missing key.
    MissingKey(&'static str),
}

impl core::fmt::Display for BodyPartsInputError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BodyPartsInputError::MissingKey(name) => {
                write!(f, "missing value for body part \"{name}\"")
            }
        }
    }
}

impl core::error::Error for BodyPartsInputError {}

/// Broadcast a single value to all 17 body segments.
/// Python: `to_array_body_parts(inp)` when `inp` is `int | float`.
#[must_use]
pub(crate) fn to_array_body_parts_scalar(value: f64) -> [f64; NUM_BODY_PARTS] {
    [value; NUM_BODY_PARTS]
}

/// Look up each of the 17 body-segment names (in `pairs`, order-independent) and
/// assemble a `[f64; 17]` in [`BODY_PART_NAMES`] order.
/// Python: `to_array_body_parts(inp)` when `inp` is a `dict`.
///
/// # Errors
///
/// Returns [`BodyPartsInputError::MissingKey`] if `pairs` doesn't contain every name in
/// [`BODY_PART_NAMES`].
pub(crate) fn to_array_body_parts_by_name(
    pairs: &[(&str, f64)],
) -> Result<[f64; NUM_BODY_PARTS], BodyPartsInputError> {
    let mut out = [0.0; NUM_BODY_PARTS];
    for (i, name) in BODY_PART_NAMES.iter().enumerate() {
        let value = pairs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, v)| *v)
            .ok_or(BodyPartsInputError::MissingKey(name))?;
        out[i] = value;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// bsa_rate / local_bsa / weight_rate / bfb_rate
// ---------------------------------------------------------------------------

/// Calculate the ratio of body surface area (BSA) to the standard body (1.87 m^2).
/// Python: `bsa_rate(height, weight, bsa_equation)`.
#[must_use]
pub(crate) fn bsa_rate(height: f64, weight: f64, bsa_equation: BsaFormula) -> f64 {
    let bsa_all = body_surface_area(
        BodySurfaceAreaInputs {
            weight: Mass::from_kilograms(weight),
            height: Length::from_meters(height),
        },
        BodySurfaceAreaOptions {
            formula: bsa_equation,
        },
    )
    .as_square_meters();
    bsa_all / defaults::LOCAL_BSA.iter().sum::<f64>()
}

/// Calculate local body surface area (BSA) in square meters for each of the 17 body
/// segments. Python: `local_bsa(height, weight, bsa_equation)`.
#[must_use]
pub(crate) fn local_bsa(
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
) -> [f64; NUM_BODY_PARTS] {
    let bsa_ratio = bsa_rate(height, weight, bsa_equation);
    let mut out = defaults::LOCAL_BSA;
    for v in &mut out {
        *v *= bsa_ratio;
    }
    out
}

/// Calculate the ratio of the body weight to the standard body.
/// Python: `weight_rate(weight)`.
#[must_use]
pub(crate) fn weight_rate(weight: f64) -> f64 {
    weight / defaults::WEIGHT
}

/// Calculate the ratio of basal blood flow (BFB) to the standard body (290 L/h).
/// Python: `bfb_rate(height, weight, bsa_equation, age, ci)`.
#[must_use]
pub(crate) fn bfb_rate(
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    ci: f64,
) -> f64 {
    // Convert unit from L/min/m^2 to L/h/m^2.
    let mut ci = ci * 60.0;

    // Adjust cardiac index based on age.
    if age < 50 {
        // no-op, matches Python's explicit `ci *= 1`
    } else if age < 60 {
        ci *= 0.85;
    } else if age < 70 {
        ci *= 0.75;
    } else {
        // age >= 70
        ci *= 0.7;
    }

    let bsa_ratio = bsa_rate(height, weight, bsa_equation);
    let bfb_all = ci * bsa_ratio * defaults::LOCAL_BSA.iter().sum::<f64>(); // Total BFB in L/h
    bfb_all / f64::from(defaults::BLOOD_FLOW_RATE) // Ratio to the standard body (290 L/h)
}

// ---------------------------------------------------------------------------
// conductance / capacity: node-layer index table
// ---------------------------------------------------------------------------

/// Flat matrix index for row `r`, column `c` of a [`NUM_NODES`]-by-[`NUM_NODES`] matrix
/// stored row-major in a `Vec<f64>`.
const fn mat_index(r: usize, c: usize) -> usize {
    r * NUM_NODES + c
}

// ---------------------------------------------------------------------------
// conductance
// ---------------------------------------------------------------------------

/// Calculate thermal conductance between layers \[W/K\].
///
/// Python: `conductance(height, weight, bsa_equation, fat)`.
///
/// Returns a `NUM_NODES`-by-`NUM_NODES` (85x85) matrix, flattened row-major
/// (`result[r * 85 + c]`), symmetric by construction.
#[must_use]
pub(crate) fn conductance(
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    fat: f64,
) -> Vec<f64> {
    // core to skin [W/K], selected by body-fat-rate bracket.
    let mut cdt_cr_sk: [f64; NUM_BODY_PARTS] = if fat < 12.5 {
        [
            1.341, 0.93, 1.879, 1.729, 2.37, 1.557, 1.018, 2.21, 1.557, 1.018, 2.21, 2.565, 1.378,
            3.404, 2.565, 1.378, 3.404,
        ]
    } else if fat < 17.5 {
        [
            1.311, 0.909, 1.785, 1.643, 2.251, 1.501, 0.982, 2.183, 1.501, 0.982, 2.183, 2.468,
            1.326, 3.37, 2.468, 1.326, 3.37,
        ]
    } else if fat < 22.5 {
        [
            1.282, 0.889, 1.698, 1.563, 2.142, 1.448, 0.947, 2.156, 1.448, 0.947, 2.156, 2.375,
            1.276, 3.337, 2.375, 1.276, 3.337,
        ]
    } else if fat < 27.5 {
        [
            1.255, 0.87, 1.618, 1.488, 2.04, 1.396, 0.913, 2.13, 1.396, 0.913, 2.13, 2.285, 1.227,
            3.304, 2.285, 1.227, 3.304,
        ]
    } else {
        // fat >= 27.5
        [
            1.227, 0.852, 1.542, 1.419, 1.945, 1.346, 0.88, 1.945, 1.346, 0.88, 1.945, 2.198,
            1.181, 3.271, 2.198, 1.181, 3.271,
        ]
    };

    let mut cdt_cr_ms = [0.0_f64; NUM_BODY_PARTS]; // core to muscle [W/K]
    let mut cdt_ms_fat = [0.0_f64; NUM_BODY_PARTS]; // muscle to fat [W/K]
    let mut cdt_fat_sk = [0.0_f64; NUM_BODY_PARTS]; // fat to skin [W/K]

    // head and pelvis consist of 65MN's conductances.
    cdt_cr_ms[0] = 1.601; // head
    cdt_ms_fat[0] = 13.222;
    cdt_fat_sk[0] = 16.008;
    cdt_cr_ms[4] = 3.0813; // pelvis
    cdt_ms_fat[4] = 10.3738;
    cdt_fat_sk[4] = 41.4954;

    // vessel to core [W/K]
    let mut cdt_ves_cr = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.586, 0.383, 1.534, 0.586, 0.383, 1.534, 0.81, 0.435, 1.816,
        0.81, 0.435, 1.816,
    ];
    // superficial vein to skin
    let mut cdt_sfv_sk = [
        0.0, 0.0, 0.0, 0.0, 0.0, 57.735, 37.768, 16.634, 57.735, 37.768, 16.634, 102.012, 54.784,
        24.277, 102.012, 54.784, 24.277,
    ];
    // art to vein (counter-flow) [W/K]
    let mut cdt_art_vein = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.537, 0.351, 0.762, 0.537, 0.351, 0.762, 0.826, 0.444, 0.992,
        0.826, 0.444, 0.992,
    ];

    // Changes values by body size based on the standard body.
    let wr = weight_rate(weight);
    let bsar = bsa_rate(height, weight, bsa_equation);
    // head, neck (sphere shape)
    let head_neck_factor = wr / bsar;
    for i in 0..2 {
        cdt_cr_sk[i] *= head_neck_factor;
        cdt_cr_ms[i] *= head_neck_factor;
        cdt_ms_fat[i] *= head_neck_factor;
        cdt_fat_sk[i] *= head_neck_factor;
        cdt_ves_cr[i] *= head_neck_factor;
        cdt_sfv_sk[i] *= head_neck_factor;
        cdt_art_vein[i] *= head_neck_factor;
    }
    // others (cylinder shape)
    let others_factor = bsar * bsar / wr;
    for i in 2..NUM_BODY_PARTS {
        cdt_cr_sk[i] *= others_factor;
        cdt_cr_ms[i] *= others_factor;
        cdt_ms_fat[i] *= others_factor;
        cdt_fat_sk[i] *= others_factor;
        cdt_ves_cr[i] *= others_factor;
        cdt_sfv_sk[i] *= others_factor;
        cdt_art_vein[i] *= others_factor;
    }

    let mut cdt_whole = vec![0.0_f64; NUM_NODES * NUM_NODES];
    for (i, li) in IDICT.iter().enumerate() {
        let artery = li.artery.expect("every body segment has an artery node");
        let vein = li.vein.expect("every body segment has a vein node");
        let core = li.core.expect("every body segment has a core node");
        let skin = li.skin.expect("every body segment has a skin node");

        // Common
        cdt_whole[mat_index(artery, vein)] = cdt_art_vein[i]; // art to vein
        cdt_whole[mat_index(artery, core)] = cdt_ves_cr[i]; // art to cr
        cdt_whole[mat_index(vein, core)] = cdt_ves_cr[i]; // vein to cr

        // Only limbs
        if i >= 5 {
            if let Some(sfvein) = li.sfvein {
                cdt_whole[mat_index(sfvein, skin)] = cdt_sfv_sk[i]; // sfv to sk
            }
        }

        // If the segment has a muscle or fat layer
        if let Some(muscle) = li.muscle {
            let fat_layer = li.fat.expect("body segments with muscle also have fat");
            cdt_whole[mat_index(core, muscle)] = cdt_cr_ms[i]; // cr to ms
            cdt_whole[mat_index(muscle, fat_layer)] = cdt_ms_fat[i]; // ms to fat
            cdt_whole[mat_index(fat_layer, skin)] = cdt_fat_sk[i]; // fat to sk
        } else {
            cdt_whole[mat_index(core, skin)] = cdt_cr_sk[i]; // cr to sk
        }
    }

    // Creates a symmetrical matrix: cdt_whole = cdt_whole + cdt_whole^T
    let mut result = vec![0.0_f64; NUM_NODES * NUM_NODES];
    for r in 0..NUM_NODES {
        for c in 0..NUM_NODES {
            result[mat_index(r, c)] = cdt_whole[mat_index(r, c)] + cdt_whole[mat_index(c, r)];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// capacity
// ---------------------------------------------------------------------------

/// Calculate the thermal capacity \[J/K\].
///
/// Python: `capacity(height, weight, bsa_equation, age, ci)`.
///
/// Returns a `NUM_NODES`-length (85) vector.
#[must_use]
pub(crate) fn capacity(
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    ci: f64,
) -> Vec<f64> {
    // Define capacities [Wh/K].
    let mut cap_art = [
        0.096, 0.025, 0.12, 0.111, 0.265, 0.0186, 0.0091, 0.0044, 0.0186, 0.0091, 0.0044, 0.0813,
        0.04, 0.0103, 0.0813, 0.04, 0.0103,
    ]; // artery
    let mut cap_vein = [
        0.321, 0.085, 0.424, 0.39, 0.832, 0.046, 0.024, 0.01, 0.046, 0.024, 0.01, 0.207, 0.1,
        0.024, 0.207, 0.1, 0.024,
    ]; // vein
    let mut cap_sfv = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.015, 0.011, 0.025, 0.015, 0.011, 0.074, 0.05, 0.021,
        0.074, 0.05, 0.021,
    ]; // superficial vein
    let mut cap_cb = 1.999_f64; // central blood
    let mut cap_cr = [
        1.7229, 0.564, 10.2975, 9.3935, 4.488, 1.6994, 1.1209, 0.1536, 1.6994, 1.1209, 0.1536,
        5.3117, 2.867, 0.2097, 5.3117, 2.867, 0.2097,
    ]; // core
    let mut cap_ms = [
        0.305, 0.0, 0.0, 0.0, 7.409, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]; // muscle
    let mut cap_fat = [
        0.203, 0.0, 0.0, 0.0, 1.947, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]; // fat
    let mut cap_sk = [
        0.1885, 0.058, 0.441, 0.406, 0.556, 0.126, 0.084, 0.088, 0.126, 0.084, 0.088, 0.334, 0.169,
        0.107, 0.334, 0.169, 0.107,
    ]; // skin

    // Adjust capacities based on body parameters.
    let bfbr = bfb_rate(height, weight, bsa_equation, age, ci);
    let wr = weight_rate(weight);
    for v in &mut cap_art {
        *v *= bfbr;
    }
    for v in &mut cap_vein {
        *v *= bfbr;
    }
    for v in &mut cap_sfv {
        *v *= bfbr;
    }
    cap_cb *= bfbr;
    for v in &mut cap_cr {
        *v *= wr;
    }
    for v in &mut cap_ms {
        *v *= wr;
    }
    for v in &mut cap_fat {
        *v *= wr;
    }
    for v in &mut cap_sk {
        *v *= wr;
    }

    // Initialize capacity array.
    let mut cap_whole = vec![0.0_f64; NUM_NODES];
    cap_whole[0] = cap_cb;

    for (i, li) in IDICT.iter().enumerate() {
        let artery = li.artery.expect("every body segment has an artery node");
        let vein = li.vein.expect("every body segment has a vein node");
        let core = li.core.expect("every body segment has a core node");
        let skin = li.skin.expect("every body segment has a skin node");

        // Common
        cap_whole[artery] = cap_art[i];
        cap_whole[vein] = cap_vein[i];
        cap_whole[core] = cap_cr[i];
        cap_whole[skin] = cap_sk[i];

        // Only limbs
        if i >= 5 {
            if let Some(sfvein) = li.sfvein {
                cap_whole[sfvein] = cap_sfv[i];
            }
        }

        // If the segment has a muscle or fat layer
        if let Some(muscle) = li.muscle {
            let fat_layer = li.fat.expect("body segments with muscle also have fat");
            cap_whole[muscle] = cap_ms[i];
            cap_whole[fat_layer] = cap_fat[i];
        }
    }

    for v in &mut cap_whole {
        *v *= 3600.0; // Convert [Wh/K] to [J/K]
    }
    cap_whole
}

#[cfg(test)]
// Reference values are pasted verbatim from the Python oracle, and rustfmt groups their
// fractional digits but not the integer part. Regrouping by hand to satisfy the lint would
// mean editing numbers whose whole value is that they were not edited.
#[allow(clippy::inconsistent_digit_grouping)]
mod tests {
    use super::*;
    use crate::{BodyFat, CardiacIndex};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -- validate_body_parameters --

    #[test]
    fn validate_body_parameters_accepts_in_range() {
        assert_eq!(validate_body_parameters(1.80, 75.0, 30, 20.0), Ok(()));
        // Boundary values, inclusive both ends.
        assert_eq!(validate_body_parameters(0.5, 20.0, 5, 1.0), Ok(()));
        assert_eq!(validate_body_parameters(3.0, 200.0, 100, 90.0), Ok(()));
    }

    #[test]
    fn validate_body_parameters_rejects_out_of_range() {
        // Cross-checked against Python:
        // validate_body_parameters(height=0.4, weight=70, age=30, body_fat=20)
        //   -> ValueError("Height must be in the range [0.5, 3.0] meters.")
        assert_eq!(
            validate_body_parameters(0.4, 70.0, 30, 20.0),
            Err(BodyParameterError::Height)
        );
        // validate_body_parameters(height=1.7, weight=10, age=30, body_fat=20)
        //   -> ValueError("Weight must be in the range [20.0, 200.0] kilograms.")
        assert_eq!(
            validate_body_parameters(1.7, 10.0, 30, 20.0),
            Err(BodyParameterError::Weight)
        );
        // validate_body_parameters(height=1.7, weight=70, age=200, body_fat=20)
        //   -> ValueError("Age must be in the range [5, 100] years.")
        assert_eq!(
            validate_body_parameters(1.7, 70.0, 200, 20.0),
            Err(BodyParameterError::Age)
        );
        // validate_body_parameters(height=1.7, weight=70, age=30, body_fat=0.5)
        //   -> ValueError("Body fat percentage must be in the range [1, 90] (1% to 90%).")
        assert_eq!(
            validate_body_parameters(1.7, 70.0, 30, 0.5),
            Err(BodyParameterError::BodyFat)
        );
    }

    // -- to_array_body_parts --

    #[test]
    fn to_array_body_parts_scalar_broadcasts() {
        // Python: to_array_body_parts(5) -> array([5, 5, ..., 5]) (17 times)
        assert_eq!(to_array_body_parts_scalar(5.0), [5.0; NUM_BODY_PARTS]);
    }

    #[test]
    fn to_array_body_parts_by_name_matches_body_part_order() {
        let pairs: Vec<(&str, f64)> = BODY_PART_NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| (*name, i as f64))
            .collect();
        let expected: [f64; NUM_BODY_PARTS] = core::array::from_fn(|i| i as f64);
        assert_eq!(to_array_body_parts_by_name(&pairs), Ok(expected));
    }

    #[test]
    fn to_array_body_parts_by_name_rejects_missing_key() {
        let pairs: Vec<(&str, f64)> = BODY_PART_NAMES[1..]
            .iter()
            .map(|name| (*name, 1.0))
            .collect();
        assert_eq!(
            to_array_body_parts_by_name(&pairs),
            Err(BodyPartsInputError::MissingKey("head"))
        );
    }

    // -- bsa_rate / local_bsa / weight_rate / bfb_rate --
    //
    // Cross-checked against pythermalcomfort 4.4.0, for
    // height=1.80, weight=75.0, age=30, fat=20.0, ci=2.59, bsa_equation="dubois":
    //
    //   bsa_rate(1.80, 75, "dubois")            == 1.0374044359155161
    //   local_bsa(1.80, 75, "dubois")[0]         == 0.11411448795070678  (head)
    //   local_bsa(1.80, 75, "dubois")[16]        == 0.05809464841126891  (right_foot)
    //   weight_rate(75)                          == 1.007658202337767
    //   bfb_rate(1.80, 75, "dubois", 30, 2.59)   == 1.03843182403274

    #[test]
    fn bsa_rate_matches_python() {
        let got = bsa_rate(1.80, 75.0, BsaFormula::DuBois);
        assert!(approx_eq(got, 1.037_404_435_915_516_1, 1e-12));
    }

    #[test]
    fn local_bsa_matches_python() {
        let got = local_bsa(1.80, 75.0, BsaFormula::DuBois);
        assert!(approx_eq(got[0], 0.114_114_487_950_706_78, 1e-12)); // head
        assert!(approx_eq(got[16], 0.058_094_648_411_268_91, 1e-12)); // right_foot
    }

    #[test]
    fn weight_rate_matches_python() {
        let got = weight_rate(75.0);
        assert!(approx_eq(got, 1.007_658_202_337_767, 1e-12));
        // Default weight -> ratio of exactly 1.
        assert!(approx_eq(weight_rate(defaults::WEIGHT), 1.0, 1e-12));
    }

    #[test]
    fn bfb_rate_matches_python() {
        let got = bfb_rate(1.80, 75.0, BsaFormula::DuBois, 30, 2.59);
        assert!(approx_eq(got, 1.038_431_824_032_74, 1e-9));
    }

    // -- conductance --
    //
    // Cross-checked against Python's `conductance(1.80, 75.0, "dubois", 20.0)`,
    // an 85x85 matrix. Values below are `cdt[row, col]` (0-indexed, node 0 = CB):
    //
    //   cdt[3,4]   (head core -> muscle)   == 1.555093390861638
    //   cdt[4,5]   (head muscle -> fat)    == 12.842876211100924
    //   cdt[5,6]   (head fat -> skin)      == 15.548991256035666
    //   cdt[1,3]   (head artery -> core)   == 0.0   (no vessel-to-core term on head/neck)
    //   cdt[9,10]  (neck core -> skin)     == 0.863509072127418
    //   cdt[25,26] (L.shoulder art->vein)  == 0.5735314565426335
    //   cdt[25,28] (L.shoulder art->core)  == 0.6258648669161698
    //   cdt[27,29] (L.shoulder sfv->skin)  == 61.6626417942066
    //   cdt[28,29] (L.shoulder core->skin) == 1.5465056779771569
    //   matrix sum                         == 1552.1394809967724

    #[test]
    fn conductance_matches_python() {
        let cdt = conductance(1.80, 75.0, BsaFormula::DuBois, 20.0);
        assert_eq!(cdt.len(), NUM_NODES * NUM_NODES);

        assert!(approx_eq(cdt[mat_index(3, 4)], 1.555_093_390_861_638, 1e-9));
        assert!(approx_eq(
            cdt[mat_index(4, 5)],
            12.842_876_211_100_924,
            1e-9
        ));
        assert!(approx_eq(
            cdt[mat_index(5, 6)],
            15.548_991_256_035_666,
            1e-9
        ));
        assert!(approx_eq(cdt[mat_index(1, 3)], 0.0, 1e-12));
        assert!(approx_eq(
            cdt[mat_index(9, 10)],
            0.863_509_072_127_418,
            1e-9
        ));
        assert!(approx_eq(
            cdt[mat_index(25, 26)],
            0.573_531_456_542_633_5,
            1e-9
        ));
        assert!(approx_eq(
            cdt[mat_index(25, 28)],
            0.625_864_866_916_169_8,
            1e-9
        ));
        assert!(approx_eq(
            cdt[mat_index(27, 29)],
            61.662_641_794_206_6,
            1e-8
        ));
        assert!(approx_eq(
            cdt[mat_index(28, 29)],
            1.546_505_677_977_156_9,
            1e-9
        ));

        let sum: f64 = cdt.iter().sum();
        assert!(approx_eq(sum, 1552.139_480_996_772_4, 1e-6));
    }

    #[test]
    fn conductance_is_symmetric() {
        let cdt = conductance(1.80, 75.0, BsaFormula::DuBois, 20.0);
        for r in 0..NUM_NODES {
            for c in 0..NUM_NODES {
                assert!(approx_eq(cdt[mat_index(r, c)], cdt[mat_index(c, r)], 1e-12));
            }
        }
    }

    // -- capacity --
    //
    // Cross-checked against Python's `capacity(1.80, 75.0, "dubois", 30, 2.59)`,
    // an 85-length vector (index 0 = CB):
    //
    //   cap[0]  (CB)                == 7472.970778469211
    //   cap[1]  (head artery)       == 358.88203838571496
    //   cap[2]  (head vein)         == 1200.0118158522343
    //   cap[3]  (head core)         == 6249.939540507859
    //   cap[4]  (head muscle)       == 1106.4087061668681
    //   cap[5]  (head fat)          == 736.3966142684402
    //   cap[6]  (head skin)         == 683.7968561064087
    //   cap[9]  (neck core)         == 2045.9492140266018
    //   cap[10] (neck skin)         == 210.39903264812574
    //   cap[25] (L.shoulder artery) == 69.53339493723226
    //   cap[27] (L.shoulder sfvein) == 93.4588641629466
    //   cap[29] (L.shoulder skin)   == 457.07376058041115
    //   vector sum                  == 250008.5857619019

    #[test]
    fn capacity_matches_python() {
        let cap = capacity(1.80, 75.0, BsaFormula::DuBois, 30, 2.59);
        assert_eq!(cap.len(), NUM_NODES);

        assert!(approx_eq(cap[0], 7472.970_778_469_211, 1e-6));
        assert!(approx_eq(cap[1], 358.882_038_385_714_96, 1e-6));
        assert!(approx_eq(cap[2], 1200.011_815_852_234_3, 1e-6));
        assert!(approx_eq(cap[3], 6249.939_540_507_859, 1e-6));
        assert!(approx_eq(cap[4], 1106.408_706_166_868_1, 1e-6));
        assert!(approx_eq(cap[5], 736.396_614_268_440_2, 1e-6));
        assert!(approx_eq(cap[6], 683.796_856_106_408_7, 1e-6));
        assert!(approx_eq(cap[9], 2045.949_214_026_601_8, 1e-6));
        assert!(approx_eq(cap[10], 210.399_032_648_125_74, 1e-6));
        assert!(approx_eq(cap[25], 69.533_394_937_232_26, 1e-6));
        assert!(approx_eq(cap[27], 93.458_864_162_946_6, 1e-6));
        assert!(approx_eq(cap[29], 457.073_760_580_411_15, 1e-6));

        let sum: f64 = cap.iter().sum();
        assert!(approx_eq(sum, 250_008.585_761_901_9, 1e-3));
    }

    // -- defaults sanity cross-check --
    //
    // Cross-checked against Python with Default.height/weight/age/cardiac_index:
    //   bsa_rate    == 1.0005194360290224
    //   weight_rate == 1.0
    //   bfb_rate    == 1.0015102952773929
    //   conductance(...).sum() == 1461.9758045286394
    //   capacity(...).sum()    == 247459.6008989836

    #[test]
    fn defaults_cross_check() {
        // `Default.cardiac_index`/`Default.body_fat` are ported as
        // `crate::CardiacIndex::DEFAULT`/`crate::BodyFat::default()`, not as
        // `defaults::` constants (this module's `defaults` no longer duplicates them).
        let cardiac_index = CardiacIndex::DEFAULT.as_liters_per_minute_per_square_meter();
        let body_fat = BodyFat::default().as_percent();

        let bsar = bsa_rate(defaults::HEIGHT, defaults::WEIGHT, BsaFormula::DuBois);
        assert!(approx_eq(bsar, 1.000_519_436_029_022_4, 1e-9));

        let wr = weight_rate(defaults::WEIGHT);
        assert!(approx_eq(wr, 1.0, 1e-12));

        let bfbr = bfb_rate(
            defaults::HEIGHT,
            defaults::WEIGHT,
            BsaFormula::DuBois,
            defaults::AGE,
            cardiac_index,
        );
        assert!(approx_eq(bfbr, 1.001_510_295_277_392_9, 1e-9));

        let cdt = conductance(
            defaults::HEIGHT,
            defaults::WEIGHT,
            BsaFormula::DuBois,
            body_fat,
        );
        let cdt_sum: f64 = cdt.iter().sum();
        assert!(approx_eq(cdt_sum, 1461.975_804_528_639_4, 1e-6));

        let cap = capacity(
            defaults::HEIGHT,
            defaults::WEIGHT,
            BsaFormula::DuBois,
            defaults::AGE,
            cardiac_index,
        );
        let cap_sum: f64 = cap.iter().sum();
        assert!(approx_eq(cap_sum, 247_459.600_898_983_6, 1e-3));
    }
}
