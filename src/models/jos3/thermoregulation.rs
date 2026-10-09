//! Thermoregulation physiology for the JOS3 model.
//!
//! Mirrors:
//! `pythermalcomfort/jos3_functions/thermoregulation.py`
//! (pythermalcomfort 4.4.0, 1575 lines).
//!
//! Covers basal metabolism, shivering, non-shivering thermogenesis, sweating,
//! vasomotion (skin and AVA blood flow), convective/radiative/evaporative heat
//! transfer coefficients, and respiratory heat loss.
//!
//! # Ported
//!
//! Every function in `thermoregulation.py` is ported, except `tetens` (see "Not
//! ported" below). Function names match the Python names 1:1 (see each function's
//! doc comment for the exact Python name), and each function's parameters are kept in
//! Python's order, so the files diff side by side.
//!
//! # Not ported
//!
//! - **`tetens`.** Dead code in the Python source too: `thermoregulation.py` defines
//!   `tetens(x)` (a saturated-vapor-pressure formula) but never calls it anywhere in
//!   that module or `jos3.py`; [`evaporation`] uses [`crate::utilities::antoine`]
//!   instead, matching Python's actual (not merely documented) behavior.
//!
//! # Deviations from the Python source
//!
//! - **`posture` is a [`Posture`] enum, not a `&str`.** Python's `natural_convection`,
//!   `conv_coef`, and `rad_coef` take a `posture: str` and raise `ValueError` for an
//!   unrecognized value. This crate follows the same pattern already used for
//!   `sex`/`bmr_equation`/`bsa_equation` elsewhere in this port (see
//!   [`crate::Sex`], [`crate::BmrEquation`], [`crate::utilities::BsaFormula`],
//!   and `models::pet::Posture` / `models::phs::PhsPosture` for other per-model
//!   `Posture` enums in this crate): the type system makes the invalid-value case
//!   unrepresentable, so there is no runtime error to model. Python's `sitting` and
//!   `sedentary` select the same coefficient table, as do `lying` and `supine`; like
//!   [`crate::BmrEquation::Japanese`] collapsing `"japanese"`/`"ganpule"`, those pairs
//!   become the single variants [`Posture::Sitting`] and [`Posture::Lying`].
//! - **All per-body-segment quantities are `[f64; 17]`, not "float or array".**
//!   Python's docstrings advertise `float | np.ndarray` for these parameters, but
//!   every call site in `jos3.py` (the top-level solver, not yet ported) always
//!   passes an already-17-long array (broadcast once at input-parsing time via
//!   `to_array_body_parts`, already ported as
//!   [`super::construction::to_array_body_parts_scalar`] and friends). There is
//!   nothing left in this port for these functions to broadcast internally.
//! - **`shivering`'s `PRE_SHIV` is a function argument/return value, not a mutable
//!   module global.** Python keeps `PRE_SHIV` (the previous timestep's shivering
//!   signal) as a `global` variable mutated in place by [`shivering`]; besides being
//!   awkward in `no_std` (no ergonomic mutable static without `unsafe`), a module
//!   global would leak state between independent simulations. [`shivering`] instead
//!   takes `pre_shiv: f64` and returns `(q_shiv, new_pre_shiv)`; callers thread
//!   `new_pre_shiv` into the next timestep's call themselves.
//! - **`sum_m` / `cr_ms_fat_blood_flow`'s muscle-layer check is a local constant, not
//!   `matrix.py`'s `IDICT`.** Python checks `IDICT[body_name]["muscle"] is not None`
//!   to decide whether work/shivering thermogenesis (or blood flow) adds to the
//!   muscle layer or the core layer. Only `head` and `pelvis` have a muscle layer in
//!   the fixed 17-segment JOS3 body plan (visible in `local_mbase`'s `mbf_ms` table
//!   and in [`super::construction`]'s `cap_ms`/`bfb_muscle` tables, which are zero
//!   everywhere else), so this port hardcodes that as [`HAS_MUSCLE`] rather than
//!   depending on `matrix.rs`'s `IDICT` (owned by a different, concurrently-written
//!   part of this port) — the same reasoning [`super::construction::LAYER_INDEX_TABLE`]
//!   already documents for its own hardcoded copy of `IDICT`-derived data.

use libm::{fabs, pow};

use super::construction;
use super::matrix::IDICT;
use super::parameters::{NUM_BODY_PARTS, defaults};
use crate::utilities::{BsaFormula, antoine, np_maximum, np_minimum, py_max, py_min};
use crate::{BmrEquation, Sex, Temperature};

// ---------------------------------------------------------------------------
// NaN-faithful min/max
// ---------------------------------------------------------------------------
//
// JOS3 legitimately produces NaN trajectories: when a subject's reference-environment
// metabolic rate falls below ISO 7730's 0.8 met floor, the operative-temperature PMV
// search returns NaN, `_reset_setpt` seeds tdb/tr with NaN, and Python's whole
// simulation is NaN from there on. A clamp that silently drops the NaN would heal an
// invalid simulation into a plausible-looking wrong number, so every clamp below has
// to reproduce the NaN behaviour of the exact Python construct it ports.
//
// `libm::fmin`/`fmax` are IEEE-754/C99 `fmin`/`fmax`, which *discard* NaN and return
// the other operand (`fmin(NaN, 1.0) == 1.0`). That matches neither Python construct
// used here, so neither is imported: use `py_min`/`py_max` (CPython's order-dependent
// builtins) or `np_minimum`/`np_maximum` (always propagate) from [`crate::utilities`].

// ---------------------------------------------------------------------------
// Posture
// ---------------------------------------------------------------------------

/// Body posture, for the convective/radiative heat transfer coefficient tables.
///
/// Re-exported as [`crate::models::jos3::Jos3Posture`], which is also the type of
/// [`crate::models::jos3::Jos3Conditions::posture`] — JOS3 has coefficient tables for
/// exactly these three postures, so the public field is this enum. Every model with its own
/// coefficient table carries its own posture enum for the same reason.
///
/// Python: a `posture: str` argument to [`natural_convection`], [`conv_coef`], and
/// [`rad_coef`], validated against `pythermalcomfort.utilities.Postures`. See the
/// module docs for why this is an enum (and why `sitting`/`sedentary` and
/// `lying`/`supine` are collapsed into one variant each) rather than a string.
///
/// Narrowing the public field to this enum also removes a failure mode Python has no
/// equivalent for. `JOS3`'s `posture` setter (`models/jos3.py:1471-1491`) matches
/// `standing`/`sitting`/`sedentary`/`lying`/`supine` in its `elif isinstance(inp, str):`
/// branch and otherwise falls through doing nothing, silently leaving `self._posture` at
/// whatever it was last set to (the `else` branch that resets to `standing` and prints a
/// warning fires only for a non-string `inp`). Expressing the restriction in the type makes an
/// unsupported posture a compile error rather than a runtime one, and nothing is lost
/// because `sedentary` and `supine` are pure aliases upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Posture {
    /// Python: `Postures.standing.value` (`"standing"`).
    Standing,
    /// Python: `Postures.sitting.value` or `Postures.sedentary.value`
    /// (`"sitting"`/`"sedentary"`) — same coefficient table in Python.
    Sitting,
    /// Python: `Postures.lying.value` or `Postures.supine.value`
    /// (`"lying"`/`"supine"`) — same coefficient table in Python.
    Lying,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// An input to a thermoregulation function violated a physical precondition.
/// Python: these functions raise `ValueError` for each of these cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermoregulationError {
    /// [`dry_r`] was given a negative `hc` or `hr` value.
    /// Python: `ValueError("Input parameters hc and hr must be non-negative.")`.
    NegativeConvectiveOrRadiativeCoefficient,
    /// [`wet_r`] was given a negative `hc` value.
    /// Python: `ValueError("Input parameters hc must be non-negative.")`.
    NegativeConvectiveCoefficient,
    /// [`local_q_work`] was given `par < 1`.
    /// Python: `ValueError("par must be 1 or more")`.
    ParTooSmall,
}

impl core::fmt::Display for ThermoregulationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NegativeConvectiveOrRadiativeCoefficient => {
                write!(f, "Input parameters hc and hr must be non-negative.")
            }
            Self::NegativeConvectiveCoefficient => {
                write!(f, "Input parameters hc must be non-negative.")
            }
            Self::ParTooSmall => write!(f, "par must be 1 or more"),
        }
    }
}

impl core::error::Error for ThermoregulationError {}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Weighted average, matching `numpy.average(values, weights=weights)`.
///
/// `pub(super)`: reused by [`super::jos3`]'s `t_skin_mean`/`w_mean`-style getters, which
/// need the exact same averaging Python's `_run` and property getters use, rather than a
/// second hand-written copy.
pub(super) fn weighted_average(values: &[f64], weights: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (v, w) in values.iter().zip(weights.iter()) {
        numerator += v * w;
        denominator += w;
    }
    numerator / denominator
}

/// Whether each of the 17 body segments has a muscle (and fat) layer.
///
/// Python asks `IDICT[body_name]["muscle"] is not None` at each use. Derived here from the
/// same `IDICT`, at compile time, rather than restated as a literal: [`matrix`] owns the
/// node layout, and a second hand-maintained copy of a fact about it is exactly what was
/// removed from `construction.rs`. As it happens only `head` and `pelvis` have one.
///
/// [`matrix`]: super::matrix
const HAS_MUSCLE: [bool; NUM_BODY_PARTS] = {
    let mut has = [false; NUM_BODY_PARTS];
    let mut i = 0;
    while i < NUM_BODY_PARTS {
        has[i] = IDICT[i].muscle.is_some();
        i += 1;
    }
    has
};

// ---------------------------------------------------------------------------
// natural_convection / forced_convection / conv_coef / rad_coef
// ---------------------------------------------------------------------------

/// Calculate the natural convection heat transfer coefficient based on posture.
/// Python: `natural_convection(posture, tdb, t_skin)`.
#[must_use]
pub(crate) fn natural_convection(
    posture: Posture,
    tdb: [f64; NUM_BODY_PARTS],
    t_skin: [f64; NUM_BODY_PARTS],
) -> [f64; NUM_BODY_PARTS] {
    match posture {
        // Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        Posture::Standing => [
            4.48, 4.48, 2.97, 2.91, 2.85, 3.61, 3.55, 3.67, 3.61, 3.55, 3.67, 2.80, 2.04, 2.04,
            2.80, 2.04, 2.04,
        ],
        // Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        Posture::Sitting => [
            4.75, 4.75, 3.12, 2.48, 1.84, 3.76, 3.62, 2.06, 3.76, 3.62, 2.06, 2.98, 2.98, 2.62,
            2.98, 2.98, 2.62,
        ],
        // Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17
        // The values are applied under cold environment.
        Posture::Lying => {
            const HC_A: [f64; NUM_BODY_PARTS] = [
                1.105, 1.105, 1.211, 1.211, 1.211, 0.913, 2.081, 2.178, 0.913, 2.081, 2.178, 0.945,
                0.385, 0.200, 0.945, 0.385, 0.200,
            ];
            const HC_B: [f64; NUM_BODY_PARTS] = [
                0.345, 0.345, 0.046, 0.046, 0.046, 0.373, 0.850, 0.297, 0.373, 0.850, 0.297, 0.447,
                0.580, 0.966, 0.447, 0.580, 0.966,
            ];
            let mut hc_natural = [0.0; NUM_BODY_PARTS];
            for i in 0..NUM_BODY_PARTS {
                hc_natural[i] = HC_A[i] * pow(fabs(tdb[i] - t_skin[i]), HC_B[i]);
            }
            hc_natural
        }
    }
}

/// Calculate the forced convection heat transfer coefficient.
///
/// Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
///
/// Python: `forced_convection(v)`.
#[must_use]
pub(crate) fn forced_convection(v: [f64; NUM_BODY_PARTS]) -> [f64; NUM_BODY_PARTS] {
    const HC_A: [f64; NUM_BODY_PARTS] = [
        15.0, 15.0, 11.0, 17.0, 13.0, 17.0, 17.0, 20.0, 17.0, 17.0, 20.0, 14.0, 15.8, 15.1, 14.0,
        15.8, 15.1,
    ];
    const HC_B: [f64; NUM_BODY_PARTS] = [
        0.62, 0.62, 0.67, 0.49, 0.60, 0.59, 0.61, 0.60, 0.59, 0.61, 0.60, 0.61, 0.74, 0.62, 0.61,
        0.74, 0.62,
    ];
    let mut hc_forced = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        hc_forced[i] = HC_A[i] * pow(v[i], HC_B[i]);
    }
    hc_forced
}

/// Calculate convective heat transfer coefficient (hc) \[W/(m2*K)\].
///
/// Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5;
/// Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17.
///
/// Python: `conv_coef(posture, v, tdb, t_skin)`.
#[must_use]
pub(crate) fn conv_coef(
    posture: Posture,
    v: [f64; NUM_BODY_PARTS],
    tdb: [f64; NUM_BODY_PARTS],
    t_skin: [f64; NUM_BODY_PARTS],
) -> [f64; NUM_BODY_PARTS] {
    let hc_natural = natural_convection(posture, tdb, t_skin);
    let hc_forced = forced_convection(v);
    let mut hc = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        hc[i] = if v[i] < 0.2 {
            hc_natural[i]
        } else {
            hc_forced[i]
        };
    }
    hc
}

/// Calculate radiative heat transfer coefficient (hr) \[W/(m2*K)\].
/// Python: `rad_coef(posture)`.
#[must_use]
pub(crate) fn rad_coef(posture: Posture) -> [f64; NUM_BODY_PARTS] {
    match posture {
        // Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        Posture::Standing => [
            4.89, 4.89, 4.32, 4.09, 4.32, 4.55, 4.43, 4.21, 4.55, 4.43, 4.21, 4.77, 5.34, 6.14,
            4.77, 5.34, 6.14,
        ],
        // Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        Posture::Sitting => [
            4.96, 4.96, 3.99, 4.64, 4.21, 4.96, 4.21, 4.74, 4.96, 4.21, 4.74, 4.10, 4.74, 6.36,
            4.10, 4.74, 6.36,
        ],
        // Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17
        Posture::Lying => [
            5.475, 5.475, 3.463, 3.463, 3.463, 4.249, 4.835, 4.119, 4.249, 4.835, 4.119, 4.440,
            5.547, 6.085, 4.440, 5.547, 6.085,
        ],
    }
}

/// Fix hc values to fit two-node-model's values.
/// Python: `fixed_hc(hc, v)`.
#[must_use]
pub(crate) fn fixed_hc(
    hc: [f64; NUM_BODY_PARTS],
    v: [f64; NUM_BODY_PARTS],
) -> [f64; NUM_BODY_PARTS] {
    let mean_hc = weighted_average(&hc, &defaults::LOCAL_BSA);
    let mean_va = weighted_average(&v, &defaults::LOCAL_BSA);
    let mean_hc_whole = py_max(3.0, 8.600_001 * pow(mean_va, 0.53));
    let mut fixed = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        fixed[i] = hc[i] * mean_hc_whole / mean_hc;
    }
    fixed
}

/// Fix hr values to fit two-node-model's values.
/// Python: `fixed_hr(hr)`.
#[must_use]
pub(crate) fn fixed_hr(hr: [f64; NUM_BODY_PARTS]) -> [f64; NUM_BODY_PARTS] {
    let mean_hr = weighted_average(&hr, &defaults::LOCAL_BSA);
    let mut fixed = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        fixed[i] = hr[i] * 4.7 / mean_hr;
    }
    fixed
}

// ---------------------------------------------------------------------------
// operative_temp / clo_area_factor / dry_r / wet_r
// ---------------------------------------------------------------------------

/// Calculate operative temperature \[°C\].
/// Python: `operative_temp(tdb, tr, hc, hr)`.
// TODO this function is a duplicate in utils (matches upstream's own TODO)
#[must_use]
pub(crate) fn operative_temp(
    tdb: [f64; NUM_BODY_PARTS],
    tr: [f64; NUM_BODY_PARTS],
    hc: [f64; NUM_BODY_PARTS],
    hr: [f64; NUM_BODY_PARTS],
) -> [f64; NUM_BODY_PARTS] {
    let mut to = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        to[i] = (hc[i] * tdb[i] + hr[i] * tr[i]) / (hc[i] + hr[i]);
    }
    to
}

/// Calculate clothing area factor.
/// Python: `clo_area_factor(clo)`.
// TODO this function is different from ISO 9920 (matches upstream's own TODO)
#[must_use]
pub(crate) fn clo_area_factor(clo: [f64; NUM_BODY_PARTS]) -> [f64; NUM_BODY_PARTS] {
    let mut fcl = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        fcl[i] = if clo[i] < 0.5 {
            clo[i] * 0.2 + 1.0
        } else {
            clo[i] * 0.1 + 1.05
        };
    }
    fcl
}

/// Calculate total sensible thermal resistance (between the skin and ambient air).
/// Python: `dry_r(hc, hr, clo)`.
///
/// # Errors
///
/// Returns [`ThermoregulationError::NegativeConvectiveOrRadiativeCoefficient`] if any
/// `hc` or `hr` value is negative.
pub(crate) fn dry_r(
    hc: [f64; NUM_BODY_PARTS],
    hr: [f64; NUM_BODY_PARTS],
    clo: [f64; NUM_BODY_PARTS],
) -> Result<[f64; NUM_BODY_PARTS], ThermoregulationError> {
    if hc.iter().any(|&v| v < 0.0) || hr.iter().any(|&v| v < 0.0) {
        return Err(ThermoregulationError::NegativeConvectiveOrRadiativeCoefficient);
    }
    let fcl = clo_area_factor(clo);
    let mut r_t = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        let r_a = 1.0 / (hc[i] + hr[i]);
        let r_cl = 0.155 * clo[i];
        r_t[i] = r_a / fcl[i] + r_cl;
    }
    Ok(r_t)
}

/// Calculate total evaporative thermal resistance (between the skin and ambient air).
/// Python: `wet_r(hc, clo, i_clo, lewis_rate)`.
///
/// # Errors
///
/// Returns [`ThermoregulationError::NegativeConvectiveCoefficient`] if any `hc` value
/// is negative.
pub(crate) fn wet_r(
    hc: [f64; NUM_BODY_PARTS],
    clo: [f64; NUM_BODY_PARTS],
    i_clo: [f64; NUM_BODY_PARTS],
    lewis_rate: f64,
) -> Result<[f64; NUM_BODY_PARTS], ThermoregulationError> {
    if hc.iter().any(|&v| v < 0.0) {
        return Err(ThermoregulationError::NegativeConvectiveCoefficient);
    }
    let fcl = clo_area_factor(clo);
    let mut r_et = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        let r_cl = 0.155 * clo[i];
        let r_ea = 1.0 / (lewis_rate * hc[i]);
        let r_ecl = r_cl / (lewis_rate * i_clo[i]);
        r_et[i] = r_ea / fcl[i] + r_ecl;
    }
    Ok(r_et)
}

// ---------------------------------------------------------------------------
// error_signals / evaporation
// ---------------------------------------------------------------------------

/// Calculate WRMS and CLDS signals of thermoregulation.
/// Python: `error_signals(err_sk)`. Returns `(wrms, clds)`.
#[must_use]
pub(crate) fn error_signals(err_sk: [f64; NUM_BODY_PARTS]) -> (f64, f64) {
    // SKINR (Distribution coefficients of thermal receptor) [-]
    const RECEPTOR: [f64; NUM_BODY_PARTS] = [
        0.0549, 0.0146, 0.1492, 0.1321, 0.2122, 0.0227, 0.0117, 0.0923, 0.0227, 0.0117, 0.0923,
        0.0501, 0.0251, 0.0167, 0.0501, 0.0251, 0.0167,
    ];
    let mut warm_signal_sum = 0.0;
    let mut cold_signal_sum = 0.0;
    for i in 0..NUM_BODY_PARTS {
        warm_signal_sum += np_maximum(err_sk[i], 0.0) * RECEPTOR[i];
        cold_signal_sum += -np_minimum(err_sk[i], 0.0) * RECEPTOR[i];
    }
    (warm_signal_sum, cold_signal_sum)
}

/// Calculate evaporative heat loss.
/// Python: `evaporation(err_cr, err_sk, t_skin, tdb, rh, ret, height, weight,
/// bsa_equation, age)`.
///
/// Returns `(wet, e_sk, e_max, e_sweat)`:
/// - `wet`: local skin wettedness \[-\].
/// - `e_sk`: evaporative heat loss at the skin by sweating and diffuse \[W\].
/// - `e_max`: maximum evaporative heat loss at the skin \[W\].
/// - `e_sweat`: evaporative heat loss at the skin by only sweating \[W\].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaporation(
    err_cr: [f64; NUM_BODY_PARTS],
    err_sk: [f64; NUM_BODY_PARTS],
    t_skin: [f64; NUM_BODY_PARTS],
    tdb: [f64; NUM_BODY_PARTS],
    rh: [f64; NUM_BODY_PARTS],
    ret: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
) -> (
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
) {
    let (wrms, clds) = error_signals(err_sk); // Thermoregulation signals
    let bsar = construction::bsa_rate(height, weight, bsa_equation); // bsa rate
    let mut bsa = defaults::LOCAL_BSA; // bsa
    for v in &mut bsa {
        *v *= bsar;
    }

    let mut e_max = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        let p_a = antoine(Temperature::from_celsius(tdb[i])) * rh[i] / 100.0; // Saturated vapor pressure of ambient [kPa]
        let p_sk_s = antoine(Temperature::from_celsius(t_skin[i])); // Saturated vapor pressure at the skin [kPa]
        let mut e_max_i = (p_sk_s - p_a) / ret[i] * bsa[i]; // Maximum evaporative heat loss
        // Replace any zero values with 0.001 to avoid causing a divide by 0 error
        if e_max_i == 0.0 {
            e_max_i = 0.001;
        }
        e_max[i] = e_max_i;
    }

    // SKINS
    const SKIN_SWEAT: [f64; NUM_BODY_PARTS] = [
        0.064, 0.017, 0.146, 0.129, 0.206, 0.051, 0.026, 0.0155, 0.051, 0.026, 0.0155, 0.073,
        0.036, 0.0175, 0.073, 0.036, 0.0175,
    ];

    let mut sig_sweat = (371.2 * err_cr[0]) + (33.64 * (wrms - clds));
    sig_sweat = py_max(sig_sweat, 0.0);
    sig_sweat *= bsar;

    // Signal decrement by aging
    let sd_sweat: [f64; NUM_BODY_PARTS] = if age < 60 {
        [1.0; NUM_BODY_PARTS]
    } else {
        // age >= 60
        [
            0.69, 0.69, 0.59, 0.52, 0.40, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.40, 0.40, 0.40,
            0.40, 0.40, 0.40,
        ]
    };

    let mut wet = [0.0; NUM_BODY_PARTS];
    let mut e_sk = [0.0; NUM_BODY_PARTS];
    let mut e_sweat = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        let e_sweat_i = SKIN_SWEAT[i] * sig_sweat * sd_sweat[i] * pow(2.0, err_sk[i] / 10.0);
        let mut wet_i = 0.06 + 0.94 * (e_sweat_i / e_max[i]);
        wet_i = np_minimum(wet_i, 1.0); // Wettedness' upper limit
        wet[i] = wet_i;
        e_sk[i] = wet_i * e_max[i];
        e_sweat[i] = (wet_i - 0.06) / 0.94 * e_max[i]; // Effective sweating
    }
    (wet, e_sk, e_max, e_sweat)
}

// ---------------------------------------------------------------------------
// skin_blood_flow / ava_blood_flow
// ---------------------------------------------------------------------------

/// Calculate skin blood flow rate (bf_skin) \[L/h\].
/// Python: `skin_blood_flow(err_cr, err_sk, height, weight, bsa_equation, age, ci)`.
#[must_use]
pub(crate) fn skin_blood_flow(
    err_cr: [f64; NUM_BODY_PARTS],
    err_sk: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    ci: f64,
) -> [f64; NUM_BODY_PARTS] {
    let (wrms, clds) = error_signals(err_sk);

    // BFBsk
    const BFB_SK: [f64; NUM_BODY_PARTS] = [
        1.754, 0.325, 1.967, 1.475, 2.272, 0.91, 0.508, 1.114, 0.91, 0.508, 1.114, 1.456, 0.651,
        0.934, 1.456, 0.651, 0.934,
    ];
    // SKIND
    const SKIN_DILAT: [f64; NUM_BODY_PARTS] = [
        0.0692, 0.0992, 0.0580, 0.0679, 0.0707, 0.0400, 0.0373, 0.0632, 0.0400, 0.0373, 0.0632,
        0.0736, 0.0411, 0.0623, 0.0736, 0.0411, 0.0623,
    ];
    // SKINC
    const SKIN_STRIC: [f64; NUM_BODY_PARTS] = [
        0.0213, 0.0213, 0.0638, 0.0638, 0.0638, 0.0213, 0.0213, 0.1489, 0.0213, 0.0213, 0.1489,
        0.0213, 0.0213, 0.1489, 0.0213, 0.0213, 0.1489,
    ];

    let mut sig_dilat = (100.5 * err_cr[0]) + (6.4 * (wrms - clds));
    let mut sig_stric = (-10.8 * err_cr[0]) + (-10.8 * (wrms - clds));
    sig_dilat = py_max(sig_dilat, 0.0);
    sig_stric = py_max(sig_stric, 0.0);

    // Signal decrement by aging
    let (sd_dilat, sd_stric): ([f64; NUM_BODY_PARTS], [f64; NUM_BODY_PARTS]) = if age < 60 {
        ([1.0; NUM_BODY_PARTS], [1.0; NUM_BODY_PARTS])
    } else {
        // age >= 60
        (
            [
                0.91, 0.91, 0.47, 0.47, 0.31, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.31, 0.31, 0.31,
                0.31, 0.31, 0.31,
            ],
            [1.0; NUM_BODY_PARTS],
        )
    };

    // Skin blood flow [L/h]
    let mut bf_skin = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        bf_skin[i] = (1.0 + SKIN_DILAT[i] * sd_dilat[i] * sig_dilat)
            / (1.0 + SKIN_STRIC[i] * sd_stric[i] * sig_stric)
            * BFB_SK[i]
            * pow(2.0, err_sk[i] / 6.0);
    }
    // Basal blood flow rate to the standard body [-]
    let bfb_rate = construction::bfb_rate(height, weight, bsa_equation, age, ci);
    for v in &mut bf_skin {
        *v *= bfb_rate;
    }
    bf_skin
}

/// Calculate areteriovenous anastmoses (AVA) blood flow rate \[L/h\] based on
/// Takemori's model, 1995.
/// Python: `ava_blood_flow(err_cr, err_sk, height, weight, bsa_equation, age, ci)`.
///
/// Returns `(bf_ava_hand, bf_ava_foot)`: AVA blood flow rate at hand and foot \[L/h\].
#[must_use]
pub(crate) fn ava_blood_flow(
    err_cr: [f64; NUM_BODY_PARTS],
    err_sk: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    ci: f64,
) -> (f64, f64) {
    // Cal. mean error body core temp.
    const CAP_BCR: [f64; 3] = [10.2975, 9.3935, 4.488]; // Thermal capacity at chest, back and pelvis
    let err_bcr = weighted_average(&err_cr[2..5], &CAP_BCR);

    // Cal. mean error skin temp.
    let err_msk = weighted_average(&err_sk, &defaults::LOCAL_BSA);

    // Openness of AVA [-]
    let mut sig_ava_hand = 0.265 * (err_msk + 0.43) + 0.953 * (err_bcr + 0.1905) + 0.9126;
    let mut sig_ava_foot = 0.265 * (err_msk - 0.997) + 0.953 * (err_bcr + 0.0095) + 0.9126;

    sig_ava_hand = py_min(sig_ava_hand, 1.0);
    sig_ava_hand = py_max(sig_ava_hand, 0.0);
    sig_ava_foot = py_min(sig_ava_foot, 1.0);
    sig_ava_foot = py_max(sig_ava_foot, 0.0);

    // Basal blood flow rate to the standard body [-]
    let bfb_rate = construction::bfb_rate(height, weight, bsa_equation, age, ci);
    // AVA blood flow rate [L/h]
    let bf_ava_hand = 1.71 * bfb_rate * sig_ava_hand; // Hand
    let bf_ava_foot = 2.16 * bfb_rate * sig_ava_foot; // Foot
    (bf_ava_hand, bf_ava_foot)
}

// ---------------------------------------------------------------------------
// basal_met / local_mbase / local_q_work
// ---------------------------------------------------------------------------

/// Calculate basal metabolic rate \[W\].
/// Python: `basal_met(height, weight, age, sex, bmr_equation)`.
#[must_use]
pub(crate) fn basal_met(
    height: f64,
    weight: f64,
    age: i32,
    sex: Sex,
    bmr_equation: BmrEquation,
) -> f64 {
    let age = f64::from(age);
    let mut bmr = match bmr_equation {
        BmrEquation::HarrisBenedict => match sex {
            Sex::Male => 88.362 + 13.397 * weight + 500.3 * height - 5.677 * age,
            Sex::Female => 447.593 + 9.247 * weight + 479.9 * height - 4.330 * age,
        },
        BmrEquation::HarrisBenedictOriginal => match sex {
            Sex::Male => 66.4730 + 13.7516 * weight + 500.33 * height - 6.7550 * age,
            Sex::Female => 655.0955 + 9.5634 * weight + 184.96 * height - 4.6756 * age,
        },
        BmrEquation::Japanese => {
            // Ganpule et al., 2007, https://doi.org/10.1038/sj.ejcn.1602645
            let mut bmr = match sex {
                Sex::Male => 0.0481 * weight + 2.34 * height - 0.0138 * age - 0.4235,
                Sex::Female => 0.0481 * weight + 2.34 * height - 0.0138 * age - 0.9708,
            };
            bmr *= 1000.0 / 4.186;
            bmr
        }
    };

    bmr *= 0.048; // [kcal/day] to [W]

    // Set minimum BMR value in W
    let min_bmr_in_w = 68.0;
    py_max(bmr, min_bmr_in_w)
}

/// Calculate local basal metabolic rate \[W\].
/// Python: `local_mbase(height, weight, age, sex, bmr_equation)`.
///
/// Returns `(mbase_cr, mbase_ms, mbase_fat, mbase_sk)`: local basal metabolic rate
/// (Mbase) \[W\] for the core, muscle, fat, and skin layers.
#[must_use]
pub(crate) fn local_mbase(
    height: f64,
    weight: f64,
    age: i32,
    sex: Sex,
    bmr_equation: BmrEquation,
) -> (
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
) {
    let mbase_all = basal_met(height, weight, age, sex, bmr_equation);
    // Distribution coefficient of basal metabolic rate
    const MBF_CR: [f64; NUM_BODY_PARTS] = [
        0.19551, 0.00324, 0.28689, 0.25677, 0.09509, 0.01435, 0.00409, 0.00106, 0.01435, 0.00409,
        0.00106, 0.01557, 0.00422, 0.00250, 0.01557, 0.00422, 0.00250,
    ];
    const MBF_MS: [f64; NUM_BODY_PARTS] = [
        0.00252, 0.0, 0.0, 0.0, 0.04804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    const MBF_FAT: [f64; NUM_BODY_PARTS] = [
        0.00127, 0.0, 0.0, 0.0, 0.00950, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    const MBF_SK: [f64; NUM_BODY_PARTS] = [
        0.00152, 0.00033, 0.00211, 0.00187, 0.00300, 0.00059, 0.00031, 0.00059, 0.00059, 0.00031,
        0.00059, 0.00144, 0.00027, 0.00118, 0.00144, 0.00027, 0.00118,
    ];

    let mut mbase_cr = [0.0; NUM_BODY_PARTS];
    let mut mbase_ms = [0.0; NUM_BODY_PARTS];
    let mut mbase_fat = [0.0; NUM_BODY_PARTS];
    let mut mbase_sk = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        mbase_cr[i] = MBF_CR[i] * mbase_all;
        mbase_ms[i] = MBF_MS[i] * mbase_all;
        mbase_fat[i] = MBF_FAT[i] * mbase_all;
        mbase_sk[i] = MBF_SK[i] * mbase_all;
    }
    (mbase_cr, mbase_ms, mbase_fat, mbase_sk)
}

/// Calculate local thermogenesis by work \[W\].
/// Python: `local_q_work(bmr, par)`.
///
/// # Errors
///
/// Returns [`ThermoregulationError::ParTooSmall`] if `par < 1`.
pub(crate) fn local_q_work(
    bmr: f64,
    par: f64,
) -> Result<[f64; NUM_BODY_PARTS], ThermoregulationError> {
    if par < 1.0 {
        return Err(ThermoregulationError::ParTooSmall);
    }

    let q_work_all = (par - 1.0) * bmr;

    // Distribution coefficient of thermogenesis by work
    const WORKF: [f64; NUM_BODY_PARTS] = [
        0.0, 0.0, 0.091, 0.08, 0.129, 0.0262, 0.0139, 0.005, 0.0262, 0.0139, 0.005, 0.2010, 0.0990,
        0.005, 0.2010, 0.0990, 0.005,
    ];
    let mut q_work = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        q_work[i] = q_work_all * WORKF[i];
    }
    Ok(q_work)
}

// ---------------------------------------------------------------------------
// shivering / nonshivering
// ---------------------------------------------------------------------------

/// Options controlling [`shivering`]'s optional thresholding behavior.
/// Python: the `options` dict passed to `shivering`, with keys `"shivering_threshold"`
/// and `"limit_dshiv/dt"`. `None` (rather than `Some(_)`) corresponds to Python's
/// `options=None` (or a falsy/absent dict), which skips both behaviors entirely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShiveringOptions {
    /// Whether to zero the shivering signal above a skin/core-temperature-dependent
    /// threshold (Asaka, 2016). Python: `options["shivering_threshold"]`.
    pub shivering_threshold: bool,
    /// Limit on the rate of change of the shivering signal \[W/s\]
    /// (Asaka, 2016: dshiv/dt < 0.0077). `None` disables the clamp; `Some(rate)`
    /// enables it with the given rate. Python: `options["limit_dshiv/dt"]`, where
    /// `False` is this crate's `None`, the sentinel `True` (default 0.0077 W/s) is
    /// `Some(0.0077)`, and a custom float rate is `Some(rate)`.
    pub limit_dshiv_dt: Option<f64>,
}

/// Calculate local thermogenesis by shivering \[W\].
/// Python: `shivering(err_cr, err_sk, t_core, t_skin, height, weight, bsa_equation,
/// age, sex, dtime, options)`.
///
/// `pre_shiv` is the previous timestep's shivering signal; see the module docs for
/// why this is a parameter rather than Python's mutable module-level `PRE_SHIV`.
///
/// Returns `(q_shiv, new_pre_shiv)`: local thermogenesis by shivering \[W\], and the
/// `pre_shiv` value to pass to the next timestep's call.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shivering(
    err_cr: [f64; NUM_BODY_PARTS],
    err_sk: [f64; NUM_BODY_PARTS],
    t_core: [f64; NUM_BODY_PARTS],
    t_skin: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    sex: Sex,
    dtime: f64,
    options: Option<ShiveringOptions>,
    pre_shiv: f64,
) -> ([f64; NUM_BODY_PARTS], f64) {
    // Integrated error signal in the warm and cold receptors
    let (_wrms, clds) = error_signals(err_sk);

    // Distribution coefficient of thermogenesis by shivering
    const SHIVF: [f64; NUM_BODY_PARTS] = [
        0.0339, 0.0436, 0.27394, 0.24102, 0.38754, 0.00243, 0.00137, 0.0002, 0.00243, 0.00137,
        0.0002, 0.0039, 0.00175, 0.00035, 0.0039, 0.00175, 0.00035,
    ];
    // integrated error signal of shivering
    let mut sig_shiv = 24.36 * clds * (-err_cr[0]);
    sig_shiv = py_max(sig_shiv, 0.0);

    if let Some(opts) = options {
        if opts.shivering_threshold {
            // Asaka, 2016
            // Threshold of starting shivering
            let tskm = weighted_average(&t_skin, &defaults::LOCAL_BSA); // Mean skin temp.
            let thres = if tskm < 31.0 {
                36.6
            } else {
                match sex {
                    Sex::Male => -0.2436 * tskm + 44.10,
                    Sex::Female => -0.2250 * tskm + 43.05,
                }
            };
            // Second threshold of starting shivering
            if thres < t_core[0] {
                sig_shiv = 0.0;
            }
        }
    }

    // Previous shivering thermogenesis [W]
    let mut new_pre_shiv = pre_shiv;
    if let Some(opts) = options {
        if let Some(rate) = opts.limit_dshiv_dt {
            let dshiv = sig_shiv - pre_shiv; // Asaka, 2016 dshiv < 0.0077 [W/s]
            let limit_dshiv = rate * dtime;
            if dshiv > limit_dshiv {
                sig_shiv = limit_dshiv + pre_shiv;
            } else if dshiv < -limit_dshiv {
                sig_shiv = -limit_dshiv + pre_shiv;
            }
        }
        new_pre_shiv = sig_shiv;
    }

    // Signal sd_shiv by aging
    let sd_shiv = if age < 30 {
        1.0
    } else if age < 40 {
        0.97514
    } else if age < 50 {
        0.95028
    } else if age < 60 {
        0.92818
    } else if age < 70 {
        0.90055
    } else if age < 80 {
        0.86188
    } else {
        // age >= 80
        0.82597
    };

    // Ratio of body surface area to the standard body [-]
    let bsar = construction::bsa_rate(height, weight, bsa_equation);

    // Local thermogenesis by shivering [W]
    let mut q_shiv = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        q_shiv[i] = SHIVF[i] * bsar * sd_shiv * sig_shiv;
    }
    (q_shiv, new_pre_shiv)
}

/// Calculate local metabolic rate by non-shivering \[W\].
/// Python: `nonshivering(err_sk, height, weight, bsa_equation, age, cold_acclimation,
/// batpositive)`.
#[must_use]
pub(crate) fn nonshivering(
    err_sk: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    cold_acclimation: bool,
    batpositive: bool,
) -> [f64; NUM_BODY_PARTS] {
    // NST (Non-Shivering Thermogenesis) model, Asaka, 2016
    let (_wrms, clds) = error_signals(err_sk);

    // BMI (Body Mass Index)
    let bmi = weight / (height * height);

    // BAT: brown adipose tissue [SUV]
    let mut bat = pow(10.0, -0.10502 * bmi + 2.7708);

    // age factor
    if age < 30 {
        bat *= 1.61;
    } else if age < 40 {
        bat *= 1.00;
    } else {
        // age >= 40
        bat *= 0.80;
    }

    if cold_acclimation {
        bat += 3.46;
    }

    if !batpositive {
        // incidence age factor: T.Yoneshiro 2011
        if age < 30 {
            // age = 20s or younger
            bat *= 44.0 / 83.0;
        } else if age < 40 {
            // age = 30s
            bat *= 15.0 / 38.0;
        } else if age < 50 {
            // age = 40s
            bat *= 7.0 / 26.0;
        } else if age < 60 {
            // age = 50s
            bat *= 1.0 / 8.0;
        } else {
            // age > 60
            bat *= 0.0;
        }
    }

    // NST limit
    let thres = (1.80 * bat + 2.43) + 5.62; // [W]

    let mut sig_nst = 2.8 * clds; // [W]
    sig_nst = py_min(sig_nst, thres);

    // Distribution coefficient of thermogenesis by non-shivering
    const NSTF: [f64; NUM_BODY_PARTS] = [
        0.000, 0.190, 0.000, 0.190, 0.190, 0.215, 0.000, 0.000, 0.215, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000,
    ];

    // Ratio of body surface area to the standard body [-]
    let bsar = construction::bsa_rate(height, weight, bsa_equation);

    // Local thermogenesis by non-shivering [W]
    let mut q_nst = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        q_nst[i] = bsar * NSTF[i] * sig_nst;
    }
    q_nst
}

// ---------------------------------------------------------------------------
// sum_m / cr_ms_fat_blood_flow / sum_bf
// ---------------------------------------------------------------------------

/// Calculate total thermogenesis in each layer \[W\].
/// Python: `sum_m(mbase, q_work, q_shiv, q_nst)`.
///
/// `mbase` is `(mbase_cr, mbase_ms, mbase_fat, mbase_sk)`, as returned by
/// [`local_mbase`].
///
/// Returns `(q_thermogenesis_core, q_thermogenesis_muscle, q_thermogenesis_fat,
/// q_thermogenesis_skin)`: total thermogenesis in core, muscle, fat, skin layers
/// \[W\].
#[must_use]
pub(crate) fn sum_m(
    mbase: (
        [f64; NUM_BODY_PARTS],
        [f64; NUM_BODY_PARTS],
        [f64; NUM_BODY_PARTS],
        [f64; NUM_BODY_PARTS],
    ),
    q_work: [f64; NUM_BODY_PARTS],
    q_shiv: [f64; NUM_BODY_PARTS],
    q_nst: [f64; NUM_BODY_PARTS],
) -> (
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
) {
    let (
        mut q_thermogenesis_core,
        mut q_thermogenesis_muscle,
        q_thermogenesis_fat,
        q_thermogenesis_skin,
    ) = mbase;

    for i in 0..NUM_BODY_PARTS {
        // If the segment has a muscle layer, muscle thermogenesis increases by the activity.
        if HAS_MUSCLE[i] {
            q_thermogenesis_muscle[i] += q_work[i] + q_shiv[i];
        } else {
            // In other segments, core thermogenesis increase, instead of muscle.
            q_thermogenesis_core[i] += q_work[i] + q_shiv[i];
        }
    }
    for i in 0..NUM_BODY_PARTS {
        q_thermogenesis_core[i] += q_nst[i]; // Non-shivering thermogenesis occurs in core layers
    }
    (
        q_thermogenesis_core,
        q_thermogenesis_muscle,
        q_thermogenesis_fat,
        q_thermogenesis_skin,
    )
}

/// Calculate core, muscle and fat blood flow rate \[L/h\].
/// Python: `cr_ms_fat_blood_flow(q_work, q_shiv, height, weight, bsa_equation, age,
/// ci)`.
///
/// Returns `(bf_core, bf_muscle, bf_fat)`: core, muscle and fat blood flow rate
/// \[L/h\].
#[must_use]
pub(crate) fn cr_ms_fat_blood_flow(
    q_work: [f64; NUM_BODY_PARTS],
    q_shiv: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa_equation: BsaFormula,
    age: i32,
    ci: f64,
) -> (
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
    [f64; NUM_BODY_PARTS],
) {
    // Basal blood flow rate [L/h]
    // core, CBFB
    const BFB_CORE: [f64; NUM_BODY_PARTS] = [
        35.251, 15.240, 89.214, 87.663, 18.686, 1.808, 0.940, 0.217, 1.808, 0.940, 0.217, 1.406,
        0.164, 0.080, 1.406, 0.164, 0.080,
    ];
    // muscle, MSBFB
    const BFB_MUSCLE: [f64; NUM_BODY_PARTS] = [
        0.682, 0.0, 0.0, 0.0, 12.614, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    // fat, FTBFB
    const BFB_FAT: [f64; NUM_BODY_PARTS] = [
        0.265, 0.0, 0.0, 0.0, 2.219, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let bfb_rate = construction::bfb_rate(height, weight, bsa_equation, age, ci);
    let mut bf_core = BFB_CORE;
    let mut bf_muscle = BFB_MUSCLE;
    let mut bf_fat = BFB_FAT;
    for i in 0..NUM_BODY_PARTS {
        bf_core[i] *= bfb_rate;
        bf_muscle[i] *= bfb_rate;
        bf_fat[i] *= bfb_rate;
    }

    for i in 0..NUM_BODY_PARTS {
        // If the segment has a muscle layer, muscle blood flow increases.
        if HAS_MUSCLE[i] {
            bf_muscle[i] += (q_work[i] + q_shiv[i]) / 1.163;
        } else {
            // In other segments, core blood flow increase, instead of muscle blood flow.
            bf_core[i] += (q_work[i] + q_shiv[i]) / 1.163;
        }
    }
    (bf_core, bf_muscle, bf_fat)
}

/// Sum the total blood flow in various body parts.
/// Python: `sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)`.
///
/// Returns `co`: cardiac output (the sum of the whole blood flow rate) \[L/h\].
#[must_use]
pub(crate) fn sum_bf(
    bf_core: [f64; NUM_BODY_PARTS],
    bf_muscle: [f64; NUM_BODY_PARTS],
    bf_fat: [f64; NUM_BODY_PARTS],
    bf_skin: [f64; NUM_BODY_PARTS],
    bf_ava_hand: f64,
    bf_ava_foot: f64,
) -> f64 {
    // Cardiac output (CO)
    let mut co = 0.0;
    co += bf_core.iter().sum::<f64>();
    co += bf_muscle.iter().sum::<f64>();
    co += bf_fat.iter().sum::<f64>();
    co += bf_skin.iter().sum::<f64>();
    co += 2.0 * bf_ava_hand;
    co += 2.0 * bf_ava_foot;
    co
}

// ---------------------------------------------------------------------------
// resp_heat_loss
// ---------------------------------------------------------------------------

/// Calculate heat loss by respiration \[W\].
/// Python: `resp_heat_loss(tdb, p_a, q_thermogenesis_total)`.
///
/// Returns `(res_sh, res_lh)`: sensible and latent heat loss by respiration \[W\].
#[must_use]
pub(crate) fn resp_heat_loss(tdb: f64, p_a: f64, q_thermogenesis_total: f64) -> (f64, f64) {
    let res_sh = 0.0014 * q_thermogenesis_total * (34.0 - tdb); // Sensible heat loss
    let res_lh = 0.0173 * q_thermogenesis_total * (5.87 - p_a); // Latent heat loss
    (res_sh, res_lh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        fabs(a - b) < tol
    }

    fn approx_eq_arr(a: [f64; NUM_BODY_PARTS], b: [f64; NUM_BODY_PARTS], tol: f64) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    // Shared test fixtures, matching a throwaway cross-check script run against
    // pythermalcomfort 4.4.0 (`thermoregulation.py`) via `.parity-venv`.
    const TDB17: [f64; NUM_BODY_PARTS] = [
        28.8, 25.0, 30.0, 22.0, 26.5, 29.0, 24.0, 20.0, 27.0, 31.0, 18.0, 33.0, 15.0, 35.0, 21.0,
        23.0, 32.0,
    ];
    const TSK17: [f64; NUM_BODY_PARTS] = [
        34.0, 33.5, 35.0, 32.0, 33.0, 34.5, 33.2, 31.0, 34.2, 35.5, 30.0, 36.0, 29.0, 36.5, 32.5,
        33.8, 34.8,
    ];
    const V17: [f64; NUM_BODY_PARTS] = [
        0.1, 0.3, 0.15, 0.5, 0.05, 1.0, 0.2, 0.01, 0.19, 0.25, 0.4, 0.6, 0.02, 0.8, 0.35, 0.1, 2.0,
    ];
    const ERR_SK17: [f64; NUM_BODY_PARTS] = [
        0.5, -0.3, 0.2, -0.1, 0.0, 0.4, -0.6, 0.15, 0.3, -0.2, 0.25, -0.4, 0.1, 0.05, -0.15, 0.35,
        -0.25,
    ];
    const ERR_CR17: [f64; NUM_BODY_PARTS] = [
        0.2, 0.1, 0.15, 0.05, 0.3, 0.0, -0.1, 0.05, 0.0, -0.1, 0.05, 0.02, -0.05, 0.01, 0.02,
        -0.05, 0.01,
    ];

    // -- natural_convection / forced_convection / conv_coef / rad_coef --

    #[test]
    fn natural_convection_matches_python() {
        // Python: natural_convection("standing", TDB17, TSK17)
        assert_eq!(
            natural_convection(Posture::Standing, TDB17, TSK17),
            [
                4.48, 4.48, 2.97, 2.91, 2.85, 3.61, 3.55, 3.67, 3.61, 3.55, 3.67, 2.8, 2.04, 2.04,
                2.8, 2.04, 2.04
            ]
        );
        // Python: natural_convection("sitting", ...) == natural_convection("sedentary", ...)
        assert_eq!(
            natural_convection(Posture::Sitting, TDB17, TSK17),
            [
                4.75, 4.75, 3.12, 2.48, 1.84, 3.76, 3.62, 2.06, 3.76, 3.62, 2.06, 2.98, 2.98, 2.62,
                2.98, 2.98, 2.62
            ]
        );
        // Python: natural_convection("lying", TDB17, TSK17) == natural_convection("supine", ...)
        let got = natural_convection(Posture::Lying, TDB17, TSK17);
        let want = [
            1.951_566_846_184_296_8,
            2.312_127_650_351_918,
            1.304_057_566_400_781_4,
            1.346_307_121_738_007,
            1.319_891_273_583_458_3,
            1.724_352_538_486_428_9,
            13.724_348_372_544_133,
            4.439_678_307_404_022,
            1.906_583_361_635_225_7,
            7.473_133_093_246_477,
            4.555_905_481_807_477,
            1.544_205_195_706_941_6,
            1.779_159_935_372_874_6,
            0.295_892_632_647_571_1,
            2.815_553_234_876_702_5,
            1.530_546_426_428_649_2,
            0.540_735_215_400_557_7,
        ];
        assert!(approx_eq_arr(got, want, 1e-12));
    }

    #[test]
    fn forced_convection_matches_python() {
        // Python: forced_convection(V17)
        let got = forced_convection(V17);
        let want = [
            3.598_249_378_529_236,
            7.110_603_085_857_498,
            3.085_843_016_037_195_2,
            12.104_426_662_575_11,
            2.154_395_111_270_991_6,
            17.0,
            6.369_087_549_399_174,
            1.261_914_688_960_386_5,
            6.381_364_411_098_867,
            7.297_806_209_720_907,
            11.541_599_247_257_711,
            10.251_804_092_468_916,
            0.873_814_573_778_139_6,
            13.149_000_419_470_028,
            7.379_204_442_416_378,
            2.875_127_356_603_774,
            23.206_815_237_448_986,
        ];
        assert!(approx_eq_arr(got, want, 1e-9));
    }

    #[test]
    fn conv_coef_matches_python() {
        // Python: conv_coef("standing", V17, TDB17, TSK17)
        let got = conv_coef(Posture::Standing, V17, TDB17, TSK17);
        let want = [
            4.48,
            7.110_603_085_857_498,
            2.97,
            12.104_426_662_575_11,
            2.85,
            17.0,
            6.369_087_549_399_174,
            3.67,
            3.61,
            7.297_806_209_720_907,
            11.541_599_247_257_711,
            10.251_804_092_468_916,
            2.04,
            13.149_000_419_470_028,
            7.379_204_442_416_378,
            2.04,
            23.206_815_237_448_986,
        ];
        assert!(approx_eq_arr(got, want, 1e-9));

        // Python: conv_coef("lying", [0.1]*17, TDB17, TSK17) == natural_convection("lying", ...)
        // since every v < 0.2.
        let low_v = [0.1; NUM_BODY_PARTS];
        assert_eq!(
            conv_coef(Posture::Lying, low_v, TDB17, TSK17),
            natural_convection(Posture::Lying, TDB17, TSK17)
        );
    }

    #[test]
    fn rad_coef_matches_python() {
        assert_eq!(
            rad_coef(Posture::Standing),
            [
                4.89, 4.89, 4.32, 4.09, 4.32, 4.55, 4.43, 4.21, 4.55, 4.43, 4.21, 4.77, 5.34, 6.14,
                4.77, 5.34, 6.14
            ]
        );
        assert_eq!(
            rad_coef(Posture::Sitting),
            [
                4.96, 4.96, 3.99, 4.64, 4.21, 4.96, 4.21, 4.74, 4.96, 4.21, 4.74, 4.1, 4.74, 6.36,
                4.1, 4.74, 6.36
            ]
        );
        assert_eq!(
            rad_coef(Posture::Lying),
            [
                5.475, 5.475, 3.463, 3.463, 3.463, 4.249, 4.835, 4.119, 4.249, 4.835, 4.119, 4.44,
                5.547, 6.085, 4.44, 5.547, 6.085
            ]
        );
    }

    #[test]
    fn fixed_hc_and_fixed_hr_match_python() {
        let hc = conv_coef(Posture::Standing, V17, TDB17, TSK17);
        let hr = rad_coef(Posture::Standing);

        // Python: fixed_hc(hc, V17)
        let got_hc = fixed_hc(hc, V17);
        let want_hc = [
            3.077_368_109_652_747,
            4.884_362_316_253_633,
            2.040_130_197_693_897,
            8.314_682_276_124_792,
            1.957_700_694_756_77,
            11.677_512_916_093_011,
            4.375_006_007_166_826,
            2.520_968_964_827_138_5,
            2.479_754_213_358_574_6,
            5.012_954_486_656_452,
            7.928_069_075_424_783,
            7.042_092_629_591_822,
            1.401_301_549_931_161_4,
            9.032_213_072_474_923,
            5.068_867_952_165_188,
            1.401_301_549_931_161_4,
            15.941_052_039_805_57,
        ];
        assert!(approx_eq_arr(got_hc, want_hc, 1e-9));

        // Python: fixed_hr(hr)
        let got_hr = fixed_hr(hr);
        let want_hr = [
            4.890_111_397_134_201,
            4.890_111_397_134_201,
            4.320_098_412_192_178,
            4.090_093_172_654_168,
            4.320_098_412_192_178,
            4.550_103_651_730_188,
            4.430_100_918_058_182,
            4.210_095_906_326_173,
            4.550_103_651_730_188,
            4.430_100_918_058_182,
            4.210_095_906_326_173,
            4.770_108_663_462_197,
            5.340_121_648_404_22,
            6.140_139_872_884_253,
            4.770_108_663_462_197,
            5.340_121_648_404_22,
            6.140_139_872_884_253,
        ];
        assert!(approx_eq_arr(got_hr, want_hr, 1e-9));
    }

    // -- operative_temp / clo_area_factor / dry_r / wet_r --

    #[test]
    fn operative_temp_clo_area_factor_dry_r_wet_r_match_python() {
        let hc = conv_coef(Posture::Standing, V17, TDB17, TSK17);
        let hr = rad_coef(Posture::Standing);
        let fhc = fixed_hc(hc, V17);
        let fhr = fixed_hr(hr);
        let tr = [28.8; NUM_BODY_PARTS];

        // Python: operative_temp(TDB17, [28.8]*17, fhc, fhr)
        let got_to = operative_temp(TDB17, tr, fhc, fhr);
        let want_to = [
            28.8,
            26.901_117_528_574_265,
            29.184_916_390_179_332,
            24.242_090_853_549_957,
            28.082_756_341_636_284,
            28.943_921_479_377_913,
            26.415_017_169_838_16,
            25.504_157_913_326_18,
            28.165_057_207_456_2,
            29.967_895_283_674_6,
            21.745_956_316_847_13,
            31.303_918_474_677_353,
            25.931_471_972_591_037,
            32.490_905_507_604_43,
            24.781_576_990_020_383,
            27.594_386_771_088_985,
            31.110_172_690_363_79,
        ];
        assert!(approx_eq_arr(got_to, want_to, 1e-9));

        let clo17 = [
            0.0, 0.1, 0.3, 0.6, 0.9, 0.2, 0.4, 0.05, 0.2, 0.4, 0.05, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1,
        ];
        // Python: clo_area_factor(clo17)
        let got_fcl = clo_area_factor(clo17);
        let want_fcl = [
            1.0,
            1.02,
            1.06,
            1.11,
            1.140_000_000_000_000_1,
            1.04,
            1.08,
            1.01,
            1.04,
            1.08,
            1.01,
            1.1,
            1.06,
            1.02,
            1.1,
            1.06,
            1.02,
        ];
        assert!(approx_eq_arr(got_fcl, want_fcl, 1e-12));

        // Python: dry_r(fhc, fhr, clo17)
        let got_rt = dry_r(fhc, fhr, clo17).unwrap();
        let want_rt = [
            0.125_510_206_728_259_38,
            0.115_801_273_051_655_79,
            0.194_827_408_381_000_38,
            0.165_625_329_222_672_8,
            0.279_229_380_872_536_7,
            0.090_253_215_499_621_83,
            0.167_157_828_722_479_08,
            0.154_843_963_414_936_6,
            0.167_779_218_014_292_05,
            0.160_053_636_904_813_56,
            0.089_319_084_897_885_95,
            0.154_462_023_126_501_04,
            0.186_440_217_170_766_2,
            0.080_117_014_934_563_69,
            0.169_896_897_015_385_43,
            0.186_440_217_170_766_2,
            0.059_899_421_948_745_63,
        ];
        assert!(approx_eq_arr(got_rt, want_rt, 1e-9));

        // Python: wet_r(fhc, clo17, [0.45]*17, 16.5)
        let i_clo17 = [0.45; NUM_BODY_PARTS];
        let got_ret = wet_r(fhc, clo17, i_clo17, 16.5).unwrap();
        let want_ret = [
            0.019_694_121_225_198_32,
            0.014_252_427_210_581_88,
            0.034_288_057_648_462_135,
            0.019_091_956_193_152_862,
            0.045_943_823_105_788_45,
            0.009_165_449_734_410_448,
            0.021_176_830_231_652_17,
            0.024_846_523_650_911_764,
            0.027_675_420_604_044_463,
            0.019_544_509_518_188_466,
            0.008_612_575_002_110_032,
            0.018_261_580_646_582_697,
            0.047_064_357_250_828_484,
            0.008_665_963_787_712_29,
            0.021_307_281_168_833_49,
            0.047_064_357_250_828_484,
            0.005_814_881_182_111_327,
        ];
        assert!(approx_eq_arr(got_ret, want_ret, 1e-9));

        // Python: dry_r([-1.0]*17, fhr, clo17) -> ValueError
        assert_eq!(
            dry_r([-1.0; NUM_BODY_PARTS], fhr, clo17),
            Err(ThermoregulationError::NegativeConvectiveOrRadiativeCoefficient)
        );
        // Python: wet_r([-1.0]*17, clo17, i_clo17, 16.5) -> ValueError
        assert_eq!(
            wet_r([-1.0; NUM_BODY_PARTS], clo17, i_clo17, 16.5),
            Err(ThermoregulationError::NegativeConvectiveCoefficient)
        );
    }

    // -- error_signals --

    #[test]
    fn error_signals_matches_python() {
        // Python: error_signals(ERR_SK17) == (0.12223, 0.05868)
        let (wrms, clds) = error_signals(ERR_SK17);
        assert!(approx_eq(wrms, 0.122_23, 1e-9));
        assert!(approx_eq(clds, 0.058_68, 1e-9));
    }

    // -- evaporation --

    #[test]
    fn evaporation_matches_python() {
        let rh17 = [50.0; NUM_BODY_PARTS];
        let ret17 = [0.05; NUM_BODY_PARTS];

        // Python: evaporation(ERR_CR17, ERR_SK17, TSK17, TDB17, rh17, ret17, 1.72, 74.43,
        // "dubois", 30)
        let (wet, e_sk, e_max, e_sweat) = evaporation(
            ERR_CR17,
            ERR_SK17,
            TSK17,
            TDB17,
            rh17,
            ret17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
        );
        let want_wet = [
            0.707_318_472_812_767_5,
            0.634_084_388_100_959,
            0.927_119_979_935_736_4,
            0.891_834_848_648_698_4,
            1.0,
            0.625_490_220_457_43,
            0.455_201_512_781_807_87,
            0.398_236_923_454_327_15,
            0.601_301_782_396_044_9,
            0.473_376_927_174_980_76,
            0.412_511_055_719_912_9,
            0.415_895_255_481_758_5,
            0.428_388_604_037_701_2,
            0.401_605_554_928_847_64,
            0.400_108_429_498_184,
            0.366_517_158_373_316_33,
            0.406_254_733_478_277_95,
        ];
        assert!(approx_eq_arr(wet, want_wet, 1e-9));
        assert!(approx_eq(e_max[4], 14.593_518_457_246_597, 1e-6)); // pelvis, wet clamped to 1
        assert!(approx_eq(e_sk[4], 14.593_518_457_246_597, 1e-6)); // wet==1 -> e_sk==e_max
        assert!(approx_eq(e_sweat[0], 5.063_190_563_054_891, 1e-6));

        // Python: evaporation(..., age=70) uses the age>=60 sd_sweat table.
        let (wet70, _e_sk70, _e_max70, e_sweat70) = evaporation(
            ERR_CR17,
            ERR_SK17,
            TSK17,
            TDB17,
            rh17,
            ret17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            70,
        );
        assert!(approx_eq(wet70[0], 0.506_649_746_240_809_7, 1e-9));
        assert!(approx_eq(e_sweat70[0], 3.493_601_488_507_876, 1e-6));

        // e_max == 0 guard: equal tdb/t_skin and rh=100 makes p_sk_s == p_a, so wet
        // saturates at its floor+ceiling of 1 regardless (0.06 + 0.94*(e_sweat/0.001)
        // clamped to 1).
        let tsk_eq = [25.0; NUM_BODY_PARTS];
        let tdb_eq = [25.0; NUM_BODY_PARTS];
        let rh100 = [100.0; NUM_BODY_PARTS];
        let (wet_eq, _, _, _) = evaporation(
            ERR_CR17,
            ERR_SK17,
            tsk_eq,
            tdb_eq,
            rh100,
            ret17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
        );
        assert!(approx_eq_arr(wet_eq, [1.0; NUM_BODY_PARTS], 1e-9));
    }

    // -- skin_blood_flow / ava_blood_flow --

    #[test]
    fn skin_blood_flow_matches_python() {
        // Python: skin_blood_flow(ERR_CR17, ERR_SK17, 1.72, 74.43, "dubois", 30, 2.59)
        let got30 = skin_blood_flow(
            ERR_CR17,
            ERR_SK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            2.59,
        );
        let want30 = [
            4.502_133_640_415_669,
            0.953_983_842_768_453_8,
            4.413_846_175_294_417,
            3.493_535_872_170_290_7,
            5.574_408_943_783_542,
            1.737_404_739_880_153,
            0.837_792_359_455_357_1,
            2.606_411_093_026_913_3,
            1.717_448_944_399_108,
            0.877_415_014_201_621_4,
            2.636_696_131_124_608,
            3.493_813_972_354_89,
            1.215_452_383_140_107_8,
            2.142_803_539_239_156_2,
            3.596_190_536_161_173,
            1.251_067_856_499_775_5,
            2.069_811_784_299_836_7,
        ];
        assert!(approx_eq_arr(got30, want30, 1e-9));

        // Python: skin_blood_flow(..., age=70) uses the age>=60 sd_dilat table.
        let got70 = skin_blood_flow(
            ERR_CR17,
            ERR_SK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            70,
            2.59,
        );
        assert!(approx_eq(got70[0], 2.985_108_734_280_808_5, 1e-9));
        assert!(approx_eq(got70[16], 0.888_090_520_907_029, 1e-9));
    }

    #[test]
    fn ava_blood_flow_matches_python() {
        // Python: ava_blood_flow(ERR_CR17, ERR_SK17, 1.72, 74.43, "dubois", 30, 2.59)
        let (hand, foot) = ava_blood_flow(
            ERR_CR17,
            ERR_SK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            2.59,
        );
        assert!(approx_eq(hand, 1.712_582_604_924_341_8, 1e-9));
        assert!(approx_eq(foot, 1.716_834_717_294_236_2, 1e-9));

        // Clamped high: err_cr = err_sk = 3.0 everywhere -> sig_ava_foot saturates at 1.
        let hot = [3.0; NUM_BODY_PARTS];
        let (hand_hi, foot_hi) =
            ava_blood_flow(hot, hot, 1.72, 74.43, BsaFormula::DuBois, 30, 2.59);
        assert!(approx_eq(hand_hi, 1.712_582_604_924_341_8, 1e-9)); // already saturated at baseline
        assert!(approx_eq(foot_hi, 2.163_262_237_799_169, 1e-9));

        // Clamped low: err_cr = err_sk = -3.0 everywhere -> both signals clamp to 0.
        let cold = [-3.0; NUM_BODY_PARTS];
        let (hand_lo, foot_lo) =
            ava_blood_flow(cold, cold, 1.72, 74.43, BsaFormula::DuBois, 30, 2.59);
        assert!(approx_eq(hand_lo, 0.0, 1e-12));
        assert!(approx_eq(foot_lo, 0.0, 1e-12));
    }

    // -- basal_met / local_mbase / local_q_work --

    #[test]
    fn basal_met_matches_python_all_equations_and_sexes() {
        // Python: basal_met(1.72, 74.43, 20, sex, equation) for every combination.
        assert!(approx_eq(
            basal_met(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict),
            87.958_882_08,
            1e-9
        ));
        assert!(approx_eq(
            basal_met(1.72, 74.43, 20, Sex::Female, BmrEquation::HarrisBenedict),
            89.984_410_08,
            1e-9
        ));
        assert!(approx_eq(
            basal_met(
                1.72,
                74.43,
                20,
                Sex::Male,
                BmrEquation::HarrisBenedictOriginal
            ),
            87.142_665_024,
            1e-9
        ));
        assert!(approx_eq(
            basal_met(
                1.72,
                74.43,
                20,
                Sex::Female,
                BmrEquation::HarrisBenedictOriginal
            ),
            76.392_890_976,
            1e-9
        ));
        // Japanese covers both Python's "japanese" and "ganpule" (identical results).
        assert!(approx_eq(
            basal_met(1.72, 74.43, 20, Sex::Male, BmrEquation::Japanese),
            79.182_604_873_387_49,
            1e-9
        ));
        assert!(approx_eq(
            basal_met(1.72, 74.43, 20, Sex::Female, BmrEquation::Japanese),
            72.906_828_475_871_96,
            1e-9
        ));

        // Python: basal_met(0.5, 20.0, 100, "female", "japanese") == 68 (clamped to the
        // minimum).
        assert!(approx_eq(
            basal_met(0.5, 20.0, 100, Sex::Female, BmrEquation::Japanese),
            68.0,
            1e-12
        ));
    }

    #[test]
    fn local_mbase_matches_python() {
        // Python: local_mbase(1.72, 74.43, 20, "male", "harris-benedict")
        let (cr, ms, fat, sk) =
            local_mbase(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict);
        assert!(approx_eq(cr[0], 17.196_841_035_460_8, 1e-9));
        assert!(approx_eq(cr[2], 25.234_523_679_931_197, 1e-9));
        assert!(approx_eq(ms[0], 0.221_656_382_841_6, 1e-9));
        assert!(approx_eq(ms[4], 4.225_544_695_123_2, 1e-9));
        assert!(approx_eq(fat[0], 0.111_707_780_241_6, 1e-9));
        assert!(approx_eq(fat[4], 0.835_609_379_759_999_9, 1e-9));
        assert!(approx_eq(sk[0], 0.133_697_500_761_6, 1e-9));
        assert!(approx_eq(sk[16], 0.103_791_480_854_4, 1e-9));
        // Only head/pelvis get nonzero muscle/fat thermogenesis.
        for i in [1usize, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] {
            assert_eq!(ms[i], 0.0);
            assert_eq!(fat[i], 0.0);
        }
    }

    #[test]
    fn local_q_work_matches_python() {
        let bmr = basal_met(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict);
        // Python: local_q_work(bmr, 1.5)
        let got = local_q_work(bmr, 1.5).unwrap();
        let want = [
            0.0,
            0.0,
            4.002_129_134_64,
            3.518_355_283_2,
            5.673_347_894_16,
            1.152_261_355_248,
            0.611_314_230_456,
            0.219_897_205_2,
            1.152_261_355_248,
            0.611_314_230_456,
            0.219_897_205_2,
            8.839_867_649_04,
            4.353_964_662_96,
            0.219_897_205_2,
            8.839_867_649_04,
            4.353_964_662_96,
            0.219_897_205_2,
        ];
        assert!(approx_eq_arr(got, want, 1e-9));

        // Python: local_q_work(bmr, 0.5) -> ValueError("par must be 1 or more")
        assert_eq!(
            local_q_work(bmr, 0.5),
            Err(ThermoregulationError::ParTooSmall)
        );
    }

    // -- shivering / nonshivering --

    #[test]
    fn shivering_matches_python() {
        let tcore17 = [37.0; NUM_BODY_PARTS];
        let err_cr_shiv = [
            -0.5, 0.1, 0.15, 0.05, 0.3, 0.0, -0.1, 0.05, 0.0, -0.1, 0.05, 0.02, -0.05, 0.01, 0.02,
            -0.05, 0.01,
        ];

        // Python (options=None): PRE_SHIV is never touched; signal is not thresholded.
        let (q_shiv, new_pre_shiv) = shivering(
            err_cr_shiv,
            ERR_SK17,
            tcore17,
            TSK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            Sex::Male,
            60.0,
            None,
            0.0,
        );
        let want = [
            0.023_639_026_785_889_96,
            0.030_402_996_102_206_557,
            0.191_022_861_289_873_06,
            0.168_067_204_599_858_38,
            0.270_238_007_097_457_1,
            0.001_694_478_911_200_961_8,
            0.000_955_323_501_376_673_9,
            0.000_139_463_284_872_507_14,
            0.001_694_478_911_200_961_8,
            0.000_955_323_501_376_673_9,
            0.000_139_463_284_872_507_14,
            0.002_719_534_055_013_889_3,
            0.001_220_303_742_634_437_5,
            0.000_244_060_748_526_887_5,
            0.002_719_534_055_013_889_3,
            0.001_220_303_742_634_437_5,
            0.000_244_060_748_526_887_5,
        ];
        assert!(approx_eq_arr(q_shiv, want, 1e-9));
        assert_eq!(new_pre_shiv, 0.0); // Python: options=None -> PRE_SHIV unchanged.

        // Python (age=85, female, options=None): different sd_shiv aging factor.
        let (q_shiv_age85, _) = shivering(
            err_cr_shiv,
            ERR_SK17,
            tcore17,
            TSK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            85,
            Sex::Female,
            60.0,
            None,
            0.0,
        );
        assert!(approx_eq(q_shiv_age85[0], 0.020_022_896_152_697_593, 1e-9));
        assert!(approx_eq(q_shiv_age85[4], 0.228_898_913_717_298_67, 1e-9));

        // Python (shivering_threshold=True): mean skin temp ~34 C for a male gives a
        // threshold below 37 C core temp, so the signal is zeroed.
        let opts_threshold = ShiveringOptions {
            shivering_threshold: true,
            limit_dshiv_dt: Some(0.0077),
        };
        let (q_shiv_thresh, pre_shiv_thresh) = shivering(
            err_cr_shiv,
            ERR_SK17,
            tcore17,
            TSK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            Sex::Male,
            60.0,
            Some(opts_threshold),
            0.0,
        );
        assert!(approx_eq_arr(q_shiv_thresh, [0.0; NUM_BODY_PARTS], 1e-12));
        assert_eq!(pre_shiv_thresh, 0.0); // Python: PRE_SHIV updated to the (zeroed) sig_shiv.

        // Python (shivering_threshold=True, tskm<31): also zeroed via the other branch
        // of the threshold computation.
        let tsk_cold = [25.0; NUM_BODY_PARTS];
        let (q_shiv_cold, _) = shivering(
            err_cr_shiv,
            ERR_SK17,
            tcore17,
            tsk_cold,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            Sex::Male,
            60.0,
            Some(ShiveringOptions {
                shivering_threshold: true,
                limit_dshiv_dt: None,
            }),
            0.0,
        );
        assert!(approx_eq_arr(q_shiv_cold, [0.0; NUM_BODY_PARTS], 1e-12));

        // Python (limit_dshiv/dt=0.001, a tight custom rate): clamps the signal well
        // below its natural value, and PRE_SHIV tracks the clamp limit (0.001*60=0.06).
        let (q_shiv_custom, pre_shiv_custom) = shivering(
            err_cr_shiv,
            ERR_SK17,
            tcore17,
            TSK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            Sex::Male,
            60.0,
            Some(ShiveringOptions {
                shivering_threshold: false,
                limit_dshiv_dt: Some(0.001),
            }),
            0.0,
        );
        let want_custom = [
            0.001_984_465_027_475_559_5,
            0.002_552_291_303_773_875_3,
            0.016_036_116_508_160_91,
            0.014_109_019_496_228_888,
            0.022_686_123_207_902_014,
            0.000_142_249_263_031_433_9,
            0.000_080_198_144_178_215_81,
            0.000_011_707_758_274_192_09,
            0.000_142_249_263_031_433_9,
            0.000_080_198_144_178_215_81,
            0.000_011_707_758_274_192_09,
            0.000_228_301_286_346_745_75,
            0.000_102_442_884_899_180_79,
            0.000_020_488_576_979_836_156,
            0.000_228_301_286_346_745_75,
            0.000_102_442_884_899_180_79,
            0.000_020_488_576_979_836_156,
        ];
        assert!(approx_eq_arr(q_shiv_custom, want_custom, 1e-9));
        assert!(approx_eq(pre_shiv_custom, 0.06, 1e-9));
    }

    #[test]
    fn nonshivering_matches_python() {
        // Python: nonshivering(ERR_SK17, 1.72, 74.43, "dubois", 25) — default
        // cold_acclimation=False, batpositive=True.
        let want = [
            0.0,
            0.031_233_975_629_289_375,
            0.0,
            0.031_233_975_629_289_375,
            0.031_233_975_629_289_375,
            0.035_343_709_264_722_19,
            0.0,
            0.0,
            0.035_343_709_264_722_19,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let got = nonshivering(ERR_SK17, 1.72, 74.43, BsaFormula::DuBois, 25, false, true);
        assert!(approx_eq_arr(got, want, 1e-9));

        // batpositive=False path is exercised (all these ages hit a different age-factor
        // branch); clds isn't large enough here to exceed the resulting NST threshold,
        // so the result is identical to the batpositive=True case at this signal level.
        for age in [25, 35, 45, 55, 65] {
            let got_batneg =
                nonshivering(ERR_SK17, 1.72, 74.43, BsaFormula::DuBois, age, false, false);
            assert!(approx_eq_arr(got_batneg, want, 1e-9));
        }

        // cold_acclimation=True is exercised; also below the NST threshold at this
        // signal level, so identical here too.
        let got_cold_acc = nonshivering(ERR_SK17, 1.72, 74.43, BsaFormula::DuBois, 25, true, true);
        assert!(approx_eq_arr(got_cold_acc, want, 1e-9));
    }

    // -- sum_m / cr_ms_fat_blood_flow / sum_bf / resp_heat_loss --

    #[test]
    fn sum_m_matches_python() {
        let mbase = local_mbase(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict);
        let bmr = basal_met(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict);
        let q_work = local_q_work(bmr, 1.5).unwrap();
        let err_cr_shiv = [
            -0.5, 0.1, 0.15, 0.05, 0.3, 0.0, -0.1, 0.05, 0.0, -0.1, 0.05, 0.02, -0.05, 0.01, 0.02,
            -0.05, 0.01,
        ];
        let (q_shiv, _) = shivering(
            err_cr_shiv,
            ERR_SK17,
            [37.0; NUM_BODY_PARTS],
            TSK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            Sex::Male,
            60.0,
            Some(ShiveringOptions {
                shivering_threshold: true,
                limit_dshiv_dt: Some(0.0077),
            }),
            0.0,
        );
        let q_nst = nonshivering(ERR_SK17, 1.72, 74.43, BsaFormula::DuBois, 25, false, true);

        // Python: sum_m(mbase, q_work, q_shiv, q_nst) with q_shiv from the
        // shivering_threshold-zeroed call above (all zeros).
        let (core, muscle, fat, skin) = sum_m(mbase, q_work, q_shiv, q_nst);
        assert!(approx_eq(core[2], 29.236_652_814_571_197, 1e-9)); // chest: mbase_cr + q_work (no muscle layer)
        assert!(approx_eq(muscle[0], 0.221_656_382_841_6, 1e-9)); // head: q_work/q_shiv don't apply to head
        assert!(approx_eq(muscle[4], 9.898_892_589_283_2, 1e-9)); // pelvis: mbase_ms + q_work (has muscle layer)
        assert!(approx_eq(core[4], 8.395_244_072_616_489, 1e-9)); // pelvis core: mbase_cr + q_nst only
        assert_eq!(fat, mbase.2); // fat is untouched by sum_m
        assert_eq!(skin, mbase.3); // skin is untouched by sum_m
    }

    #[test]
    fn cr_ms_fat_blood_flow_matches_python() {
        let bmr = basal_met(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict);
        let q_work = local_q_work(bmr, 1.5).unwrap();
        let q_shiv = [0.0; NUM_BODY_PARTS];

        // Python: cr_ms_fat_blood_flow(q_work, [0]*17, 1.72, 74.43, "dubois", 30, 2.59)
        let (bf_core, bf_muscle, bf_fat) =
            cr_ms_fat_blood_flow(q_work, q_shiv, 1.72, 74.43, BsaFormula::DuBois, 30, 2.59);
        assert!(approx_eq(bf_core[0], 35.304_239_418_823_37, 1e-9));
        assert!(approx_eq(bf_core[2], 92.789_951_120_572_94, 1e-9));
        assert!(approx_eq(bf_muscle[0], 0.683_030_021_379_182, 1e-9));
        assert!(approx_eq(bf_muscle[4], 17.511_251_977_406_335, 1e-9));
        assert!(approx_eq(bf_fat[0], 0.265_400_228_248_509_1, 1e-9));
        assert!(approx_eq(bf_fat[4], 2.222_351_345_220_534_6, 1e-9));
    }

    #[test]
    fn sum_bf_matches_python() {
        let bmr = basal_met(1.72, 74.43, 20, Sex::Male, BmrEquation::HarrisBenedict);
        let q_work = local_q_work(bmr, 1.5).unwrap();
        let q_shiv = [0.0; NUM_BODY_PARTS];
        let (bf_core, bf_muscle, bf_fat) =
            cr_ms_fat_blood_flow(q_work, q_shiv, 1.72, 74.43, BsaFormula::DuBois, 30, 2.59);
        let bf_skin = skin_blood_flow(
            ERR_CR17,
            ERR_SK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            2.59,
        );
        let (bf_ava_hand, bf_ava_foot) = ava_blood_flow(
            ERR_CR17,
            ERR_SK17,
            1.72,
            74.43,
            BsaFormula::DuBois,
            30,
            2.59,
        );

        // Python: sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)
        // == 359.27551375637347
        let co = sum_bf(
            bf_core,
            bf_muscle,
            bf_fat,
            bf_skin,
            bf_ava_hand,
            bf_ava_foot,
        );
        assert!(approx_eq(co, 359.275_513_756_373_47, 1e-6));
    }

    #[test]
    fn resp_heat_loss_matches_python() {
        // Python: resp_heat_loss(20.0, 1.5, 85.0) == (1.666, 6.426085)
        let (res_sh, res_lh) = resp_heat_loss(20.0, 1.5, 85.0);
        assert!(approx_eq(res_sh, 1.666, 1e-12));
        assert!(approx_eq(res_lh, 6.426_085, 1e-12));
    }
}
