//! Constant tables and default coefficients for the JOS3 thermoregulation model.
//!
//! Mirrors:
//! `pythermalcomfort/jos3_functions/parameters.py`
//! (pythermalcomfort 4.4.0, ~1627 lines).
//!
//! Only data (default coefficients and the `local_clo_typical_ensembles` clothing
//! table) is ported here. The Python module's `ALL_OUT_PARAMS` dict (human-readable
//! meaning/unit metadata for output fields) is NOT ported: the Rust port returns a
//! typed [`crate::models::jos3::Jos3Results`] struct whose fields carry doc comments,
//! so that metadata has no consumer. Likewise the doc-generation helpers
//! `show_out_param_docs()` and `add_prompt_to_code()` from the Python module are NOT
//! ported: they are string-formatting conveniences for building human-readable
//! documentation (using `textwrap`, dict sorting, regex) rather than model data, and do
//! not fit a `no_std` data module. See the crate's module docs for the equivalent Rust
//! documentation.
//!
//! # Body segment order
//!
//! The JOS3 model divides the body into 17 segments. Every per-segment table in this
//! file (and in the upstream Python module) uses this exact order, taken from
//! `pythermalcomfort.classes_return.JOS3BodyParts`:
//!
//! head, neck, chest, back, pelvis, left_shoulder, left_arm, left_hand,
//! right_shoulder, right_arm, right_hand, left_thigh, left_leg, left_foot,
//! right_thigh, right_leg, right_foot.
//!
//! Downstream indexing into per-segment arrays depends on this order being preserved.

/// Number of body segments in the JOS3 model. Python: `Default.num_body_parts`.
pub const NUM_BODY_PARTS: usize = 17;

/// Names of the 17 body segments, in the exact order used by every per-segment table
/// in this module. Mirrors the field order of `classes_return.JOS3BodyParts` (not
/// itself defined in `parameters.py`, but the order every table here relies on).
pub const BODY_PART_NAMES: [&str; NUM_BODY_PARTS] = [
    "head",
    "neck",
    "chest",
    "back",
    "pelvis",
    "left_shoulder",
    "left_arm",
    "left_hand",
    "right_shoulder",
    "right_arm",
    "right_hand",
    "left_thigh",
    "left_leg",
    "left_foot",
    "right_thigh",
    "right_leg",
    "right_foot",
];

/// Default input coefficients for the JOS3 model.
///
/// Mirrors the Python `Default` frozen dataclass. Each constant name is the
/// `SCREAMING_SNAKE_CASE` form of the corresponding Python `snake_case` attribute
/// (e.g. `Default.height` -> [`defaults::HEIGHT`]).
pub mod defaults {
    use super::NUM_BODY_PARTS;

    // -- Body information --

    /// Body height, \[m\]. Python: `Default.height`.
    pub const HEIGHT: f64 = 1.72;
    /// Body weight, \[kg\]. Python: `Default.weight`.
    pub const WEIGHT: f64 = 74.43;
    /// Age, \[years\]. Python: `Default.age` (annotated `int`).
    pub const AGE: i32 = 20;
    /// Blood flow rate, \[L/h\]. Python: `Default.blood_flow_rate` (annotated `int`).
    pub const BLOOD_FLOW_RATE: i32 = 290;
    /// Physical activity ratio, \[-\]. Python: `Default.physical_activity_ratio`.
    pub const PHYSICAL_ACTIVITY_RATIO: f64 = 1.25;
    /// Local body surface area of each of the 17 body segments, \[m2\].
    /// Python: `Default.local_bsa`.
    pub const LOCAL_BSA: [f64; NUM_BODY_PARTS] = [
        0.11, 0.029, 0.175, 0.161, 0.221, 0.096, 0.063, 0.05, 0.096, 0.063, 0.05, 0.209, 0.112,
        0.056, 0.209, 0.112, 0.056,
    ];

    // -- Environment information --

    /// Core temperature, \[°C\]. Python: `Default.core_temperature`.
    pub const CORE_TEMPERATURE: f64 = 37.0;
    /// Skin temperature, \[°C\]. Python: `Default.skin_temperature`.
    pub const SKIN_TEMPERATURE: f64 = 34.0;
    /// Temperature of tissues other than core/skin, \[°C\].
    /// Python: `Default.other_body_temperature`.
    pub const OTHER_BODY_TEMPERATURE: f64 = 36.0;
    /// Dry bulb air temperature, \[°C\]. Python: `Default.dry_bulb_air_temperature`.
    pub const DRY_BULB_AIR_TEMPERATURE: f64 = 28.8;
    /// Mean radiant temperature, \[°C\]. Python: `Default.mean_radiant_temperature`.
    pub const MEAN_RADIANT_TEMPERATURE: f64 = 28.8;
    /// Relative humidity, \[%\]. Python: `Default.relative_humidity`.
    pub const RELATIVE_HUMIDITY: f64 = 50.0;
    /// Air speed, \[m/s\]. Python: `Default.air_speed`.
    pub const AIR_SPEED: f64 = 0.1;

    // -- Clothing information --

    /// Clothing insulation, \[clo\]. Python: `Default.clothing_insulation`.
    pub const CLOTHING_INSULATION: f64 = 0.0;
    /// Clothing vapor permeation efficiency, \[-\].
    /// Python: `Default.clothing_vapor_permeation_efficiency`.
    pub const CLOTHING_VAPOR_PERMEATION_EFFICIENCY: f64 = 0.45;
    /// Lewis relation rate, \[K/kPa\]. Python: `Default.lewis_rate`.
    pub const LEWIS_RATE: f64 = 16.5;
}

/// One entry of [`LOCAL_CLO_TYPICAL_ENSEMBLES`]: the whole-body clothing insulation
/// value plus the per-segment breakdown for a named clothing ensemble. Mirrors one
/// value of the Python `local_clo_typical_ensembles` dict
/// (`{"whole_body": ..., "local_body_part": JOS3BodyParts(...)}`), with the dict key
/// carried as `name`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClothingEnsemble {
    /// Name of the clothing ensemble, e.g. `"nude (mesh chair)"`.
    pub name: &'static str,
    /// Whole-body clothing insulation, \[clo\].
    pub whole_body: f64,
    /// Per-segment clothing insulation, \[clo\], in `BODY_PART_NAMES` order.
    pub local_body_part: [f64; NUM_BODY_PARTS],
}

/// Local and whole-body clothing insulation of typical clothing ensembles.
///
/// A lookup table of measured per-body-segment clo values for 52 named clothing
/// ensembles, plus each ensemble's whole-body clo value. Each entry's
/// [`ClothingEnsemble::local_body_part`] has exactly `NUM_BODY_PARTS` (17) values, in
/// `BODY_PART_NAMES` order. This is distinct from [`crate::utilities::CLO_TYPICAL_ENSEMBLES`],
/// which only gives the whole-body value for the same ensembles.
///
/// Based on the study by Juyoun et al. (<https://escholarship.org/uc/item/18f0r375>)
/// and by Nomoto et al. (<https://doi.org/10.1002/2475-8876.12124>).
///
/// The value for the neck is the same as the measured value for the head, and it does
/// not take into account the insulation effect of the hair. Typically, the clothing
/// insulation for the hair is quantified by assuming a head covering of approximately
/// 0.6 to 1.0 clo.
///
/// Mirrors the Python `local_clo_typical_ensembles` dict (order: Python dict
/// insertion order).
// Several garments happen to have a local clo value of 3.14, which clippy reads as a
// misspelled pi. They are measured insulation values from upstream's table, not constants.
#[allow(clippy::approx_constant)]
pub const LOCAL_CLO_TYPICAL_ENSEMBLES: &[ClothingEnsemble] = &[
    ClothingEnsemble {
        name: "nude (mesh chair)",
        whole_body: 0.01,
        local_body_part: [
            0.13, 0.13, 0.01, 0.01, 0.04, 0.02, 0.0, 0.01, 0.02, 0.0, 0.01, 0.01, 0.03, 0.05, 0.01,
            0.03, 0.05,
        ],
    },
    ClothingEnsemble {
        name: "nude (nude chair)",
        whole_body: -0.02,
        local_body_part: [
            0.13, 0.13, 0.05, -0.14, -0.01, -0.01, -0.01, -0.02, -0.01, -0.01, -0.02, -0.1, 0.0,
            0.0, -0.1, 0.0, 0.0,
        ],
    },
    ClothingEnsemble {
        name: "panty",
        whole_body: 0.03,
        local_body_part: [
            0.0, 0.0, 0.0, 0.0, 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.05, 0.0,
            0.05,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty",
        whole_body: 0.05,
        local_body_part: [
            0.0, 0.0, 0.22, 0.0, 0.18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.08, 0.03, 0.03,
            0.08,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, tanktop, shorts, sandals",
        whole_body: 0.22,
        local_body_part: [
            0.0, 0.0, 0.57, 0.27, 0.92, 0.04, 0.02, 0.02, 0.04, 0.02, 0.02, 0.51, 0.01, 0.38, 0.51,
            0.01, 0.38,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long-sleeve shirt, shorts, sandals",
        whole_body: 0.43,
        local_body_part: [
            0.0, 0.0, 1.43, 1.02, 1.45, 0.29, 0.22, 0.01, 0.29, 0.22, 0.01, 0.57, 0.01, 0.4, 0.57,
            0.01, 0.4,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, sleeveless dress, sandals",
        whole_body: 0.29,
        local_body_part: [
            0.0, 0.0, 0.85, 0.48, 0.94, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.72, 0.0, 0.41, 0.72, 0.0,
            0.41,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, long pants, socks, sneakers",
        whole_body: 0.52,
        local_body_part: [
            0.0, 0.0, 1.14, 0.84, 1.04, 0.42, 0.0, 0.0, 0.42, 0.0, 0.0, 0.58, 0.62, 0.82, 0.58,
            0.62, 0.82,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, sleeveless dress, cardigan, sandals",
        whole_body: 0.53,
        local_body_part: [
            0.0, 0.0, 1.78, 1.42, 1.19, 0.65, 0.41, 0.05, 0.65, 0.41, 0.05, 0.77, 0.0, 0.39, 0.77,
            0.0, 0.39,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, song-sleeve dress, socks, sneakers",
        whole_body: 0.54,
        local_body_part: [
            0.0, 0.0, 1.49, 1.1, 0.91, 0.72, 0.58, 0.03, 0.72, 0.58, 0.03, 0.73, 0.07, 0.77, 0.73,
            0.07, 0.77,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long-sleeve dress, cardigan, socks, sneakers",
        whole_body: 0.67,
        local_body_part: [
            0.0, 0.0, 2.05, 1.32, 1.39, 1.14, 0.63, 0.04, 1.14, 0.63, 0.04, 0.84, 0.05, 0.78, 0.84,
            0.05, 0.78,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, tank top, skirt, sandals",
        whole_body: 0.31,
        local_body_part: [
            0.0, 0.0, 0.83, 0.22, 0.99, 0.0, 0.0, 0.03, 0.0, 0.0, 0.03, 0.88, 0.05, 0.44, 0.88,
            0.05, 0.44,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long sleeve shirts, skirt, sandals",
        whole_body: 0.52,
        local_body_part: [
            0.0, 0.0, 1.62, 0.99, 1.41, 0.31, 0.28, 0.03, 0.31, 0.28, 0.03, 0.82, 0.04, 0.41, 0.82,
            0.04, 0.41,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, dress shirts, skirt, stocking, formal shoes",
        whole_body: 0.62,
        local_body_part: [
            0.0, 0.0, 1.58, 0.99, 1.31, 0.91, 0.64, 0.04, 0.91, 0.64, 0.04, 0.87, 0.05, 0.81, 0.87,
            0.05, 0.81,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, dress shirts, skirt, leggings, sandals",
        whole_body: 0.65,
        local_body_part: [
            0.13, 0.13, 1.59, 1.04, 1.36, 0.91, 0.67, 0.07, 0.91, 0.67, 0.07, 1.26, 0.12, 0.43,
            1.26, 0.12, 0.43,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, thin dress shirts, long pants, socks, sneakers",
        whole_body: 0.82,
        local_body_part: [
            0.0, 0.0, 3.35, 1.73, 1.63, 1.99, 1.49, 0.11, 1.99, 1.49, 0.11, 0.6, 0.43, 0.68, 0.6,
            0.43, 0.68,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long sleeve shirts, long pants, socks, sneakers",
        whole_body: 0.8,
        local_body_part: [
            0.0, 0.0, 2.47, 1.48, 1.58, 0.98, 0.58, 0.04, 0.98, 0.58, 0.04, 0.69, 0.65, 0.89, 0.69,
            0.65, 0.89,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, long sleeve shirts, long pants, socks, sneakers",
        whole_body: 0.83,
        local_body_part: [
            0.25, 0.25, 3.88, 2.28, 2.07, 1.89, 1.41, 0.16, 1.89, 1.41, 0.16, 0.83, 0.66, 0.86,
            0.83, 0.66, 0.86,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, jeans, socks, sneakers",
        whole_body: 0.57,
        local_body_part: [
            0.0, 0.0, 1.29, 0.93, 1.3, 0.68, 0.0, 0.0, 0.68, 0.0, 0.0, 0.65, 0.47, 0.73, 0.65,
            0.47, 0.73,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long sleeve shirts, jeans, socks, sneakers",
        whole_body: 0.74,
        local_body_part: [
            0.0, 0.0, 1.58, 0.98, 1.35, 0.86, 0.71, 0.07, 0.86, 0.71, 0.07, 0.74, 0.48, 0.74, 0.74,
            0.48, 0.74,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, oxford shirts, long thin pants, socks, sneakers",
        whole_body: 0.83,
        local_body_part: [
            0.16, 0.16, 1.39, 1.02, 1.34, 0.83, 0.69, 0.22, 0.83, 0.69, 0.22, 1.02, 0.68, 0.8,
            1.02, 0.68, 0.8,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, thin dress shirts (roll-up), long pants, socks, sneakers",
        whole_body: 0.81,
        local_body_part: [
            0.0, 0.0, 3.6, 1.83, 1.71, 2.16, 1.49, 0.13, 2.16, 1.49, 0.13, 0.64, 0.43, 0.69, 0.64,
            0.43, 0.69,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, short sleeve shirt, long pants, socks, sneakers",
        whole_body: 0.71,
        local_body_part: [
            0.12, 0.12, 2.15, 1.4, 1.71, 1.22, 0.02, 0.05, 1.22, 0.02, 0.05, 0.79, 0.48, 0.67,
            0.79, 0.48, 0.67,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, sports shirts, long pants, socks, sneakers",
        whole_body: 0.8,
        local_body_part: [
            0.05, 0.05, 1.92, 1.31, 1.41, 1.14, 0.86, 0.18, 1.14, 0.86, 0.18, 0.59, 0.49, 0.75,
            0.59, 0.49, 0.75,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, sports shirts, sports pants, sports socks, sports shoes",
        whole_body: 0.87,
        local_body_part: [
            0.07, 0.07, 1.87, 1.17, 1.26, 1.2, 1.07, 0.09, 1.2, 1.07, 0.09, 0.62, 0.77, 1.58, 0.62,
            0.77, 1.58,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, thin dress shirts, long pants, wool sweater, socks, sneakers",
        whole_body: 0.92,
        local_body_part: [
            0.09, 0.09, 2.39, 1.64, 1.71, 1.36, 1.29, 0.21, 1.36, 1.29, 0.21, 0.7, 0.52, 0.77, 0.7,
            0.52, 0.77,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, thin dress shirts, long pants, cashmere sweater, socks, sneakers",
        whole_body: 0.87,
        local_body_part: [
            0.1, 0.1, 2.4, 1.72, 1.67, 1.33, 1.23, 0.08, 1.33, 1.23, 0.08, 0.61, 0.47, 0.77, 0.61,
            0.47, 0.77,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, long sleeve shirts, long pants, winter jacket, socks, sneakers",
        whole_body: 1.18,
        local_body_part: [
            0.65, 0.65, 5.26, 3.07, 2.2, 3.14, 2.07, 0.08, 3.14, 2.07, 0.08, 0.67, 0.54, 0.77,
            0.67, 0.54, 0.77,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, long sleeve shirts, jeans, sports jumper, socks, sneakers",
        whole_body: 1.07,
        local_body_part: [
            0.28, 0.28, 3.99, 2.12, 2.0, 1.7, 1.36, 0.1, 1.7, 1.36, 0.1, 0.92, 0.48, 1.07, 0.92,
            0.48, 1.07,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, long sleeve shirts, long pants, ventura jacket, socks, sneakers",
        whole_body: 0.9,
        local_body_part: [
            0.09, 0.09, 2.66, 1.42, 1.57, 1.32, 0.99, 0.14, 1.32, 0.99, 0.14, 0.73, 0.66, 0.85,
            0.73, 0.66, 0.85,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtle neck, long pants, short trench coat, socks, sneakers",
        whole_body: 1.24,
        local_body_part: [
            0.06, 0.06, 3.22, 1.99, 2.03, 1.62, 1.5, 0.37, 1.62, 1.5, 0.37, 1.51, 0.65, 0.8, 1.51,
            0.65, 0.8,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, tank top, long sleeve shirts, blazer, skirt, sandals",
        whole_body: 0.86,
        local_body_part: [
            0.0, 0.0, 3.24, 1.81, 2.06, 1.98, 1.13, 0.07, 1.98, 1.13, 0.07, 1.19, 0.04, 0.44, 1.19,
            0.04, 0.44,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long sleeve shirts, wool skirt, socks, formal shoes",
        whole_body: 0.59,
        local_body_part: [
            0.0, 0.0, 1.21, 0.74, 1.56, 0.44, 0.24, 0.17, 0.44, 0.24, 0.17, 1.52, 0.09, 0.74, 1.52,
            0.09, 0.74,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtleneck, wool skirt, socks, formal shoes",
        whole_body: 0.7,
        local_body_part: [
            0.0, 0.0, 1.11, 0.94, 1.52, 0.73, 0.62, 0.14, 0.73, 0.62, 0.14, 1.53, 0.09, 0.85, 1.53,
            0.09, 0.85,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long sleeve shirt, wool skirt, sweater, socks, formal shoes",
        whole_body: 0.91,
        local_body_part: [
            0.14, 0.14, 2.82, 1.53, 1.79, 1.22, 0.97, 0.08, 1.22, 0.97, 0.08, 1.53, 0.11, 0.83,
            1.53, 0.11, 0.83,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, thin dress shirts, slacks, tie, socks, sneakers",
        whole_body: 0.57,
        local_body_part: [
            0.0, 0.0, 1.69, 0.8, 1.08, 0.67, 0.58, 0.07, 0.67, 0.58, 0.07, 0.36, 0.39, 0.74, 0.36,
            0.39, 0.74,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, thin dress shirts, slacks, blazer, tie, belt, socks, formal shoes",
        whole_body: 0.93,
        local_body_part: [
            0.0, 0.0, 3.6, 1.83, 1.71, 2.16, 1.49, 0.13, 2.16, 1.49, 0.13, 0.64, 0.43, 0.69, 0.64,
            0.43, 0.69,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, long sleeve shirts, long pants, blazer, socks, sneakers",
        whole_body: 0.96,
        local_body_part: [
            0.04, 0.04, 3.3, 1.67, 2.2, 2.1, 1.43, 0.09, 2.1, 1.43, 0.09, 0.72, 0.42, 0.67, 0.72,
            0.42, 0.67,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, T-shirt, long sleeve shirts, long pants, winter jacket (Notica)",
        whole_body: 1.05,
        local_body_part: [
            0.04, 0.04, 3.88, 2.26, 1.97, 1.82, 1.46, 0.17, 1.82, 1.46, 0.17, 0.81, 0.57, 0.78,
            0.81, 0.57, 0.78,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtle neck, ski-jumper, skin pants, sports socks, sports shoes",
        whole_body: 1.84,
        local_body_part: [
            0.89, 0.89, 5.24, 2.87, 2.64, 2.55, 2.16, 0.46, 2.55, 2.16, 0.46, 1.49, 1.82, 1.56,
            1.49, 1.82, 1.56,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtle neck, ski-jumper and hood, skin pants, sports socks, sports shoes",
        whole_body: 1.87,
        local_body_part: [
            1.63, 1.63, 5.12, 2.7, 2.57, 2.58, 2.16, 0.49, 2.58, 2.16, 0.49, 1.44, 1.76, 1.54,
            1.44, 1.76, 1.54,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtle neck, goose down, ski pants, sports socks, sports shoes",
        whole_body: 2.53,
        local_body_part: [
            1.17, 1.17, 15.44, 5.5, 5.2, 6.55, 5.58, 0.35, 6.55, 5.58, 0.35, 2.12, 1.7, 1.54, 2.12,
            1.7, 1.54,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtle neck, goose down-with hood, ski pants, sports socks, sports shoes",
        whole_body: 2.75,
        local_body_part: [
            3.52, 3.52, 12.62, 3.99, 5.05, 6.2, 5.73, 0.53, 6.2, 5.73, 0.53, 2.11, 1.81, 1.58,
            2.11, 1.81, 1.58,
        ],
    },
    ClothingEnsemble {
        name: "bra+panty, turtle neck, goose down-with hood and gloves, ski pants, sports socks, sports shoes",
        whole_body: 3.27,
        local_body_part: [
            3.92, 3.92, 16.13, 4.47, 5.71, 7.12, 5.37, 2.54, 7.12, 5.37, 2.54, 2.14, 1.82, 1.61,
            2.14, 1.82, 1.61,
        ],
    },
    ClothingEnsemble {
        name: "briefs, socks, T-shirt, half pants, sneakers",
        whole_body: 0.53,
        local_body_part: [
            0.0, 0.0, 0.5, 1.13, 1.21, 0.39, 0.0, 0.0, 0.39, 0.0, 0.0, 0.94, 0.07, 0.62, 0.94,
            0.07, 0.62,
        ],
    },
    ClothingEnsemble {
        name: "briefs, undershirt, sports t-shirts, sports shorts",
        whole_body: 0.7,
        local_body_part: [
            0.0, 0.0, 0.9, 1.76, 2.14, 0.48, 0.0, 0.0, 0.48, 0.0, 0.0, 1.18, 0.0, 0.01, 1.18, 0.0,
            0.01,
        ],
    },
    ClothingEnsemble {
        name: "briefs, socks, polo shirt, long pants, sneakers",
        whole_body: 0.54,
        local_body_part: [
            0.02, 0.02, 0.52, 0.97, 1.12, 0.34, 0.0, 0.0, 0.34, 0.0, 0.0, 0.79, 0.56, 0.65, 0.79,
            0.56, 0.65,
        ],
    },
    ClothingEnsemble {
        name: "briefs, under shirt, long-sleeved shirt, long pants",
        whole_body: 1.01,
        local_body_part: [
            0.0, 0.0, 1.2, 1.65, 2.29, 0.98, 0.78, 0.03, 0.98, 0.78, 0.03, 1.46, 0.62, 0.02, 1.46,
            0.62, 0.02,
        ],
    },
    ClothingEnsemble {
        name: "briefs, socks, undershirt, short-sleeved shirt, long pants, belt, shoes",
        whole_body: 0.72,
        local_body_part: [
            0.01, 0.01, 0.8, 1.4, 1.55, 0.54, 0.0, 0.0, 0.54, 0.0, 0.0, 0.89, 0.64, 0.99, 0.89,
            0.64, 0.99,
        ],
    },
    ClothingEnsemble {
        name: "briefs, socks, undershirt, long-sleeved shirt, long pants, belt",
        whole_body: 0.77,
        local_body_part: [
            0.0, 0.0, 0.86, 1.45, 1.54, 0.82, 0.6, 0.01, 0.82, 0.6, 0.01, 0.9, 0.66, 0.64, 0.9,
            0.66, 0.64,
        ],
    },
    ClothingEnsemble {
        name: "briefs, socks, undershirt, long-sleeved shirt, jacket, long pants, belt, shoes",
        whole_body: 1.39,
        local_body_part: [
            0.02, 0.02, 2.13, 2.28, 3.04, 1.8, 1.54, 0.15, 1.8, 1.54, 0.15, 1.33, 0.69, 0.97, 1.33,
            0.69, 0.97,
        ],
    },
    ClothingEnsemble {
        name: "briefs, socks, undershirt, work jacket, work pants, safety shoes",
        whole_body: 0.8,
        local_body_part: [
            0.0, 0.0, 1.25, 1.39, 1.78, 0.84, 0.71, 0.08, 0.84, 0.71, 0.08, 0.65, 0.59, 1.12, 0.65,
            0.59, 1.12,
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_body_parts_is_17() {
        assert_eq!(NUM_BODY_PARTS, 17);
        assert_eq!(BODY_PART_NAMES.len(), 17);
        assert_eq!(defaults::LOCAL_BSA.len(), 17);
    }

    #[test]
    fn body_part_order_matches_python_jos3bodyparts() {
        assert_eq!(
            BODY_PART_NAMES,
            [
                "head",
                "neck",
                "chest",
                "back",
                "pelvis",
                "left_shoulder",
                "left_arm",
                "left_hand",
                "right_shoulder",
                "right_arm",
                "right_hand",
                "left_thigh",
                "left_leg",
                "left_foot",
                "right_thigh",
                "right_leg",
                "right_foot",
            ]
        );
    }

    #[test]
    fn local_clo_typical_ensembles_len_matches_python() {
        assert_eq!(LOCAL_CLO_TYPICAL_ENSEMBLES.len(), 52);
        for ensemble in LOCAL_CLO_TYPICAL_ENSEMBLES {
            assert_eq!(ensemble.local_body_part.len(), 17);
        }
    }

    #[test]
    fn local_bsa_sum_matches_python() {
        let sum: f64 = defaults::LOCAL_BSA.iter().sum();
        assert!((sum - 1.8680000000000005).abs() < 1e-12);
    }

    #[test]
    fn local_clo_typical_ensembles_first_and_last_match_python() {
        let first = LOCAL_CLO_TYPICAL_ENSEMBLES.first().unwrap();
        assert_eq!(first.name, "nude (mesh chair)");
        assert!((first.whole_body - 0.01).abs() < 1e-12);
        let last = LOCAL_CLO_TYPICAL_ENSEMBLES.last().unwrap();
        assert_eq!(
            last.name,
            "briefs, socks, undershirt, work jacket, work pants, safety shoes"
        );
        assert!((last.whole_body - 0.8).abs() < 1e-12);
    }

    #[test]
    fn local_clo_typical_ensembles_grand_total_matches_python() {
        // Python: sum(whole_body) + sum(all local_body_part values), computed by
        // iterating pythermalcomfort.jos3_functions.parameters.local_clo_typical_ensembles.
        let mut total = 0.0_f64;
        for ensemble in LOCAL_CLO_TYPICAL_ENSEMBLES {
            total += ensemble.whole_body;
            total += ensemble.local_body_part.iter().sum::<f64>();
        }
        assert!((total - 865.4900000000004).abs() < 1e-9);
    }
}
