//! Utility functions for thermal comfort calculations

use crate::ClothingInsulation;
use core::time::Duration;
use libm::{copysign, fabs, pow, round, trunc};
pub use measurements::{Area, Humidity, Length, Mass, Pressure, Speed, Temperature};

/// Convert Temperature to Celsius (f64)
#[inline]
pub fn temp_to_celsius(temp: Temperature) -> f64 {
    temp.as_celsius()
}

/// Convert f64 Celsius to Temperature
#[inline]
pub fn celsius_to_temp(temp_c: f64) -> Temperature {
    Temperature::from_celsius(temp_c)
}

/// Convert Speed to m/s (f64)
#[inline]
pub fn speed_to_ms(speed: Speed) -> f64 {
    speed.as_meters_per_second()
}

/// Convert f64 m/s to Speed
#[inline]
pub fn ms_to_speed(speed_ms: f64) -> Speed {
    Speed::from_meters_per_second(speed_ms)
}

/// Units for thermal comfort calculations
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Units {
    /// SI (International System of Units)
    #[default]
    SI,
    /// IP (Imperial Units)
    IP,
}

/// Model standards
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Model {
    /// ASHRAE 55-2023
    Ashrae552023,
    /// ISO 7730-2005
    #[default]
    Iso77302005,
    /// ISO 9920-2007
    Iso99202007,
    /// ISO 7933-2004
    Iso79332004,
    /// ISO 7933-2023
    Iso79332023,
}

/// Whole minutes in a duration for a model that steps once per minute, rounded to the
/// nearest minute. A duration under thirty seconds is zero steps, which is what upstream
/// does with a zero duration: empty histories, limits reported as zero.
pub(crate) fn whole_minutes(duration: Duration) -> usize {
    round(duration.as_secs_f64() / 60.0) as usize
}

/// A `Duration` from a (possibly fractional) number of minutes.
pub(crate) fn minutes(minutes: f64) -> Duration {
    Duration::from_secs_f64(minutes * 60.0)
}

/// Mirrors upstream's private `_valid_range`: the value inside `[min, max]`, else NaN.
#[inline]
pub(crate) fn valid_range(value: f64, min: f64, max: f64) -> f64 {
    if value >= min && value <= max {
        value
    } else {
        f64::NAN
    }
}

// ---------------------------------------------------------------------------
// NaN-faithful min/max
// ---------------------------------------------------------------------------
//
// Several models legitimately produce NaN: `valid_range` above returns NaN for any
// out-of-range input, and JOS3's operative-temperature PMV search returns NaN whenever a
// subject's reference metabolic rate falls below ISO 7730's 0.8 met floor. A clamp that
// silently drops the NaN would heal an invalid calculation into a plausible-looking wrong
// number, so every clamp has to reproduce the NaN behaviour of the exact Python construct
// it ports.
//
// `libm::fmin`/`fmax` are IEEE-754/C99 `fmin`/`fmax`, which *discard* NaN and return the
// other operand (`fmin(NaN, 1.0) == 1.0`). That matches neither Python construct, so
// neither should be used for a ported min/max: use one of the four helpers below.
//
// Rust's `f64::min`/`max` are also wrong here — they too return the non-NaN operand.
// `f64::clamp` is the one built-in that already propagates, and it matches Python's
// `min(max(x, lo), hi)` exactly, so clamp sites need no helper.

/// Mirrors CPython's two-argument builtin `min(a, b)`, which is `b if b < a else a`.
///
/// NaN handling is therefore *order-dependent*: `min(nan, 1) == nan` but
/// `min(1, nan) == 1`, because a NaN in `b` makes `b < a` false and loses. Callers
/// must preserve Python's argument order. This is not `libm::fmin`, which always
/// discards NaN, nor `np.minimum`, which always propagates it.
#[inline]
pub(crate) fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Mirrors CPython's two-argument builtin `max(a, b)`, which is `b if b > a else a`.
///
/// NaN handling is therefore *order-dependent*: `max(nan, 1) == nan` but
/// `max(1, nan) == 1`, because a NaN in `b` makes `b > a` false and loses. Callers
/// must preserve Python's argument order. This is not `libm::fmax`, which always
/// discards NaN, nor `np.maximum`, which always propagates it.
#[inline]
pub(crate) fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Mirrors `np.minimum(a, b)`, which propagates NaN from *either* operand.
///
/// This is not `libm::fmin`, which returns the non-NaN operand instead, and not the
/// builtin `min`, whose NaN behaviour depends on argument order.
#[inline]
pub(crate) fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// Mirrors `np.maximum(a, b)`, which propagates NaN from *either* operand.
///
/// This is not `libm::fmax`, which returns the non-NaN operand instead, and not the
/// builtin `max`, whose NaN behaviour depends on argument order.
#[inline]
pub(crate) fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// Round to specified decimal places, half-to-even.
///
/// Matches `numpy.around`, which pythermalcomfort uses for every rounded output.
/// `libm::round` rounds half *away from zero*, which disagrees on ties — and ties are
/// common here because these values are products of already-2-decimal quantities. For
/// example `adaptive_en` at `t_running_mean = 25` yields a raw 29.05, which numpy
/// renders 29.0 and half-away-from-zero renders 29.1.
#[inline]
pub fn round_to(value: f64, decimals: i32) -> f64 {
    let multiplier = pow(10.0, decimals as f64);
    round_half_even(value * multiplier) / multiplier
}

/// Round to specified decimal places the way CPython's *builtin* `round(x, n)` does.
///
/// This is a genuinely different function from [`round_to`], not a stylistic variant.
/// `round_to` (and `numpy.round`) multiplies by `10^n` and rounds the result; the builtin
/// rounds the *exact* decimal expansion of the binary value. The two disagree whenever
/// `value * 10^n` lands on the far side of a half-way point from where the exact value
/// sits — e.g. `-22.285` is really `-22.28500000000000085…`, which the builtin rounds to
/// `-22.29` while `numpy.round` computes `-2228.4999999999995` and returns `-22.28`.
/// Measured against CPython over 140,000 values spanning the tie grid and its
/// neighbourhood: this function agrees on all of them, `round_to` on 136,989.
///
/// pythermalcomfort mixes both in one function. Most of its builtin `round(…)` calls take
/// a `numpy.float64`, which overrides `__round__` to numpy's rule, so they are
/// [`round_to`] in disguise. `JOS3.t_cb` is the exception: its property casts through
/// `float(...)` first (`models/jos3.py:1606-1608`), so `round(self.t_cb, 2)` at
/// `models/jos3.py:1073` really is the builtin's decimal rounding.
///
/// Implemented by formatting and re-parsing. That is inelegant and not free — it is the
/// only correctly-rounded decimal conversion available here, since matching the builtin
/// requires the exact decimal expansion of the binary value and no amount of `f64`
/// arithmetic reproduces that. `core`'s float `Display` is correctly rounded with
/// ties-to-even, which is precisely CPython's `_Py_dg_dtoa` contract. It writes into a
/// stack buffer via `heapless`, so this stays `no_std` and allocation-free.
pub fn round_to_exact_decimal(value: f64, decimals: usize) -> f64 {
    use core::fmt::Write;

    // Above this magnitude every f64 is already an integer, so rounding is the identity
    // and the formatted string would overflow the buffer below. Also covers NaN and
    // infinity, which have no decimal expansion to round.
    let magnitude = fabs(value);
    if magnitude.is_nan() || magnitude >= 1e15 {
        return value;
    }

    // 16 integer digits at most (1e15), plus sign, point, and up to 5 decimals: 32 is
    // ample for every call site in this crate.
    let mut buffer: heapless::String<32> = heapless::String::new();
    if write!(buffer, "{value:.decimals$}").is_err() {
        // Unreachable given the magnitude guard; degrade to numpy's rule rather than
        // panic in a library that may be running on a microcontroller.
        return round_to(value, decimals as i32);
    }
    buffer.parse().unwrap_or(value)
}

/// Round to the nearest integer, ties to even — `numpy.around`'s rule.
#[inline]
pub fn round_half_even(value: f64) -> f64 {
    let rounded = round(value);
    // A tie is exactly .5 away from an integer; send it to the even neighbour.
    // libm rather than f64 methods: `trunc`, `abs` and `signum` are std-only and this
    // crate is no_std by default.
    if fabs(value - trunc(value)) == 0.5 && fabs(rounded % 2.0) != 0.0 {
        rounded - copysign(1.0, value)
    } else {
        rounded
    }
}

/// Formula options for body surface area calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BsaFormula {
    /// DuBois formula (1916) - most widely used
    #[default]
    DuBois,
    /// Takahira formula (1925)
    Takahira,
    /// Fujimoto formula (1968)
    Fujimoto,
    /// Kurazumi formula (1994)
    Kurazumi,
}

/// Calculate body surface area using DuBois formula (m²)
///
/// # Arguments
///
/// * `weight` - Body weight (kg)
/// * `height` - Body height (m)
///
/// # Returns
///
/// Body surface area
#[inline]
pub fn body_surface_area_dubois(weight: Mass, height: Length) -> Area {
    let weight_kg = weight.as_kilograms();
    let height_m = height.as_meters();
    Area::from_square_meters(0.202 * pow(weight_kg, 0.425) * pow(height_m, 0.725))
}

/// The inputs to [`body_surface_area`]: weight and height are both required in
/// pythermalcomfort's signature (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodySurfaceAreaInputs {
    /// Body weight
    pub weight: Mass,
    /// Body height
    pub height: Length,
}

/// Optional parameters for [`body_surface_area`], with pythermalcomfort's default
/// (`formula: str = BodySurfaceAreaEquations.dubois.value`, `utilities.py:610-616`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BodySurfaceAreaOptions {
    /// Formula to use for calculation
    pub formula: BsaFormula,
}

/// Calculate body surface area using various formulas
///
/// # Returns
///
/// Body surface area
///
/// # Formulas
///
/// - **DuBois (1916)**: BSA = 0.202 * weight^0.425 * height^0.725
/// - **Takahira (1925)**: BSA = 0.2042 * weight^0.425 * height^0.725
/// - **Fujimoto (1968)**: BSA = 0.1882 * weight^0.444 * height^0.663
/// - **Kurazumi (1994)**: BSA = 0.2440 * weight^0.383 * height^0.693
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{body_surface_area, BodySurfaceAreaInputs, BodySurfaceAreaOptions, BsaFormula};
/// use thermalcomfort::{Mass, Length};
///
/// let bsa = body_surface_area(
///     BodySurfaceAreaInputs {
///         weight: Mass::from_kilograms(70.0),
///         height: Length::from_meters(1.75),
///     },
///     BodySurfaceAreaOptions { formula: BsaFormula::DuBois },
/// );
/// assert!((bsa.as_square_meters() - 1.844).abs() < 0.01);
/// ```
pub fn body_surface_area(inputs: BodySurfaceAreaInputs, options: BodySurfaceAreaOptions) -> Area {
    let BodySurfaceAreaInputs { weight, height } = inputs;
    let BodySurfaceAreaOptions { formula } = options;
    let weight_kg = weight.as_kilograms();
    let height_m = height.as_meters();
    let area_m2 = match formula {
        BsaFormula::DuBois => 0.202 * pow(weight_kg, 0.425) * pow(height_m, 0.725),
        BsaFormula::Takahira => 0.2042 * pow(weight_kg, 0.425) * pow(height_m, 0.725),
        BsaFormula::Fujimoto => 0.1882 * pow(weight_kg, 0.444) * pow(height_m, 0.663),
        BsaFormula::Kurazumi => 0.2440 * pow(weight_kg, 0.383) * pow(height_m, 0.693),
    };
    Area::from_square_meters(area_m2)
}

/// Calculate representative clothing insulation based on outdoor temperature
///
/// Estimates clothing insulation (clo) based on outdoor air temperature at 06:00 a.m.
/// according to Schiavon et al. (2013). This is useful for estimating typical indoor
/// clothing levels in mechanically conditioned buildings.
///
/// # Arguments
///
/// * `tout` - Outdoor air temperature at 06:00 a.m. (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Representative clothing insulation
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_tout;
/// use thermalcomfort::Temperature;
///
/// let clo = clo_tout(Temperature::from_celsius(27.0));
/// assert!((clo.as_clo() - 0.46).abs() < 0.01);
/// ```
///
/// # References
///
/// - Schiavon et al. (2013)
/// - ASHRAE 55-2020
///
/// # Notes
///
/// - For tout < -5°C: clo = 1.0
/// - For -5°C ≤ tout < 5°C: clo = 0.818 - 0.0364 * tout
/// - For 5°C ≤ tout < 26°C: clo = 10^(-0.1635 - 0.0066 * tout)
/// - For tout ≥ 26°C: clo = 0.46
pub fn clo_tout(tout: Temperature) -> ClothingInsulation {
    let tout_celsius = tout.as_celsius();
    let clo = if tout_celsius < -5.0 {
        1.0
    } else if tout_celsius < 5.0 {
        0.818 - 0.0364 * tout_celsius
    } else if tout_celsius < 26.0 {
        pow(10.0, -0.1635 - 0.0066 * tout_celsius)
    } else {
        0.46
    };

    ClothingInsulation::from_clo(round(clo * 100.0) / 100.0)
}

/// Typical clothing ensembles with predefined insulation values
///
/// These represent common clothing combinations based on ASHRAE 55 and ISO 9920.
/// Each tuple contains (ensemble_name, insulation_clo).
pub const CLO_TYPICAL_ENSEMBLES: &[(&str, ClothingInsulation)] = &[
    (
        "Walking shorts, short-sleeve shirt",
        ClothingInsulation::from_clo(0.36),
    ),
    (
        "Typical summer indoor clothing",
        ClothingInsulation::from_clo(0.5),
    ),
    (
        "Knee-length skirt, short-sleeve shirt, sandals, underwear",
        ClothingInsulation::from_clo(0.54),
    ),
    (
        "Trousers, short-sleeve shirt, socks, shoes, underwear",
        ClothingInsulation::from_clo(0.57),
    ),
    (
        "Trousers, long-sleeve shirt",
        ClothingInsulation::from_clo(0.61),
    ),
    (
        "Knee-length skirt, long-sleeve shirt, full slip",
        ClothingInsulation::from_clo(0.67),
    ),
    (
        "Sweat pants, long-sleeve sweatshirt",
        ClothingInsulation::from_clo(0.74),
    ),
    (
        "Jacket, Trousers, long-sleeve shirt",
        ClothingInsulation::from_clo(0.96),
    ),
    (
        "Typical winter indoor clothing",
        ClothingInsulation::from_clo(1.0),
    ),
];

/// Individual garment insulation values
///
/// Insulation values for individual clothing items based on ISO 9920:2009.
/// Each tuple contains (garment_name, insulation).
pub const CLO_INDIVIDUAL_GARMENTS: &[(&str, ClothingInsulation)] = &[
    ("Metal chair", ClothingInsulation::from_clo(0.0)),
    ("Bra", ClothingInsulation::from_clo(0.01)),
    ("Wooden stool", ClothingInsulation::from_clo(0.01)),
    ("Ankle socks", ClothingInsulation::from_clo(0.02)),
    ("Shoes or sandals", ClothingInsulation::from_clo(0.02)),
    ("Slippers", ClothingInsulation::from_clo(0.03)),
    ("Panty hose", ClothingInsulation::from_clo(0.02)),
    ("Calf length socks", ClothingInsulation::from_clo(0.03)),
    ("Women's underwear", ClothingInsulation::from_clo(0.03)),
    ("Men's underwear", ClothingInsulation::from_clo(0.04)),
    ("Knee socks (thick)", ClothingInsulation::from_clo(0.06)),
    ("Short shorts", ClothingInsulation::from_clo(0.06)),
    ("Walking shorts", ClothingInsulation::from_clo(0.08)),
    ("T-shirt", ClothingInsulation::from_clo(0.08)),
    ("Standard office chair", ClothingInsulation::from_clo(0.1)),
    ("Executive chair", ClothingInsulation::from_clo(0.15)),
    ("Boots", ClothingInsulation::from_clo(0.1)),
    (
        "Sleeveless scoop-neck blouse",
        ClothingInsulation::from_clo(0.12),
    ),
    ("Half slip", ClothingInsulation::from_clo(0.14)),
    ("Long underwear bottoms", ClothingInsulation::from_clo(0.15)),
    ("Full slip", ClothingInsulation::from_clo(0.16)),
    (
        "Short-sleeve knit shirt",
        ClothingInsulation::from_clo(0.17),
    ),
    ("Sleeveless vest (thin)", ClothingInsulation::from_clo(0.1)),
    (
        "Sleeveless vest (thick)",
        ClothingInsulation::from_clo(0.17),
    ),
    (
        "Sleeveless short gown (thin)",
        ClothingInsulation::from_clo(0.18),
    ),
    (
        "Short-sleeve dress shirt",
        ClothingInsulation::from_clo(0.19),
    ),
    (
        "Sleeveless long gown (thin)",
        ClothingInsulation::from_clo(0.2),
    ),
    ("Long underwear top", ClothingInsulation::from_clo(0.2)),
    ("Thick skirt", ClothingInsulation::from_clo(0.23)),
    (
        "Long-sleeve dress shirt",
        ClothingInsulation::from_clo(0.25),
    ),
    (
        "Long-sleeve flannel shirt",
        ClothingInsulation::from_clo(0.34),
    ),
    (
        "Long-sleeve sweat shirt",
        ClothingInsulation::from_clo(0.34),
    ),
    (
        "Short-sleeve hospital gown",
        ClothingInsulation::from_clo(0.31),
    ),
    (
        "Short-sleeve short robe (thin)",
        ClothingInsulation::from_clo(0.34),
    ),
    ("Short-sleeve pajamas", ClothingInsulation::from_clo(0.42)),
    ("Long-sleeve long gown", ClothingInsulation::from_clo(0.46)),
    (
        "Long-sleeve short wrap robe (thick)",
        ClothingInsulation::from_clo(0.48),
    ),
    (
        "Long-sleeve pajamas (thick)",
        ClothingInsulation::from_clo(0.57),
    ),
    (
        "Long-sleeve long wrap robe (thick)",
        ClothingInsulation::from_clo(0.69),
    ),
    ("Thin trousers", ClothingInsulation::from_clo(0.15)),
    ("Thick trousers", ClothingInsulation::from_clo(0.24)),
    ("Sweatpants", ClothingInsulation::from_clo(0.28)),
    ("Overalls", ClothingInsulation::from_clo(0.3)),
    ("Coveralls", ClothingInsulation::from_clo(0.49)),
    ("Thin skirt", ClothingInsulation::from_clo(0.14)),
    (
        "Long-sleeve shirt dress (thin)",
        ClothingInsulation::from_clo(0.33),
    ),
    (
        "Long-sleeve shirt dress (thick)",
        ClothingInsulation::from_clo(0.47),
    ),
    (
        "Short-sleeve shirt dress",
        ClothingInsulation::from_clo(0.29),
    ),
    (
        "Sleeveless, scoop-neck shirt (thin)",
        ClothingInsulation::from_clo(0.23),
    ),
    (
        "Sleeveless, scoop-neck shirt (thick)",
        ClothingInsulation::from_clo(0.27),
    ),
    (
        "Long sleeve shirt (thin)",
        ClothingInsulation::from_clo(0.25),
    ),
    (
        "Long sleeve shirt (thick)",
        ClothingInsulation::from_clo(0.36),
    ),
    (
        "Single-breasted coat (thin)",
        ClothingInsulation::from_clo(0.36),
    ),
    (
        "Single-breasted coat (thick)",
        ClothingInsulation::from_clo(0.44),
    ),
    (
        "Double-breasted coat (thin)",
        ClothingInsulation::from_clo(0.42),
    ),
    (
        "Double-breasted coat (thick)",
        ClothingInsulation::from_clo(0.48),
    ),
];

/// Look up clothing insulation value for a typical ensemble
///
/// Returns the insulation of a typical clothing ensemble.
///
/// # Arguments
///
/// * `ensemble_name` - Name of the clothing ensemble (case-sensitive)
///
/// # Returns
///
/// Some(insulation) if the ensemble is found, None otherwise
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_typical_ensemble;
/// use thermalcomfort::ClothingInsulation;
///
/// let clo = clo_typical_ensemble("Typical summer indoor clothing");
/// assert_eq!(clo, Some(ClothingInsulation::from_clo(0.5)));
///
/// let not_found = clo_typical_ensemble("Unknown ensemble");
/// assert_eq!(not_found, None);
/// ```
pub fn clo_typical_ensemble(ensemble_name: &str) -> Option<ClothingInsulation> {
    CLO_TYPICAL_ENSEMBLES
        .iter()
        .find(|(name, _)| *name == ensemble_name)
        .map(|(_, clo)| *clo)
}

/// Look up clothing insulation value for an individual garment
///
/// Returns the insulation of an individual garment.
///
/// # Arguments
///
/// * `garment_name` - Name of the garment (case-sensitive)
///
/// # Returns
///
/// Some(insulation) if the garment is found, None otherwise
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_individual_garment;
/// use thermalcomfort::ClothingInsulation;
///
/// let clo = clo_individual_garment("Long-sleeve dress shirt");
/// assert_eq!(clo, Some(ClothingInsulation::from_clo(0.25)));
///
/// let not_found = clo_individual_garment("Unknown garment");
/// assert_eq!(not_found, None);
/// ```
pub fn clo_individual_garment(garment_name: &str) -> Option<ClothingInsulation> {
    CLO_INDIVIDUAL_GARMENTS
        .iter()
        .find(|(name, _)| *name == garment_name)
        .map(|(_, clo)| *clo)
}

// ---------------------------------------------------------------------------
// Paths pythermalcomfort 4.6.0 moved elsewhere. Each forwards to its new home and is
// removed after two minor releases, as upstream's compatibility shims are.
// ---------------------------------------------------------------------------

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
pub type RunningMeanOutdoorTemperatureOptions =
    crate::environment::RunningMeanOutdoorTemperatureOptions;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
#[inline]
pub fn running_mean_outdoor_temperature(
    temp_array: &[Temperature],
    options: crate::environment::RunningMeanOutdoorTemperatureOptions,
) -> Temperature {
    crate::environment::running_mean_outdoor_temperature(temp_array, options)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
#[inline]
pub fn v_relative(v: Speed, met: crate::MetabolicRate) -> Speed {
    crate::environment::v_relative(v, met)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::psychrometrics`")]
#[inline]
pub fn antoine(tdb: Temperature) -> Pressure {
    crate::psychrometrics::antoine(tdb)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::psychrometrics`")]
#[inline]
pub fn p_sat_torr(tdb: Temperature) -> Pressure {
    crate::psychrometrics::p_sat_torr(tdb)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::psychrometrics`")]
#[inline]
pub fn p_sat(tdb: Temperature) -> Pressure {
    crate::psychrometrics::p_sat(tdb)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::psychrometrics`")]
#[inline]
pub fn hr_to_rh(hr: f64, tdb: Temperature, p_atm: Pressure) -> Humidity {
    crate::psychrometrics::hr_to_rh(hr, tdb, p_atm)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type Ashrae55Model = crate::clothing::Ashrae55Model;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type Iso9920Model = crate::clothing::Iso9920Model;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloDynamicAshraeInputs = crate::clothing::CloDynamicAshraeInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloDynamicAshraeOptions = crate::clothing::CloDynamicAshraeOptions;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloDynamicIsoInputs = crate::clothing::CloDynamicIsoInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloDynamicIsoOptions = crate::clothing::CloDynamicIsoOptions;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloInsulationAirLayerInputs = crate::clothing::CloInsulationAirLayerInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloTotalInsulationInputs = crate::clothing::CloTotalInsulationInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
pub type CloCorrectionFactorEnvironmentInputs =
    crate::clothing::CloCorrectionFactorEnvironmentInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_area_factor(i_cl: ClothingInsulation) -> f64 {
    crate::clothing::clo_area_factor(i_cl)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_dynamic_ashrae(
    inputs: crate::clothing::CloDynamicAshraeInputs,
    options: crate::clothing::CloDynamicAshraeOptions,
) -> ClothingInsulation {
    crate::clothing::clo_dynamic_ashrae(inputs, options)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_dynamic_iso(
    inputs: crate::clothing::CloDynamicIsoInputs,
    options: crate::clothing::CloDynamicIsoOptions,
) -> ClothingInsulation {
    crate::clothing::clo_dynamic_iso(inputs, options)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_insulation_air_layer(
    inputs: crate::clothing::CloInsulationAirLayerInputs,
) -> ClothingInsulation {
    crate::clothing::clo_insulation_air_layer(inputs)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_total_insulation(
    inputs: crate::clothing::CloTotalInsulationInputs,
) -> ClothingInsulation {
    crate::clothing::clo_total_insulation(inputs)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_correction_factor_environment(
    inputs: crate::clothing::CloCorrectionFactorEnvironmentInputs,
) -> f64 {
    crate::clothing::clo_correction_factor_environment(inputs)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::clothing`")]
#[inline]
pub fn clo_intrinsic_insulation_ensemble(
    clo_garments: &[ClothingInsulation],
) -> ClothingInsulation {
    crate::clothing::clo_intrinsic_insulation_ensemble(clo_garments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `round_to_exact_decimal` reproduces CPython's builtin `round(x, n)`, which is a
    /// different function from `round_to`/`numpy.round`. Expected values were taken from
    /// CPython 3.10 directly; the `round_to` column records where the two part company,
    /// so a future "simplification" that collapses them back into one helper fails here.
    #[test]
    fn round_to_exact_decimal_matches_cpython_builtin_round() {
        // (value, decimals, CPython round(value, decimals), numpy-rule round_to)
        for (value, decimals, cpython, numpy_rule) in [
            // Off the tie grid the two agree.
            (36.758_094_325_901_3_f64, 2, 36.76, 36.76),
            (0.029_628_715_270_251_08, 5, 0.02963, 0.02963),
            // Exact decimal ties: both round half to even, and agree.
            (36.125, 2, 36.12, 36.12),
            (36.375, 2, 36.38, 36.38),
            // Not ties at all: the exact expansion of these is just above the half-way
            // point, but multiplying by 100 first drops below it. CPython rounds up,
            // numpy down.
            (-22.285, 2, -22.29, -22.28),
            (70.685, 2, 70.69, 70.68),
            (49.185, 2, 49.19, 49.18),
            // 5e-06 is really 5.0000000000000004e-06, so the builtin rounds it away.
            (5e-06, 5, 1e-05, 0.0),
        ] {
            assert_eq!(
                round_to_exact_decimal(value, decimals),
                cpython,
                "round_to_exact_decimal({value}, {decimals})"
            );
            assert_eq!(
                round_to(value, decimals as i32),
                numpy_rule,
                "round_to({value}, {decimals})"
            );
        }
    }

    /// Values too large to have a fractional part, and non-finite ones, pass through.
    #[test]
    fn round_to_exact_decimal_passes_through_values_it_cannot_change() {
        assert_eq!(round_to_exact_decimal(1e300, 2), 1e300);
        assert_eq!(round_to_exact_decimal(-1e300, 5), -1e300);
        assert_eq!(round_to_exact_decimal(f64::INFINITY, 2), f64::INFINITY);
        assert!(round_to_exact_decimal(f64::NAN, 2).is_nan());
    }

    #[test]
    fn test_valid_range() {
        assert_eq!(valid_range(15.0, 10.0, 30.0), 15.0);
        assert!(valid_range(5.0, 10.0, 30.0).is_nan());
        assert!(valid_range(35.0, 10.0, 30.0).is_nan());
    }

    #[test]
    fn test_clo_typical_ensemble() {
        // Test valid lookups
        assert_eq!(
            clo_typical_ensemble("Typical summer indoor clothing"),
            Some(ClothingInsulation::from_clo(0.5))
        );
        assert_eq!(
            clo_typical_ensemble("Typical winter indoor clothing"),
            Some(ClothingInsulation::from_clo(1.0))
        );
        assert_eq!(
            clo_typical_ensemble("Walking shorts, short-sleeve shirt"),
            Some(ClothingInsulation::from_clo(0.36))
        );

        // Test invalid lookup
        assert_eq!(clo_typical_ensemble("Non-existent ensemble"), None);
    }

    #[test]
    fn test_clo_individual_garment() {
        // Test valid lookups
        assert_eq!(
            clo_individual_garment("Long-sleeve dress shirt"),
            Some(ClothingInsulation::from_clo(0.25))
        );
        assert_eq!(
            clo_individual_garment("Thick trousers"),
            Some(ClothingInsulation::from_clo(0.24))
        );
        assert_eq!(
            clo_individual_garment("T-shirt"),
            Some(ClothingInsulation::from_clo(0.08))
        );

        // Test invalid lookup
        assert_eq!(clo_individual_garment("Non-existent garment"), None);
    }
}
