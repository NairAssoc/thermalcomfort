//! Utility functions for thermal comfort calculations

use crate::constants::*;
use crate::{ClothingInsulation, MetabolicRate};
use libm::{copysign, exp, fabs, log, pow, round, trunc};
pub use measurements::{Area, Length, Mass, Pressure, Speed, Temperature};

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

/// Body postures for thermal comfort calculations
///
/// Different postures affect the radiative heat transfer coefficient
/// and body surface area exposed to the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Posture {
    /// Standing posture (0.73 radiation area ratio)
    #[default]
    Standing,
    /// Sitting posture (0.7 radiation area ratio)
    Sitting,
    /// Sedentary posture
    Sedentary,
    /// Reclining posture
    Reclining,
    /// Lying down posture
    Lying,
    /// Supine (lying face up) posture
    Supine,
    /// Crouching posture
    Crouching,
}

impl Posture {}

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

/// Calculate running mean outdoor temperature (prevailing mean)
///
/// Estimates the exponentially weighted running mean temperature from an array
/// of daily mean temperatures. Used by adaptive comfort models.
///
/// # Arguments
///
/// * `temp_array` - Array of daily mean temperatures in descending order
///   (newest/yesterday first: [t_day-1, t_day-2, ..., t_day-n])
/// * `alpha` - Weighting constant between 0 and 1 (default: 0.8)
///   - EN 16798-1 recommends 0.8
///   - ASHRAE 55 recommends 0.6-0.9 (slow to fast response)
///   - Use 0.9 for stable climates, 0.6 for variable climates
///
/// # Returns
///
/// Running mean outdoor temperature
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::running_mean_outdoor_temperature;
/// use thermalcomfort::Temperature;
///
/// // Last 7 days of daily mean temperatures (yesterday to 7 days ago)
/// let temps = vec![
///     Temperature::from_celsius(22.0),
///     Temperature::from_celsius(20.5),
///     Temperature::from_celsius(19.0),
///     Temperature::from_celsius(21.0),
///     Temperature::from_celsius(18.5),
///     Temperature::from_celsius(17.0),
///     Temperature::from_celsius(16.5),
/// ];
/// let t_rm = running_mean_outdoor_temperature(&temps, 0.8);
/// // Rounded to one decimal, matching pythermalcomfort
/// assert!((t_rm.as_celsius() - 19.9).abs() < 0.01);
/// ```
pub fn running_mean_outdoor_temperature(temp_array: &[Temperature], alpha: f64) -> Temperature {
    if temp_array.is_empty() {
        return Temperature::from_celsius(0.0);
    }

    let mut sum_weighted = 0.0;
    let mut sum_weights = 0.0;

    for (i, temp) in temp_array.iter().enumerate() {
        let weight = pow(alpha, i as f64);
        sum_weighted += weight * temp.as_celsius();
        sum_weights += weight;
    }

    // pythermalcomfort rounds this to one decimal before returning; without it the two
    // implementations disagree in the second decimal for every input.
    Temperature::from_celsius(round(sum_weighted / sum_weights * 10.0) / 10.0)
}

/// Calculate relative air speed which combines average air speed plus body movement
///
/// # Arguments
///
/// * `v` - Air speed measured by sensor (use `Speed::from_meters_per_second()` or similar)
/// * `met` - Metabolic rate
///
/// # Returns
///
/// Relative air speed
///
/// # Example
///
/// ```
/// use thermalcomfort::utilities::v_relative;
/// use thermalcomfort::{Speed, MetabolicRate};
///
/// let v = Speed::from_meters_per_second(0.1);
/// let met = MetabolicRate::from_met(1.4);
/// let vr = v_relative(v, met);
/// assert!((vr.as_meters_per_second() - 0.22).abs() < 0.01);
/// ```
#[inline]
pub fn v_relative(v: Speed, met: MetabolicRate) -> Speed {
    let v_ms = v.as_meters_per_second();
    let met_val = met.as_met();
    let vr_ms = if met_val > 1.0 {
        // Relative air speed accounts for increased air movement from body motion
        // 0.3 m/s per met above 1.0 (ISO 7730 and ASHRAE 55)
        // Round to 3 decimal places
        round((v_ms + 0.3 * (met_val - 1.0)) * 1000.0) / 1000.0
    } else {
        v_ms
    };
    Speed::from_meters_per_second(vr_ms)
}

/// Check if value is within valid range, return f64::NAN if not
#[inline]
pub fn valid_range(value: f64, min: f64, max: f64) -> f64 {
    if value >= min && value <= max {
        value
    } else {
        f64::NAN
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

/// Calculate saturation vapor pressure using Antoine equation
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure \[Pa\]
#[inline]
pub fn p_sat_antoine(tdb: Temperature) -> Pressure {
    let tdb_celsius = tdb.as_celsius();
    let p_pa = exp(16.6536 - 4030.183 / (tdb_celsius + 235.0)) * 1000.0; // Convert kPa to Pa
    Pressure::from_pascals(p_pa)
}

/// Calculate saturation vapor pressure using Antoine equation \[kPa\]
///
/// This is an alias for p_sat_antoine that returns the value in kPa
/// to match the Python pythermalcomfort.utilities.antoine function.
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure \[kPa\]
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::antoine;
/// use thermalcomfort::Temperature;
///
/// let p_kpa = antoine(Temperature::from_celsius(25.0));
/// assert!((p_kpa - 3.167).abs() < 0.01); // ~3.167 kPa at 25°C
/// ```
#[inline]
pub fn antoine(tdb: Temperature) -> f64 {
    p_sat_antoine(tdb).as_pascals() / 1000.0 // Convert Pa to kPa
}

/// Saturation vapor pressure using exponential equation
///
/// This is used in the two-node Gagge model and related calculations.
/// Returns pressure in torr units.
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure (torr)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::p_sat_torr;
/// use thermalcomfort::Temperature;
///
/// let p = p_sat_torr(Temperature::from_celsius(25.0));
/// assert!((p.as_torrs() - 23.8).abs() < 0.1);
/// ```
#[inline]
pub fn p_sat_torr(tdb: Temperature) -> Pressure {
    let tdb_celsius = tdb.as_celsius();
    let p_torr = exp(18.6686 - 4030.183 / (tdb_celsius + 235.0));
    // Convert torr to Pa (1 torr = 133.322 Pa)
    Pressure::from_pascals(p_torr * 133.322)
}

/// Calculate saturation vapor pressure
///
/// Uses different formulas for temperatures above and below freezing
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure
pub fn p_sat(tdb: Temperature) -> Pressure {
    const C1: f64 = -5674.5359;
    const C2: f64 = 6.3925247;
    const C3: f64 = -0.9677843e-2;
    const C4: f64 = 0.62215701e-6;
    const C5: f64 = 0.20747825e-8;
    const C6: f64 = -0.9484024e-12;
    const C7: f64 = 4.1635019;
    const C8: f64 = -5800.2206;
    const C9: f64 = 1.3914993;
    const C10: f64 = -0.048640239;
    const C11: f64 = 0.41764768e-4;
    const C12: f64 = -0.14452093e-7;
    const C13: f64 = 6.5459673;

    let ta_k = tdb.as_celsius() + C_TO_K;
    let log_ta_k = log(ta_k);

    let p_pa = if ta_k < C_TO_K {
        exp(C1 / ta_k + C2 + ta_k * (C3 + ta_k * (C4 + ta_k * (C5 + C6 * ta_k))) + C7 * log_ta_k)
    } else {
        exp(C8 / ta_k + C9 + ta_k * (C10 + ta_k * (C11 + ta_k * C12)) + C13 * log_ta_k)
    };
    Pressure::from_pascals(p_pa)
}

/// Convert humidity ratio to relative humidity
///
/// Algebraic inverse of the humidity-ratio formula used by
/// [`crate::psychrometrics::psy_ta_rh`].
///
/// # Arguments
///
/// * `hr` - Humidity ratio [kg water / kg dry air]
/// * `tdb` - Dry bulb air temperature
/// * `p_atm` - Atmospheric pressure
///
/// # Returns
///
/// Relative humidity [%]
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::hr_to_rh;
/// use thermalcomfort::{Temperature, Pressure};
///
/// let rh = hr_to_rh(0.01, Temperature::from_celsius(25.0), Pressure::from_pascals(101325.0));
/// assert!((rh - 50.5).abs() < 0.5);
/// ```
pub fn hr_to_rh(hr: f64, tdb: Temperature, p_atm: Pressure) -> f64 {
    // 0.62198 = ratio of molecular weights (M_water / M_air = 18.015 / 28.965)
    let p_vap = hr * p_atm.as_pascals() / (0.62198 + hr);
    p_vap / p_sat(tdb).as_pascals() * 100.0
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

/// Calculate body surface area using various formulas
///
/// # Arguments
///
/// * `weight` - Body weight
/// * `height` - Body height
/// * `formula` - Formula to use for calculation
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
/// use thermalcomfort::utilities::{body_surface_area, BsaFormula};
/// use thermalcomfort::{Mass, Length};
///
/// let bsa = body_surface_area(
///     Mass::from_kilograms(70.0),
///     Length::from_meters(1.75),
///     BsaFormula::DuBois
/// );
/// assert!((bsa.as_square_meters() - 1.844).abs() < 0.01);
/// ```
pub fn body_surface_area(weight: Mass, height: Length, formula: BsaFormula) -> Area {
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

/// Calculate clothing area factor
///
/// # Arguments
///
/// * `i_cl` - Intrinsic clothing insulation (clo)
///
/// # Returns
///
/// Clothing area factor
///
/// # Source
/// ISO 9920: f_cl = 1.0 + 0.28 * i_cl
/// where 0.28 is the empirical coefficient relating clothing insulation to increased surface area
#[inline]
pub fn clo_area_factor(i_cl: ClothingInsulation) -> f64 {
    1.0 + 0.28 * i_cl.as_clo()
}

/// Edition selector for ASHRAE 55-derived calculations.
///
/// pythermalcomfort's `clo_dynamic_ashrae` and `pmv_ppd_ashrae` both take a `model: str`
/// parameter (default `"55-2023"`) that is validated against a single legal value and
/// never branched on — today there is exactly one ASHRAE 55 edition these functions
/// support. The selector is still ported (rather than dropped) because it is a
/// documented, user-facing part of upstream's signature; a future edition would add a
/// variant here, and every match on this enum is exhaustive so that addition is a
/// compile error everywhere it needs handling, not a silent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ashrae55Model {
    /// ASHRAE 55-2023 (the only edition currently supported by either function).
    #[default]
    Ashrae552023,
}

/// Edition selector for [`clo_dynamic_iso`].
///
/// Mirrors [`Ashrae55Model`]: pythermalcomfort's `clo_dynamic_iso` takes a `model: str`
/// parameter (default `"9920-2007"`) validated against a single legal value. Ported as a
/// single-variant enum for the same reason — it is part of upstream's public signature,
/// and a future edition becomes a compile error to ignore rather than a silent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Iso9920Model {
    /// ISO 9920:2007 (the only edition currently supported).
    #[default]
    Iso99202007,
}

/// The inputs to [`clo_dynamic_ashrae`]: both parameters are required in
/// pythermalcomfort's signature (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloDynamicAshraeInputs {
    /// Static (intrinsic) clothing insulation
    pub clothing_insulation: ClothingInsulation,
    /// Metabolic rate
    pub metabolic_rate: MetabolicRate,
}

/// Optional parameters for [`clo_dynamic_ashrae`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CloDynamicAshraeOptions {
    /// Standard edition. Behaviourally identical to the only other legal value today
    /// (there is only one), see [`Ashrae55Model`].
    pub model: Ashrae55Model,
}

/// Calculate dynamic clothing insulation for ASHRAE 55
///
/// # Returns
///
/// Dynamic clothing insulation
///
/// # Source
/// ASHRAE 55: For met > 1.2, clo_dyn = clo * (0.6 + 0.4/met)
/// - 1.2 met: threshold for walking/active movement
/// - 0.6: base reduction factor
/// - 0.4: adjustment factor (accounts for increased ventilation with activity)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{clo_dynamic_ashrae, CloDynamicAshraeInputs, CloDynamicAshraeOptions};
/// use thermalcomfort::{ClothingInsulation, MetabolicRate};
///
/// let clo_dyn = clo_dynamic_ashrae(
///     CloDynamicAshraeInputs {
///         clothing_insulation: ClothingInsulation::from_clo(0.5),
///         metabolic_rate: MetabolicRate::from_met(1.4),
///     },
///     CloDynamicAshraeOptions::default(),
/// );
/// assert!(clo_dyn.as_clo() < 0.5);
/// ```
#[inline]
pub fn clo_dynamic_ashrae(
    inputs: CloDynamicAshraeInputs,
    options: CloDynamicAshraeOptions,
) -> ClothingInsulation {
    let CloDynamicAshraeInputs {
        clothing_insulation,
        metabolic_rate,
    } = inputs;
    // Exhaustive match: today there is only one edition, but this stops a future
    // variant from being silently ignored.
    match options.model {
        Ashrae55Model::Ashrae552023 => {}
    }
    let met_val = metabolic_rate.as_met();
    if met_val > 1.2 {
        ClothingInsulation::from_clo(round_to(
            clothing_insulation.as_clo() * (0.6 + 0.4 / met_val),
            3,
        ))
    } else {
        clothing_insulation
    }
}

/// Calculate insulation of the boundary air layer (I_a,r) - ISO 9920:2007
///
/// The static boundary air value is 0.7 clo for air velocities around 0.1-0.15 m/s.
/// For walking conditions, the boundary air layer insulation is calculated based on
/// the walking speed and relative air speed.
///
/// # Arguments
///
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_a_static` - Static boundary air layer insulation (typically 0.7 clo)
///
/// # Returns
///
/// Boundary air layer insulation (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_insulation_air_layer;
/// use thermalcomfort::{Speed, ClothingInsulation};
///
/// let i_a_r = clo_insulation_air_layer(
///     Speed::from_meters_per_second(0.1),
///     Speed::from_meters_per_second(0.0),
///     ClothingInsulation::from_clo(0.7)
/// );
/// assert!((i_a_r - 0.719).abs() < 0.01);
/// ```
#[inline]
pub fn clo_insulation_air_layer(vr: Speed, v_walk: Speed, i_a_static: ClothingInsulation) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.533 * (vr_ms - 0.15) + 0.069 * pow(vr_ms - 0.15, 2.0) - 0.462 * v_walk_ms
            + 0.201 * pow(v_walk_ms, 2.0),
    ) * i_a_static.as_clo()
}

/// Correction factor for nude person - ISO 9920:2007
///
/// Empirical coefficients from ISO 9920:
/// - -0.533: linear velocity coefficient
/// - 0.15 m/s: reference air speed offset
/// - 0.069: quadratic velocity coefficient
/// - -0.462: linear walking speed coefficient
/// - 0.201: quadratic walking speed coefficient
#[inline]
fn correction_nude(vr: Speed, v_walk: Speed) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.533 * (vr_ms - 0.15) + 0.069 * pow(vr_ms - 0.15, 2.0) - 0.462 * v_walk_ms
            + 0.201 * pow(v_walk_ms, 2.0),
    )
}

/// Correction factor for normal clothing - ISO 9920:2007
///
/// Empirical coefficients from ISO 9920 for clothed persons:
/// - -0.281: linear velocity coefficient
/// - 0.15 m/s: reference air speed offset
/// - 0.044: quadratic velocity coefficient
/// - -0.492: linear walking speed coefficient
#[inline]
fn correction_normal_clothing(vr: Speed, v_walk: Speed) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.281 * (vr_ms - 0.15) + 0.044 * pow(vr_ms - 0.15, 2.0) - 0.492 * v_walk_ms
            + 0.176 * pow(v_walk_ms, 2.0),
    )
}

/// Calculate total insulation of clothing ensemble (I_T,r) - ISO 9920:2007
///
/// The total insulation (I_T,r) is the actual thermal insulation from the body surface
/// to the environment, considering all clothing, enclosed air layers, and boundary air
/// layers under given environmental conditions and activities.
///
/// Different equations are used based on clothing level:
/// - Nude: i_cl = 0 clo
/// - Low clothing: i_cl < 0.6 clo
/// - Normal clothing: 0.6 clo <= i_cl <= 1.4 clo
///
/// # Arguments
///
/// * `i_t` - Total thermal insulation under static conditions
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_a_static` - Static boundary air layer insulation
/// * `i_cl` - Intrinsic clothing insulation
///
/// # Returns
///
/// Total insulation of clothing ensemble (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_total_insulation;
/// use thermalcomfort::{Speed, ClothingInsulation};
///
/// let i_t_r = clo_total_insulation(
///     ClothingInsulation::from_clo(1.5),
///     Speed::from_meters_per_second(0.1),
///     Speed::from_meters_per_second(0.0),
///     ClothingInsulation::from_clo(0.7),
///     ClothingInsulation::from_clo(1.0)
/// );
/// assert!(i_t_r > 0.0);
/// ```
pub fn clo_total_insulation(
    i_t: ClothingInsulation,
    vr: Speed,
    v_walk: Speed,
    i_a_static: ClothingInsulation,
    i_cl: ClothingInsulation,
) -> f64 {
    let i_t = i_t.as_clo();
    let i_a_static = i_a_static.as_clo();
    let i_cl = i_cl.as_clo();
    // Calculate insulation for different clothing levels
    let nude_insulation = i_a_static * correction_nude(vr, v_walk);
    let normal_insulation = i_t * correction_normal_clothing(vr, v_walk);

    if i_cl == 0.0 {
        // Nude
        nude_insulation
    } else if i_cl <= 0.6 {
        // Low clothing - interpolate between nude and normal
        ((0.6 - i_cl) * nude_insulation + i_cl * normal_insulation) / 0.6
    } else {
        // Normal clothing
        normal_insulation
    }
}

/// The inputs to [`clo_dynamic_iso`]: all three are required in pythermalcomfort's
/// signature (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloDynamicIsoInputs {
    /// Static (intrinsic) clothing insulation
    pub clothing_insulation: ClothingInsulation,
    /// Metabolic rate
    pub metabolic_rate: MetabolicRate,
    /// Air speed (not yet the relative air speed; computed internally via
    /// [`v_relative`])
    pub air_speed: Speed,
}

/// Optional parameters for [`clo_dynamic_iso`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloDynamicIsoOptions {
    /// Thermal insulation of the boundary (surface) air layer around the outer
    /// clothing or, when nude, around the skin surface. Defaults to 0.7 clo.
    pub boundary_air_layer_insulation: ClothingInsulation,
    /// Standard edition. Behaviourally identical to the only other legal value today
    /// (there is only one), see [`Iso9920Model`].
    pub model: Iso9920Model,
}

impl Default for CloDynamicIsoOptions {
    fn default() -> Self {
        Self {
            boundary_air_layer_insulation: ClothingInsulation::from_clo(0.7),
            model: Iso9920Model::default(),
        }
    }
}

/// Calculate dynamic clothing insulation for ISO 9920:2007
///
/// Estimates the dynamic intrinsic clothing insulation (I_cl,r). The activity
/// as well as the air speed modify the insulation characteristics of the clothing.
///
/// # Returns
///
/// Dynamic clothing insulation (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::{clo_dynamic_iso, CloDynamicIsoInputs, CloDynamicIsoOptions};
/// use thermalcomfort::{Speed, ClothingInsulation, MetabolicRate};
///
/// let clo_dyn = clo_dynamic_iso(
///     CloDynamicIsoInputs {
///         clothing_insulation: ClothingInsulation::from_clo(1.0),
///         metabolic_rate: MetabolicRate::from_met(1.2),
///         air_speed: Speed::from_meters_per_second(0.1),
///     },
///     CloDynamicIsoOptions::default(),
/// );
/// assert!(clo_dyn > 0.0 && clo_dyn <= 1.0);
/// ```
pub fn clo_dynamic_iso(inputs: CloDynamicIsoInputs, options: CloDynamicIsoOptions) -> f64 {
    let CloDynamicIsoInputs {
        clothing_insulation: clo,
        metabolic_rate: met,
        air_speed: v,
    } = inputs;
    let CloDynamicIsoOptions {
        boundary_air_layer_insulation: i_a,
        model,
    } = options;
    // Exhaustive match: today there is only one edition, but this stops a future
    // variant from being silently ignored.
    match model {
        Iso9920Model::Iso99202007 => {}
    }

    let clo_val = clo.as_clo();
    // Calculate clothing area factor
    let f_cl = clo_area_factor(clo);

    // Total insulation under static conditions
    let i_t = clo_val + i_a.as_clo() / f_cl;

    // Relative air speed for the whole-body heat balance
    let v_r = v_relative(v, met);

    // Walking speed when undefined, per ISO 7730 Annex C / ISO 9920:
    // v_walk = 0.0052 * (M - 58), clipped to [0, 0.7] m/s, where M is metabolic rate
    // in W/m². This is a distinct formula from `v_relative`'s activity-generated air
    // speed, which the previous implementation incorrectly reused here.
    let v_walk_ms = (0.0052 * (met.as_met() * MET_TO_W_M2 - 58.0)).clamp(0.0, 0.7);
    let v_walk = Speed::from_meters_per_second(v_walk_ms);

    // Calculate total dynamic insulation
    let i_t_r = clo_total_insulation(ClothingInsulation::from_clo(i_t), v_r, v_walk, i_a, clo);

    // Calculate dynamic air layer insulation
    let i_a_r = clo_insulation_air_layer(v_r, v_walk, i_a);

    // Return dynamic clothing insulation
    i_t_r - i_a_r / f_cl
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
/// Representative clothing insulation (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_tout;
/// use thermalcomfort::Temperature;
///
/// let clo = clo_tout(Temperature::from_celsius(27.0));
/// assert!((clo - 0.46).abs() < 0.01);
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
pub fn clo_tout(tout: Temperature) -> f64 {
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

    round(clo * 100.0) / 100.0
}

/// Calculate environmental correction factor for clothing insulation
///
/// Returns the correction factor for total clothing insulation to account for
/// real environmental conditions (walking, air movement, etc.) vs. static
/// measurement conditions. Based on ISO 9920:2007.
///
/// # Arguments
///
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_cl` - Intrinsic clothing insulation
///
/// # Returns
///
/// Correction factor for clothing insulation
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_correction_factor_environment;
/// use thermalcomfort::{Speed, ClothingInsulation};
///
/// let cf = clo_correction_factor_environment(
///     Speed::from_meters_per_second(0.3),
///     Speed::from_meters_per_second(0.5),
///     ClothingInsulation::from_clo(0.8)
/// );
/// assert!(cf > 0.0 && cf <= 1.0);
/// ```
///
/// # References
///
/// - ISO 9920:2007
pub fn clo_correction_factor_environment(
    vr: Speed,
    v_walk: Speed,
    i_cl: ClothingInsulation,
) -> f64 {
    let i_cl = i_cl.as_clo();
    if i_cl == 0.0 {
        return correction_nude(vr, v_walk);
    }

    if i_cl <= 0.6 {
        let nude_corr = correction_nude(vr, v_walk);
        let normal_corr = correction_normal_clothing(vr, v_walk);
        ((0.6 - i_cl) * nude_corr + i_cl * normal_corr) / 0.6
    } else {
        correction_normal_clothing(vr, v_walk)
    }
}

/// Calculate intrinsic insulation of clothing ensemble
///
/// Calculates the intrinsic insulation of a clothing ensemble based on the
/// sum of individual garment insulation values. Valid for ensembles with
/// rather uniform insulation distribution across the body.
///
/// # Arguments
///
/// * `clo_garments` - Slice of clothing insulation values for each garment (clo)
///
/// # Returns
///
/// Total intrinsic insulation of ensemble (clo)
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_intrinsic_insulation_ensemble;
/// use thermalcomfort::ClothingInsulation;
///
/// let garments = [
///     ClothingInsulation::from_clo(0.25), // shirt
///     ClothingInsulation::from_clo(0.15), // pants
///     ClothingInsulation::from_clo(0.10), // underwear
/// ];
/// let total = clo_intrinsic_insulation_ensemble(&garments);
/// assert!(total > 0.0);
/// ```
///
/// # References
///
/// - ISO 9920:2009 Section 4.3
pub fn clo_intrinsic_insulation_ensemble(clo_garments: &[ClothingInsulation]) -> f64 {
    let sum: f64 = clo_garments.iter().map(|c| c.as_clo()).sum();
    sum * 0.835 + 0.161
}

/// Typical clothing ensembles with predefined insulation values
///
/// These represent common clothing combinations based on ASHRAE 55 and ISO 9920.
/// Each tuple contains (ensemble_name, insulation_clo).
pub const CLO_TYPICAL_ENSEMBLES: &[(&str, f64)] = &[
    ("Walking shorts, short-sleeve shirt", 0.36),
    ("Typical summer indoor clothing", 0.5),
    (
        "Knee-length skirt, short-sleeve shirt, sandals, underwear",
        0.54,
    ),
    (
        "Trousers, short-sleeve shirt, socks, shoes, underwear",
        0.57,
    ),
    ("Trousers, long-sleeve shirt", 0.61),
    ("Knee-length skirt, long-sleeve shirt, full slip", 0.67),
    ("Sweat pants, long-sleeve sweatshirt", 0.74),
    ("Jacket, Trousers, long-sleeve shirt", 0.96),
    ("Typical winter indoor clothing", 1.0),
];

/// Individual garment insulation values
///
/// Insulation values for individual clothing items based on ISO 9920:2009.
/// Each tuple contains (garment_name, insulation_clo).
pub const CLO_INDIVIDUAL_GARMENTS: &[(&str, f64)] = &[
    ("Metal chair", 0.0),
    ("Bra", 0.01),
    ("Wooden stool", 0.01),
    ("Ankle socks", 0.02),
    ("Shoes or sandals", 0.02),
    ("Slippers", 0.03),
    ("Panty hose", 0.02),
    ("Calf length socks", 0.03),
    ("Women's underwear", 0.03),
    ("Men's underwear", 0.04),
    ("Knee socks (thick)", 0.06),
    ("Short shorts", 0.06),
    ("Walking shorts", 0.08),
    ("T-shirt", 0.08),
    ("Standard office chair", 0.1),
    ("Executive chair", 0.15),
    ("Boots", 0.1),
    ("Sleeveless scoop-neck blouse", 0.12),
    ("Half slip", 0.14),
    ("Long underwear bottoms", 0.15),
    ("Full slip", 0.16),
    ("Short-sleeve knit shirt", 0.17),
    ("Sleeveless vest (thin)", 0.1),
    ("Sleeveless vest (thick)", 0.17),
    ("Sleeveless short gown (thin)", 0.18),
    ("Short-sleeve dress shirt", 0.19),
    ("Sleeveless long gown (thin)", 0.2),
    ("Long underwear top", 0.2),
    ("Thick skirt", 0.23),
    ("Long-sleeve dress shirt", 0.25),
    ("Long-sleeve flannel shirt", 0.34),
    ("Long-sleeve sweat shirt", 0.34),
    ("Short-sleeve hospital gown", 0.31),
    ("Short-sleeve short robe (thin)", 0.34),
    ("Short-sleeve pajamas", 0.42),
    ("Long-sleeve long gown", 0.46),
    ("Long-sleeve short wrap robe (thick)", 0.48),
    ("Long-sleeve pajamas (thick)", 0.57),
    ("Long-sleeve long wrap robe (thick)", 0.69),
    ("Thin trousers", 0.15),
    ("Thick trousers", 0.24),
    ("Sweatpants", 0.28),
    ("Overalls", 0.3),
    ("Coveralls", 0.49),
    ("Thin skirt", 0.14),
    ("Long-sleeve shirt dress (thin)", 0.33),
    ("Long-sleeve shirt dress (thick)", 0.47),
    ("Short-sleeve shirt dress", 0.29),
    ("Sleeveless, scoop-neck shirt (thin)", 0.23),
    ("Sleeveless, scoop-neck shirt (thick)", 0.27),
    ("Long sleeve shirt (thin)", 0.25),
    ("Long sleeve shirt (thick)", 0.36),
    ("Single-breasted coat (thin)", 0.36),
    ("Single-breasted coat (thick)", 0.44),
    ("Double-breasted coat (thin)", 0.42),
    ("Double-breasted coat (thick)", 0.48),
];

/// Look up clothing insulation value for a typical ensemble
///
/// Returns the insulation value in clo for a typical clothing ensemble.
///
/// # Arguments
///
/// * `ensemble_name` - Name of the clothing ensemble (case-sensitive)
///
/// # Returns
///
/// Some(clo) if the ensemble is found, None otherwise
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_typical_ensemble;
///
/// let clo = clo_typical_ensemble("Typical summer indoor clothing");
/// assert_eq!(clo, Some(0.5));
///
/// let not_found = clo_typical_ensemble("Unknown ensemble");
/// assert_eq!(not_found, None);
/// ```
pub fn clo_typical_ensemble(ensemble_name: &str) -> Option<f64> {
    CLO_TYPICAL_ENSEMBLES
        .iter()
        .find(|(name, _)| *name == ensemble_name)
        .map(|(_, clo)| *clo)
}

/// Look up clothing insulation value for an individual garment
///
/// Returns the insulation value in clo for an individual garment.
///
/// # Arguments
///
/// * `garment_name` - Name of the garment (case-sensitive)
///
/// # Returns
///
/// Some(clo) if the garment is found, None otherwise
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_individual_garment;
///
/// let clo = clo_individual_garment("Long-sleeve dress shirt");
/// assert_eq!(clo, Some(0.25));
///
/// let not_found = clo_individual_garment("Unknown garment");
/// assert_eq!(not_found, None);
/// ```
pub fn clo_individual_garment(garment_name: &str) -> Option<f64> {
    CLO_INDIVIDUAL_GARMENTS
        .iter()
        .find(|(name, _)| *name == garment_name)
        .map(|(_, clo)| *clo)
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
    fn test_clo_dynamic_iso_matches_python() {
        // Reference values from pythermalcomfort 4.4.0 clo_dynamic_iso (i_a defaults to 0.7).
        // These exercise the ISO 7730 Annex C / ISO 9920 walking-speed formula.
        for (clo, met, v, expected) in [
            (0.5, 1.2, 0.1, 0.417585),
            (1.0, 2.0, 0.3, 0.817727),
            (0.7, 1.5, 0.2, 0.640019),
            (1.5, 3.0, 0.5, 1.000017),
        ] {
            let got = clo_dynamic_iso(
                CloDynamicIsoInputs {
                    clothing_insulation: ClothingInsulation::from_clo(clo),
                    metabolic_rate: MetabolicRate::from_met(met),
                    air_speed: Speed::from_meters_per_second(v),
                },
                CloDynamicIsoOptions::default(),
            );
            assert!(
                (got - expected).abs() < 1e-5,
                "clo_dynamic_iso({clo}, {met}, {v}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_hr_to_rh_matches_python() {
        // Reference values from pythermalcomfort 4.4.0 hr_to_rh
        for (hr, tdb, expected) in [
            (0.01, 25.0, 50.589615),
            (0.005, 20.0, 34.549292),
            (0.02, 30.0, 74.343333),
        ] {
            let got = hr_to_rh(
                hr,
                Temperature::from_celsius(tdb),
                Pressure::from_pascals(101325.0),
            );
            assert!(
                (got - expected).abs() < 1e-4,
                "hr_to_rh({hr}, {tdb}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_v_relative() {
        let v1 = v_relative(
            Speed::from_meters_per_second(0.1),
            MetabolicRate::from_met(1.0),
        );
        assert_eq!(v1.as_meters_per_second(), 0.1);

        let v2 = v_relative(
            Speed::from_meters_per_second(0.1),
            MetabolicRate::from_met(1.4),
        );
        assert!((v2.as_meters_per_second() - 0.22).abs() < 0.001);

        let v3 = v_relative(
            Speed::from_meters_per_second(0.15),
            MetabolicRate::from_met(2.0),
        );
        assert!((v3.as_meters_per_second() - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_valid_range() {
        assert_eq!(valid_range(15.0, 10.0, 30.0), 15.0);
        assert!(valid_range(5.0, 10.0, 30.0).is_nan());
        assert!(valid_range(35.0, 10.0, 30.0).is_nan());
    }

    #[test]
    fn test_clo_area_factor() {
        assert!((clo_area_factor(ClothingInsulation::from_clo(0.5)) - 1.14).abs() < 0.01);
        assert!((clo_area_factor(ClothingInsulation::from_clo(1.0)) - 1.28).abs() < 0.01);
    }

    #[test]
    fn test_clo_typical_ensemble() {
        // Test valid lookups
        assert_eq!(
            clo_typical_ensemble("Typical summer indoor clothing"),
            Some(0.5)
        );
        assert_eq!(
            clo_typical_ensemble("Typical winter indoor clothing"),
            Some(1.0)
        );
        assert_eq!(
            clo_typical_ensemble("Walking shorts, short-sleeve shirt"),
            Some(0.36)
        );

        // Test invalid lookup
        assert_eq!(clo_typical_ensemble("Non-existent ensemble"), None);
    }

    #[test]
    fn test_clo_individual_garment() {
        // Test valid lookups
        assert_eq!(
            clo_individual_garment("Long-sleeve dress shirt"),
            Some(0.25)
        );
        assert_eq!(clo_individual_garment("Thick trousers"), Some(0.24));
        assert_eq!(clo_individual_garment("T-shirt"), Some(0.08));

        // Test invalid lookup
        assert_eq!(clo_individual_garment("Non-existent garment"), None);
    }

    #[test]
    fn test_clo_intrinsic_insulation_ensemble() {
        // Test with typical garments - shirt + pants + underwear
        let garments = [
            ClothingInsulation::from_clo(0.25), // long-sleeve shirt
            ClothingInsulation::from_clo(0.24), // thick trousers
            ClothingInsulation::from_clo(0.04), // underwear
        ];
        let total = clo_intrinsic_insulation_ensemble(&garments);
        // Formula: sum * 0.835 + 0.161 = 0.53 * 0.835 + 0.161 = 0.604
        assert!((total - 0.604).abs() < 0.01);

        // Test with single garment
        let single = [ClothingInsulation::from_clo(0.5)];
        let total_single = clo_intrinsic_insulation_ensemble(&single);
        // Formula: 0.5 * 0.835 + 0.161 = 0.579
        assert!((total_single - 0.579).abs() < 0.01);
    }

    #[test]
    fn test_antoine() {
        // Test antoine function returns kPa
        let p_kpa = antoine(Temperature::from_celsius(25.0));
        // At 25°C, saturated vapor pressure is ~3.167 kPa
        assert!((p_kpa - 3.167).abs() < 0.01);

        let p_kpa_0 = antoine(Temperature::from_celsius(0.0));
        // At 0°C, saturated vapor pressure is ~0.609 kPa
        assert!((p_kpa_0 - 0.609).abs() < 0.01);
    }
}
