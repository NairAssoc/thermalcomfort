//! Utility functions for thermal comfort calculations

use crate::constants::*;
use libm::{exp, pow, log, round};
pub use measurements::{Temperature, Speed, Pressure, Area, Mass, Length};

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Units {
    /// SI (International System of Units)
    SI,
    /// IP (Imperial Units)
    IP,
}

impl Default for Units {
    fn default() -> Self {
        Units::SI
    }
}

/// Body postures for thermal comfort calculations
///
/// Different postures affect the radiative heat transfer coefficient
/// and body surface area exposed to the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Posture {
    /// Standing posture (0.73 radiation area ratio)
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

impl Default for Posture {
    fn default() -> Self {
        Posture::Standing
    }
}

impl Posture {
    /// Get the radiation area ratio for this posture
    ///
    /// This is the ratio between the radiation area of the body
    /// and the total body surface area.
    pub fn radiation_area_ratio(&self) -> f64 {
        match self {
            Posture::Standing => 0.73,
            Posture::Sitting => 0.70,
            // For other postures, use standing as default
            _ => 0.73,
        }
    }
}

/// Model standards
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Model {
    /// ASHRAE 55-2023
    Ashrae552023,
    /// ISO 7730-2005
    Iso77302005,
    /// ISO 9920-2007
    Iso99202007,
    /// ISO 7933-2004
    Iso79332004,
    /// ISO 7933-2023
    Iso79332023,
}

impl Default for Model {
    fn default() -> Self {
        Model::Iso77302005
    }
}

/// Calculate running mean outdoor temperature (prevailing mean)
///
/// Estimates the exponentially weighted running mean temperature from an array
/// of daily mean temperatures. Used by adaptive comfort models.
///
/// # Arguments
///
/// * `temp_array` - Array of daily mean temperatures in descending order
///                  (newest/yesterday first: [t_day-1, t_day-2, ..., t_day-n])
/// * `alpha` - Weighting constant between 0 and 1 (default: 0.8)
///             - EN 16798-1 recommends 0.8
///             - ASHRAE 55 recommends 0.6-0.9 (slow to fast response)
///             - Use 0.9 for stable climates, 0.6 for variable climates
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
/// assert!((t_rm.as_celsius() - 19.94).abs() < 0.01);
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

    Temperature::from_celsius(sum_weighted / sum_weights)
}

/// Calculate relative air speed which combines average air speed plus body movement
///
/// # Arguments
///
/// * `v` - Air speed measured by sensor (use `Speed::from_meters_per_second()` or similar)
/// * `met` - Metabolic rate [met]
///
/// # Returns
///
/// Relative air speed
///
/// # Example
///
/// ```
/// use thermalcomfort::utilities::v_relative;
/// use thermalcomfort::Speed;
///
/// let v = Speed::from_meters_per_second(0.1);
/// let met = 1.4; // metabolic rate [met]
/// let vr = v_relative(v, met);
/// assert!((vr.as_meters_per_second() - 0.22).abs() < 0.01);
/// ```
#[inline]
pub fn v_relative(v: Speed, met: f64) -> Speed {
    let v_ms = v.as_meters_per_second();
    let vr_ms = if met > 1.0 {
        // Round to 3 decimal places
        round((v_ms + 0.3 * (met - 1.0)) * 1000.0) / 1000.0
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

/// Round to specified decimal places
#[inline]
pub fn round_to(value: f64, decimals: i32) -> f64 {
    let multiplier = pow(10.0, decimals as f64);
    round(value * multiplier) / multiplier
}

/// Calculate saturation vapor pressure using Antoine equation
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure
#[inline]
pub fn p_sat_antoine(tdb: Temperature) -> Pressure {
    let tdb_celsius = tdb.as_celsius();
    let p_pa = exp(16.6536 - 4030.183 / (tdb_celsius + 235.0)) * 1000.0; // Convert kPa to Pa
    Pressure::from_pascals(p_pa)
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
        exp(
            C1 / ta_k
                + C2
                + ta_k * (C3 + ta_k * (C4 + ta_k * (C5 + C6 * ta_k)))
                + C7 * log_ta_k,
        )
    } else {
        exp(
            C8 / ta_k + C9 + ta_k * (C10 + ta_k * (C11 + ta_k * C12)) + C13 * log_ta_k,
        )
    };
    Pressure::from_pascals(p_pa)
}

/// Formula options for body surface area calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BsaFormula {
    /// DuBois formula (1916) - most widely used
    DuBois,
    /// Takahira formula (1925)
    Takahira,
    /// Fujimoto formula (1968)
    Fujimoto,
    /// Kurazumi formula (1994)
    Kurazumi,
}

impl Default for BsaFormula {
    fn default() -> Self {
        BsaFormula::DuBois
    }
}

/// Calculate body surface area using DuBois formula [m²]
///
/// # Arguments
///
/// * `weight` - Body weight [kg]
/// * `height` - Body height [m]
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
/// * `i_cl` - Intrinsic clothing insulation [clo]
///
/// # Returns
///
/// Clothing area factor
#[inline]
pub fn clo_area_factor(i_cl: f64) -> f64 {
    1.0 + 0.28 * i_cl
}

/// Calculate dynamic clothing insulation for ASHRAE 55
///
/// # Arguments
///
/// * `clo` - Static clothing insulation [clo]
/// * `met` - Metabolic rate [met]
///
/// # Returns
///
/// Dynamic clothing insulation [clo]
#[inline]
pub fn clo_dynamic_ashrae(clo: f64, met: f64) -> f64 {
    if met > 1.2 {
        round_to(clo * (0.6 + 0.4 / met), 3)
    } else {
        clo
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
/// * `i_a_static` - Static boundary air layer insulation [clo] (typically 0.7)
///
/// # Returns
///
/// Boundary air layer insulation [clo]
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_insulation_air_layer;
/// use thermalcomfort::Speed;
///
/// let i_a_r = clo_insulation_air_layer(
///     Speed::from_meters_per_second(0.1),
///     Speed::from_meters_per_second(0.0),
///     0.7
/// );
/// assert!((i_a_r - 0.719).abs() < 0.01);
/// ```
#[inline]
pub fn clo_insulation_air_layer(vr: Speed, v_walk: Speed, i_a_static: f64) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.533 * (vr_ms - 0.15)
        + 0.069 * pow(vr_ms - 0.15, 2.0)
        - 0.462 * v_walk_ms
        + 0.201 * pow(v_walk_ms, 2.0)
    ) * i_a_static
}

/// Correction factor for nude person - ISO 9920:2007
#[inline]
fn correction_nude(vr: Speed, v_walk: Speed) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.533 * (vr_ms - 0.15)
        + 0.069 * pow(vr_ms - 0.15, 2.0)
        - 0.462 * v_walk_ms
        + 0.201 * pow(v_walk_ms, 2.0)
    )
}

/// Correction factor for normal clothing - ISO 9920:2007
#[inline]
fn correction_normal_clothing(vr: Speed, v_walk: Speed) -> f64 {
    let vr_ms = vr.as_meters_per_second();
    let v_walk_ms = v_walk.as_meters_per_second();
    exp(
        -0.281 * (vr_ms - 0.15)
        + 0.044 * pow(vr_ms - 0.15, 2.0)
        - 0.492 * v_walk_ms
        + 0.176 * pow(v_walk_ms, 2.0)
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
/// * `i_t` - Total thermal insulation under static conditions [clo]
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `v_walk` - Walking speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_a_static` - Static boundary air layer insulation [clo]
/// * `i_cl` - Intrinsic clothing insulation [clo]
///
/// # Returns
///
/// Total insulation of clothing ensemble [clo]
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_total_insulation;
/// use thermalcomfort::Speed;
///
/// let i_t_r = clo_total_insulation(
///     1.5,
///     Speed::from_meters_per_second(0.1),
///     Speed::from_meters_per_second(0.0),
///     0.7,
///     1.0
/// );
/// assert!(i_t_r > 0.0);
/// ```
pub fn clo_total_insulation(
    i_t: f64,
    vr: Speed,
    v_walk: Speed,
    i_a_static: f64,
    i_cl: f64,
) -> f64 {
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

/// Calculate dynamic clothing insulation for ISO 9920:2007
///
/// Estimates the dynamic intrinsic clothing insulation (I_cl,r). The activity
/// as well as the air speed modify the insulation characteristics of the clothing.
///
/// # Arguments
///
/// * `clo` - Static clothing insulation [clo]
/// * `met` - Metabolic rate [met]
/// * `v` - Air speed (use `Speed::from_meters_per_second()` or similar)
/// * `i_a` - Thermal insulation of boundary air layer [clo] (typically 0.7)
///
/// # Returns
///
/// Dynamic clothing insulation [clo]
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_dynamic_iso;
/// use thermalcomfort::Speed;
///
/// let clo_dyn = clo_dynamic_iso(1.0, 1.2, Speed::from_meters_per_second(0.1), 0.7);
/// assert!(clo_dyn > 0.0 && clo_dyn <= 1.0);
/// ```
pub fn clo_dynamic_iso(clo: f64, met: f64, v: Speed, i_a: f64) -> f64 {
    // Calculate clothing area factor
    let f_cl = clo_area_factor(clo);

    // Total insulation under static conditions
    let i_t = clo + i_a / f_cl;

    // Calculate walking speed and relative air speed
    let v_r = v_relative(v, met);
    let v_walk_ms = v_r.as_meters_per_second() - v.as_meters_per_second();
    let v_walk = Speed::from_meters_per_second(v_walk_ms);

    // Calculate total dynamic insulation
    let i_t_r = clo_total_insulation(i_t, v_r, v_walk, i_a, clo);

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
/// Representative clothing insulation [clo]
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
/// * `i_cl` - Intrinsic clothing insulation [clo]
///
/// # Returns
///
/// Correction factor for clothing insulation
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_correction_factor_environment;
/// use thermalcomfort::Speed;
///
/// let cf = clo_correction_factor_environment(
///     Speed::from_meters_per_second(0.3),
///     Speed::from_meters_per_second(0.5),
///     0.8
/// );
/// assert!(cf > 0.0 && cf <= 1.0);
/// ```
///
/// # References
///
/// - ISO 9920:2007
pub fn clo_correction_factor_environment(vr: Speed, v_walk: Speed, i_cl: f64) -> f64 {
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
/// * `clo_garments` - Slice of clothing insulation values for each garment [clo]
///
/// # Returns
///
/// Total intrinsic insulation of ensemble [clo]
///
/// # Examples
///
/// ```
/// use thermalcomfort::utilities::clo_intrinsic_insulation_ensemble;
///
/// let garments = vec![0.25, 0.15, 0.1]; // shirt, pants, underwear
/// let total = clo_intrinsic_insulation_ensemble(&garments);
/// assert!(total > 0.0);
/// ```
///
/// # References
///
/// - ISO 9920:2009 Section 4.3
pub fn clo_intrinsic_insulation_ensemble(clo_garments: &[f64]) -> f64 {
    let sum: f64 = clo_garments.iter().sum();
    sum * 0.835 + 0.161
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v_relative() {
        let v1 = v_relative(Speed::from_meters_per_second(0.1), 1.0);
        assert_eq!(v1.as_meters_per_second(), 0.1);

        let v2 = v_relative(Speed::from_meters_per_second(0.1), 1.4);
        assert!((v2.as_meters_per_second() - 0.22).abs() < 0.001);

        let v3 = v_relative(Speed::from_meters_per_second(0.15), 2.0);
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
        assert!((clo_area_factor(0.5) - 1.14).abs() < 0.01);
        assert!((clo_area_factor(1.0) - 1.28).abs() < 0.01);
    }
}
