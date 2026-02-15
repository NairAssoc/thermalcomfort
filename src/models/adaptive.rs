//! Adaptive thermal comfort models
//!
//! Adaptive models relate indoor design temperatures to outdoor climate parameters.
//! Only applicable to naturally conditioned spaces without mechanical cooling/heating.

use crate::psychrometrics::operative_temperature;
use measurements::{Speed, Temperature};

/// Result from ASHRAE 55 adaptive comfort model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveAshraeResult {
    /// Comfort temperature [°C]
    pub tmp_cmf: f64,
    /// Lower bound of 80% acceptability [°C]
    pub tmp_cmf_80_low: f64,
    /// Upper bound of 80% acceptability [°C]
    pub tmp_cmf_80_up: f64,
    /// Lower bound of 90% acceptability [°C]
    pub tmp_cmf_90_low: f64,
    /// Upper bound of 90% acceptability [°C]
    pub tmp_cmf_90_up: f64,
    /// Whether conditions meet 80% acceptability
    pub acceptability_80: bool,
    /// Whether conditions meet 90% acceptability
    pub acceptability_90: bool,
}

/// Result from EN 16798-1 adaptive comfort model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveEnResult {
    /// Comfort temperature [°C]
    pub tmp_cmf: f64,
    /// Category I lower limit [°C]
    pub tmp_cmf_cat_i_low: f64,
    /// Category I upper limit [°C]
    pub tmp_cmf_cat_i_up: f64,
    /// Category II lower limit [°C]
    pub tmp_cmf_cat_ii_low: f64,
    /// Category II upper limit [°C]
    pub tmp_cmf_cat_ii_up: f64,
    /// Category III lower limit [°C]
    pub tmp_cmf_cat_iii_low: f64,
    /// Category III upper limit [°C]
    pub tmp_cmf_cat_iii_up: f64,
    /// Whether conditions meet Category I
    pub acceptability_cat_i: bool,
    /// Whether conditions meet Category II
    pub acceptability_cat_ii: bool,
    /// Whether conditions meet Category III
    pub acceptability_cat_iii: bool,
}

/// Options for adaptive comfort calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveOptions {
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
}

impl Default for AdaptiveOptions {
    fn default() -> Self {
        Self { limit_inputs: true }
    }
}

/// Calculate adaptive thermal comfort based on ASHRAE 55
///
/// The adaptive model can only be used in occupant-controlled naturally conditioned
/// spaces that meet ALL the following criteria:
/// - No mechanical cooling or heating system in operation
/// - Occupants have metabolic rate between 1.0 and 1.5 met
/// - Occupants can adapt clothing within 0.5 to 1.0 clo range
/// - Prevailing mean outdoor temperature is between 10 and 33.5 °C
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (recommended range: 10-40°C)
/// * `mean_radiant_temp` - Mean radiant temperature (recommended range: 10-40°C)
/// * `running_mean_outdoor_temp` - Running mean outdoor temperature (recommended range: 10-33.5°C)
/// * `air_speed` - Air speed (recommended range: 0-2 m/s)
/// * `options` - Adaptive comfort options
///
/// # Returns
///
/// AdaptiveAshraeResult with comfort temperature and acceptability limits
///
/// # Applicability Limits (when limit_inputs = true)
///
/// * 10 < tdb [°C] < 40
/// * 10 < tr [°C] < 40
/// * 0 < v [m/s] < 2
/// * 10 < t_running_mean [°C] < 33.5
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::adaptive::{adaptive_ashrae, AdaptiveOptions};
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = adaptive_ashrae(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(20.0),
///     Speed::from_meters_per_second(0.1),
///     Default::default()
/// );
/// assert!(result.acceptability_80);
/// println!("Comfort temp: {:.1}°C", result.tmp_cmf);
/// ```
pub fn adaptive_ashrae(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    running_mean_outdoor_temp: Temperature,
    air_speed: Speed,
    options: AdaptiveOptions,
) -> AdaptiveAshraeResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let running_mean_celsius = running_mean_outdoor_temp.as_celsius();
    let speed_mps = air_speed.as_meters_per_second();

    // Calculate operative temperature (use_ashrae=true for adaptive models)
    let to = operative_temperature(dry_bulb_temp, mean_radiant_temp, air_speed, true);

    // Calculate cooling effect for elevated air speed when to > 25°C
    // From ASHRAE 55-2023 Section 5.4.3
    // Thresholds: 0.6 m/s, 0.9 m/s, 1.2 m/s
    // Cooling effects: 1.2°C, 1.8°C, 2.2°C respectively
    // Only applies when operative temperature >= 25°C
    let ce = if speed_mps >= 0.6 && to.as_celsius() >= 25.0 {
        if speed_mps < 0.9 {
            1.2 // First tier cooling effect
        } else if speed_mps < 1.2 {
            1.8 // Second tier cooling effect
        } else {
            2.2 // Third tier cooling effect
        }
    } else {
        0.0
    };

    // Comfort temperature based on running mean outdoor temperature
    // ASHRAE 55-2023 adaptive comfort equation:
    // t_cmf = 0.31 * t_running_mean + 17.8
    // where 0.31 is the climate adaptation coefficient
    // and 17.8°C is the base comfort temperature
    let mut t_cmf = 0.31 * running_mean_celsius + 17.8;

    // Apply input limits if requested (ASHRAE 55-2023 applicability limits)
    // Dry bulb and radiant temperature: 10-40°C
    // Air speed: 0-2 m/s
    // Running mean outdoor temperature: 10-33.5°C
    if options.limit_inputs
        && (!(10.0..=40.0).contains(&dry_bulb_celsius)
            || !(10.0..=40.0).contains(&radiant_celsius)
            || !(0.0..=2.0).contains(&speed_mps)
            || !(10.0..=33.5).contains(&running_mean_celsius))
    {
        t_cmf = f64::NAN;
    }

    // Round to 1 decimal place
    t_cmf = libm::round(t_cmf * 10.0) / 10.0;

    // Calculate acceptability bounds (ASHRAE 55-2023)
    // 80% acceptability: ±3.5°C from comfort temperature
    // 90% acceptability: ±2.5°C from comfort temperature
    let tmp_cmf_80_low = t_cmf - 3.5;
    let tmp_cmf_90_low = t_cmf - 2.5;
    let tmp_cmf_80_up = t_cmf + 3.5 + ce;
    let tmp_cmf_90_up = t_cmf + 2.5 + ce;

    // Check acceptability
    let to_celsius = to.as_celsius();
    let acceptability_80 =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_80_low && to_celsius <= tmp_cmf_80_up;
    let acceptability_90 =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_90_low && to_celsius <= tmp_cmf_90_up;

    AdaptiveAshraeResult {
        tmp_cmf: t_cmf,
        tmp_cmf_80_low,
        tmp_cmf_80_up,
        tmp_cmf_90_low,
        tmp_cmf_90_up,
        acceptability_80,
        acceptability_90,
    }
}

/// Calculate adaptive thermal comfort based on EN 16798-1
///
/// The adaptive model can only be used in buildings without mechanical cooling
/// systems where occupants can freely adapt their clothing and open windows.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (recommended range: 10-30°C)
/// * `mean_radiant_temp` - Mean radiant temperature (recommended range: 10-40°C)
/// * `running_mean_outdoor_temp` - Running mean outdoor temperature (recommended range: 10-30°C)
/// * `air_speed` - Air speed (recommended range: 0-2 m/s)
/// * `options` - Adaptive comfort options
///
/// # Returns
///
/// AdaptiveEnResult with comfort temperature and category limits
///
/// # Applicability Limits (when limit_inputs = true)
///
/// * 10 < tdb [°C] < 30
/// * 10 < tr [°C] < 40
/// * 0 < v [m/s] < 2
/// * 10 < t_running_mean [°C] < 30
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::adaptive::{adaptive_en, AdaptiveOptions};
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = adaptive_en(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(20.0),
///     Speed::from_meters_per_second(0.1),
///     Default::default()
/// );
/// assert!(result.acceptability_cat_ii);
/// println!("Comfort temp: {:.1}°C", result.tmp_cmf);
/// ```
pub fn adaptive_en(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    running_mean_outdoor_temp: Temperature,
    air_speed: Speed,
    options: AdaptiveOptions,
) -> AdaptiveEnResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let running_mean_celsius = running_mean_outdoor_temp.as_celsius();
    let speed_mps = air_speed.as_meters_per_second();

    // Calculate operative temperature (use_ashrae=true for adaptive models)
    let to = operative_temperature(dry_bulb_temp, mean_radiant_temp, air_speed, true);

    // Comfort temperature based on running mean outdoor temperature
    // EN 16798-1:2019 adaptive comfort equation:
    // t_cmf = 0.33 * t_running_mean + 18.8
    // where 0.33 is the climate adaptation coefficient
    // and 18.8°C is the base comfort temperature
    let mut t_cmf = 0.33 * running_mean_celsius + 18.8;

    // Apply input limits if requested (EN 16798-1:2019 applicability limits)
    // Dry bulb temperature: 10-30°C
    // Mean radiant temperature: 10-40°C
    // Air speed: 0-2 m/s
    // Running mean outdoor temperature: 10-30°C
    if options.limit_inputs
        && (!(10.0..=30.0).contains(&dry_bulb_celsius)
            || !(10.0..=40.0).contains(&radiant_celsius)
            || !(0.0..=2.0).contains(&speed_mps)
            || !(10.0..=30.0).contains(&running_mean_celsius))
    {
        t_cmf = f64::NAN;
    }

    // Round to 1 decimal place
    t_cmf = libm::round(t_cmf * 10.0) / 10.0;

    // Calculate category bounds (EN 16798-1:2019)
    // Category I (high expectation): ±2°C from comfort temperature
    // Category II (medium expectation): ±3°C from comfort temperature
    // Category III (moderate expectation): ±4°C from comfort temperature
    let tmp_cmf_cat_i_low = t_cmf - 2.0;
    let tmp_cmf_cat_i_up = t_cmf + 2.0;
    let tmp_cmf_cat_ii_low = t_cmf - 3.0;
    let tmp_cmf_cat_ii_up = t_cmf + 3.0;
    let tmp_cmf_cat_iii_low = t_cmf - 4.0;
    let tmp_cmf_cat_iii_up = t_cmf + 4.0;

    // Check acceptability for each category
    let to_celsius = to.as_celsius();
    let acceptability_cat_i =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_cat_i_low && to_celsius <= tmp_cmf_cat_i_up;
    let acceptability_cat_ii =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_cat_ii_low && to_celsius <= tmp_cmf_cat_ii_up;
    let acceptability_cat_iii =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_cat_iii_low && to_celsius <= tmp_cmf_cat_iii_up;

    AdaptiveEnResult {
        tmp_cmf: t_cmf,
        tmp_cmf_cat_i_low,
        tmp_cmf_cat_i_up,
        tmp_cmf_cat_ii_low,
        tmp_cmf_cat_ii_up,
        tmp_cmf_cat_iii_low,
        tmp_cmf_cat_iii_up,
        acceptability_cat_i,
        acceptability_cat_ii,
        acceptability_cat_iii,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_ashrae_comfortable() {
        let result = adaptive_ashrae(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(20.0),
            Speed::from_meters_per_second(0.1),
            Default::default(),
        );
        assert!((result.tmp_cmf - 24.0).abs() < 0.1);
        assert!(result.acceptability_80);
        assert!(result.acceptability_90);
    }

    #[test]
    fn test_adaptive_ashrae_limits() {
        // Test invalid running mean (too low)
        let result = adaptive_ashrae(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(5.0),
            Speed::from_meters_per_second(0.1),
            Default::default(),
        );
        assert!(result.tmp_cmf.is_nan());
        assert!(!result.acceptability_80);

        // Test with limits disabled
        let options = AdaptiveOptions {
            limit_inputs: false,
        };
        let result = adaptive_ashrae(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(5.0),
            Speed::from_meters_per_second(0.1),
            options,
        );
        assert!(!result.tmp_cmf.is_nan());
    }

    #[test]
    fn test_adaptive_ashrae_cooling_effect() {
        // High air speed with high temperature
        let result = adaptive_ashrae(
            Temperature::from_celsius(28.0),
            Temperature::from_celsius(28.0),
            Temperature::from_celsius(20.0),
            Speed::from_meters_per_second(1.0),
            Default::default(),
        );
        // Upper limit should be extended by cooling effect
        assert!(result.tmp_cmf_80_up > result.tmp_cmf + 3.5);
    }

    #[test]
    fn test_adaptive_en_comfortable() {
        let result = adaptive_en(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(20.0),
            Speed::from_meters_per_second(0.1),
            Default::default(),
        );
        assert!((result.tmp_cmf - 25.4).abs() < 0.1);
        assert!(result.acceptability_cat_ii);
    }

    #[test]
    fn test_adaptive_en_categories() {
        let result = adaptive_en(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(20.0),
            Speed::from_meters_per_second(0.1),
            Default::default(),
        );

        // Check category bounds are properly ordered
        assert!(result.tmp_cmf_cat_i_low > result.tmp_cmf_cat_ii_low);
        assert!(result.tmp_cmf_cat_ii_low > result.tmp_cmf_cat_iii_low);
        assert!(result.tmp_cmf_cat_i_up < result.tmp_cmf_cat_ii_up);
        assert!(result.tmp_cmf_cat_ii_up < result.tmp_cmf_cat_iii_up);
    }

    #[test]
    fn test_adaptive_en_limits() {
        // Test invalid running mean
        let result = adaptive_en(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(5.0),
            Speed::from_meters_per_second(0.1),
            Default::default(),
        );
        assert!(result.tmp_cmf.is_nan());
        assert!(!result.acceptability_cat_ii);
    }
}
