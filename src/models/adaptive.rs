//! Adaptive thermal comfort models
//!
//! Adaptive models relate indoor design temperatures to outdoor climate parameters.
//! Only applicable to naturally conditioned spaces without mechanical cooling/heating.

use crate::UnitSystem;
use crate::environment::operative_temperature_celsius;
use crate::unit_system::celsius_in_units;
use crate::utilities::{Units, round_to};
use measurements::{Speed, Temperature};

/// Result from ASHRAE 55 adaptive comfort model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveAshraeResult {
    /// Comfort temperature
    pub tmp_cmf: Temperature,
    /// Lower bound of 80% acceptability
    pub tmp_cmf_80_low: Temperature,
    /// Upper bound of 80% acceptability
    pub tmp_cmf_80_up: Temperature,
    /// Lower bound of 90% acceptability
    pub tmp_cmf_90_low: Temperature,
    /// Upper bound of 90% acceptability
    pub tmp_cmf_90_up: Temperature,
    /// Whether conditions meet 80% acceptability; `None` where the inputs are outside the
    /// applicability limits or an input is NaN (upstream's NaN flag)
    pub acceptability_80: Option<bool>,
    /// Whether conditions meet 90% acceptability; `None` where the inputs are outside the
    /// applicability limits or an input is NaN (upstream's NaN flag)
    pub acceptability_90: Option<bool>,
}

/// Result from EN 16798-1 adaptive comfort model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveEnResult {
    /// Comfort temperature
    pub tmp_cmf: Temperature,
    /// Category I lower limit
    pub tmp_cmf_cat_i_low: Temperature,
    /// Category I upper limit
    pub tmp_cmf_cat_i_up: Temperature,
    /// Category II lower limit
    pub tmp_cmf_cat_ii_low: Temperature,
    /// Category II upper limit
    pub tmp_cmf_cat_ii_up: Temperature,
    /// Category III lower limit
    pub tmp_cmf_cat_iii_low: Temperature,
    /// Category III upper limit
    pub tmp_cmf_cat_iii_up: Temperature,
    /// Whether conditions meet Category I; `None` where the inputs are outside the
    /// applicability limits or an input is NaN (upstream's NaN flag)
    pub acceptability_cat_i: Option<bool>,
    /// Whether conditions meet Category II; `None` as for Category I
    pub acceptability_cat_ii: Option<bool>,
    /// Whether conditions meet Category III; `None` as for Category I
    pub acceptability_cat_iii: Option<bool>,
}

/// The comfort inputs shared by [`adaptive_ashrae`] and [`adaptive_en`].
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveInputs {
    /// Dry bulb air temperature (recommended range: 10-40°C)
    pub tdb: Temperature,
    /// Mean radiant temperature (recommended range: 10-40°C)
    pub tr: Temperature,
    /// Running mean outdoor temperature (applicability: 10-33.5 °C for ASHRAE 55,
    /// 10-30 °C for EN 16798-1)
    pub t_running_mean: Temperature,
    /// Air speed (recommended range: 0-2 m/s)
    pub v: Speed,
}

/// Options for adaptive comfort calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveOptions {
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round the reported temperatures to one decimal place, in the output unit. The
    /// acceptability flags always come from unrounded values, so this never changes them.
    pub round_output: bool,
    /// Unit system, SI or IP (inch-pound), the result is rounded in. Inputs are typed, so
    /// this never changes what a value means: read the result with `as_si()` or `as_ip()`
    /// ([`crate::UnitSystem`]) in either case. Pass the system you read in to
    /// get pythermalcomfort's rounded number, to within the storage round-off described in
    /// [`crate::unit_system`].
    pub units: Units,
}

impl Default for AdaptiveOptions {
    fn default() -> Self {
        Self {
            limit_inputs: true,
            round_output: true,
            units: Units::SI,
        }
    }
}

/// Cooling effect of elevated air speed, shared by the ASHRAE 55 and EN 16798 adaptive
/// models.
///
/// Per ASHRAE 55-2023 Section 5.4.3 the allowance is a three-tier step function of air
/// speed, and applies only once the operative temperature reaches 25 °C.
///
/// # Arguments
///
/// * `v_ms` - Air speed [m/s]
/// * `to_celsius` - Operative temperature [°C]
///
/// # Returns
///
/// Cooling effect [°C]: 0.0, 1.2, 1.8 or 2.2
fn adaptive_cooling_effect(v_ms: f64, to_celsius: f64) -> f64 {
    // A NaN operative temperature gets no allowance, as upstream's
    // `np.where(to >= 25, ...)` gives it none
    if to_celsius.is_nan() || to_celsius < 25.0 {
        return 0.0;
    }
    if v_ms >= 1.2 {
        2.2
    } else if v_ms >= 0.9 {
        1.8
    } else if v_ms >= 0.6 {
        1.2
    } else {
        0.0
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
/// * `inputs` - Required environmental inputs
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
/// use thermalcomfort::models::adaptive::{adaptive_ashrae, AdaptiveInputs, AdaptiveOptions};
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = adaptive_ashrae(
///     AdaptiveInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         t_running_mean: Temperature::from_celsius(20.0),
///         v: Speed::from_meters_per_second(0.1),
///     },
///     Default::default()
/// );
/// assert_eq!(result.acceptability_80, Some(true));
/// println!("Comfort temp: {:.1}°C", result.tmp_cmf.as_celsius());
/// ```
pub fn adaptive_ashrae(inputs: AdaptiveInputs, options: AdaptiveOptions) -> AdaptiveAshraeResult {
    let AdaptiveInputs {
        tdb,
        tr,
        t_running_mean,
        v,
    } = inputs;
    let dry_bulb_celsius = tdb.as_celsius();
    let radiant_celsius = tr.as_celsius();
    let running_mean_celsius = t_running_mean.as_celsius();
    let speed_mps = v.as_meters_per_second();

    // Calculate operative temperature (use_ashrae=true for adaptive models).
    // The plain-f64 core, NOT `operative_temperature`: `to` is compared against the
    // comfort bounds below, and a `Temperature` round-trip through kelvin shifts it by
    // ~2e-14, which flips `acceptability_*` for a `to` sitting exactly on a bound.
    let to_celsius =
        operative_temperature_celsius(dry_bulb_celsius, radiant_celsius, speed_mps, true);

    let ce = adaptive_cooling_effect(speed_mps, to_celsius);

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

    // Calculate acceptability bounds (ASHRAE 55-2023), still in SI Celsius
    // 80% acceptability: ±3.5°C from comfort temperature
    // 90% acceptability: ±2.5°C from comfort temperature
    let tmp_cmf_80_low = t_cmf - 3.5;
    let tmp_cmf_90_low = t_cmf - 2.5;
    let tmp_cmf_80_up = t_cmf + 3.5 + ce;
    let tmp_cmf_90_up = t_cmf + 2.5 + ce;

    // Acceptability comes from the unrounded SI values. It is undefined where the inputs
    // are outside the applicability limits (t_cmf is NaN) or an input is NaN; the
    // cooling effect treats a NaN v as still air, so v is checked directly.
    let valid = !(t_cmf.is_nan() || to_celsius.is_nan() || speed_mps.is_nan());
    let within = |low: f64, up: f64| valid.then_some(low <= to_celsius && to_celsius <= up);
    let acceptability_80 = within(tmp_cmf_80_low, tmp_cmf_80_up);
    let acceptability_90 = within(tmp_cmf_90_low, tmp_cmf_90_up);

    // Only the reported temperatures are rounded, after the conversion to the output unit.
    let units = options.units;
    let report = |value_celsius: f64| {
        let value = celsius_in_units(value_celsius, units);
        Temperature::from_units(
            if options.round_output {
                round_to(value, 1)
            } else {
                value
            },
            units,
        )
    };
    AdaptiveAshraeResult {
        tmp_cmf: report(t_cmf),
        tmp_cmf_80_low: report(tmp_cmf_80_low),
        tmp_cmf_80_up: report(tmp_cmf_80_up),
        tmp_cmf_90_low: report(tmp_cmf_90_low),
        tmp_cmf_90_up: report(tmp_cmf_90_up),
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
/// * `inputs` - Required environmental inputs
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
/// use thermalcomfort::models::adaptive::{adaptive_en, AdaptiveInputs, AdaptiveOptions};
/// use thermalcomfort::{Temperature, Speed};
///
/// let result = adaptive_en(
///     AdaptiveInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         t_running_mean: Temperature::from_celsius(20.0),
///         v: Speed::from_meters_per_second(0.1),
///     },
///     Default::default()
/// );
/// assert_eq!(result.acceptability_cat_ii, Some(true));
/// println!("Comfort temp: {:.1}°C", result.tmp_cmf.as_celsius());
/// ```
pub fn adaptive_en(inputs: AdaptiveInputs, options: AdaptiveOptions) -> AdaptiveEnResult {
    let AdaptiveInputs {
        tdb,
        tr,
        t_running_mean,
        v,
    } = inputs;
    let running_mean_celsius = t_running_mean.as_celsius();
    let speed_mps = v.as_meters_per_second();

    // EN 16798 uses the ISO operative temperature formulation, unlike adaptive_ashrae.
    // Plain-f64 core for the same reason as adaptive_ashrae above.
    let to_celsius =
        operative_temperature_celsius(tdb.as_celsius(), tr.as_celsius(), speed_mps, false);

    let ce = adaptive_cooling_effect(speed_mps, to_celsius);

    // Comfort temperature based on running mean outdoor temperature
    // EN 16798-1:2019 adaptive comfort equation:
    // t_cmf = 0.33 * t_running_mean + 18.8
    // where 0.33 is the climate adaptation coefficient
    // and 18.8°C is the base comfort temperature
    let mut t_cmf = 0.33 * running_mean_celsius + 18.8;

    // Apply input limits if requested. EN 16798-1:2019 bounds only the running mean
    // outdoor temperature, to 10-30 °C; tdb, tr and v are not gated.
    if options.limit_inputs && !(10.0..=30.0).contains(&running_mean_celsius) {
        t_cmf = f64::NAN;
    }

    // Category bounds (EN 16798-1:2019), unrounded and still in SI Celsius. The bands
    // are asymmetric, and the elevated air speed allowance widens only the upper bound.
    // Category I (high expectation): -3 / +2 °C
    // Category II (medium expectation): -4 / +3 °C
    // Category III (moderate expectation): -5 / +4 °C
    let tmp_cmf_cat_i_low = t_cmf - 3.0;
    let tmp_cmf_cat_i_up = t_cmf + 2.0 + ce;
    let tmp_cmf_cat_ii_low = t_cmf - 4.0;
    let tmp_cmf_cat_ii_up = t_cmf + 3.0 + ce;
    let tmp_cmf_cat_iii_low = t_cmf - 5.0;
    let tmp_cmf_cat_iii_up = t_cmf + 4.0 + ce;

    // Acceptability is evaluated against the unrounded SI bounds, and is undefined where
    // the inputs are outside the applicability limits (t_cmf is NaN) or `to` is NaN.
    let valid = !(t_cmf.is_nan() || to_celsius.is_nan());
    let within = |low: f64, up: f64| valid.then_some(low <= to_celsius && to_celsius <= up);
    let acceptability_cat_i = within(tmp_cmf_cat_i_low, tmp_cmf_cat_i_up);
    let acceptability_cat_ii = within(tmp_cmf_cat_ii_low, tmp_cmf_cat_ii_up);
    let acceptability_cat_iii = within(tmp_cmf_cat_iii_low, tmp_cmf_cat_iii_up);

    // Unlike adaptive_ashrae, EN converts to the output unit FIRST, and only then
    // rounds — each bound independently, in whatever unit it ends up in
    // (adaptive_en.py:140-162).
    let units = options.units;
    let mut t_cmf_out = celsius_in_units(t_cmf, units);
    let mut cat_i_low_out = celsius_in_units(tmp_cmf_cat_i_low, units);
    let mut cat_i_up_out = celsius_in_units(tmp_cmf_cat_i_up, units);
    let mut cat_ii_low_out = celsius_in_units(tmp_cmf_cat_ii_low, units);
    let mut cat_ii_up_out = celsius_in_units(tmp_cmf_cat_ii_up, units);
    let mut cat_iii_low_out = celsius_in_units(tmp_cmf_cat_iii_low, units);
    let mut cat_iii_up_out = celsius_in_units(tmp_cmf_cat_iii_up, units);

    if options.round_output {
        t_cmf_out = round_to(t_cmf_out, 1);
        cat_i_low_out = round_to(cat_i_low_out, 1);
        cat_i_up_out = round_to(cat_i_up_out, 1);
        cat_ii_low_out = round_to(cat_ii_low_out, 1);
        cat_ii_up_out = round_to(cat_ii_up_out, 1);
        cat_iii_low_out = round_to(cat_iii_low_out, 1);
        cat_iii_up_out = round_to(cat_iii_up_out, 1);
    }

    AdaptiveEnResult {
        tmp_cmf: Temperature::from_units(t_cmf_out, units),
        tmp_cmf_cat_i_low: Temperature::from_units(cat_i_low_out, units),
        tmp_cmf_cat_i_up: Temperature::from_units(cat_i_up_out, units),
        tmp_cmf_cat_ii_low: Temperature::from_units(cat_ii_low_out, units),
        tmp_cmf_cat_ii_up: Temperature::from_units(cat_ii_up_out, units),
        tmp_cmf_cat_iii_low: Temperature::from_units(cat_iii_low_out, units),
        tmp_cmf_cat_iii_up: Temperature::from_units(cat_iii_up_out, units),
        acceptability_cat_i,
        acceptability_cat_ii,
        acceptability_cat_iii,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(tdb: f64, tr: f64, t_running_mean: f64, v: f64) -> AdaptiveInputs {
        AdaptiveInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            t_running_mean: Temperature::from_celsius(t_running_mean),
            v: Speed::from_meters_per_second(v),
        }
    }

    #[test]
    fn test_adaptive_ashrae_comfortable() {
        let result = adaptive_ashrae(inputs(25.0, 25.0, 20.0, 0.1), Default::default());
        assert!((result.tmp_cmf.as_celsius() - 24.0).abs() < 0.1);
        assert_eq!(result.acceptability_80, Some(true));
        assert_eq!(result.acceptability_90, Some(true));
    }

    #[test]
    fn test_adaptive_ashrae_limits() {
        // Test invalid running mean (too low)
        let result = adaptive_ashrae(inputs(25.0, 25.0, 5.0, 0.1), Default::default());
        assert!(result.tmp_cmf.as_celsius().is_nan());
        assert_eq!(result.acceptability_80, None);
        assert_eq!(result.acceptability_90, None);

        // Test with limits disabled
        let options = AdaptiveOptions {
            limit_inputs: false,
            ..Default::default()
        };
        let result = adaptive_ashrae(inputs(25.0, 25.0, 5.0, 0.1), options);
        assert!(!result.tmp_cmf.as_celsius().is_nan());
    }

    #[test]
    fn test_adaptive_ashrae_cooling_effect() {
        // High air speed with high temperature
        let result = adaptive_ashrae(inputs(28.0, 28.0, 20.0, 1.0), Default::default());
        // Upper limit should be extended by cooling effect
        assert!(result.tmp_cmf_80_up.as_celsius() > result.tmp_cmf.as_celsius() + 3.5);
    }

    #[test]
    fn test_adaptive_en_comfortable() {
        let result = adaptive_en(inputs(25.0, 25.0, 20.0, 0.1), Default::default());
        assert!((result.tmp_cmf.as_celsius() - 25.4).abs() < 0.1);
        assert_eq!(result.acceptability_cat_ii, Some(true));
    }

    #[test]
    fn test_adaptive_en_categories() {
        let result = adaptive_en(inputs(25.0, 25.0, 20.0, 0.1), Default::default());

        // Check category bounds are properly ordered
        assert!(result.tmp_cmf_cat_i_low.as_celsius() > result.tmp_cmf_cat_ii_low.as_celsius());
        assert!(result.tmp_cmf_cat_ii_low.as_celsius() > result.tmp_cmf_cat_iii_low.as_celsius());
        assert!(result.tmp_cmf_cat_i_up.as_celsius() < result.tmp_cmf_cat_ii_up.as_celsius());
        assert!(result.tmp_cmf_cat_ii_up.as_celsius() < result.tmp_cmf_cat_iii_up.as_celsius());
    }

    #[test]
    fn test_adaptive_en_limits() {
        // Test invalid running mean
        let result = adaptive_en(inputs(25.0, 25.0, 5.0, 0.1), Default::default());
        assert!(result.tmp_cmf.as_celsius().is_nan());
        assert_eq!(result.acceptability_cat_ii, None);
    }

    #[test]
    fn test_adaptive_en_hot_indoor_conditions() {
        // Indoor air above 30 °C is the condition the model has to assess, not an
        // input error: only the running mean outdoor temperature is limited.
        let result = adaptive_en(inputs(30.5, 30.5, 24.0, 0.1), Default::default());
        assert!((result.tmp_cmf.as_celsius() - 26.7).abs() < 1e-9);
        assert!((result.tmp_cmf_cat_iii_up.as_celsius() - 30.7).abs() < 1e-9);
        assert_eq!(result.acceptability_cat_iii, Some(true));
        assert_eq!(result.acceptability_cat_ii, Some(false));

        // EN 16798-1 accepts running mean outdoor temperatures up to 30 °C, not
        // ASHRAE 55's 33.5 °C
        let result = adaptive_en(inputs(30.0, 30.0, 30.0, 0.1), Default::default());
        assert!(!result.tmp_cmf.as_celsius().is_nan());
        let result = adaptive_en(inputs(30.0, 30.0, 30.5, 0.1), Default::default());
        assert!(result.tmp_cmf.as_celsius().is_nan());
        assert_eq!(result.acceptability_cat_iii, None);
    }

    #[test]
    fn test_adaptive_en_category_limits() {
        // trm = 20 °C: t_cmf = 25.4 °C
        let result = adaptive_en(inputs(25.0, 25.0, 20.0, 0.1), Default::default());
        assert!((result.tmp_cmf_cat_i_low.as_celsius() - 22.4).abs() < 1e-9);
        assert!((result.tmp_cmf_cat_ii_low.as_celsius() - 21.4).abs() < 1e-9);
        assert!((result.tmp_cmf_cat_iii_low.as_celsius() - 20.4).abs() < 1e-9);
        assert!((result.tmp_cmf_cat_i_up.as_celsius() - 27.4).abs() < 1e-9);
        assert!((result.tmp_cmf_cat_ii_up.as_celsius() - 28.4).abs() < 1e-9);
        assert!((result.tmp_cmf_cat_iii_up.as_celsius() - 29.4).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_en_cooling_effect() {
        // Elevated air speed raises the upper limits when to >= 25 °C
        let still = adaptive_en(inputs(28.0, 28.0, 20.0, 0.1), Default::default());
        let fan = adaptive_en(inputs(28.0, 28.0, 20.0, 1.0), Default::default());
        let up_delta = fan.tmp_cmf_cat_ii_up.as_celsius() - still.tmp_cmf_cat_ii_up.as_celsius();
        let low_delta = fan.tmp_cmf_cat_ii_low.as_celsius() - still.tmp_cmf_cat_ii_low.as_celsius();
        assert!((up_delta - 1.8).abs() < 1e-9);
        assert!(low_delta.abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_en_iso_operative_temperature() {
        // With tdb != tr and v = 0.5 m/s the ISO operative temperature is
        // (30 * sqrt(5) + 26) / (1 + sqrt(5)) = 28.76 °C, inside category III
        // (upper limit 25.4 + 4 = 29.4 °C). The ASHRAE weighting would give 28.4 °C.
        let result = adaptive_en(
            inputs(30.0, 26.0, 20.0, 0.5),
            AdaptiveOptions {
                round_output: false,
                ..Default::default()
            },
        );
        assert_eq!(result.acceptability_cat_iii, Some(true));
        assert_eq!(result.acceptability_cat_ii, Some(false));
    }

    #[test]
    fn test_adaptive_ashrae_round_output() {
        // trm=27 yields t_cmf = 0.31*27 + 17.8 = 26.17, which rounds to 26.2.
        // The 0.03 gap lets us detect whether rounding was applied.
        let case = inputs(25.0, 25.0, 27.0, 0.1);

        let rounded = adaptive_ashrae(
            case,
            AdaptiveOptions {
                round_output: true,
                ..Default::default()
            },
        );
        assert!((rounded.tmp_cmf.as_celsius() - 26.2).abs() < 1e-9);
        // Each bound is rounded on its own: 26.17 - 3.5 = 22.67 -> 22.7.
        assert!((rounded.tmp_cmf_80_low.as_celsius() - 22.7).abs() < 1e-9);
        assert!((rounded.tmp_cmf_80_up.as_celsius() - 29.7).abs() < 1e-9);

        let unrounded = adaptive_ashrae(
            case,
            AdaptiveOptions {
                round_output: false,
                ..Default::default()
            },
        );
        assert!((unrounded.tmp_cmf.as_celsius() - 26.17).abs() < 1e-9);
        assert!((unrounded.tmp_cmf_80_low.as_celsius() - (26.17 - 3.5)).abs() < 1e-9);
        assert!((unrounded.tmp_cmf_80_up.as_celsius() - (26.17 + 3.5)).abs() < 1e-9);

        // The two paths must differ — proves round_output=false actually changes behavior.
        assert!((rounded.tmp_cmf.as_celsius() - unrounded.tmp_cmf.as_celsius()).abs() > 0.01);
    }

    #[test]
    fn test_adaptive_en_round_output() {
        // trm=22 yields t_cmf = 0.33*22 + 18.8 = 26.06, which rounds to 26.1.
        let case = inputs(25.0, 25.0, 22.0, 0.1);

        let rounded = adaptive_en(
            case,
            AdaptiveOptions {
                round_output: true,
                ..Default::default()
            },
        );
        assert!((rounded.tmp_cmf.as_celsius() - 26.1).abs() < 1e-9);

        let unrounded = adaptive_en(
            case,
            AdaptiveOptions {
                round_output: false,
                ..Default::default()
            },
        );
        assert!((unrounded.tmp_cmf.as_celsius() - 26.06).abs() < 1e-9);

        // The two paths must differ.
        assert!((rounded.tmp_cmf.as_celsius() - unrounded.tmp_cmf.as_celsius()).abs() > 0.01);
    }

    #[test]
    fn test_adaptive_ashrae_units_ip_rounds_after_conversion() {
        // The IP temperatures are rounded to one decimal in °F, after the conversion of
        // the unrounded SI value: trm=27°C gives t_cmf = 26.17°C = 79.106°F -> 79.1°F.
        let ip = adaptive_ashrae(
            inputs(25.0, 25.0, 27.0, 0.1),
            AdaptiveOptions {
                units: Units::IP,
                ..Default::default()
            },
        );
        assert!((ip.tmp_cmf.as_fahrenheit() - 79.1).abs() < 1e-9);
        // 26.17 - 3.5 = 22.67°C = 72.806°F -> 72.8°F; rounding in SI first would give
        // 22.7°C = 72.86°F.
        assert!((ip.tmp_cmf_80_low.as_fahrenheit() - 72.8).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_ashrae_round_output_never_changes_acceptability() {
        // tdb = tr = 21.11, trm = 22: t_cmf = 24.62, so the 80 % lower limit is 21.12
        // and `to` is just outside it. Rounding t_cmf first (to 24.6, limit 21.1) used
        // to accept the same point.
        for round_output in [true, false] {
            let result = adaptive_ashrae(
                inputs(21.11, 21.11, 22.0, 0.1),
                AdaptiveOptions {
                    round_output,
                    ..Default::default()
                },
            );
            assert_eq!(
                result.acceptability_80,
                Some(false),
                "round_output={round_output}"
            );
        }
    }

    #[test]
    fn test_adaptive_nan_input_gives_undefined_acceptability() {
        let nan_v = AdaptiveInputs {
            v: Speed::from_meters_per_second(f64::NAN),
            ..inputs(25.0, 25.0, 20.0, 0.1)
        };
        let options = AdaptiveOptions {
            limit_inputs: false,
            ..Default::default()
        };
        assert_eq!(adaptive_ashrae(nan_v, options).acceptability_80, None);
        assert_eq!(adaptive_en(nan_v, options).acceptability_cat_i, None);
    }

    #[test]
    fn test_adaptive_en_units_ip_rounds_after_bounds() {
        // adaptive_en converts to °F BEFORE rounding, then rounds each bound
        // independently in the output unit, so the IP result should land on a clean
        // 1-decimal °F boundary (as adaptive_ashrae's IP output above).
        let ip = adaptive_en(
            inputs(25.0, 25.0, 22.0, 0.1),
            AdaptiveOptions {
                units: Units::IP,
                ..Default::default()
            },
        );
        let f = ip.tmp_cmf.as_fahrenheit();
        assert!(((f * 10.0) - (f * 10.0).round()).abs() < 1e-9);
    }
}
