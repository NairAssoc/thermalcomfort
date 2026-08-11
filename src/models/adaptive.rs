//! Adaptive thermal comfort models
//!
//! Adaptive models relate indoor design temperatures to outdoor climate parameters.
//! Only applicable to naturally conditioned spaces without mechanical cooling/heating.

use crate::psychrometrics::{
    OperativeTemperatureInputs, OperativeTemperatureOptions, operative_temperature,
};
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
    /// Whether conditions meet 80% acceptability
    pub acceptability_80: bool,
    /// Whether conditions meet 90% acceptability
    pub acceptability_90: bool,
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
    /// Whether conditions meet Category I
    pub acceptability_cat_i: bool,
    /// Whether conditions meet Category II
    pub acceptability_cat_ii: bool,
    /// Whether conditions meet Category III
    pub acceptability_cat_iii: bool,
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
    /// Running mean outdoor temperature (recommended range: 10-33.5°C)
    pub t_running_mean: Temperature,
    /// Air speed (recommended range: 0-2 m/s)
    pub v: Speed,
}

/// Options for adaptive comfort calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveOptions {
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round `tmp_cmf` to one decimal place before deriving comfort bounds
    pub round_output: bool,
    /// Unit system the result is expressed in.
    ///
    /// Upstream's IP branch (`adaptive_ashrae.py`/`adaptive_en.py`) is a genuine
    /// affine °C→°F conversion (`value * 9 / 5 + 32`), unlike `cooling_effect`'s
    /// literal, non-physical rescale — so there is no "meaningless as_celsius" caveat
    /// here; `as_fahrenheit()` on an `IP` result is the real Fahrenheit value.
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

/// Convert a Celsius value to the requested output unit, matching pythermalcomfort's
/// `units_converter`'s literal `(value * 9 / 5) + 32` formula exactly (rather than
/// routing through `Temperature`'s Kelvin storage, which would lose the last ULP —
/// see 28adfa8).
fn to_output_unit(value_celsius: f64, units: Units) -> f64 {
    match units {
        Units::SI => value_celsius,
        Units::IP => value_celsius * 9.0 / 5.0 + 32.0,
    }
}

/// Wrap a value already expressed in the given output unit back into a [`Temperature`].
fn wrap_temperature(value_in_unit: f64, units: Units) -> Temperature {
    match units {
        Units::SI => Temperature::from_celsius(value_in_unit),
        Units::IP => Temperature::from_fahrenheit(value_in_unit),
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
    if to_celsius < 25.0 {
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
/// assert!(result.acceptability_80);
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

    // Calculate operative temperature (use_ashrae=true for adaptive models)
    let to = operative_temperature(
        OperativeTemperatureInputs { tdb, tr, v },
        OperativeTemperatureOptions { use_ashrae: true },
    );

    let ce = adaptive_cooling_effect(speed_mps, to.as_celsius());

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

    // pythermalcomfort rounds `t_cmf` here, in SI, BEFORE deriving the comfort bounds
    // — unlike adaptive_en, which rounds each bound independently at the very end.
    if options.round_output {
        t_cmf = crate::utilities::round_half_even(t_cmf * 10.0) / 10.0;
    }

    // Calculate acceptability bounds (ASHRAE 55-2023), still in SI Celsius
    // 80% acceptability: ±3.5°C from comfort temperature
    // 90% acceptability: ±2.5°C from comfort temperature
    let tmp_cmf_80_low = t_cmf - 3.5;
    let tmp_cmf_90_low = t_cmf - 2.5;
    let tmp_cmf_80_up = t_cmf + 3.5 + ce;
    let tmp_cmf_90_up = t_cmf + 2.5 + ce;

    // Check acceptability against the SI values (unaffected by the `units` output
    // rescale below)
    let to_celsius = to.as_celsius();
    let acceptability_80 =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_80_low && to_celsius <= tmp_cmf_80_up;
    let acceptability_90 =
        !t_cmf.is_nan() && to_celsius >= tmp_cmf_90_low && to_celsius <= tmp_cmf_90_up;

    // Convert to the output unit as the last step; no further rounding happens after
    // this in pythermalcomfort, so an IP result carries the extra decimals from the
    // °C-to-°F conversion of the already-rounded SI value.
    let units = options.units;
    AdaptiveAshraeResult {
        tmp_cmf: wrap_temperature(to_output_unit(t_cmf, units), units),
        tmp_cmf_80_low: wrap_temperature(to_output_unit(tmp_cmf_80_low, units), units),
        tmp_cmf_80_up: wrap_temperature(to_output_unit(tmp_cmf_80_up, units), units),
        tmp_cmf_90_low: wrap_temperature(to_output_unit(tmp_cmf_90_low, units), units),
        tmp_cmf_90_up: wrap_temperature(to_output_unit(tmp_cmf_90_up, units), units),
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
/// assert!(result.acceptability_cat_ii);
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

    // EN 16798 uses the ISO operative temperature formulation, unlike adaptive_ashrae
    let to = operative_temperature(
        OperativeTemperatureInputs { tdb, tr, v },
        OperativeTemperatureOptions { use_ashrae: false },
    );

    let ce = adaptive_cooling_effect(speed_mps, to.as_celsius());

    // Comfort temperature based on running mean outdoor temperature
    // EN 16798-1:2019 adaptive comfort equation:
    // t_cmf = 0.33 * t_running_mean + 18.8
    // where 0.33 is the climate adaptation coefficient
    // and 18.8°C is the base comfort temperature
    let mut t_cmf = 0.33 * running_mean_celsius + 18.8;

    // Apply input limits if requested. EN 16798-1:2019 bounds only the running mean
    // outdoor temperature; tdb, tr and v are not gated.
    if options.limit_inputs && !(10.0..=33.5).contains(&running_mean_celsius) {
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

    // Acceptability is evaluated against the unrounded SI bounds
    let to_celsius = to.as_celsius();
    let acceptability_cat_i = to_celsius >= tmp_cmf_cat_i_low && to_celsius <= tmp_cmf_cat_i_up;
    let acceptability_cat_ii = to_celsius >= tmp_cmf_cat_ii_low && to_celsius <= tmp_cmf_cat_ii_up;
    let acceptability_cat_iii =
        to_celsius >= tmp_cmf_cat_iii_low && to_celsius <= tmp_cmf_cat_iii_up;

    // Unlike adaptive_ashrae, EN converts to the output unit FIRST, and only then
    // rounds — each bound independently, in whatever unit it ends up in
    // (adaptive_en.py:140-162).
    let units = options.units;
    let mut t_cmf_out = to_output_unit(t_cmf, units);
    let mut cat_i_low_out = to_output_unit(tmp_cmf_cat_i_low, units);
    let mut cat_i_up_out = to_output_unit(tmp_cmf_cat_i_up, units);
    let mut cat_ii_low_out = to_output_unit(tmp_cmf_cat_ii_low, units);
    let mut cat_ii_up_out = to_output_unit(tmp_cmf_cat_ii_up, units);
    let mut cat_iii_low_out = to_output_unit(tmp_cmf_cat_iii_low, units);
    let mut cat_iii_up_out = to_output_unit(tmp_cmf_cat_iii_up, units);

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
        tmp_cmf: wrap_temperature(t_cmf_out, units),
        tmp_cmf_cat_i_low: wrap_temperature(cat_i_low_out, units),
        tmp_cmf_cat_i_up: wrap_temperature(cat_i_up_out, units),
        tmp_cmf_cat_ii_low: wrap_temperature(cat_ii_low_out, units),
        tmp_cmf_cat_ii_up: wrap_temperature(cat_ii_up_out, units),
        tmp_cmf_cat_iii_low: wrap_temperature(cat_iii_low_out, units),
        tmp_cmf_cat_iii_up: wrap_temperature(cat_iii_up_out, units),
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
        assert!(result.acceptability_80);
        assert!(result.acceptability_90);
    }

    #[test]
    fn test_adaptive_ashrae_limits() {
        // Test invalid running mean (too low)
        let result = adaptive_ashrae(inputs(25.0, 25.0, 5.0, 0.1), Default::default());
        assert!(result.tmp_cmf.as_celsius().is_nan());
        assert!(!result.acceptability_80);

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
        assert!(result.acceptability_cat_ii);
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
        assert!(!result.acceptability_cat_ii);
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
        // Derived bounds inherit the rounded value.
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
    fn test_adaptive_ashrae_units_ip_rounds_before_bounds() {
        // trm=27°C -> t_cmf rounds to 26.2°C in SI *before* the bounds are derived,
        // then that already-rounded value (and the derived bounds) convert to °F as
        // the last step, with no further rounding — so IP results carry extra decimals.
        let case = inputs(25.0, 25.0, 27.0, 0.1);
        let si = adaptive_ashrae(
            case,
            AdaptiveOptions {
                units: Units::SI,
                ..Default::default()
            },
        );
        let ip = adaptive_ashrae(
            case,
            AdaptiveOptions {
                units: Units::IP,
                ..Default::default()
            },
        );

        // Converting the rounded SI value should reproduce the IP value exactly
        // (same formula, same order of operations).
        let expected_f = si.tmp_cmf.as_celsius() * 9.0 / 5.0 + 32.0;
        assert!((ip.tmp_cmf.as_fahrenheit() - expected_f).abs() < 1e-9);
        // Not a round-numbered °F value, proving no further rounding happened in IP.
        assert!((ip.tmp_cmf.as_fahrenheit() * 10.0).round() / 10.0 != ip.tmp_cmf.as_fahrenheit());
    }

    #[test]
    fn test_adaptive_en_units_ip_rounds_after_bounds() {
        // adaptive_en converts to °F BEFORE rounding, then rounds each bound
        // independently in the output unit, so the IP result should land on a clean
        // 1-decimal °F boundary (unlike adaptive_ashrae's IP output above).
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
