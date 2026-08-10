//! Wet Bulb Globe Temperature (WBGT) Index
//!
//! The WBGT is a heat stress index that measures the thermal environment to which a person is
//! exposed. It should be used as a screening tool to determine whether heat stress is present.
//! Implements ISO 7243:2017 standard.

use measurements::Temperature;

/// The comfort inputs to [`wbgt`].
///
/// `wet_bulb_temp` and `globe_temp` are consecutive [`Temperature`]s; naming every field
/// forecloses a silent transposition between them.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WbgtInputs {
    /// Natural (no forced air flow) wet bulb temperature
    pub wet_bulb_temp: Temperature,
    /// Globe temperature
    pub globe_temp: Temperature,
}

/// Optional parameters for [`wbgt`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WbgtOptions {
    /// Dry bulb air temperature, required if `with_solar_load` is true
    pub dry_bulb_temp: Option<Temperature>,
    /// Whether the person is exposed to direct solar radiation
    pub with_solar_load: bool,
    /// Whether to round output to 1 decimal place
    pub round_output: bool,
}

impl Default for WbgtOptions {
    fn default() -> Self {
        Self {
            dry_bulb_temp: None,
            with_solar_load: false,
            round_output: true,
        }
    }
}

/// Calculate Wet Bulb Globe Temperature (WBGT) index
///
/// The WBGT is a heat stress index in compliance with ISO 7243:2017.
/// It measures the thermal environment to which a person is exposed and should
/// be used as a screening tool to determine whether heat stress is present.
///
/// The WBGT determines the impact of heat on a person throughout the course of
/// a working day (up to 8 hours). It does not apply to very brief heat exposures.
///
/// # Returns
///
/// WBGT index [°C]
///
/// # Formulas
///
/// - **Without solar load**: WBGT = 0.7 * twb + 0.3 * tg
/// - **With solar load**: WBGT = 0.7 * twb + 0.2 * tg + 0.1 * tdb
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::wbgt::{wbgt, WbgtInputs, WbgtOptions};
/// use thermalcomfort::Temperature;
///
/// // Indoor environment (no direct solar radiation)
/// let result = wbgt(
///     WbgtInputs {
///         wet_bulb_temp: Temperature::from_celsius(25.0),
///         globe_temp: Temperature::from_celsius(32.0),
///     },
///     Default::default(),
/// );
/// assert!((result - 27.1).abs() < 0.1);
///
/// // Outdoor environment (with solar load)
/// let options = WbgtOptions {
///     dry_bulb_temp: Some(Temperature::from_celsius(20.0)),
///     with_solar_load: true,
///     round_output: true,
/// };
/// let result = wbgt(
///     WbgtInputs {
///         wet_bulb_temp: Temperature::from_celsius(25.0),
///         globe_temp: Temperature::from_celsius(32.0),
///     },
///     options,
/// );
/// assert!((result - 25.9).abs() < 0.1);
/// ```
///
/// # References
///
/// - ISO 7243:2017 - Ergonomics of the thermal environment
pub fn wbgt(inputs: WbgtInputs, options: WbgtOptions) -> f64 {
    let WbgtInputs {
        wet_bulb_temp,
        globe_temp,
    } = inputs;
    let wet_bulb_celsius = wet_bulb_temp.as_celsius();
    let globe_celsius = globe_temp.as_celsius();
    let dry_bulb_celsius_opt = options.dry_bulb_temp.map(|t| t.as_celsius());

    // Validate that tdb is provided when solar load is present
    if options.with_solar_load && dry_bulb_celsius_opt.is_none() {
        return f64::NAN;
    }

    // Calculate WBGT based on solar load condition
    // WBGT formulas from ISO 7243 and ACGIH:
    // - Outdoors (with solar load): WBGT = 0.7*Twb + 0.2*Tg + 0.1*Tdb
    // - Indoors (no solar load): WBGT = 0.7*Twb + 0.3*Tg
    // Coefficients represent weighting of each temperature component
    let mut wbgt_value = if options.with_solar_load {
        let dry_bulb_celsius = dry_bulb_celsius_opt.unwrap();
        0.7 * wet_bulb_celsius + 0.2 * globe_celsius + 0.1 * dry_bulb_celsius
    } else {
        0.7 * wet_bulb_celsius + 0.3 * globe_celsius
    };

    // Round to 1 decimal place if requested
    if options.round_output {
        wbgt_value = crate::utilities::round_half_even(wbgt_value * 10.0) / 10.0;
    }

    wbgt_value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(twb: f64, tg: f64) -> WbgtInputs {
        WbgtInputs {
            wet_bulb_temp: Temperature::from_celsius(twb),
            globe_temp: Temperature::from_celsius(tg),
        }
    }

    #[test]
    fn test_wbgt_indoor() {
        // Test without solar load (indoor environment)
        let result = wbgt(inputs(25.0, 32.0), Default::default());
        assert!((result - 27.1).abs() < 0.1);
    }

    #[test]
    fn test_wbgt_outdoor() {
        // Test with solar load (outdoor environment)
        let options = WbgtOptions {
            dry_bulb_temp: Some(Temperature::from_celsius(20.0)),
            with_solar_load: true,
            round_output: true,
        };
        let result = wbgt(inputs(25.0, 32.0), options);
        assert!((result - 25.9).abs() < 0.1);
    }

    #[test]
    fn test_wbgt_no_rounding() {
        let options = WbgtOptions {
            dry_bulb_temp: None,
            with_solar_load: false,
            round_output: false,
        };
        let result = wbgt(inputs(25.0, 32.0), options);
        // 0.7 * 25 + 0.3 * 32 = 17.5 + 9.6 = 27.1
        assert!((result - 27.1).abs() < 0.001);
    }

    #[test]
    fn test_wbgt_missing_tdb() {
        // Should return NaN when solar load is true but tdb is None
        let options = WbgtOptions {
            dry_bulb_temp: None,
            with_solar_load: true,
            round_output: true,
        };
        let result = wbgt(inputs(25.0, 32.0), options);
        assert!(result.is_nan());
    }

    #[test]
    fn test_wbgt_formulas() {
        // Test exact formulas

        // Without solar: 0.7 * 30 + 0.3 * 35 = 21 + 10.5 = 31.5
        let result = wbgt(
            inputs(30.0, 35.0),
            WbgtOptions {
                dry_bulb_temp: None,
                with_solar_load: false,
                round_output: false,
            },
        );
        assert!((result - 31.5).abs() < 0.001);

        // With solar: 0.7 * 30 + 0.2 * 35 + 0.1 * 28 = 21 + 7 + 2.8 = 30.8
        let result = wbgt(
            inputs(30.0, 35.0),
            WbgtOptions {
                dry_bulb_temp: Some(Temperature::from_celsius(28.0)),
                with_solar_load: true,
                round_output: false,
            },
        );
        assert!((result - 30.8).abs() < 0.001);
    }
}
