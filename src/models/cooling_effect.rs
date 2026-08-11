//! Cooling effect of elevated air speed
//!
//! This module calculates the cooling effect when air speed is elevated above
//! the still air threshold (0.1 m/s).

use crate::models::set_tmp::{SetInputs, SetOptions, set_tmp};
use crate::numerical::brentq;
use crate::utilities::{Posture, Units};
use crate::{ClothingInsulation, MetabolicRate, TemperatureDelta};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

// pythermalcomfort's cooling_effect() hardcodes these as private module constants
// (`_BODY_SURFACE_AREA`, `_P_ATM`, `_POSITION_STANDING_CODE`) rather than exposing them
// as parameters, even though the `set_tmp`/`two_nodes_gagge` machinery it calls into
// defaults them differently in general. An earlier port exposed all four as
// `CoolingEffectOptions` fields, which let a caller ask for combinations upstream can
// never produce — the opposite of the "don't invent parameters upstream lacks" rule.
// They are now fixed here instead.
const STILL_AIR_THRESHOLD_MS: f64 = 0.1;
const BODY_SURFACE_AREA_M2: f64 = 1.8258;
const P_ATM_PA: f64 = 101_325.0;
// two_nodes_gagge's calculate_ce=True path hardcodes position=1 (the "standing" branch)
// regardless of any posture argument, so cooling_effect always uses it too.
const POSTURE: Posture = Posture::Standing;

/// The comfort inputs to [`cooling_effect`].
///
/// `tdb` and `tr` are consecutive [`Temperature`]s; naming
/// every field forecloses a silent transposition between them.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoolingEffectInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Relative air speed
    pub vr: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
}

/// Optional parameters for [`cooling_effect`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoolingEffectOptions {
    /// External work
    pub wme: MetabolicRate,
    /// Unit system the result is expressed in.
    ///
    /// Upstream's IP branch is `_ce / 1.8 * 3.28`, not a standard delta-°C-to-°F
    /// conversion (`* 9 / 5`) — 3.28 is a feet-per-metre factor, not a temperature one.
    /// It is not derivable from the SI answer, so it is ported verbatim rather than
    /// reconstructed from first principles. Because the result is not a genuine
    /// Fahrenheit delta, the `Units::IP` result is carried in the returned
    /// [`TemperatureDelta`] via `from_fahrenheit`/`as_fahrenheit` purely so the exact
    /// upstream number round-trips; `as_celsius()` on that value is not physically
    /// meaningful.
    pub units: Units,
}

impl Default for CoolingEffectOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            units: Units::SI,
        }
    }
}

/// Calculate the cooling effect of elevated air speed
///
/// Returns the temperature difference that would equalize the Standard Effective
/// Temperature (SET) between the actual environment (with elevated air speed) and
/// a reference environment at still air conditions.
///
/// The cooling effect is only applicable when air speed exceeds the still air
/// threshold (0.1 m/s, matching pythermalcomfort's hardcoded value).
///
/// # Returns
///
/// The cooling effect as a [`TemperatureDelta`] — the temperature reduction that
/// produces equivalent SET at still air conditions. Zero if `vr` is at
/// or below the still air threshold.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::cooling_effect::{
///     cooling_effect, CoolingEffectInputs, CoolingEffectOptions,
/// };
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// // Calculate cooling effect with elevated air speed
/// let ce = cooling_effect(
///     CoolingEffectInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         vr: Speed::from_meters_per_second(0.5),
///         rh: Humidity::from_percent(50.0),
///         met: MetabolicRate::from_met(1.2),
///         clo: ClothingInsulation::from_clo(0.5),
///     },
///     Default::default(),
/// );
/// println!("Cooling effect: {:.2}°C", ce.as_celsius());
/// ```
pub fn cooling_effect(
    inputs: CoolingEffectInputs,
    options: CoolingEffectOptions,
) -> TemperatureDelta {
    let CoolingEffectInputs {
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
    } = inputs;

    let air_speed = vr.as_meters_per_second();

    // No cooling effect if air speed is at or below still air threshold
    if air_speed <= STILL_AIR_THRESHOLD_MS {
        return TemperatureDelta::from_celsius(0.0);
    }

    // Calculate SET at the actual air speed
    let set_options = SetOptions {
        wme: options.wme,
        body_surface_area: Area::from_square_meters(BODY_SURFACE_AREA_M2),
        p_atm: Pressure::from_pascals(P_ATM_PA),
        position: POSTURE,
        limit_inputs: false, // Don't limit inputs for cooling effect calculation
        round_output: false, // Need exact values for root finding
        // The reduced SET-only solver path. pythermalcomfort's cooling_effect is the
        // one caller that opts into this; set_tmp itself defaults it off.
        calculate_ce: true,
    };

    let initial_set = set_tmp(
        SetInputs {
            tdb,
            tr,
            v: vr,
            rh,
            met,
            clo,
        },
        set_options,
    )
    .as_celsius();

    // If SET calculation failed, return 0
    if initial_set.is_nan() {
        return TemperatureDelta::from_celsius(0.0);
    }

    let dry_bulb_celsius = tdb.as_celsius();
    let radiant_celsius = tr.as_celsius();

    // Define the function to find the root of:
    // We want to find ce such that SET(tdb-ce, tr-ce, still_air) = SET(tdb, tr, vr)
    let function = |cooling_effect_delta: f64| -> f64 {
        let set_still = set_tmp(
            SetInputs {
                tdb: Temperature::from_celsius(dry_bulb_celsius - cooling_effect_delta),
                tr: Temperature::from_celsius(radiant_celsius - cooling_effect_delta),
                v: Speed::from_meters_per_second(STILL_AIR_THRESHOLD_MS),
                rh,
                met,
                clo,
            },
            set_options,
        )
        .as_celsius();
        set_still - initial_set
    };

    // Use Brent's method to find the cooling effect
    // Search in range [0, 40] °C
    // scipy's brentq defaults to xtol=2e-12; the previous 1e-3 here was nine orders of
    // magnitude looser and converged to a visibly different root (ce 15.49 vs 13.67 at
    // tdb=8.1, tr=31.5, vr=0.58, rh=5.6, met=4.6), which then shifted pmv_ppd_ashrae.
    let ce_celsius = brentq(function, 0.0, 40.0, Some(2e-12), Some(100)).unwrap_or(0.0);

    // pythermalcomfort rounds to two decimals AFTER the unit rescale below, not before.
    match options.units {
        Units::SI => {
            let ce = crate::utilities::round_half_even(ce_celsius * 100.0) / 100.0;
            TemperatureDelta::from_celsius(ce)
        }
        Units::IP => {
            // Upstream's literal `_ce / 1.8 * 3.28`, not a delta-°C-to-°F conversion.
            let ce_ip = ce_celsius / 1.8 * 3.28;
            let ce_ip = crate::utilities::round_half_even(ce_ip * 100.0) / 100.0;
            TemperatureDelta::from_fahrenheit(ce_ip)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(tdb: f64, tr: f64, vr: f64, rh: f64, met: f64, clo: f64) -> CoolingEffectInputs {
        CoolingEffectInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            vr: Speed::from_meters_per_second(vr),
            rh: Humidity::from_percent(rh),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
        }
    }

    #[test]
    fn test_cooling_effect_no_effect() {
        // At still air threshold, should have no cooling effect
        let ce = cooling_effect(inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5), Default::default());
        assert_eq!(ce.as_celsius(), 0.0);

        // Below still air threshold, should have no cooling effect
        let ce = cooling_effect(inputs(25.0, 25.0, 0.05, 50.0, 1.2, 0.5), Default::default());
        assert_eq!(ce.as_celsius(), 0.0);
    }

    #[test]
    fn test_cooling_effect_elevated_speed() {
        // With elevated air speed, should have positive cooling effect
        let ce = cooling_effect(inputs(25.0, 25.0, 0.5, 50.0, 1.2, 0.5), Default::default());
        assert!(ce.as_celsius() > 0.0);
        assert!(ce.as_celsius() < 5.0); // Reasonable range for cooling effect
    }

    #[test]
    fn test_cooling_effect_high_speed() {
        // Higher air speed should produce larger cooling effect
        let ce1 = cooling_effect(inputs(25.0, 25.0, 0.3, 50.0, 1.2, 0.5), Default::default());
        let ce2 = cooling_effect(inputs(25.0, 25.0, 0.8, 50.0, 1.2, 0.5), Default::default());

        assert!(ce2.as_celsius() > ce1.as_celsius());
    }

    #[test]
    fn test_cooling_effect_hot_conditions() {
        // Test in hot conditions
        let ce = cooling_effect(inputs(30.0, 30.0, 0.5, 50.0, 1.2, 0.5), Default::default());
        assert!(ce.as_celsius() > 0.0);
    }

    /// `units` controls how the result is packaged; upstream's IP rescale is a literal
    /// `/1.8*3.28` on the Celsius value, not a physical delta-°C-to-°F conversion, so
    /// the two must not agree via the standard `TemperatureDelta` accessors.
    #[test]
    fn units_ip_matches_upstreams_literal_rescale() {
        let si = cooling_effect(
            inputs(25.0, 25.0, 0.5, 50.0, 1.2, 0.5),
            CoolingEffectOptions {
                units: Units::SI,
                ..Default::default()
            },
        );
        let ip = cooling_effect(
            inputs(25.0, 25.0, 0.5, 50.0, 1.2, 0.5),
            CoolingEffectOptions {
                units: Units::IP,
                ..Default::default()
            },
        );

        // `si` is already rounded to 2 decimals; rescaling that (rather than the
        // unrounded root the IP branch actually rescales) can differ by up to one
        // rounding step at the second decimal, so compare with that slack rather than
        // exactly.
        let expected_ip = si.as_celsius() / 1.8 * 3.28;
        assert!(
            (ip.as_fahrenheit() - expected_ip).abs() < 0.01,
            "ip={}, expected~{expected_ip}",
            ip.as_fahrenheit()
        );
    }
}
