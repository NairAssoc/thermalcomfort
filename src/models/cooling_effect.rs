//! Cooling effect of elevated air speed
//!
//! This module calculates the cooling effect when air speed is elevated above
//! the still air threshold (0.1 m/s).

use crate::models::set_tmp::{SetOptions, set_tmp};
use crate::numerical::brentq;
use crate::utilities::Posture;
use crate::{Clo, Met};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// Options for cooling effect calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoolingEffectOptions {
    /// External work
    pub wme: Met,
    /// Still air threshold [m/s]
    pub still_air_threshold: f64,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body posture
    pub posture: Posture,
}

impl Default for CoolingEffectOptions {
    fn default() -> Self {
        Self {
            wme: Met::new(0.0),
            still_air_threshold: 0.1,
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            posture: Posture::Standing,
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
/// threshold (default 0.1 m/s).
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `metabolic_rate` - Metabolic rate (met)
/// * `clothing_insulation` - Clothing insulation (clo)
/// * `options` - Cooling effect options
///
/// # Returns
///
/// Cooling effect [°C] - the temperature reduction that produces equivalent SET
/// at still air conditions. Returns 0.0 if vr <= still_air_threshold.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::cooling_effect::{cooling_effect, CoolingEffectOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity, Met, Clo};
///
/// // Calculate cooling effect with elevated air speed
/// let ce = cooling_effect(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(0.5),
///     Humidity::from_percent(50.0),
///     Met::new(1.2),
///     Clo::new(0.5),
///     Default::default()
/// );
/// println!("Cooling effect: {:.2}°C", ce);
/// ```
pub fn cooling_effect(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: Met,
    clothing_insulation: Clo,
    options: CoolingEffectOptions,
) -> f64 {
    let air_speed = relative_air_speed.as_meters_per_second();

    // No cooling effect if air speed is at or below still air threshold
    if air_speed <= options.still_air_threshold {
        return 0.0;
    }

    // Calculate SET at the actual air speed
    let set_options = SetOptions {
        wme: options.wme,
        body_surface_area: options.body_surface_area,
        p_atm: options.p_atm,
        posture: options.posture,
        limit_inputs: false, // Don't limit inputs for cooling effect calculation
        round_output: false, // Need exact values for root finding
    };

    let initial_set = set_tmp(
        dry_bulb_temp,
        mean_radiant_temp,
        relative_air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        set_options,
    );

    // If SET calculation failed, return 0
    if initial_set.is_nan() {
        return 0.0;
    }

    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();

    // Define the function to find the root of:
    // We want to find ce such that SET(tdb-ce, tr-ce, still_air) = SET(tdb, tr, vr)
    let function = |cooling_effect_delta: f64| -> f64 {
        let set_still = set_tmp(
            Temperature::from_celsius(dry_bulb_celsius - cooling_effect_delta),
            Temperature::from_celsius(radiant_celsius - cooling_effect_delta),
            Speed::from_meters_per_second(options.still_air_threshold),
            relative_humidity,
            metabolic_rate,
            clothing_insulation,
            set_options,
        );
        set_still - initial_set
    };

    // Use Brent's method to find the cooling effect
    // Search in range [0, 40] °C
    brentq(function, 0.0, 40.0, Some(0.001), Some(100)).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooling_effect_no_effect() {
        // At still air threshold, should have no cooling effect
        let ce = cooling_effect(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            Met::new(1.2),
            Clo::new(0.5),
            Default::default(),
        );
        assert_eq!(ce, 0.0);

        // Below still air threshold, should have no cooling effect
        let ce = cooling_effect(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.05),
            Humidity::from_percent(50.0),
            Met::new(1.2),
            Clo::new(0.5),
            Default::default(),
        );
        assert_eq!(ce, 0.0);
    }

    #[test]
    fn test_cooling_effect_elevated_speed() {
        // With elevated air speed, should have positive cooling effect
        let ce = cooling_effect(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.5),
            Humidity::from_percent(50.0),
            Met::new(1.2),
            Clo::new(0.5),
            Default::default(),
        );
        assert!(ce > 0.0);
        assert!(ce < 5.0); // Reasonable range for cooling effect
    }

    #[test]
    fn test_cooling_effect_high_speed() {
        // Higher air speed should produce larger cooling effect
        let ce1 = cooling_effect(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.3),
            Humidity::from_percent(50.0),
            Met::new(1.2),
            Clo::new(0.5),
            Default::default(),
        );
        let ce2 = cooling_effect(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.8),
            Humidity::from_percent(50.0),
            Met::new(1.2),
            Clo::new(0.5),
            Default::default(),
        );

        assert!(ce2 > ce1);
    }

    #[test]
    fn test_cooling_effect_hot_conditions() {
        // Test in hot conditions
        let ce = cooling_effect(
            Temperature::from_celsius(30.0),
            Temperature::from_celsius(30.0),
            Speed::from_meters_per_second(0.5),
            Humidity::from_percent(50.0),
            Met::new(1.2),
            Clo::new(0.5),
            Default::default(),
        );
        assert!(ce > 0.0);
    }
}
