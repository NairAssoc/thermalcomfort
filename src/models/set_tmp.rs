//! Standard Effective Temperature (SET) calculation
//!
//! This module provides a wrapper around the two-node Gagge model
//! to calculate SET values.

use crate::models::two_nodes_gagge::{GaggeTwoNodesOptions, two_nodes_gagge};
use crate::utilities::Posture;
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// Options for SET calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetOptions {
    /// External work [met]
    pub wme: f64,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body posture
    pub posture: Posture,
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round output value
    pub round_output: bool,
}

impl Default for SetOptions {
    fn default() -> Self {
        Self {
            wme: 0.0,
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            posture: Posture::Standing,
            limit_inputs: true,
            round_output: true,
        }
    }
}

/// Calculate Standard Effective Temperature (SET)
///
/// The SET is the temperature of a hypothetical isothermal environment at 50% RH,
/// <0.1 m/s air speed, and tr = tdb, in which the total heat loss from the skin
/// of an imaginary occupant wearing clothing standardized for the activity concerned
/// is the same as that from a person in the actual environment with actual clothing
/// and activity level [Gagge1986].
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `air_speed` - Air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `metabolic_rate` - Metabolic rate [met]
/// * `clothing_insulation` - Clothing insulation [clo]
/// * `options` - SET calculation options
///
/// # Returns
///
/// Standard Effective Temperature [°C], or NaN if inputs are outside valid ranges
/// and limit_inputs is true
///
/// # Standard Applicability Limits (when limit_inputs = true)
///
/// * 10 < tdb [°C] < 40
/// * 10 < tr [°C] < 40
/// * 0 < v [m/s] < 2
/// * 1 < met [met] < 4
/// * 0 < clo [clo] < 1.5
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::set_tmp::{set_tmp, SetOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let set = set_tmp(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(0.1),
///     Humidity::from_percent(50.0),
///     1.2,
///     0.5,
///     Default::default()
/// );
/// println!("SET: {:.1}°C", set);
/// ```
pub fn set_tmp(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: f64,
    clothing_insulation: f64,
    options: SetOptions,
) -> f64 {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let speed_mps = air_speed.as_meters_per_second();

    // Check standard compliance if limit_inputs is true
    if options.limit_inputs {
        if dry_bulb_celsius < 10.0 || dry_bulb_celsius > 40.0 {
            return f64::NAN;
        }
        if radiant_celsius < 10.0 || radiant_celsius > 40.0 {
            return f64::NAN;
        }
        if speed_mps < 0.0 || speed_mps > 2.0 {
            return f64::NAN;
        }
        if metabolic_rate < 1.0 || metabolic_rate > 4.0 {
            return f64::NAN;
        }
        if clothing_insulation < 0.0 || clothing_insulation > 1.5 {
            return f64::NAN;
        }
    }

    // Call two_nodes_gagge with calculate_ce = true for faster calculation
    let gagge_options = GaggeTwoNodesOptions {
        wme: options.wme,
        body_surface_area: options.body_surface_area,
        p_atm: options.p_atm,
        posture: options.posture,
        max_skin_blood_flow: 90.0,
        round_output: false, // Don't round in Gagge, we'll round here if needed
        max_sweating: 500.0,
        w_max: None,
        calculate_ce: true, // Only calculate SET, not all outputs
    };

    let result = two_nodes_gagge(
        dry_bulb_temp,
        mean_radiant_temp,
        air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        gagge_options,
    );
    let set = result.set;

    if options.round_output {
        libm::round(set * 10.0) / 10.0
    } else {
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_tmp_basic() {
        let set = set_tmp(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            Default::default(),
        );
        assert!(set > 20.0 && set < 30.0);
        assert!(!set.is_nan());
    }

    #[test]
    fn test_set_tmp_limits() {
        // Test invalid tdb (too low)
        let set = set_tmp(
            Temperature::from_celsius(5.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            Default::default(),
        );
        assert!(set.is_nan());

        // Test invalid tdb (too high)
        let set = set_tmp(
            Temperature::from_celsius(45.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            Default::default(),
        );
        assert!(set.is_nan());

        // Test invalid met (too low)
        let set = set_tmp(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            0.5,
            0.5,
            Default::default(),
        );
        assert!(set.is_nan());

        // Test with limit_inputs = false
        let options = SetOptions {
            limit_inputs: false,
            ..Default::default()
        };
        let set = set_tmp(
            Temperature::from_celsius(5.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            options,
        );
        assert!(!set.is_nan());
    }

    #[test]
    fn test_set_tmp_rounding() {
        let options_round = SetOptions {
            round_output: true,
            ..Default::default()
        };
        let set_rounded = set_tmp(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            options_round,
        );

        let options_no_round = SetOptions {
            round_output: false,
            ..Default::default()
        };
        let set_exact = set_tmp(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            options_no_round,
        );

        // Rounded value should be close to exact value but rounded to 1 decimal
        assert!((set_rounded - set_exact).abs() < 0.1);
    }
}
