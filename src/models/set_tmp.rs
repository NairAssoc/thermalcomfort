//! Standard Effective Temperature (SET) calculation
//!
//! This module provides a wrapper around the two-node Gagge model
//! to calculate SET values.

use crate::models::two_nodes_gagge::GaggePosture;
use crate::models::two_nodes_gagge::{GaggeTwoNodesInputs, GaggeTwoNodesOptions, two_nodes_gagge};
use crate::{ClothingInsulation, MetabolicRate};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// The comfort inputs to [`set_tmp`]: pythermalcomfort requires all six (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Air speed
    pub v: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
}

/// Options for SET calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetOptions {
    /// External work
    pub wme: MetabolicRate,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body position
    pub position: GaggePosture,
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round output value
    pub round_output: bool,
    /// Use the reduced solver path that only produces SET.
    ///
    /// Defaults to `false`, matching pythermalcomfort. This is **not** merely a speed
    /// switch: the `calculate_ce` path is a different calculation (it forces a standing
    /// position internally), so leaving it on changes the returned SET by up to ~3.5 °C
    /// at higher air speeds and metabolic rates. Only [`cooling_effect`] should enable
    /// it, which is exactly what Python does.
    ///
    /// [`cooling_effect`]: fn@crate::models::cooling_effect
    pub calculate_ce: bool,
}

impl Default for SetOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            position: GaggePosture::Standing,
            limit_inputs: true,
            round_output: true,
            calculate_ce: false,
        }
    }
}

/// Calculate Standard Effective Temperature (SET)
///
/// The SET is the temperature of a hypothetical isothermal environment at 50% RH,
/// <0.1 m/s air speed, and tr = tdb, in which the total heat loss from the skin
/// of an imaginary occupant wearing clothing standardized for the activity concerned
/// is the same as that from a person in the actual environment with actual clothing
/// and activity level (Gagge1986).
///
/// # Returns
///
/// Standard Effective Temperature, or NaN if inputs are outside valid ranges
/// and limit_inputs is true
///
/// # Standard Applicability Limits (when limit_inputs = true)
///
/// * 10 < tdb [°C] < 40
/// * 10 < tr [°C] < 40
/// * 0 < v [m/s] < 2
/// * 1 < met < 4
/// * 0 < clo < 1.5
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::set_tmp::{set_tmp, SetInputs, SetOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let set = set_tmp(
///     SetInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         v: Speed::from_meters_per_second(0.1),
///         rh: Humidity::from_percent(50.0),
///         met: MetabolicRate::from_met(1.2),
///         clo: ClothingInsulation::from_clo(0.5),
///     },
///     Default::default(),
/// );
/// println!("SET: {:.1}°C", set.as_celsius());
/// ```
pub fn set_tmp(inputs: SetInputs, options: SetOptions) -> Temperature {
    let SetInputs {
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
    } = inputs;

    let dry_bulb_celsius = tdb.as_celsius();
    let radiant_celsius = tr.as_celsius();
    let speed_mps = v.as_meters_per_second();
    let met_val = met.as_met();
    let clo_val = clo.as_clo();

    // Check standard compliance if limit_inputs is true
    if options.limit_inputs {
        if !(10.0..=40.0).contains(&dry_bulb_celsius) {
            return Temperature::from_celsius(f64::NAN);
        }
        if !(10.0..=40.0).contains(&radiant_celsius) {
            return Temperature::from_celsius(f64::NAN);
        }
        if !(0.0..=2.0).contains(&speed_mps) {
            return Temperature::from_celsius(f64::NAN);
        }
        if !(1.0..=4.0).contains(&met_val) {
            return Temperature::from_celsius(f64::NAN);
        }
        if !(0.0..=1.5).contains(&clo_val) {
            return Temperature::from_celsius(f64::NAN);
        }
    }

    let gagge_options = GaggeTwoNodesOptions {
        wme: options.wme,
        body_surface_area: options.body_surface_area,
        p_atm: options.p_atm,
        position: options.position,
        max_skin_blood_flow: 90.0,
        round_output: false, // Don't round in Gagge, we'll round here if needed
        max_sweating: 500.0,
        w_max: None,
        calculate_ce: options.calculate_ce,
    };

    let result = two_nodes_gagge(
        GaggeTwoNodesInputs {
            tdb,
            tr,
            v,
            rh,
            met,
            clo,
        },
        gagge_options,
    );
    let set = result.set.as_celsius();

    if options.round_output {
        Temperature::from_celsius(crate::utilities::round_half_even(set * 10.0) / 10.0)
    } else {
        Temperature::from_celsius(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(tdb: f64, tr: f64, v: f64, rh: f64, met: f64, clo: f64) -> SetInputs {
        SetInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(v),
            rh: Humidity::from_percent(rh),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
        }
    }

    #[test]
    fn test_set_tmp_basic() {
        let set = set_tmp(inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5), Default::default());
        let set = set.as_celsius();
        assert!(set > 20.0 && set < 30.0);
        assert!(!set.is_nan());
    }

    #[test]
    fn test_set_tmp_limits() {
        // Test invalid tdb (too low)
        let set = set_tmp(inputs(5.0, 25.0, 0.1, 50.0, 1.2, 0.5), Default::default());
        assert!(set.as_celsius().is_nan());

        // Test invalid tdb (too high)
        let set = set_tmp(inputs(45.0, 25.0, 0.1, 50.0, 1.2, 0.5), Default::default());
        assert!(set.as_celsius().is_nan());

        // Test invalid met (too low)
        let set = set_tmp(inputs(25.0, 25.0, 0.1, 50.0, 0.5, 0.5), Default::default());
        assert!(set.as_celsius().is_nan());

        // Test with limit_inputs = false
        let options = SetOptions {
            limit_inputs: false,
            ..Default::default()
        };
        let set = set_tmp(inputs(5.0, 25.0, 0.1, 50.0, 1.2, 0.5), options);
        assert!(!set.as_celsius().is_nan());
    }

    #[test]
    fn test_set_tmp_reference_values() {
        // pythermalcomfort with round_output=False:
        // set_tmp(25, 25, 0.1, 50, 1.2, 0.5) = 24.3149 and set_tmp(28, 28, 0.3, 60, 1.5, 0.3) = 26.3632
        let unrounded = SetOptions {
            round_output: false,
            ..Default::default()
        };
        let set = set_tmp(inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5), unrounded).as_celsius();
        assert!((set - 24.3149).abs() < 0.05, "set = {set}");

        let set = set_tmp(inputs(28.0, 28.0, 0.3, 60.0, 1.5, 0.3), unrounded).as_celsius();
        assert!((set - 26.3632).abs() < 0.05, "set = {set}");
    }

    #[test]
    fn test_set_tmp_calculate_ce() {
        // Above 0.85 met, calculate_ce skips the metabolic increase of the convective
        // coefficient (used by the cooling effect), so the two results differ
        let default = set_tmp(inputs(28.0, 28.0, 0.3, 60.0, 1.5, 0.3), Default::default());
        let ce = set_tmp(
            inputs(28.0, 28.0, 0.3, 60.0, 1.5, 0.3),
            SetOptions {
                calculate_ce: true,
                ..Default::default()
            },
        );
        let (default, ce) = (default.as_celsius(), ce.as_celsius());
        assert!(
            (default - ce).abs() > 0.5,
            "default {default}, calculate_ce {ce}"
        );
    }

    #[test]
    fn test_set_tmp_rounding() {
        let options_round = SetOptions {
            round_output: true,
            ..Default::default()
        };
        let set_rounded = set_tmp(inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5), options_round);

        let options_no_round = SetOptions {
            round_output: false,
            ..Default::default()
        };
        let set_exact = set_tmp(inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5), options_no_round);

        // Rounded value should be close to exact value but rounded to 1 decimal
        assert!((set_rounded.as_celsius() - set_exact.as_celsius()).abs() < 0.1);
    }
}
