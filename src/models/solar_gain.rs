//! Solar gain calculations for thermal comfort assessment
//!
//! Calculate the solar gain to the human body using the Effective Radiant Field (ERF).

use crate::{HeatFluxDensity, TemperatureDelta};
use measurements::Angle;

/// The postures `solar_gain` has a projected-area-factor table for.
///
/// pythermalcomfort accepts exactly `standing`, `sitting` and `supine` and raises
/// `ValueError` for the rest of its `Postures` enum, so the other members are not
/// representable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolarGainPosture {
    /// Standing, 0.725 of the body surface exposed to radiation
    Standing,
    /// Sitting, 0.696 exposed; pythermalcomfort's default
    #[default]
    Sitting,
    /// Lying face up; the solar angles are transposed onto the standing table
    Supine,
}

/// The sun's position and the radiation reaching the occupant.
///
/// These are the six values `solar_gain` cannot supply a default for. Named rather than
/// positional because the underlying call is seven consecutive `f64`s, five of them
/// fractions on `[0, 1]`, and no type system distinguishes those from one another — a
/// transposed `f_svv` and `f_bes` is a silently wrong answer, not a compile error.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarGainInputs {
    /// Solar altitude above the horizontal, 0-90°
    pub sol_altitude: Angle,
    /// Solar horizontal angle relative to the front of the person, 0-180°
    pub sharp: Angle,
    /// Direct-beam solar radiation
    pub sol_radiation_dir: HeatFluxDensity,
    /// Total solar transmittance of the window `[0, 1]`
    pub sol_transmittance: f64,
    /// Sky-vault view fraction `[0, 1]`
    pub f_svv: f64,
    /// Fraction of body surface exposed to sun `[0, 1]`
    pub f_bes: f64,
}

/// Optional parameters for [`solar_gain`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarGainOptions {
    /// Average short-wave absorptivity of the occupant, 0.57-0.84
    pub asw: f64,
    /// Body posture
    pub posture: SolarGainPosture,
    /// Floor reflectance `[0, 1]`
    pub floor_reflectance: f64,
    /// Round both outputs to one decimal place
    pub round_output: bool,
}

impl Default for SolarGainOptions {
    fn default() -> Self {
        Self {
            asw: 0.7,
            posture: SolarGainPosture::Sitting,
            floor_reflectance: 0.6,
            round_output: true,
        }
    }
}

/// Result of solar gain calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarGainResult {
    /// Effective Radiant Field
    pub erf: HeatFluxDensity,
    /// Amount by which the mean radiant temperature is raised by solar radiation.
    ///
    /// A [`TemperatureDelta`], not a [`Temperature`](measurements::Temperature): it is a
    /// difference to be added to an absolute reading, and typing it as an absolute value
    /// invited exactly that confusion.
    pub delta_mrt: TemperatureDelta,
}

/// Calculate solar gain using the Effective Radiant Field
///
/// Calculates the solar gain to the human body using the Effective Radiant Field (ERF).
/// The ERF is a measure of the net energy flux to or from the human body, expressed in W/m².
/// Also calculates the delta mean radiant temperature, which is the amount by which the
/// mean radiant temperature should be increased if no solar radiation is present.
///
/// # Returns
///
/// [`SolarGainResult`] containing the ERF and the delta MRT.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::solar_gain::{solar_gain, SolarGainInputs};
/// use thermalcomfort::HeatFluxDensity;
/// use measurements::Angle;
///
/// let result = solar_gain(
///     SolarGainInputs {
///         sol_altitude: Angle::from_degrees(0.0),
///         sharp: Angle::from_degrees(120.0),
///         sol_radiation_dir: HeatFluxDensity::from_watts_per_square_meter(800.0),
///         sol_transmittance: 0.5,
///         f_svv: 0.5,
///         f_bes: 0.5,
///     },
///     Default::default(),
/// );
/// assert!(result.erf.as_watts_per_square_meter() > 0.0);
/// ```
///
/// # References
///
/// - ASHRAE 55-2023 Appendix C
pub fn solar_gain(inputs: SolarGainInputs, options: SolarGainOptions) -> SolarGainResult {
    // Convert once here, then calculate in plain f64 throughout.
    let sol_altitude = inputs.sol_altitude.as_degrees();
    let sharp = inputs.sharp.as_degrees();
    let sol_radiation_dir = inputs.sol_radiation_dir.as_watts_per_square_meter();
    let SolarGainInputs {
        sol_transmittance,
        f_svv,
        f_bes,
        ..
    } = inputs;
    let SolarGainOptions {
        asw,
        posture,
        floor_reflectance,
        round_output,
    } = options;

    // The fp lookup table only covers altitudes 0-90° and azimuths 0-180°. Outside that
    // there is no valid span to interpolate within, so return NaN rather than
    // extrapolating from the first span.
    if !(0.0..=90.0).contains(&sol_altitude) || !(0.0..=180.0).contains(&sharp) {
        return SolarGainResult {
            erf: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            delta_mrt: TemperatureDelta::from_celsius(f64::NAN),
        };
    }

    let deg_to_rad = core::f64::consts::PI / 180.0;
    // Radiative heat transfer coefficient (W/(m²·K))
    // Typical value for human body in indoor environment
    let hr = 6.0;
    // Diffuse solar radiation fraction
    // Assumes diffuse radiation is 20% of direct beam radiation (typical for clear sky)
    let i_diff = 0.2 * sol_radiation_dir;

    // Get projected area factor table based on posture
    // Tables contain empirical f_p values from ASHRAE 55 for different
    // solar altitudes (rows) and azimuths (columns)
    let fp_table: [[f64; 7]; 13] = match posture {
        SolarGainPosture::Sitting => [
            [0.29, 0.324, 0.305, 0.303, 0.262, 0.224, 0.177],
            [0.292, 0.328, 0.294, 0.288, 0.268, 0.227, 0.177],
            [0.288, 0.332, 0.298, 0.29, 0.264, 0.222, 0.177],
            [0.274, 0.326, 0.294, 0.289, 0.252, 0.214, 0.177],
            [0.254, 0.308, 0.28, 0.276, 0.241, 0.202, 0.177],
            [0.23, 0.282, 0.262, 0.26, 0.233, 0.193, 0.177],
            [0.216, 0.26, 0.248, 0.244, 0.22, 0.186, 0.177],
            [0.234, 0.258, 0.236, 0.227, 0.208, 0.18, 0.177],
            [0.262, 0.26, 0.224, 0.208, 0.196, 0.176, 0.177],
            [0.28, 0.26, 0.21, 0.192, 0.184, 0.17, 0.177],
            [0.298, 0.256, 0.194, 0.174, 0.168, 0.168, 0.177],
            [0.306, 0.25, 0.18, 0.156, 0.156, 0.166, 0.177],
            [0.3, 0.24, 0.168, 0.152, 0.152, 0.164, 0.177],
        ],
        SolarGainPosture::Supine => [
            // For supine, we use standing table but will transpose angles
            [0.35, 0.35, 0.314, 0.258, 0.206, 0.144, 0.082],
            [0.342, 0.342, 0.31, 0.252, 0.2, 0.14, 0.082],
            [0.33, 0.33, 0.3, 0.244, 0.19, 0.132, 0.082],
            [0.31, 0.31, 0.275, 0.228, 0.175, 0.124, 0.082],
            [0.283, 0.283, 0.251, 0.208, 0.16, 0.114, 0.082],
            [0.252, 0.252, 0.228, 0.188, 0.15, 0.108, 0.082],
            [0.23, 0.23, 0.214, 0.18, 0.148, 0.108, 0.082],
            [0.242, 0.242, 0.222, 0.18, 0.153, 0.112, 0.082],
            [0.274, 0.274, 0.245, 0.203, 0.165, 0.116, 0.082],
            [0.304, 0.304, 0.27, 0.22, 0.174, 0.121, 0.082],
            [0.328, 0.328, 0.29, 0.234, 0.183, 0.125, 0.082],
            [0.344, 0.344, 0.304, 0.244, 0.19, 0.128, 0.082],
            [0.347, 0.347, 0.308, 0.246, 0.191, 0.128, 0.082],
        ],
        _ => [
            // Standing and other postures
            [0.35, 0.35, 0.314, 0.258, 0.206, 0.144, 0.082],
            [0.342, 0.342, 0.31, 0.252, 0.2, 0.14, 0.082],
            [0.33, 0.33, 0.3, 0.244, 0.19, 0.132, 0.082],
            [0.31, 0.31, 0.275, 0.228, 0.175, 0.124, 0.082],
            [0.283, 0.283, 0.251, 0.208, 0.16, 0.114, 0.082],
            [0.252, 0.252, 0.228, 0.188, 0.15, 0.108, 0.082],
            [0.23, 0.23, 0.214, 0.18, 0.148, 0.108, 0.082],
            [0.242, 0.242, 0.222, 0.18, 0.153, 0.112, 0.082],
            [0.274, 0.274, 0.245, 0.203, 0.165, 0.116, 0.082],
            [0.304, 0.304, 0.27, 0.22, 0.174, 0.121, 0.082],
            [0.328, 0.328, 0.29, 0.234, 0.183, 0.125, 0.082],
            [0.344, 0.344, 0.304, 0.244, 0.19, 0.128, 0.082],
            [0.347, 0.347, 0.308, 0.246, 0.191, 0.128, 0.082],
        ],
    };

    // Transpose angles for supine posture
    let (sharp_adj, alt_adj) = if posture == SolarGainPosture::Supine {
        crate::models::specialty::transpose_sharp_altitude_degrees(sharp, sol_altitude)
    } else {
        (sharp, sol_altitude)
    };

    // Find span in lookup tables
    let alt_range = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0];
    let az_range = [
        0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0,
    ];

    let alt_i = find_span(&alt_range, alt_adj);
    let az_i = find_span(&az_range, sharp_adj);

    // Bilinear interpolation
    let fp11 = fp_table[az_i][alt_i];
    let fp12 = fp_table[az_i][alt_i + 1];
    let fp21 = fp_table[az_i + 1][alt_i];
    let fp22 = fp_table[az_i + 1][alt_i + 1];

    let az1 = az_range[az_i];
    let az2 = az_range[az_i + 1];
    let alt1 = alt_range[alt_i];
    let alt2 = alt_range[alt_i + 1];

    let mut fp = fp11 * (az2 - sharp_adj) * (alt2 - alt_adj);
    fp += fp21 * (sharp_adj - az1) * (alt2 - alt_adj);
    fp += fp12 * (az2 - sharp_adj) * (alt_adj - alt1);
    fp += fp22 * (sharp_adj - az1) * (alt_adj - alt1);
    fp /= (az2 - az1) * (alt2 - alt1);

    // Effective fraction of body surface for radiation exchange
    // From ASHRAE 55 (fraction of body surface area exposed to radiation):
    // Sitting: 0.696 (larger surface area exposed while seated)
    // Standing: 0.725 (slightly more surface exposed when standing)
    let f_eff = if posture == SolarGainPosture::Sitting {
        0.696
    } else {
        0.725
    };

    let sw_abs = asw;
    // Longwave (thermal) absorptivity of clothed human body
    // 0.95 is typical for most clothing and skin (ASHRAE 55)
    let lw_abs = 0.95;

    // Calculate ERF components
    // 0.5 factor accounts for hemispherical distribution of diffuse radiation
    let e_diff = f_eff * f_svv * 0.5 * sol_transmittance * i_diff;
    let e_direct = f_eff * fp * sol_transmittance * f_bes * sol_radiation_dir;
    let e_reflected = f_eff
        * f_svv
        * 0.5
        * sol_transmittance
        // alt_adj, not sol_altitude: for a supine occupant the sun's position is
        // rotated into the body's frame, and Python reassigns sol_altitude in place so
        // the transposed value feeds this term too. Reading the original here made 94%
        // of supine cases wrong.
        * (sol_radiation_dir * libm::sin(alt_adj * deg_to_rad) + i_diff)
        * floor_reflectance;

    let e_solar = e_diff + e_direct + e_reflected;
    let erf = e_solar * (sw_abs / lw_abs);
    let delta_mrt = erf / (hr * f_eff);

    // Upstream exposes this as `round_output`; rounding unconditionally, as this did
    // before, left the full-precision values unobtainable.
    let (erf, delta_mrt) = if round_output {
        (
            crate::utilities::round_half_even(erf * 10.0) / 10.0,
            crate::utilities::round_half_even(delta_mrt * 10.0) / 10.0,
        )
    } else {
        (erf, delta_mrt)
    };

    SolarGainResult {
        erf: HeatFluxDensity::from_watts_per_square_meter(erf),
        delta_mrt: TemperatureDelta::from_celsius(delta_mrt),
    }
}

/// Find the span index in a sorted array
fn find_span(arr: &[f64], x: f64) -> usize {
    for i in 0..arr.len() - 1 {
        if x >= arr[i] && x <= arr[i + 1] {
            return i;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs for a case, with the options left at their defaults.
    fn inputs(
        alt: f64,
        sharp: f64,
        dir: f64,
        trans: f64,
        f_svv: f64,
        f_bes: f64,
    ) -> SolarGainInputs {
        SolarGainInputs {
            sol_altitude: Angle::from_degrees(alt),
            sharp: Angle::from_degrees(sharp),
            sol_radiation_dir: HeatFluxDensity::from_watts_per_square_meter(dir),
            sol_transmittance: trans,
            f_svv,
            f_bes,
        }
    }

    #[test]
    fn test_solar_gain_out_of_range_is_nan() {
        // The fp table covers altitude 0-90 degrees and azimuth 0-180 degrees; outside
        // that both outputs are NaN, matching pythermalcomfort 4.4.0.
        for (alt, sharp) in [(-10.0, 120.0), (100.0, 120.0), (45.0, 200.0), (45.0, -5.0)] {
            let result = solar_gain(inputs(alt, sharp, 800.0, 0.5, 0.5, 0.5), Default::default());
            assert!(
                result.erf.as_watts_per_square_meter().is_nan()
                    && result.delta_mrt.as_celsius().is_nan(),
                "expected NaN at sol_altitude={alt}, sharp={sharp}"
            );
        }

        // In-range case still computes (reference: erf=59.5, delta_mrt=14.2)
        let result = solar_gain(
            inputs(45.0, 120.0, 800.0, 0.5, 0.5, 0.5),
            Default::default(),
        );
        let erf = result.erf.as_watts_per_square_meter();
        let delta_mrt = result.delta_mrt.as_celsius();
        assert!((erf - 59.5).abs() < 0.5, "erf = {erf}");
        assert!((delta_mrt - 14.2).abs() < 0.5, "delta_mrt = {delta_mrt}");
    }

    #[test]
    fn test_solar_gain_sitting() {
        let result = solar_gain(inputs(0.0, 120.0, 800.0, 0.5, 0.5, 0.5), Default::default());
        assert!(result.erf.as_watts_per_square_meter() > 0.0);
        assert!(result.delta_mrt.as_celsius() > 0.0);
    }

    #[test]
    fn test_solar_gain_standing() {
        let result = solar_gain(
            inputs(45.0, 90.0, 600.0, 0.7, 0.6, 0.7),
            SolarGainOptions {
                posture: SolarGainPosture::Standing,
                ..Default::default()
            },
        );
        assert!(result.erf.as_watts_per_square_meter() > 0.0);
        assert!(result.delta_mrt.as_celsius() > 0.0);
    }

    /// `round_output` is upstream's, and was hardcoded to `true` here. Turning it off
    /// must expose digits that rounding to one decimal place would have removed.
    #[test]
    fn round_output_can_be_turned_off() {
        let rounded = solar_gain(
            inputs(45.0, 120.0, 800.0, 0.5, 0.5, 0.5),
            Default::default(),
        );
        let exact = solar_gain(
            inputs(45.0, 120.0, 800.0, 0.5, 0.5, 0.5),
            SolarGainOptions {
                round_output: false,
                ..Default::default()
            },
        );

        let r = rounded.erf.as_watts_per_square_meter();
        let e = exact.erf.as_watts_per_square_meter();
        assert!(
            (r * 10.0 - (r * 10.0).round()).abs() < 1e-9,
            "rounded erf {r} is not at one decimal"
        );
        assert!(
            (e - r).abs() > 1e-12,
            "unrounded erf {e} equals the rounded {r}"
        );
        assert!(
            (e - r).abs() < 0.05,
            "unrounded erf {e} is not within rounding of {r}"
        );
    }
}
