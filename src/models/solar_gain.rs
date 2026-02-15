//! Solar gain calculations for thermal comfort assessment
//!
//! Calculate the solar gain to the human body using the Effective Radiant Field (ERF).

use crate::utilities::Posture;

/// Result of solar gain calculation
#[derive(Debug, Clone, Copy)]
pub struct SolarGainResult {
    /// Effective Radiant Field [W/m²]
    pub erf: f64,
    /// Delta mean radiant temperature [°C]
    pub delta_mrt: f64,
}

/// Calculate solar gain using the Effective Radiant Field
///
/// Calculates the solar gain to the human body using the Effective Radiant Field (ERF).
/// The ERF is a measure of the net energy flux to or from the human body, expressed in W/m².
/// Also calculates the delta mean radiant temperature, which is the amount by which the
/// mean radiant temperature should be increased if no solar radiation is present.
///
/// # Arguments
///
/// * `sol_altitude` - Solar altitude [degrees from horizontal, 0-90]
/// * `sharp` - Solar horizontal angle relative to front of person [degrees, 0-180]
/// * `sol_radiation_dir` - Direct-beam solar radiation [W/m², typically 200-1000]
/// * `sol_transmittance` - Total solar transmittance [0-1]
/// * `f_svv` - Sky-vault view fraction [0-1]
/// * `f_bes` - Fraction of body surface exposed to sun [0-1]
/// * `asw` - Average short-wave absorptivity [0.57-0.84, default 0.7]
/// * `posture` - Body posture
/// * `floor_reflectance` - Floor reflectance [0-1, default 0.6]
///
/// # Returns
///
/// SolarGainResult containing ERF and delta MRT
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::solar_gain::solar_gain;
/// use thermalcomfort::utilities::Posture;
///
/// let result = solar_gain(0.0, 120.0, 800.0, 0.5, 0.5, 0.5, 0.7, Posture::Sitting, 0.6);
/// assert!(result.erf > 0.0);
/// ```
///
/// # References
///
/// - ASHRAE 55-2023 Appendix C
#[allow(clippy::too_many_arguments)]
pub fn solar_gain(
    sol_altitude: f64,
    sharp: f64,
    sol_radiation_dir: f64,
    sol_transmittance: f64,
    f_svv: f64,
    f_bes: f64,
    asw: f64,
    posture: Posture,
    floor_reflectance: f64,
) -> SolarGainResult {
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
        Posture::Sitting => [
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
        Posture::Supine => [
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
    let (sharp_adj, alt_adj) = if posture == Posture::Supine {
        crate::models::transpose_sharp_altitude(sharp, sol_altitude)
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
    let f_eff = if posture == Posture::Sitting {
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
        * (sol_radiation_dir * libm::sin(sol_altitude * deg_to_rad) + i_diff)
        * floor_reflectance;

    let e_solar = e_diff + e_direct + e_reflected;
    let erf = e_solar * (sw_abs / lw_abs);
    let delta_mrt = erf / (hr * f_eff);

    SolarGainResult {
        erf: libm::round(erf * 10.0) / 10.0,
        delta_mrt: libm::round(delta_mrt * 10.0) / 10.0,
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

    #[test]
    fn test_solar_gain_sitting() {
        let result = solar_gain(0.0, 120.0, 800.0, 0.5, 0.5, 0.5, 0.7, Posture::Sitting, 0.6);
        assert!(result.erf > 0.0);
        assert!(result.delta_mrt > 0.0);
    }

    #[test]
    fn test_solar_gain_standing() {
        let result = solar_gain(
            45.0,
            90.0,
            600.0,
            0.7,
            0.6,
            0.7,
            0.7,
            Posture::Standing,
            0.6,
        );
        assert!(result.erf > 0.0);
        assert!(result.delta_mrt > 0.0);
    }
}
