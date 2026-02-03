//! Psychrometric functions for calculating properties of moist air

use crate::constants::*;
use crate::utilities::p_sat;
use libm::{exp, pow, sqrt, atan, log, fabs as abs};

/// Psychrometric values result
#[derive(Debug, Clone, Copy)]
pub struct PsychrometricValues {
    /// Saturation vapor pressure [Pa]
    pub p_sat: f64,
    /// Partial pressure of water vapor [Pa]
    pub p_vap: f64,
    /// Humidity ratio [kg water/kg dry air]
    pub hr: f64,
    /// Wet bulb temperature [°C]
    pub t_wb: f64,
    /// Dew point temperature [°C]
    pub t_dp: f64,
    /// Enthalpy [J/kg dry air]
    pub h: f64,
}

/// Calculate psychrometric values from dry bulb temperature and relative humidity
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature [°C]
/// * `rh` - Relative humidity [%]
/// * `p_atm` - Atmospheric pressure [Pa], default 101325 Pa
///
/// # Returns
///
/// `PsychrometricValues` containing all psychrometric properties
///
/// # Example
///
/// ```
/// use thermalcomfort::psychrometrics::psy_ta_rh;
///
/// let result = psy_ta_rh(25.0, 50.0, 101325.0);
/// // result.t_wb ≈ 17.7°C
/// // result.t_dp ≈ 13.9°C
/// ```
pub fn psy_ta_rh(tdb: f64, rh: f64, p_atm: f64) -> PsychrometricValues {
    let p_saturation = p_sat(tdb);
    let p_vap = rh / 100.0 * p_saturation;
    let hr = 0.62198 * p_vap / (p_atm - p_vap);
    let t_dp = dew_point_temperature(tdb, rh);
    let t_wb = wet_bulb_temperature(tdb, rh);
    let h = enthalpy_air(tdb, hr);

    PsychrometricValues {
        p_sat: p_saturation,
        p_vap,
        hr,
        t_wb,
        t_dp,
        h,
    }
}

/// Calculate air enthalpy
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature [°C]
/// * `hr` - Humidity ratio [kg water/kg dry air]
///
/// # Returns
///
/// Enthalpy [J/kg dry air]
#[inline]
pub fn enthalpy_air(tdb: f64, hr: f64) -> f64 {
    let h_dry_air = CP_AIR * tdb;
    let h_sat_vap = H_FG + CP_VAPOUR * tdb;
    h_dry_air + hr * h_sat_vap
}

/// Calculate wet bulb temperature using Stull equation
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature [°C]
/// * `rh` - Relative humidity [%]
///
/// # Returns
///
/// Wet bulb temperature [°C]
///
/// # Reference
///
/// Stull, R., 2011: Wet-Bulb Temperature from Relative Humidity and Air Temperature.
/// J. Appl. Meteor. Climatol., 50, 2267–2269
#[inline]
pub fn wet_bulb_temperature(tdb: f64, rh: f64) -> f64 {
    tdb * atan(0.151977 * pow(rh + 8.313659, 0.5))
        + atan(tdb + rh)
        - atan(rh - 1.676331)
        + 0.00391838 * pow(rh, 1.5) * atan(0.023101 * rh)
        - 4.686035
}

/// Calculate dew point temperature using Magnus formula
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature [°C]
/// * `rh` - Relative humidity [%]
///
/// # Returns
///
/// Dew point temperature [°C]
///
/// # Reference
///
/// World Meteorological Organization, 2024: Guide to Instruments and
/// Methods of Observation (WMO-No. 8)
pub fn dew_point_temperature(tdb: f64, rh: f64) -> f64 {
    if rh < 0.0 || rh > 100.0 {
        return f64::NAN;
    }

    // Saturation vapor pressure in hPa
    let e_w = 6.112 * exp((17.62 * tdb) / (243.12 + tdb));

    // Actual vapor pressure in hPa
    let e_s = (rh / 100.0) * e_w;

    // Dew point temperature
    243.12 * log(e_s / 6.112) / (17.62 - log(e_s / 6.112))
}

/// Calculate mean radiant temperature from globe temperature
///
/// Converts globe temperature reading to mean radiant temperature using either
/// Mixed Convection or ISO 7726:1998 standard
///
/// # Arguments
///
/// * `tg` - Globe temperature [°C]
/// * `tdb` - Air temperature [°C]
/// * `v` - Air speed [m/s]
/// * `d` - Globe diameter [m], default 0.15 m
/// * `emissivity` - Globe emissivity, default 0.95
/// * `use_iso` - If true, use ISO formula; if false, use Mixed Convection
///
/// # Returns
///
/// Mean radiant temperature [°C]
///
/// # Note
///
/// The Mixed Convection formulation by Teitelbaum et al. (2022) is only
/// validated for globe diameters between 0.04 and 0.15 m
pub fn mean_radiant_temperature(
    tg: f64,
    tdb: f64,
    v: f64,
    d: f64,
    emissivity: f64,
    use_iso: bool,
) -> f64 {
    if use_iso {
        mean_radiant_temperature_iso(tg, tdb, v, d, emissivity)
    } else {
        mean_radiant_temperature_mixed(tg, tdb, v, d, emissivity)
    }
}

#[inline]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Calculate mean radiant temperature using ISO 7726:1998 method
fn mean_radiant_temperature_iso(tg: f64, tdb: f64, v: f64, d: f64, emissivity: f64) -> f64 {
    let tg_k = tg + C_TO_K;
    let tdb_k = tdb + C_TO_K;

    // Heat transfer coefficients
    let h_n = 1.4 * pow(abs(tg_k - tdb_k) / d, 0.25); // natural convection
    let h_f = 6.3 * pow(v, 0.6) / pow(d, 0.4); // forced convection

    // Use maximum of the two
    let h = fmax(h_n, h_f);

    pow(
        pow(tg_k, 4.0) + h * (tg_k - tdb_k) / (emissivity * 5.67e-8),
        0.25,
    ) - C_TO_K
}

/// Calculate mean radiant temperature using Mixed Convection method
fn mean_radiant_temperature_mixed(
    tg: f64,
    tdb: f64,
    v: f64,
    d: f64,
    emissivity: f64,
) -> f64 {
    // Check diameter validity
    if d < 0.04 || d > 0.15 {
        return f64::NAN;
    }

    const MU: f64 = 0.0000181; // Pa·s
    const K_AIR: f64 = 0.02662; // W/(m·K)
    const BETA: f64 = 0.0034; // 1/K
    const NU: f64 = 0.0000148; // m²/s
    const ALPHA: f64 = 0.00002591; // m²/s

    let pr = CP_AIR * MU / K_AIR; // Prandtl number

    let o = 5.67e-8; // Stefan-Boltzmann constant
    let n = 1.27 * d + 0.57;

    let ra = G * BETA * abs(tg - tdb) * pow(d, 3.0) / NU / ALPHA;
    let re = v * d / NU;

    let nu_natural = 2.0 + (0.589 * pow(ra, 0.25))
        / pow(1.0 + pow(0.469 / pr, 9.0 / 16.0), 4.0 / 9.0);

    let nu_forced = 2.0
        + (0.4 * pow(re, 0.5) + 0.06 * pow(re, 2.0 / 3.0)) * pow(pr, 0.4);

    let nu_combined = pow(pow(nu_forced, n) + pow(nu_natural, n), 1.0 / n);

    pow(
        pow(tg + C_TO_K, 4.0)
            - ((nu_combined * K_AIR / d) * (-tg + tdb)) / emissivity / o,
        0.25,
    ) - C_TO_K
}

/// Calculate operative temperature
///
/// # Arguments
///
/// * `tdb` - Air temperature [°C]
/// * `tr` - Mean radiant temperature [°C]
/// * `v` - Air speed [m/s]
/// * `use_ashrae` - If true, use ASHRAE method; if false, use ISO method
///
/// # Returns
///
/// Operative temperature [°C]
pub fn operative_temperature(tdb: f64, tr: f64, v: f64, use_ashrae: bool) -> f64 {
    if use_ashrae {
        let a = if v < 0.2 {
            0.5
        } else if v < 0.6 {
            0.6
        } else {
            0.7
        };
        a * tdb + (1.0 - a) * tr
    } else {
        // ISO method
        (tdb * sqrt(10.0 * v) + tr) / (1.0 + sqrt(10.0 * v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wet_bulb_temperature() {
        let t_wb = wet_bulb_temperature(25.0, 50.0);
        // Expected value approximately 17.7°C
        assert!((t_wb - 17.7).abs() < 0.5);
    }

    #[test]
    fn test_dew_point_temperature() {
        let t_dp = dew_point_temperature(25.0, 50.0);
        // Expected value approximately 13.9°C
        assert!((t_dp - 13.9).abs() < 0.5);

        // Test invalid RH
        assert!(dew_point_temperature(25.0, -10.0).is_nan());
        assert!(dew_point_temperature(25.0, 110.0).is_nan());
    }

    #[test]
    fn test_psy_ta_rh() {
        let result = psy_ta_rh(25.0, 50.0, 101325.0);

        assert!(result.p_sat > 0.0);
        assert!(result.p_vap > 0.0);
        assert!(result.hr > 0.0);
        assert!((result.t_wb - 17.7).abs() < 0.5);
        assert!((result.t_dp - 13.9).abs() < 0.5);
        assert!(result.h > 0.0);
    }

    #[test]
    fn test_operative_temperature() {
        // ISO method
        let t_op = operative_temperature(25.0, 25.0, 0.1, false);
        assert!((t_op - 25.0).abs() < 0.01);

        // ASHRAE method
        let t_op = operative_temperature(22.0, 26.0, 0.1, true);
        assert!(t_op > 22.0 && t_op < 26.0);
    }
}
