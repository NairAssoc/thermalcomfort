//! Psychrometric functions for calculating properties of moist air

use crate::constants::*;
use crate::utilities::p_sat;
use libm::{atan, exp, fabs as abs, log, pow, sqrt};
use measurements::{Humidity, Pressure, Speed, Temperature};

/// Psychrometric values result
#[derive(Debug, Clone, Copy)]
pub struct PsychrometricValues {
    /// Saturation vapor pressure
    pub p_sat: Pressure,
    /// Partial pressure of water vapor
    pub p_vap: Pressure,
    /// Humidity ratio [kg water/kg dry air]
    pub hr: f64,
    /// Wet bulb temperature
    pub t_wb: Temperature,
    /// Dew point temperature
    pub t_dp: Temperature,
    /// Enthalpy [J/kg dry air]
    pub h: f64,
}

/// Calculate psychrometric values from dry bulb temperature and relative humidity
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `rh` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `p_atm` - Atmospheric pressure (use `Pressure::from_pascals()` or similar), default 101325 Pa
///
/// # Returns
///
/// `PsychrometricValues` containing all psychrometric properties
///
/// # Example
///
/// ```
/// use thermalcomfort::psychrometrics::psy_ta_rh;
/// use thermalcomfort::{Temperature, Humidity, Pressure};
///
/// let result = psy_ta_rh(
///     Temperature::from_celsius(25.0),
///     Humidity::from_percent(50.0),
///     Pressure::from_pascals(101325.0)
/// );
/// // result.t_wb ≈ 17.7°C
/// // result.t_dp ≈ 13.9°C
/// ```
pub fn psy_ta_rh(tdb: Temperature, rh: Humidity, p_atm: Pressure) -> PsychrometricValues {
    let p_saturation = p_sat(tdb);
    let p_sat_pa = p_saturation.as_pascals();
    let p_atm_pa = p_atm.as_pascals();
    let rh_percent = rh.as_percent();

    let p_vap_pa = rh_percent / 100.0 * p_sat_pa;
    let p_vap = Pressure::from_pascals(p_vap_pa);
    let hr = 0.62198 * p_vap_pa / (p_atm_pa - p_vap_pa);
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
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `hr` - Humidity ratio [kg water/kg dry air]
///
/// # Returns
///
/// Enthalpy [J/kg dry air]
#[inline]
pub fn enthalpy_air(tdb: Temperature, hr: f64) -> f64 {
    let tdb_celsius = tdb.as_celsius();
    let h_dry_air = CP_AIR * tdb_celsius;
    let h_sat_vap = H_FG + CP_VAPOUR * tdb_celsius;
    h_dry_air + hr * h_sat_vap
}

/// Calculate wet bulb temperature using Stull equation
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `rh` - Relative humidity (use `Humidity::from_percent()` for RH%)
///
/// # Returns
///
/// Wet bulb temperature
///
/// # Reference
///
/// Stull, R., 2011: Wet-Bulb Temperature from Relative Humidity and Air Temperature.
/// J. Appl. Meteor. Climatol., 50, 2267–2269
#[inline]
pub fn wet_bulb_temperature(tdb: Temperature, rh: Humidity) -> Temperature {
    let tdb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();
    let t_wb_celsius = tdb_celsius * atan(0.151977 * pow(rh_percent + 8.313659, 0.5))
        + atan(tdb_celsius + rh_percent)
        - atan(rh_percent - 1.676331)
        + 0.00391838 * pow(rh_percent, 1.5) * atan(0.023101 * rh_percent)
        - 4.686035;
    Temperature::from_celsius(t_wb_celsius)
}

/// Calculate dew point temperature using Magnus formula
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `rh` - Relative humidity (use `Humidity::from_percent()` for RH%)
///
/// # Returns
///
/// Dew point temperature
///
/// # Reference
///
/// World Meteorological Organization, 2024: Guide to Instruments and
/// Methods of Observation (WMO-No. 8)
pub fn dew_point_temperature(tdb: Temperature, rh: Humidity) -> Temperature {
    let tdb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

    if !(0.0..=100.0).contains(&rh_percent) {
        return Temperature::from_celsius(f64::NAN);
    }

    // Saturation vapor pressure in hPa
    let e_w = 6.112 * exp((17.62 * tdb_celsius) / (243.12 + tdb_celsius));

    // Actual vapor pressure in hPa
    let e_s = (rh_percent / 100.0) * e_w;

    // Dew point temperature
    let t_dp_celsius = 243.12 * log(e_s / 6.112) / (17.62 - log(e_s / 6.112));
    Temperature::from_celsius(t_dp_celsius)
}

/// Calculate mean radiant temperature from globe temperature
///
/// Converts globe temperature reading to mean radiant temperature using either
/// Mixed Convection or ISO 7726:1998 standard
///
/// # Arguments
///
/// * `tg` - Globe temperature (use `Temperature::from_celsius()` or similar)
/// * `tdb` - Air temperature (use `Temperature::from_celsius()` or similar)
/// * `v` - Air speed (use `Speed::from_meters_per_second()` or similar)
/// * `d` - Globe diameter [m], default 0.15 m
/// * `emissivity` - Globe emissivity, default 0.95
/// * `use_iso` - If true, use ISO formula; if false, use Mixed Convection
///
/// # Returns
///
/// Mean radiant temperature
///
/// # Note
///
/// The Mixed Convection formulation by Teitelbaum et al. (2022) is only
/// validated for globe diameters between 0.04 and 0.15 m
pub fn mean_radiant_temperature(
    tg: Temperature,
    tdb: Temperature,
    v: Speed,
    d: f64,
    emissivity: f64,
    use_iso: bool,
) -> Temperature {
    let tg_celsius = tg.as_celsius();
    let tdb_celsius = tdb.as_celsius();
    let v_ms = v.as_meters_per_second();

    let tr_celsius = if use_iso {
        mean_radiant_temperature_iso(tg_celsius, tdb_celsius, v_ms, d, emissivity)
    } else {
        mean_radiant_temperature_mixed(tg_celsius, tdb_celsius, v_ms, d, emissivity)
    };
    Temperature::from_celsius(tr_celsius)
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
fn mean_radiant_temperature_mixed(tg: f64, tdb: f64, v: f64, d: f64, emissivity: f64) -> f64 {
    // Check diameter validity
    if !(0.04..=0.15).contains(&d) {
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

    let nu_natural =
        2.0 + (0.589 * pow(ra, 0.25)) / pow(1.0 + pow(0.469 / pr, 9.0 / 16.0), 4.0 / 9.0);

    let nu_forced = 2.0 + (0.4 * pow(re, 0.5) + 0.06 * pow(re, 2.0 / 3.0)) * pow(pr, 0.4);

    let nu_combined = pow(pow(nu_forced, n) + pow(nu_natural, n), 1.0 / n);

    pow(
        pow(tg + C_TO_K, 4.0) - ((nu_combined * K_AIR / d) * (-tg + tdb)) / emissivity / o,
        0.25,
    ) - C_TO_K
}

/// Calculate operative temperature
///
/// # Arguments
///
/// * `tdb` - Air temperature (use `Temperature::from_celsius()` or similar)
/// * `tr` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `v` - Air speed (use `Speed::from_meters_per_second()` or similar)
/// * `use_ashrae` - If true, use ASHRAE method; if false, use ISO method
///
/// # Returns
///
/// Operative temperature
pub fn operative_temperature(
    tdb: Temperature,
    tr: Temperature,
    v: Speed,
    use_ashrae: bool,
) -> Temperature {
    let tdb_celsius = tdb.as_celsius();
    let tr_celsius = tr.as_celsius();
    let v_ms = v.as_meters_per_second();

    let to_celsius = if use_ashrae {
        let a = if v_ms < 0.2 {
            0.5
        } else if v_ms < 0.6 {
            0.6
        } else {
            0.7
        };
        a * tdb_celsius + (1.0 - a) * tr_celsius
    } else {
        // ISO method
        (tdb_celsius * sqrt(10.0 * v_ms) + tr_celsius) / (1.0 + sqrt(10.0 * v_ms))
    };
    Temperature::from_celsius(to_celsius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wet_bulb_temperature() {
        let t_wb = wet_bulb_temperature(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
        );
        // Expected value approximately 17.7°C
        assert!((t_wb.as_celsius() - 17.7).abs() < 0.5);
    }

    #[test]
    fn test_dew_point_temperature() {
        let t_dp = dew_point_temperature(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
        );
        // Expected value approximately 13.9°C
        assert!((t_dp.as_celsius() - 13.9).abs() < 0.5);

        // Test invalid RH
        let t_dp_invalid1 = dew_point_temperature(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(-10.0),
        );
        assert!(t_dp_invalid1.as_celsius().is_nan());

        let t_dp_invalid2 = dew_point_temperature(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(110.0),
        );
        assert!(t_dp_invalid2.as_celsius().is_nan());
    }

    #[test]
    fn test_psy_ta_rh() {
        let result = psy_ta_rh(
            Temperature::from_celsius(25.0),
            Humidity::from_percent(50.0),
            Pressure::from_pascals(101325.0),
        );

        assert!(result.p_sat.as_pascals() > 0.0);
        assert!(result.p_vap.as_pascals() > 0.0);
        assert!(result.hr > 0.0);
        assert!((result.t_wb.as_celsius() - 17.7).abs() < 0.5);
        assert!((result.t_dp.as_celsius() - 13.9).abs() < 0.5);
        assert!(result.h > 0.0);
    }

    #[test]
    fn test_operative_temperature() {
        // ISO method
        let t_op = operative_temperature(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            false,
        );
        assert!((t_op.as_celsius() - 25.0).abs() < 0.01);

        // ASHRAE method
        let t_op2 = operative_temperature(
            Temperature::from_celsius(22.0),
            Temperature::from_celsius(26.0),
            Speed::from_meters_per_second(0.1),
            true,
        );
        assert!(t_op2.as_celsius() > 22.0 && t_op2.as_celsius() < 26.0);
    }
}
