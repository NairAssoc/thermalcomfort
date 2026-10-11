//! Psychrometric functions for calculating properties of moist air

use crate::constants::*;
use libm::{atan, exp, log, pow};
use measurements::{Humidity, Pressure, Temperature};

/// The comfort inputs to [`psy_ta_rh`]: pythermalcomfort requires both (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsyTaRhInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Relative humidity
    pub rh: Humidity,
}

/// Optional parameters for [`psy_ta_rh`], with pythermalcomfort's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsyTaRhOptions {
    /// Atmospheric pressure
    pub p_atm: Pressure,
}

impl Default for PsyTaRhOptions {
    fn default() -> Self {
        Self {
            p_atm: Pressure::from_pascals(101325.0),
        }
    }
}

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
/// # Returns
///
/// `PsychrometricValues` containing all psychrometric properties
///
/// # Example
///
/// ```
/// use thermalcomfort::psychrometrics::{psy_ta_rh, PsyTaRhInputs, PsyTaRhOptions};
/// use thermalcomfort::{Temperature, Humidity, Pressure};
///
/// let result = psy_ta_rh(
///     PsyTaRhInputs {
///         tdb: Temperature::from_celsius(25.0),
///         rh: Humidity::from_percent(50.0),
///     },
///     PsyTaRhOptions {
///         p_atm: Pressure::from_pascals(101325.0),
///     },
/// );
/// // result.t_wb ≈ 17.7°C
/// // result.t_dp ≈ 13.9°C
/// ```
pub fn psy_ta_rh(inputs: PsyTaRhInputs, options: PsyTaRhOptions) -> PsychrometricValues {
    let PsyTaRhInputs { tdb, rh } = inputs;
    let p_atm = options.p_atm;
    let p_saturation = p_sat(tdb);
    let p_sat_pa = p_saturation.as_pascals();
    let p_atm_pa = p_atm.as_pascals();
    let rh_percent = rh.as_percent();

    let p_vap_pa = rh_percent / 100.0 * p_sat_pa;
    let p_vap = Pressure::from_pascals(p_vap_pa);
    // Humidity ratio calculation using psychrometric constant
    // 0.62198 = ratio of molecular weights (M_water / M_air = 18.015 / 28.965)
    // Source: ASHRAE Fundamentals
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
///
/// All constants (0.151977, 8.313659, 1.676331, 0.00391838, 1.5, 0.023101, 4.686035)
/// are empirical coefficients from Stull's polynomial approximation
#[inline]
pub fn wet_bulb_temperature(tdb: Temperature, rh: Humidity) -> Temperature {
    let tdb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();
    // Stull equation empirical coefficients:
    // 0.151977, 8.313659: RH scaling factors
    // 1.676331: RH offset term
    // 0.00391838: higher-order RH coefficient
    // 1.5: RH power exponent
    // 0.023101: RH interaction coefficient
    // 4.686035: constant offset term
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
///
/// Uses the Magnus-Tetens formula with constants:
/// - 6.112 hPa: reference saturation vapor pressure at 0°C
/// - 17.62: empirical coefficient (unitless)
/// - 243.12°C: empirical temperature coefficient
pub fn dew_point_temperature(tdb: Temperature, rh: Humidity) -> Temperature {
    let tdb_celsius = tdb.as_celsius();
    let rh_percent = rh.as_percent();

    if !(0.0..=100.0).contains(&rh_percent) {
        return Temperature::from_celsius(f64::NAN);
    }

    // Saturation vapor pressure in hPa using Magnus formula
    // 6.112 hPa = reference vapor pressure at 0°C
    // 17.62 and 243.12 = Magnus formula empirical constants
    let e_w = 6.112 * exp((17.62 * tdb_celsius) / (243.12 + tdb_celsius));

    // Actual vapor pressure in hPa
    let e_s = (rh_percent / 100.0) * e_w;

    // Dew point temperature (inverse Magnus formula)
    let t_dp_celsius = 243.12 * log(e_s / 6.112) / (17.62 - log(e_s / 6.112));
    Temperature::from_celsius(t_dp_celsius)
}

/// Saturation vapour pressure by the Antoine equation
///
/// pythermalcomfort's `antoine` returns kPa; read the result in whatever unit the caller
/// needs with [`Pressure::as_kilopascals`] or [`Pressure::as_pascals`].
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapour pressure
///
/// # Examples
///
/// ```
/// use thermalcomfort::psychrometrics::antoine;
/// use thermalcomfort::Temperature;
///
/// let p = antoine(Temperature::from_celsius(25.0));
/// assert!((p.as_kilopascals() - 3.167).abs() < 0.01); // ~3.167 kPa at 25°C
/// ```
#[inline]
pub fn antoine(tdb: Temperature) -> Pressure {
    let tdb_celsius = tdb.as_celsius();
    let p_kpa = exp(16.6536 - 4030.183 / (tdb_celsius + 235.0));
    Pressure::from_kilopascals(p_kpa)
}

/// Saturation vapor pressure using exponential equation
///
/// This is used in the two-node Gagge model and related calculations.
/// Returns pressure in torr units.
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure (torr)
///
/// # Examples
///
/// ```
/// use thermalcomfort::psychrometrics::p_sat_torr;
/// use thermalcomfort::Temperature;
///
/// let p = p_sat_torr(Temperature::from_celsius(25.0));
/// assert!((p.as_torrs() - 23.8).abs() < 0.1);
/// ```
#[inline]
pub fn p_sat_torr(tdb: Temperature) -> Pressure {
    let tdb_celsius = tdb.as_celsius();
    let p_torr = exp(18.6686 - 4030.183 / (tdb_celsius + 235.0));
    // Built from torr so `.as_torrs()` returns upstream's value exactly; a hand-rolled
    // torr-to-pascal factor would disagree with the one `Pressure` converts back with.
    Pressure::from_torrs(p_torr)
}

/// Calculate saturation vapor pressure
///
/// Uses different formulas for temperatures above and below freezing
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// Saturation vapor pressure
pub fn p_sat(tdb: Temperature) -> Pressure {
    const C1: f64 = -5674.5359;
    const C2: f64 = 6.3925247;
    const C3: f64 = -0.9677843e-2;
    const C4: f64 = 0.62215701e-6;
    const C5: f64 = 0.20747825e-8;
    const C6: f64 = -0.9484024e-12;
    const C7: f64 = 4.1635019;
    const C8: f64 = -5800.2206;
    const C9: f64 = 1.3914993;
    const C10: f64 = -0.048640239;
    const C11: f64 = 0.41764768e-4;
    const C12: f64 = -0.14452093e-7;
    const C13: f64 = 6.5459673;

    let ta_k = tdb.as_celsius() + C_TO_K;
    let log_ta_k = log(ta_k);

    let p_pa = if ta_k < C_TO_K {
        exp(C1 / ta_k + C2 + ta_k * (C3 + ta_k * (C4 + ta_k * (C5 + C6 * ta_k))) + C7 * log_ta_k)
    } else {
        exp(C8 / ta_k + C9 + ta_k * (C10 + ta_k * (C11 + ta_k * C12)) + C13 * log_ta_k)
    };
    Pressure::from_pascals(p_pa)
}

/// Convert humidity ratio to relative humidity
///
/// Algebraic inverse of the humidity-ratio formula used by
/// [`crate::psychrometrics::psy_ta_rh`].
///
/// # Arguments
///
/// * `hr` - Humidity ratio [kg water / kg dry air]
/// * `tdb` - Dry bulb air temperature
/// * `p_atm` - Atmospheric pressure
///
/// # Returns
///
/// Relative humidity
///
/// # Examples
///
/// ```
/// use thermalcomfort::psychrometrics::hr_to_rh;
/// use thermalcomfort::{Temperature, Pressure};
///
/// let rh = hr_to_rh(0.01, Temperature::from_celsius(25.0), Pressure::from_pascals(101325.0));
/// assert!((rh.as_percent() - 50.5).abs() < 0.5);
/// ```
pub fn hr_to_rh(hr: f64, tdb: Temperature, p_atm: Pressure) -> Humidity {
    // 0.62198 = ratio of molecular weights (M_water / M_air = 18.015 / 28.965)
    let p_vap = hr * p_atm.as_pascals() / (0.62198 + hr);
    Humidity::from_percent(p_vap / p_sat(tdb).as_pascals() * 100.0)
}

// ---------------------------------------------------------------------------
// Paths pythermalcomfort 4.6.0 moved elsewhere. Each forwards to its new home and is
// removed after two minor releases, as upstream's compatibility shims are.
// ---------------------------------------------------------------------------

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
pub type MeanRadiantTemperatureInputs = crate::environment::MeanRadiantTemperatureInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
pub type MeanRadiantTemperatureOptions = crate::environment::MeanRadiantTemperatureOptions;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
pub type OperativeTemperatureInputs = crate::environment::OperativeTemperatureInputs;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
pub type OperativeTemperatureOptions = crate::environment::OperativeTemperatureOptions;

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
#[inline]
pub fn mean_radiant_temperature(
    inputs: crate::environment::MeanRadiantTemperatureInputs,
    options: crate::environment::MeanRadiantTemperatureOptions,
) -> Temperature {
    crate::environment::mean_radiant_temperature(inputs, options)
}

#[deprecated(since = "4.6.1", note = "moved to `thermalcomfort::environment`")]
#[inline]
pub fn operative_temperature(
    inputs: crate::environment::OperativeTemperatureInputs,
    options: crate::environment::OperativeTemperatureOptions,
) -> Temperature {
    crate::environment::operative_temperature(inputs, options)
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
            PsyTaRhInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
        );

        assert!(result.p_sat.as_pascals() > 0.0);
        assert!(result.p_vap.as_pascals() > 0.0);
        assert!(result.hr > 0.0);
        assert!((result.t_wb.as_celsius() - 17.7).abs() < 0.5);
        assert!((result.t_dp.as_celsius() - 13.9).abs() < 0.5);
        assert!(result.h > 0.0);
    }

    #[test]
    fn test_psy_ta_rh_p_atm_default() {
        // Python defaults p_atm to 101325 Pa; confirm the Rust default matches.
        let default_p_atm = psy_ta_rh(
            PsyTaRhInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            Default::default(),
        );
        let explicit_p_atm = psy_ta_rh(
            PsyTaRhInputs {
                tdb: Temperature::from_celsius(25.0),
                rh: Humidity::from_percent(50.0),
            },
            PsyTaRhOptions {
                p_atm: Pressure::from_pascals(101325.0),
            },
        );
        assert_eq!(default_p_atm.hr, explicit_p_atm.hr);
    }

    #[test]
    fn test_hr_to_rh_matches_python() {
        // Reference values from pythermalcomfort 4.4.0 hr_to_rh
        for (hr, tdb, expected) in [
            (0.01, 25.0, 50.589615),
            (0.005, 20.0, 34.549292),
            (0.02, 30.0, 74.343333),
        ] {
            let got = hr_to_rh(
                hr,
                Temperature::from_celsius(tdb),
                Pressure::from_pascals(101325.0),
            )
            .as_percent();
            assert!(
                (got - expected).abs() < 1e-4,
                "hr_to_rh({hr}, {tdb}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_antoine() {
        let p_kpa = antoine(Temperature::from_celsius(25.0)).as_kilopascals();
        // At 25°C, saturated vapor pressure is ~3.167 kPa
        assert!((p_kpa - 3.167).abs() < 0.01);

        let p_kpa_0 = antoine(Temperature::from_celsius(0.0)).as_kilopascals();
        // At 0°C, saturated vapor pressure is ~0.609 kPa
        assert!((p_kpa_0 - 0.609).abs() < 0.01);
    }
}
