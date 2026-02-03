//! Type-safe PMV/PPD calculations using the measurements crate
//!
//! This module provides wrappers around the core PMV/PPD functions that use
//! strongly-typed measurements instead of raw f64 values.

use measurements::{Temperature, Speed};
use crate::models::pmv::{pmv_ppd_iso as pmv_ppd_iso_f64, pmv_ppd_ashrae as pmv_ppd_ashrae_f64, PmvPpdOptions, PmvPpdResult};

/// Calculate PMV and PPD according to ISO 7730:2005 using type-safe measurements
///
/// This is a type-safe wrapper around `pmv_ppd_iso` that uses the `measurements` crate
/// for temperature and air speed values.
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `tr` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `vr` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `rh` - Relative humidity [%]
/// * `met` - Metabolic rate [met]
/// * `clo` - Clothing insulation [clo]
/// * `options` - Additional calculation options
///
/// # Returns
///
/// `PmvPpdResult` containing PMV, PPD, and thermal sensation category
///
/// # Example
///
/// ```
/// use thermalcomfort::models::pmv_typed::pmv_ppd_iso_typed;
/// use thermalcomfort::utilities::v_relative;
/// use measurements::{Temperature, Speed};
///
/// let tdb = Temperature::from_celsius(25.0);
/// let tr = Temperature::from_celsius(25.0);
/// let v = Speed::from_meters_per_second(0.1);
/// let rh = 50.0;
/// let met = 1.4;
/// let clo = 0.5;
///
/// // Calculate relative air speed
/// let vr_ms = v_relative(v.as_meters_per_second(), met);
/// let vr = Speed::from_meters_per_second(vr_ms);
///
/// let result = pmv_ppd_iso_typed(tdb, tr, vr, rh, met, clo, Default::default());
/// ```
pub fn pmv_ppd_iso_typed(
    tdb: Temperature,
    tr: Temperature,
    vr: Speed,
    rh: f64,
    met: f64,
    clo: f64,
    options: PmvPpdOptions,
) -> PmvPpdResult {
    // The main function now takes measurement types directly
    pmv_ppd_iso_f64(tdb, tr, vr, rh, met, clo, options)
}

/// Calculate PMV and PPD according to ASHRAE 55 using type-safe measurements
///
/// This is a type-safe wrapper around `pmv_ppd_ashrae` that uses the `measurements` crate
/// for temperature and air speed values.
///
/// # Arguments
///
/// Same as `pmv_ppd_iso_typed`
///
/// # Example
///
/// ```
/// use thermalcomfort::models::pmv_typed::pmv_ppd_ashrae_typed;
/// use measurements::{Temperature, Speed};
///
/// let tdb = Temperature::from_celsius(25.0);
/// let tr = Temperature::from_celsius(25.0);
/// let vr = Speed::from_meters_per_second(0.1);
/// let rh = 50.0;
/// let met = 1.2;
/// let clo = 0.5;
///
/// let result = pmv_ppd_ashrae_typed(tdb, tr, vr, rh, met, clo, Default::default());
/// ```
pub fn pmv_ppd_ashrae_typed(
    tdb: Temperature,
    tr: Temperature,
    vr: Speed,
    rh: f64,
    met: f64,
    clo: f64,
    options: PmvPpdOptions,
) -> PmvPpdResult {
    // The main function now takes measurement types directly
    pmv_ppd_ashrae_f64(tdb, tr, vr, rh, met, clo, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use measurements::{Temperature, Speed};

    #[test]
    fn test_pmv_ppd_iso_typed() {
        let tdb = Temperature::from_celsius(25.0);
        let tr = Temperature::from_celsius(25.0);
        let vr = Speed::from_meters_per_second(0.1);
        let rh = 50.0;
        let met = 1.2;
        let clo = 0.5;

        let result = pmv_ppd_iso_typed(tdb, tr, vr, rh, met, clo, Default::default());

        // Should be approximately neutral comfort
        assert!(result.pmv.abs() < 0.5);
        assert!(result.ppd < 10.0);
    }

    #[test]
    fn test_temperature_units() {
        // Test that we can use Fahrenheit and it converts correctly
        let tdb_f = Temperature::from_fahrenheit(77.0);
        let tdb_c = Temperature::from_celsius(25.0);

        let tr = Temperature::from_celsius(25.0);
        let vr = Speed::from_meters_per_second(0.1);
        let rh = 50.0;
        let met = 1.2;
        let clo = 0.5;

        let result_f = pmv_ppd_iso_typed(tdb_f, tr, vr, rh, met, clo, Default::default());
        let result_c = pmv_ppd_iso_typed(tdb_c, tr, vr, rh, met, clo, Default::default());

        // Results should be very close (within floating point precision)
        assert!((result_f.pmv - result_c.pmv).abs() < 0.01);
    }

    #[test]
    fn test_speed_units() {
        // Test that we can use different speed units
        let tdb = Temperature::from_celsius(25.0);
        let tr = Temperature::from_celsius(25.0);
        let vr_mps = Speed::from_meters_per_second(0.1);
        let vr_kmh = Speed::from_kilometers_per_hour(0.1 * 3.6); // 0.1 m/s = 0.36 km/h
        let rh = 50.0;
        let met = 1.2;
        let clo = 0.5;

        let result_mps = pmv_ppd_iso_typed(tdb, tr, vr_mps, rh, met, clo, Default::default());
        let result_kmh = pmv_ppd_iso_typed(tdb, tr, vr_kmh, rh, met, clo, Default::default());

        // Results should be very close
        assert!((result_mps.pmv - result_kmh.pmv).abs() < 0.01);
    }
}
