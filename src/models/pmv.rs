//! PMV (Predicted Mean Vote) and PPD (Predicted Percentage Dissatisfied) models
//!
//! Implementation of thermal comfort models according to ISO 7730 and ASHRAE 55 standards

use crate::constants::*;
use crate::utilities::{valid_range, round_to};
use libm::{exp, pow, sqrt, fabs as abs, fmax};
use measurements::{Temperature, Speed};

/// Result of PMV/PPD calculation
#[derive(Debug, Clone, Copy)]
pub struct PmvPpdResult {
    /// Predicted Mean Vote (PMV)
    pub pmv: f64,
    /// Predicted Percentage of Dissatisfied (PPD) [%]
    pub ppd: f64,
    /// Thermal sensation vote category
    pub tsv: ThermalSensation,
}

/// Thermal sensation categories based on PMV value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalSensation {
    Cold,
    Cool,
    SlightlyCool,
    Neutral,
    SlightlyWarm,
    Warm,
    Hot,
}

impl ThermalSensation {
    /// Map PMV value to thermal sensation category
    pub fn from_pmv(pmv: f64) -> Self {
        if pmv.is_nan() {
            return ThermalSensation::Neutral;
        }

        match pmv {
            p if p < -2.5 => ThermalSensation::Cold,
            p if p < -1.5 => ThermalSensation::Cool,
            p if p < -0.5 => ThermalSensation::SlightlyCool,
            p if p < 0.5 => ThermalSensation::Neutral,
            p if p < 1.5 => ThermalSensation::SlightlyWarm,
            p if p < 2.5 => ThermalSensation::Warm,
            _ => ThermalSensation::Hot,
        }
    }
}

/// PMV/PPD calculation options
#[derive(Debug, Clone, Copy)]
pub struct PmvPpdOptions {
    /// External work [met], default 0.0
    pub wme: f64,
    /// Limit inputs to standard compliance ranges
    pub limit_inputs: bool,
    /// Round output values
    pub round_output: bool,
}

impl Default for PmvPpdOptions {
    fn default() -> Self {
        Self {
            wme: 0.0,
            limit_inputs: true,
            round_output: true,
        }
    }
}

/// Calculate PMV and PPD according to ISO 7730:2005
///
/// Returns the Predicted Mean Vote (PMV) and Predicted Percentage of Dissatisfied (PPD)
/// calculated in accordance with ISO 7730. The ISO uses the same formulation of PMV
/// as published by Fanger (1970).
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity [%]
/// * `metabolic_rate` - Metabolic rate [met]
/// * `clothing_insulation` - Clothing insulation [clo]
/// * `options` - Additional calculation options
///
/// # Returns
///
/// `PmvPpdResult` containing PMV, PPD, and thermal sensation category
///
/// # Standard Compliance Limits (ISO 7730:2005)
///
/// When `limit_inputs` is true:
/// - 10 < tdb [°C] < 30
/// - 10 < tr [°C] < 40
/// - 0 < vr [m/s] < 1
/// - 0.8 < met [met] < 4
/// - 0 < clo [clo] < 2
/// - -2 < PMV < 2
///
/// # Example
///
/// ```
/// use thermalcomfort::models::pmv_ppd_iso;
/// use thermalcomfort::utilities::v_relative;
/// use measurements::{Temperature, Speed};
///
/// let tdb = 25.0;
/// let tr = 25.0;
/// let rh = 50.0;
/// let v = 0.1;
/// let met = 1.4;
/// let clo = 0.5;
///
/// let vr = v_relative(v, met);
/// let result = pmv_ppd_iso(
///     Temperature::from_celsius(tdb),
///     Temperature::from_celsius(tr),
///     Speed::from_meters_per_second(vr),
///     rh,
///     met,
///     clo,
///     Default::default()
/// );
/// // result.pmv ≈ 0.17, result.ppd ≈ 5.6
/// ```
pub fn pmv_ppd_iso(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: f64,
    metabolic_rate: f64,
    clothing_insulation: f64,
    options: PmvPpdOptions,
) -> PmvPpdResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let air_speed = relative_air_speed.as_meters_per_second();

    // Check standard compliance if requested
    if options.limit_inputs {
        let dry_bulb_valid = valid_range(dry_bulb_celsius, 10.0, 30.0);
        let radiant_valid = valid_range(radiant_celsius, 10.0, 40.0);
        let speed_valid = valid_range(air_speed, 0.0, 1.0);
        let metabolic_valid = valid_range(metabolic_rate, 0.8, 4.0);
        let clothing_valid = valid_range(clothing_insulation, 0.0, 2.0);

        if dry_bulb_valid.is_nan() || radiant_valid.is_nan() || speed_valid.is_nan()
            || metabolic_valid.is_nan() || clothing_valid.is_nan() {
            return PmvPpdResult {
                pmv: f64::NAN,
                ppd: f64::NAN,
                tsv: ThermalSensation::Neutral,
            };
        }
    }

    // Calculate PMV using optimized algorithm
    let pmv = pmv_optimized(
        dry_bulb_celsius,
        radiant_celsius,
        air_speed,
        relative_humidity,
        metabolic_rate,
        clothing_insulation,
        options.wme
    );

    // Check PMV range if limiting inputs
    let pmv_final = if options.limit_inputs {
        let pmv_valid = valid_range(pmv, -2.0, 2.0);
        if pmv_valid.is_nan() {
            return PmvPpdResult {
                pmv: f64::NAN,
                ppd: f64::NAN,
                tsv: ThermalSensation::Neutral,
            };
        }
        pmv_valid
    } else {
        pmv
    };

    // Calculate PPD from PMV
    let ppd = 100.0 - 95.0 * exp(-0.03353 * pow(pmv_final, 4.0) - 0.2179 * pow(pmv_final, 2.0));

    // Round if requested
    let (pmv_out, ppd_out) = if options.round_output {
        (round_to(pmv_final, 2), round_to(ppd, 1))
    } else {
        (pmv_final, ppd)
    };

    PmvPpdResult {
        pmv: pmv_out,
        ppd: ppd_out,
        tsv: ThermalSensation::from_pmv(pmv_out),
    }
}

/// Calculate PMV and PPD according to ASHRAE 55
///
/// Similar to ISO 7730 but with different applicability limits
///
/// # Arguments
///
/// Same as `pmv_ppd_iso`
///
/// # Standard Compliance Limits (ASHRAE 55-2023)
///
/// When `limit_inputs` is true:
/// - 10 < tdb [°C] < 40
/// - 10 < tr [°C] < 40
/// - 0 < vr [m/s] < 2
/// - 1.0 < met [met] < 4
/// - 0 < clo [clo] < 1.5
pub fn pmv_ppd_ashrae(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: f64,
    metabolic_rate: f64,
    clothing_insulation: f64,
    options: PmvPpdOptions,
) -> PmvPpdResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let air_speed = relative_air_speed.as_meters_per_second();

    // Check ASHRAE standard compliance if requested
    if options.limit_inputs {
        let dry_bulb_valid = valid_range(dry_bulb_celsius, 10.0, 40.0);
        let radiant_valid = valid_range(radiant_celsius, 10.0, 40.0);
        let speed_valid = valid_range(air_speed, 0.0, 2.0);
        let metabolic_valid = valid_range(metabolic_rate, 1.0, 4.0);
        let clothing_valid = valid_range(clothing_insulation, 0.0, 1.5);

        if dry_bulb_valid.is_nan() || radiant_valid.is_nan() || speed_valid.is_nan()
            || metabolic_valid.is_nan() || clothing_valid.is_nan() {
            return PmvPpdResult {
                pmv: f64::NAN,
                ppd: f64::NAN,
                tsv: ThermalSensation::Neutral,
            };
        }
    }

    // Calculate PMV (same algorithm as ISO)
    let pmv = pmv_optimized(dry_bulb_celsius, radiant_celsius, air_speed, relative_humidity, metabolic_rate, clothing_insulation, options.wme);

    // Calculate PPD from PMV
    let ppd = 100.0 - 95.0 * exp(-0.03353 * pow(pmv, 4.0) - 0.2179 * pow(pmv, 2.0));

    // Round if requested
    let (pmv_out, ppd_out) = if options.round_output {
        (round_to(pmv, 2), round_to(ppd, 1))
    } else {
        (pmv, ppd)
    };

    PmvPpdResult {
        pmv: pmv_out,
        ppd: ppd_out,
        tsv: ThermalSensation::from_pmv(pmv_out),
    }
}

/// Optimized PMV calculation core algorithm
///
/// This is the core PMV calculation from Fanger's model,
/// ported from the numba-optimized Python version
fn pmv_optimized(tdb: f64, tr: f64, vr: f64, rh: f64, met: f64, clo: f64, wme: f64) -> f64 {
    // Calculate partial vapor pressure
    let pa = rh * 10.0 * exp(16.6536 - 4030.183 / (tdb + 235.0));

    // Thermal insulation of clothing [m²K/W]
    let icl = 0.155 * clo;

    // Metabolic rate [W/m²]
    let m = met * MET_TO_W_M2;

    // External work [W/m²]
    let w = wme * MET_TO_W_M2;

    // Internal heat production
    let mw = m - w;

    // Clothing area factor
    let fcl = if icl <= 0.078 {
        1.0 + 1.29 * icl
    } else {
        1.05 + 0.645 * icl
    };

    // Heat transfer coefficient by forced convection
    let hcf = 12.1 * sqrt(vr);
    let mut hc = hcf;

    let taa = tdb + 273.0;
    let tra = tr + 273.0;
    let tcla = taa + (35.5 - tdb) / (3.5 * icl + 0.1);

    let p1 = icl * fcl;
    let p2 = p1 * 3.96;
    let p3 = p1 * 100.0;
    let p4 = p1 * taa;
    let p5 = 308.7 - 0.028 * mw + p2 * pow(tra / 100.0, 4.0);

    let mut xn = tcla / 100.0;
    let mut xf = tcla / 50.0;
    let eps = 0.00015;

    let mut n = 0;
    while abs(xn - xf) > eps {
        xf = (xf + xn) / 2.0;
        let hcn = 2.38 * pow(abs(100.0 * xf - taa), 0.25);
        hc = fmax(hcn, hcf);
        xn = (p5 + p4 * hc - p2 * pow(xf, 4.0)) / (100.0 + p3 * hc);
        n += 1;
        if n > 150 {
            // Max iterations exceeded, return NaN
            return f64::NAN;
        }
    }

    let tcl = 100.0 * xn - 273.0;

    // Heat losses
    // Heat loss diff through skin
    let hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa);

    // Heat loss by sweating
    let hl2 = if mw > MET_TO_W_M2 {
        0.42 * (mw - MET_TO_W_M2)
    } else {
        0.0
    };

    // Latent respiration heat loss
    let hl3 = 1.7 * 0.00001 * m * (5867.0 - pa);

    // Dry respiration heat loss
    let hl4 = 0.0014 * m * (34.0 - tdb);

    // Heat loss by radiation
    let hl5 = 3.96 * fcl * (pow(xn, 4.0) - pow(tra / 100.0, 4.0));

    // Heat loss by convection
    let hl6 = fcl * hc * (tcl - tdb);

    // PMV calculation
    let ts = 0.303 * exp(-0.036 * m) + 0.028;
    ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_sensation_mapping() {
        assert_eq!(ThermalSensation::from_pmv(-3.0), ThermalSensation::Cold);
        assert_eq!(ThermalSensation::from_pmv(-2.0), ThermalSensation::Cool);
        assert_eq!(ThermalSensation::from_pmv(-1.0), ThermalSensation::SlightlyCool);
        assert_eq!(ThermalSensation::from_pmv(0.0), ThermalSensation::Neutral);
        assert_eq!(ThermalSensation::from_pmv(1.0), ThermalSensation::SlightlyWarm);
        assert_eq!(ThermalSensation::from_pmv(2.0), ThermalSensation::Warm);
        assert_eq!(ThermalSensation::from_pmv(3.0), ThermalSensation::Hot);
    }

    #[test]
    fn test_pmv_ppd_iso_basic() {
        // Example from ISO 7730
        let result = pmv_ppd_iso(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            50.0,
            1.2,
            0.5,
            Default::default()
        );

        // Should be approximately neutral comfort
        assert!(result.pmv.abs() < 0.5);
        assert!(result.ppd < 10.0);
    }

    #[test]
    fn test_pmv_ppd_iso_limits() {
        // Test with values outside limits
        let result = pmv_ppd_iso(
            Temperature::from_celsius(5.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            50.0,
            1.2,
            0.5,
            Default::default()
        );
        assert!(result.pmv.is_nan());
        assert!(result.ppd.is_nan());

        // Test with limits disabled
        let mut options = PmvPpdOptions::default();
        options.limit_inputs = false;
        let result = pmv_ppd_iso(
            Temperature::from_celsius(5.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            50.0,
            1.2,
            0.5,
            options
        );
        assert!(!result.pmv.is_nan());
    }

    #[cfg(test)]
    #[test]
    fn test_compare_with_python() {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;

        Python::with_gil(|py| {
            // Import pythermalcomfort
            let pythermal = PyModule::import(py, "pythermalcomfort.models").unwrap();

            // Test case 1: Standard conditions
            let tdb = 25.0;
            let tr = 25.0;
            let vr = 0.22; // v_relative(0.1, 1.4)
            let rh = 50.0;
            let met = 1.4;
            let clo = 0.5;

            // Call Python function
            let py_result = pythermal
                .getattr("pmv_ppd_iso").unwrap()
                .call1((tdb, tr, vr, rh, met, clo)).unwrap();

            let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
            let py_ppd: f64 = py_result.getattr("ppd").unwrap().extract().unwrap();

            // Call Rust function
            let rust_result = pmv_ppd_iso(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                rh,
                met,
                clo,
                Default::default()
            );

            // Compare results (allow small floating point differences)
            assert!((rust_result.pmv - py_pmv).abs() < 0.01,
                "PMV mismatch: Rust={}, Python={}", rust_result.pmv, py_pmv);
            assert!((rust_result.ppd - py_ppd).abs() < 0.1,
                "PPD mismatch: Rust={}, Python={}", rust_result.ppd, py_ppd);
        });
    }
}

/// Calculate Adaptive Predicted Mean Vote (aPMV)
///
/// This index was developed by Yao et al. (2009) and takes into account factors such as
/// culture, climate, social, psychological, and behavioral adaptations.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity [%]
/// * `metabolic_rate` - Metabolic rate [met]
/// * `clothing_insulation` - Clothing insulation [clo]
/// * `a_coefficient` - Adaptive coefficient (λ)
/// * `options` - PMV calculation options
///
/// # Returns
///
/// Adaptive PMV value
///
/// # Formula
///
/// aPMV = PMV / (1 + λ * PMV)
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::pmv_a;
/// use measurements::{Temperature, Speed};
///
/// let a_pmv = pmv_a(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(0.1),
///     50.0,
///     1.2,
///     0.5,
///     0.5,  // adaptive coefficient
///     Default::default()
/// );
/// // Adaptive PMV adjusts standard PMV based on expectancy
/// ```
///
/// # References
///
/// - Yao R, Li B, Liu J (2009) Indoor Built Environ 18(5):394-411
pub fn pmv_a(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: f64,
    metabolic_rate: f64,
    clothing_insulation: f64,
    a_coefficient: f64,
    options: PmvPpdOptions,
) -> f64 {
    let pmv = pmv_ppd_iso(dry_bulb_temp, mean_radiant_temp, relative_air_speed, relative_humidity, metabolic_rate, clothing_insulation, options).pmv;
    let a_pmv = pmv / (1.0 + a_coefficient * pmv);
    libm::round(a_pmv * 100.0) / 100.0 // Round to 2 decimal places
}

/// Calculate Adjusted PMV with Expectancy Factor (ePMV)
///
/// Developed by Fanger et al. (2002) for non-air-conditioned buildings in warm climates.
/// Accounts for occupants' low expectations in naturally ventilated spaces.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity [%]
/// * `metabolic_rate` - Metabolic rate [met]
/// * `clothing_insulation` - Clothing insulation [clo]
/// * `e_coefficient` - Expectancy factor
/// * `options` - PMV calculation options
///
/// # Returns
///
/// Adjusted PMV with expectancy factor
///
/// # Formula
///
/// ePMV = PMV * e_coefficient
///
/// For warm conditions (PMV > 0), metabolic rate is adjusted:
/// met_adjusted = met * (1 + PMV * (-0.067))
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::pmv_e;
/// use measurements::{Temperature, Speed};
///
/// let e_pmv = pmv_e(
///     Temperature::from_celsius(28.0),
///     Temperature::from_celsius(28.0),
///     Speed::from_meters_per_second(0.2),
///     60.0,
///     1.2,
///     0.5,
///     0.7,  // expectancy factor
///     Default::default()
/// );
/// // Expectancy PMV for naturally ventilated buildings
/// ```
///
/// # References
///
/// - Fanger PO, Toftum J (2002) Energy Build 34(2):153-9
pub fn pmv_e(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: f64,
    metabolic_rate: f64,
    clothing_insulation: f64,
    e_coefficient: f64,
    options: PmvPpdOptions,
) -> f64 {
    // First PMV calculation
    let pmv1 = pmv_ppd_iso(dry_bulb_temp, mean_radiant_temp, relative_air_speed, relative_humidity, metabolic_rate, clothing_insulation, options).pmv;

    // Adjust metabolic rate if warm (PMV > 0)
    let met_adjusted = if pmv1 > 0.0 {
        metabolic_rate * (1.0 + pmv1 * (-0.067))
    } else {
        metabolic_rate
    };

    // Recalculate PMV with adjusted metabolic rate
    let pmv2 = pmv_ppd_iso(dry_bulb_temp, mean_radiant_temp, relative_air_speed, relative_humidity, met_adjusted, clothing_insulation, options).pmv;

    let e_pmv = pmv2 * e_coefficient;
    libm::round(e_pmv * 100.0) / 100.0 // Round to 2 decimal places
}

/// Calculate PMV using Adaptive Thermal Heat Balance (ATHB) framework
///
/// Developed by Schweiker et al. (2022). Accounts for physiological, behavioral,
/// and psychological adaptation within heat balance models.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `relative_air_speed` - Relative air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity [%]
/// * `metabolic_rate` - Metabolic rate [met]
/// * `clothing_insulation` - Clothing insulation [clo] (if None, calculated from running mean)
/// * `running_mean_outdoor_temp` - Running mean outdoor temperature (use `Temperature::from_celsius()` or similar)
///
/// # Returns
///
/// ATHB-adjusted PMV value
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::pmv_athb;
/// use measurements::{Temperature, Speed};
///
/// let athb_pmv = pmv_athb(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(0.1),
///     50.0,
///     1.2,
///     Some(0.6),  // clothing insulation
///     Temperature::from_celsius(20.0)  // running mean outdoor temp
/// );
/// // ATHB PMV accounts for physiological and behavioral adaptation
/// ```
///
/// # References
///
/// - Schweiker M et al. (2022) Build Environ 216:109017
pub fn pmv_athb(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    relative_air_speed: Speed,
    relative_humidity: f64,
    metabolic_rate: f64,
    clothing_insulation: Option<f64>,
    running_mean_outdoor_temp: Temperature,
) -> f64 {
    let running_mean_celsius = running_mean_outdoor_temp.as_celsius();

    // Adapt metabolic rate for psychological adaptation
    let met_adapted = metabolic_rate - (0.234 * running_mean_celsius) / 58.2;

    // Calculate or use provided clothing insulation
    let clo_adapted = if let Some(c) = clothing_insulation {
        c
    } else {
        // Behavioral adaptation: calculate clothing from conditions
        let exponent = -0.17168 - 0.000485 * running_mean_celsius + 0.08176 * met_adapted
                      - 0.00527 * running_mean_celsius * met_adapted;
        libm::pow(10.0, exponent)
    };

    // Calculate base PMV with adapted parameters
    let mut options = PmvPpdOptions::default();
    options.limit_inputs = false; // ATHB may use values outside standard limits
    let pmv_result = pmv_ppd_iso(dry_bulb_temp, mean_radiant_temp, relative_air_speed, relative_humidity, met_adapted, clo_adapted, options).pmv;

    // Calculate thermal sensation coefficient
    let ts = 0.303 * libm::exp(-0.036 * met_adapted * 58.2) + 0.028;
    let l_adapted = pmv_result / ts;

    // Calculate ATHB PMV
    let athb_pmv = 1.484
        + 0.0276 * l_adapted
        - 0.9602 * met_adapted
        - 0.0342 * running_mean_celsius
        + 0.0002264 * l_adapted * running_mean_celsius
        + 0.018696 * met_adapted * running_mean_celsius
        - 0.0002909 * l_adapted * met_adapted * running_mean_celsius;

    libm::round(athb_pmv * 1000.0) / 1000.0 // Round to 3 decimal places
}
