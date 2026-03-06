//! Ridge regression model for predicting rectal and skin temperature
//!
//! Based on Forbes et al. (2025), doi:10.1016/j.jtherbio.2025.104078.
//! Trained on adults aged 60-100 in minimal clothing under stationary conditions.

extern crate alloc;
use crate::Sex;
use alloc::vec;
use alloc::vec::Vec;
use measurements::{Humidity, Length, Mass, Temperature};

/// Result from ridge regression body temperature prediction
#[derive(Debug, Clone, PartialEq)]
pub struct PredictedBodyTemperatures {
    /// Predicted rectal temperature history [°C] over time
    pub t_re: Vec<f64>,
    /// Predicted mean skin temperature history [°C] over time
    pub t_sk: Vec<f64>,
}

/// Options for ridge regression prediction
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgeRegressionOptions {
    /// Initial rectal temperature (None = run baseline simulation)
    pub t_re_initial: Option<Temperature>,
    /// Initial skin temperature (None = run baseline simulation)
    pub t_sk_initial: Option<Temperature>,
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round output to 2 decimal places
    pub round_output: bool,
}

impl Default for RidgeRegressionOptions {
    fn default() -> Self {
        Self {
            t_re_initial: None,
            t_sk_initial: None,
            limit_inputs: true,
            round_output: true,
        }
    }
}

// Model constants - Min-Max scaling offsets for 8 input features
const FEATURES_SCALER_OFFSET: [f64; 8] = [
    0.0,                  // sex
    -0.3114754098360656,  // age
    -3.020833333333333,   // height
    -0.6183587248751761,  // weight
    -1.222222222222222,   // tdb
    -0.21951219512195122, // rh
    -11.197107405358395,  // t_re
    -2.105553500973682,   // t_sk
];

// Model constants - Min-Max scaling factors (1 / (max - min))
const FEATURES_SCALER_SCALE: [f64; 8] = [
    1.0,                  // sex
    0.01639344262295082,  // age
    0.020833333333333332, // height
    0.012802458071949815, // weight
    0.05555555555555555,  // tdb
    0.024390243902439025, // rh
    0.31613056192207334,  // t_re
    0.08119094120192633,  // t_sk
];

// Output inverse scaling offsets
const OUTPUT_SCALER_OFFSET: [f64; 2] = [
    -11.197107405358395, // t_re
    -2.0197777680408033, // t_sk
];

// Output inverse scaling factors
const OUTPUT_SCALER_SCALE: [f64; 2] = [
    0.31613056192207334, // t_re
    0.07894843838015173, // t_sk
];

// Rectal temperature regression coefficients
const T_RE_COEFFS: [f64; 8] = [
    0.00016261586852849347,  // sex
    0.0007368142143779594,   // age
    -0.00043916987857211637, // height
    0.00046532701146677997,  // weight
    0.0008443934806620367,   // tdb
    0.0006663379066237714,   // rh
    0.9932810428489056,      // t_re (previous)
    0.006016233208250791,    // t_sk (previous)
];
const T_RE_INTERCEPT: f64 = -0.0013528489525256315;

// Mean skin temperature regression coefficients
const T_SK_COEFFS: [f64; 8] = [
    0.0006157845452869151,  // sex
    0.00014854705372386215, // age
    -0.0004329826169348138, // height
    -0.0011471088118388912, // weight
    0.018904677058503336,   // tdb
    0.003188995712763656,   // rh
    -0.0010477636196332153, // t_re (previous)
    0.933918210580563,      // t_sk (previous)
];
const T_SK_INTERCEPT: f64 = 0.04356328728329839;

// Baseline/thermoneutral environment parameters
const BASELINE_DURATION: usize = 120; // minutes
const THERMONEUTRAL_TDB: f64 = 23.0; // °C
const THERMONEUTRAL_RH: f64 = 50.0; // %
const BASELINE_T_RE_INITIAL: f64 = 37.0; // °C
const BASELINE_T_SK_INITIAL: f64 = 32.0; // °C

/// Scale input features using Min-Max scaling
#[inline]
fn scale_features(features: &[f64; 8]) -> [f64; 8] {
    let mut scaled = [0.0; 8];
    for i in 0..8 {
        scaled[i] = features[i] * FEATURES_SCALER_SCALE[i] + FEATURES_SCALER_OFFSET[i];
    }
    scaled
}

/// Apply inverse scaling to model output
#[inline]
fn inverse_scale_output(scaled_t_re: f64, scaled_t_sk: f64) -> (f64, f64) {
    let t_re = (scaled_t_re - OUTPUT_SCALER_OFFSET[0]) / OUTPUT_SCALER_SCALE[0];
    let t_sk = (scaled_t_sk - OUTPUT_SCALER_OFFSET[1]) / OUTPUT_SCALER_SCALE[1];
    (t_re, t_sk)
}

/// Run temperature prediction simulation
#[allow(clippy::too_many_arguments)]
fn predict_temperature_simulation(
    sex: f64,
    age: f64,
    height_cm: f64,
    weight: f64,
    tdb: f64,
    rh: f64,
    initial_t_re: f64,
    initial_t_sk: f64,
    duration: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Build feature vector
    let features = [
        sex,
        age,
        height_cm,
        weight,
        tdb,
        rh,
        initial_t_re,
        initial_t_sk,
    ];

    // Scale features
    let scaled = scale_features(&features);

    // Precompute static components (first 6 features don't change)
    let static_t_re = scaled[0] * T_RE_COEFFS[0]
        + scaled[1] * T_RE_COEFFS[1]
        + scaled[2] * T_RE_COEFFS[2]
        + scaled[3] * T_RE_COEFFS[3]
        + scaled[4] * T_RE_COEFFS[4]
        + scaled[5] * T_RE_COEFFS[5]
        + T_RE_INTERCEPT;

    let static_t_sk = scaled[0] * T_SK_COEFFS[0]
        + scaled[1] * T_SK_COEFFS[1]
        + scaled[2] * T_SK_COEFFS[2]
        + scaled[3] * T_SK_COEFFS[3]
        + scaled[4] * T_SK_COEFFS[4]
        + scaled[5] * T_SK_COEFFS[5]
        + T_SK_INTERCEPT;

    // Initialize with scaled initial temperatures
    let mut prev_t_re = scaled[6];
    let mut prev_t_sk = scaled[7];

    // Storage for history
    let mut t_re_history = Vec::with_capacity(duration);
    let mut t_sk_history = Vec::with_capacity(duration);

    // Run simulation
    for _ in 0..duration {
        // Predict next timestep
        let new_t_re = static_t_re + T_RE_COEFFS[6] * prev_t_re + T_RE_COEFFS[7] * prev_t_sk;
        let new_t_sk = static_t_sk + T_SK_COEFFS[6] * prev_t_re + T_SK_COEFFS[7] * prev_t_sk;

        // Inverse scale and store
        let (t_re, t_sk) = inverse_scale_output(new_t_re, new_t_sk);
        t_re_history.push(t_re);
        t_sk_history.push(t_sk);

        // Update for next iteration
        prev_t_re = new_t_re;
        prev_t_sk = new_t_sk;
    }

    (t_re_history, t_sk_history)
}

/// Predict rectal and skin temperature using ridge regression model
///
/// This model predicts core (rectal) and mean skin temperatures over time
/// based on personal and environmental characteristics. It's trained on data
/// from adults aged 60-100 in minimal clothing under stationary conditions.
///
/// # Arguments
///
/// * `sex` - Biological sex
/// * `age` - Age [years]
/// * `height` - Body height (use `Length::from_meters()` or similar)
/// * `weight` - Body weight (use `Mass::from_kilograms()` or similar)
/// * `tdb` - Ambient air temperature (use `Temperature::from_celsius()` or similar)
/// * `rh` - Relative humidity (use `Humidity::from_percent()` or similar)
/// * `duration` - Simulation duration [minutes]
/// * `options` - Model options
///
/// # Returns
///
/// PredictedBodyTemperatures with time series of t_re and t_sk
///
/// # Applicability Limits
///
/// When `limit_inputs` is true:
/// - Age: 60-100 years
/// - Height: 1.30-2.30 m
/// - Weight: 40-140 kg
/// - Temperature: 0-60°C
/// - Relative humidity: 0-100%
///
/// # Notes
///
/// - Model assumes minimal clothing and stationary conditions
/// - Does not account for air velocity, radiation, clothing level, or activity
/// - If initial temperatures not provided, runs 120-min baseline at 23°C, 50% RH
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::ridge_regression::{
///     ridge_regression_predict_t_re_t_sk, RidgeRegressionOptions
/// };
/// use thermalcomfort::{Sex, Temperature, Length, Mass, Humidity};
///
/// let result = ridge_regression_predict_t_re_t_sk(
///     Sex::Male,
///     60.0,          // age
///     Length::from_meters(1.8),
///     Mass::from_kilograms(75.0),
///     Temperature::from_celsius(35.0),
///     Humidity::from_percent(60.0),
///     540,           // duration [min]
///     Default::default()
/// );
///
/// println!("Final rectal temp: {:.2}°C", result.t_re.last().unwrap());
/// println!("Final skin temp: {:.2}°C", result.t_sk.last().unwrap());
/// ```
///
/// # References
///
/// - Forbes et al. (2025), doi:10.1016/j.jtherbio.2025.104078
#[allow(clippy::too_many_arguments)]
pub fn ridge_regression_predict_t_re_t_sk(
    sex: Sex,
    age: f64,
    height: Length,
    weight: Mass,
    tdb: Temperature,
    rh: Humidity,
    duration: usize,
    options: RidgeRegressionOptions,
) -> PredictedBodyTemperatures {
    let tdb_c = tdb.as_celsius();
    let height_m = height.as_meters();
    let height_cm = height_m * 100.0;
    let weight_kg = weight.as_kilograms();
    let rh_pct = rh.as_percent();
    let sex_value = sex.as_value();

    // Check applicability limits if requested
    if options.limit_inputs {
        let age_valid = (60.0..=100.0).contains(&age);
        let height_valid = (1.30..=2.30).contains(&height_m);
        let weight_valid = (40.0..=140.0).contains(&weight_kg);
        let tdb_valid = (0.0..=60.0).contains(&tdb_c);
        let rh_valid = (0.0..=100.0).contains(&rh_pct);

        if !age_valid || !height_valid || !weight_valid || !tdb_valid || !rh_valid {
            // Return NaN-filled arrays for out-of-range inputs
            return PredictedBodyTemperatures {
                t_re: vec![f64::NAN; duration],
                t_sk: vec![f64::NAN; duration],
            };
        }
    }

    // Determine initial temperatures
    let (initial_t_re, initial_t_sk) =
        if let (Some(t_re), Some(t_sk)) = (options.t_re_initial, options.t_sk_initial) {
            // Use provided initial temperatures
            (t_re.as_celsius(), t_sk.as_celsius())
        } else {
            // Run baseline simulation in thermoneutral environment
            let (baseline_t_re, baseline_t_sk) = predict_temperature_simulation(
                sex_value,
                age,
                height_cm,
                weight_kg,
                THERMONEUTRAL_TDB,
                THERMONEUTRAL_RH,
                BASELINE_T_RE_INITIAL,
                BASELINE_T_SK_INITIAL,
                BASELINE_DURATION,
            );

            // Use final values from baseline as initial conditions
            (
                *baseline_t_re.last().unwrap(),
                *baseline_t_sk.last().unwrap(),
            )
        };

    // Run main simulation
    let (mut t_re_history, mut t_sk_history) = predict_temperature_simulation(
        sex_value,
        age,
        height_cm,
        weight_kg,
        tdb_c,
        rh_pct,
        initial_t_re,
        initial_t_sk,
        duration,
    );

    // Round if requested
    if options.round_output {
        for val in &mut t_re_history {
            *val = libm::round(*val * 100.0) / 100.0;
        }
        for val in &mut t_sk_history {
            *val = libm::round(*val * 100.0) / 100.0;
        }
    }

    PredictedBodyTemperatures {
        t_re: t_re_history,
        t_sk: t_sk_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression_basic() {
        let result = ridge_regression_predict_t_re_t_sk(
            Sex::Male,
            60.0,
            Length::from_meters(1.8),
            Mass::from_kilograms(75.0),
            Temperature::from_celsius(35.0),
            Humidity::from_percent(60.0),
            540,
            Default::default(),
        );

        // Should have correct duration
        assert_eq!(result.t_re.len(), 540);
        assert_eq!(result.t_sk.len(), 540);

        // Temperatures should be in realistic range
        assert!(result.t_re.iter().all(|&t| t > 30.0 && t < 45.0));
        assert!(result.t_sk.iter().all(|&t| t > 25.0 && t < 45.0));

        // Temperature should increase over time in hot environment
        assert!(result.t_re.last().unwrap() > result.t_re.first().unwrap());
    }

    #[test]
    fn test_ridge_regression_out_of_range() {
        let result = ridge_regression_predict_t_re_t_sk(
            Sex::Male,
            50.0, // Too young (< 60)
            Length::from_meters(1.8),
            Mass::from_kilograms(75.0),
            Temperature::from_celsius(35.0),
            Humidity::from_percent(60.0),
            10,
            Default::default(),
        );

        // Should return NaN for out-of-range inputs
        assert!(result.t_re.iter().all(|t| t.is_nan()));
        assert!(result.t_sk.iter().all(|t| t.is_nan()));
    }

    #[test]
    fn test_ridge_regression_with_initial_temps() {
        let options = RidgeRegressionOptions {
            t_re_initial: Some(Temperature::from_celsius(37.0)),
            t_sk_initial: Some(Temperature::from_celsius(33.0)),
            limit_inputs: true,
            round_output: true,
        };

        let result = ridge_regression_predict_t_re_t_sk(
            Sex::Female,
            65.0,
            Length::from_meters(1.65),
            Mass::from_kilograms(60.0),
            Temperature::from_celsius(40.0),
            Humidity::from_percent(50.0),
            60,
            options,
        );

        // Should have correct duration
        assert_eq!(result.t_re.len(), 60);
        assert_eq!(result.t_sk.len(), 60);

        // Values should be reasonable
        assert!(result.t_re.iter().all(|&t| t > 35.0 && t < 42.0));
    }
}
