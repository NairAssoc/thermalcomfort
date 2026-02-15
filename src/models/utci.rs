//! Universal Thermal Climate Index (UTCI)
//!
//! The UTCI is the equivalent temperature for the environment derived from a
//! reference environment, widely used for outdoor thermal comfort assessment.

use libm::{exp, pow};
use measurements::{Humidity, Speed, Temperature};

/// UTCI result with stress category
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtciResult {
    /// Universal Thermal Climate Index [°C]
    pub utci: f64,
    /// Thermal stress category
    pub stress_category: StressCategory,
}

/// Thermal stress categories based on UTCI value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressCategory {
    ExtremeColdStress,
    VeryStrongColdStress,
    StrongColdStress,
    ModerateColdStress,
    SlightColdStress,
    NoThermalStress,
    ModerateHeatStress,
    StrongHeatStress,
    VeryStrongHeatStress,
    ExtremeHeatStress,
}

impl StressCategory {
    /// Get stress category from UTCI value
    pub fn from_utci(utci: f64) -> Self {
        if utci.is_nan() {
            return StressCategory::NoThermalStress;
        }

        if utci < -40.0 {
            StressCategory::ExtremeColdStress
        } else if utci < -27.0 {
            StressCategory::VeryStrongColdStress
        } else if utci < -13.0 {
            StressCategory::StrongColdStress
        } else if utci < 0.0 {
            StressCategory::ModerateColdStress
        } else if utci < 9.0 {
            StressCategory::SlightColdStress
        } else if utci < 26.0 {
            StressCategory::NoThermalStress
        } else if utci < 32.0 {
            StressCategory::ModerateHeatStress
        } else if utci < 38.0 {
            StressCategory::StrongHeatStress
        } else if utci < 46.0 {
            StressCategory::VeryStrongHeatStress
        } else {
            StressCategory::ExtremeHeatStress
        }
    }

    /// Get description string
    pub fn as_str(&self) -> &'static str {
        match self {
            StressCategory::ExtremeColdStress => "extreme cold stress",
            StressCategory::VeryStrongColdStress => "very strong cold stress",
            StressCategory::StrongColdStress => "strong cold stress",
            StressCategory::ModerateColdStress => "moderate cold stress",
            StressCategory::SlightColdStress => "slight cold stress",
            StressCategory::NoThermalStress => "no thermal stress",
            StressCategory::ModerateHeatStress => "moderate heat stress",
            StressCategory::StrongHeatStress => "strong heat stress",
            StressCategory::VeryStrongHeatStress => "very strong heat stress",
            StressCategory::ExtremeHeatStress => "extreme heat stress",
        }
    }
}

/// Options for UTCI calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtciOptions {
    /// Limit inputs to standard applicability ranges
    pub limit_inputs: bool,
    /// Round output value to 1 decimal place
    pub round_output: bool,
}

impl Default for UtciOptions {
    fn default() -> Self {
        Self {
            limit_inputs: true,
            round_output: true,
        }
    }
}

/// Calculate the Universal Thermal Climate Index (UTCI)
///
/// The UTCI is the equivalent temperature for the environment derived from a reference
/// environment. It is defined as the air temperature of the reference environment which
/// produces the same strain index value in comparison with the reference individual's
/// response to the real environment.
///
/// UTCI is regarded as one of the most comprehensive indices for calculating heat stress
/// in outdoor spaces, taking into account dry bulb temperature, mean radiation temperature,
/// water vapor pressure (via relative humidity), and wind speed at 10m elevation.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (recommended range: -50 to 50°C)
/// * `mean_radiant_temp` - Mean radiant temperature (recommended range: tdb-70 to tdb+30°C)
/// * `wind_speed` - Wind speed at 10m above ground (recommended range: 0.5-17 m/s)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `options` - UTCI calculation options
///
/// # Returns
///
/// UtciResult containing UTCI value and stress category. Returns NaN for UTCI if inputs
/// are outside valid ranges and limit_inputs is true.
///
/// # Applicability Limits (when limit_inputs = true)
///
/// * -50 < tdb [°C] < 50
/// * tdb - 70 < tr [°C] < tdb + 30
/// * 0.5 < v [m/s] < 17.0
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::utci::{utci, UtciOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let result = utci(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(1.0),
///     Humidity::from_percent(50.0),
///     Default::default()
/// );
/// println!("UTCI: {:.1}°C", result.utci);
/// println!("Stress: {}", result.stress_category.as_str());
/// ```
pub fn utci(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    wind_speed: Speed,
    relative_humidity: Humidity,
    options: UtciOptions,
) -> UtciResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let wind_speed_mps = wind_speed.as_meters_per_second();
    let rh_percent = relative_humidity.as_percent();

    // Calculate saturation vapor pressure using exponential formula
    let tk = dry_bulb_celsius + 273.15; // air temp in K

    let g = [
        -2836.5744,
        -6028.076559,
        19.54263612,
        -0.02737830188,
        0.000016261698,
        7.0229056e-10,
        -1.8680009e-13,
    ];

    let mut es = 2.7150305 * libm::log1p(tk);
    for (i, &coef) in g.iter().enumerate() {
        es += coef * pow(tk, (i as i32 - 2) as f64);
    }
    es = exp(es) * 0.01; // convert Pa to hPa

    let eh_pa = es * (rh_percent / 100.0);
    let delta_t_tr = radiant_celsius - dry_bulb_celsius;
    let pa = eh_pa / 10.0; // convert vapour pressure to kPa

    // Calculate UTCI using polynomial regression
    let mut utci_value = utci_polynomial(dry_bulb_celsius, wind_speed_mps, delta_t_tr, pa);

    // Check validity if requested
    if options.limit_inputs {
        if !(-50.0..=50.0).contains(&dry_bulb_celsius) {
            utci_value = f64::NAN;
        }
        if !(-30.0..=70.0).contains(&delta_t_tr) {
            utci_value = f64::NAN;
        }
        if !(0.5..=17.0).contains(&wind_speed_mps) {
            utci_value = f64::NAN;
        }
    }

    // Round if requested
    if options.round_output && !utci_value.is_nan() {
        utci_value = libm::round(utci_value * 10.0) / 10.0;
    }

    let stress_category = StressCategory::from_utci(utci_value);

    UtciResult {
        utci: utci_value,
        stress_category,
    }
}

/// Core UTCI polynomial calculation
///
/// This is a 6th-order polynomial regression model with interaction terms.
#[inline]
fn utci_polynomial(tdb: f64, v: f64, delta_t_tr: f64, pa: f64) -> f64 {
    // Pre-compute powers for efficiency
    let tdb2 = tdb * tdb;
    let tdb3 = tdb2 * tdb;
    let tdb4 = tdb3 * tdb;
    let tdb5 = tdb4 * tdb;
    let tdb6 = tdb5 * tdb;

    let v2 = v * v;
    let v3 = v2 * v;
    let v4 = v3 * v;
    let v5 = v4 * v;
    let v6 = v5 * v;

    let d = delta_t_tr;
    let d2 = d * d;
    let d3 = d2 * d;
    let d4 = d3 * d;
    let d5 = d4 * d;
    let d6 = d5 * d;

    let pa2 = pa * pa;
    let pa3 = pa2 * pa;
    let pa4 = pa3 * pa;
    let pa5 = pa4 * pa;
    let pa6 = pa5 * pa;

    tdb + 0.607562052
        + (-0.0227712343) * tdb
        + (8.06470249e-4) * tdb2
        + (-1.54271372e-4) * tdb3
        + (-3.24651735e-6) * tdb4
        + (7.32602852e-8) * tdb5
        + (1.35959073e-9) * tdb6
        + (-2.25836520) * v
        + 0.0880326035 * tdb * v
        + 0.00216844454 * tdb2 * v
        + (-1.53347087e-5) * tdb3 * v
        + (-5.72983704e-7) * tdb4 * v
        + (-2.55090145e-9) * tdb5 * v
        + (-0.751269505) * v2
        + (-0.00408350271) * tdb * v2
        + (-5.21670675e-5) * tdb2 * v2
        + (1.94544667e-6) * tdb3 * v2
        + (1.14099531e-8) * tdb4 * v2
        + 0.158137256 * v3
        + (-6.57263143e-5) * tdb * v3
        + (2.22697524e-7) * tdb2 * v3
        + (-4.16117031e-8) * tdb3 * v3
        + (-0.0127762753) * v4
        + (9.66891875e-6) * tdb * v4
        + (2.52785852e-9) * tdb2 * v4
        + (4.56306672e-4) * v5
        + (-1.74202546e-7) * tdb * v5
        + (-5.91491269e-6) * v6
        + 0.398374029 * d
        + (1.83945314e-4) * tdb * d
        + (-1.73754510e-4) * tdb2 * d
        + (-7.60781159e-7) * tdb3 * d
        + (3.77830287e-8) * tdb4 * d
        + (5.43079673e-10) * tdb5 * d
        + (-0.0200518269) * v * d
        + (8.92859837e-4) * tdb * v * d
        + (3.45433048e-6) * tdb2 * v * d
        + (-3.77925774e-7) * tdb3 * v * d
        + (-1.69699377e-9) * tdb4 * v * d
        + (1.69992415e-4) * v2 * d
        + (-4.99204314e-5) * tdb * v2 * d
        + (2.47417178e-7) * tdb2 * v2 * d
        + (1.07596466e-8) * tdb3 * v2 * d
        + (8.49242932e-5) * v3 * d
        + (1.35191328e-6) * tdb * v3 * d
        + (-6.21531254e-9) * tdb2 * v3 * d
        + (-4.99410301e-6) * v4 * d
        + (-1.89489258e-8) * tdb * v4 * d
        + (8.15300114e-8) * v5 * d
        + (7.55043090e-4) * d2
        + (-5.65095215e-5) * tdb * d2
        + (-4.52166564e-7) * tdb2 * d2
        + (2.46688878e-8) * tdb3 * d2
        + (2.42674348e-10) * tdb4 * d2
        + (1.54547250e-4) * v * d2
        + (5.24110970e-6) * tdb * v * d2
        + (-8.75874982e-8) * tdb2 * v * d2
        + (-1.50743064e-9) * tdb3 * v * d2
        + (-1.56236307e-5) * v2 * d2
        + (-1.33895614e-7) * tdb * v2 * d2
        + (2.49709824e-9) * tdb2 * v2 * d2
        + (6.51711721e-7) * v3 * d2
        + (1.94960053e-9) * tdb * v3 * d2
        + (-1.00361113e-8) * v4 * d2
        + (-1.21206673e-5) * d3
        + (-2.18203660e-7) * tdb * d3
        + (7.51269482e-9) * tdb2 * d3
        + (9.79063848e-11) * tdb3 * d3
        + (1.25006734e-6) * v * d3
        + (-1.81584736e-9) * tdb * v * d3
        + (-3.52197671e-10) * tdb2 * v * d3
        + (-3.36514630e-8) * v2 * d3
        + (1.35908359e-10) * tdb * v2 * d3
        + (4.17032620e-10) * v3 * d3
        + (-1.30369025e-9) * d4
        + (4.13908461e-10) * tdb * d4
        + (9.22652254e-12) * tdb2 * d4
        + (-5.08220384e-9) * v * d4
        + (-2.24730961e-11) * tdb * v * d4
        + (1.17139133e-10) * v2 * d4
        + (6.62154879e-10) * d5
        + (4.03863260e-13) * tdb * d5
        + (1.95087203e-12) * v * d5
        + (-4.73602469e-12) * d6
        + 5.12733497 * pa
        + (-0.312788561) * tdb * pa
        + (-0.0196701861) * tdb2 * pa
        + (9.99690870e-4) * tdb3 * pa
        + (9.51738512e-6) * tdb4 * pa
        + (-4.66426341e-7) * tdb5 * pa
        + 0.548050612 * v * pa
        + (-0.00330552823) * tdb * v * pa
        + (-0.00164119440) * tdb2 * v * pa
        + (-5.16670694e-6) * tdb3 * v * pa
        + (9.52692432e-7) * tdb4 * v * pa
        + (-0.0429223622) * v2 * pa
        + 0.00500845667 * tdb * v2 * pa
        + (1.00601257e-6) * tdb2 * v2 * pa
        + (-1.81748644e-6) * tdb3 * v2 * pa
        + (-1.25813502e-3) * v3 * pa
        + (-1.79330391e-4) * tdb * v3 * pa
        + (2.34994441e-6) * tdb2 * v3 * pa
        + (1.29735808e-4) * v4 * pa
        + (1.29064870e-6) * tdb * v4 * pa
        + (-2.28558686e-6) * v5 * pa
        + (-0.0369476348) * d * pa
        + 0.00162325322 * tdb * d * pa
        + (-3.14279680e-5) * tdb2 * d * pa
        + (2.59835559e-6) * tdb3 * d * pa
        + (-4.77136523e-8) * tdb4 * d * pa
        + (8.64203390e-3) * v * d * pa
        + (-6.87405181e-4) * tdb * v * d * pa
        + (-9.13863872e-6) * tdb2 * v * d * pa
        + (5.15916806e-7) * tdb3 * v * d * pa
        + (-3.59217476e-5) * v2 * d * pa
        + (3.28696511e-5) * tdb * v2 * d * pa
        + (-7.10542454e-7) * tdb2 * v2 * d * pa
        + (-1.24382300e-5) * v3 * d * pa
        + (-7.38584400e-9) * tdb * v3 * d * pa
        + (2.20609296e-7) * v4 * d * pa
        + (-7.32469180e-4) * d2 * pa
        + (-1.87381964e-5) * tdb * d2 * pa
        + (4.80925239e-6) * tdb2 * d2 * pa
        + (-8.75492040e-8) * tdb3 * d2 * pa
        + (2.77862930e-5) * v * d2 * pa
        + (-5.06004592e-6) * tdb * v * d2 * pa
        + (1.14325367e-7) * tdb2 * v * d2 * pa
        + (2.53016723e-6) * v2 * d2 * pa
        + (-1.72857035e-8) * tdb * v2 * d2 * pa
        + (-3.95079398e-8) * v3 * d2 * pa
        + (-3.59413173e-7) * d3 * pa
        + (7.04388046e-7) * tdb * d3 * pa
        + (-1.89309167e-8) * tdb2 * d3 * pa
        + (-4.79768731e-7) * v * d3 * pa
        + (7.96079978e-9) * tdb * v * d3 * pa
        + (1.62897058e-9) * v2 * d3 * pa
        + (3.94367674e-8) * d4 * pa
        + (-1.18566247e-9) * tdb * d4 * pa
        + (3.34678041e-10) * v * d4 * pa
        + (-1.15606447e-10) * d5 * pa
        + (-2.80626406) * pa2
        + 0.548712484 * tdb * pa2
        + (-0.00399428410) * tdb2 * pa2
        + (-9.54009191e-4) * tdb3 * pa2
        + (1.93090978e-5) * tdb4 * pa2
        + (-0.308806365) * v * pa2
        + 0.0116952364 * tdb * v * pa2
        + (4.95271903e-4) * tdb2 * v * pa2
        + (-1.90710882e-5) * tdb3 * v * pa2
        + 0.00210787756 * v2 * pa2
        + (-6.98445738e-4) * tdb * v2 * pa2
        + (2.30109073e-5) * tdb2 * v2 * pa2
        + (4.17856590e-4) * v3 * pa2
        + (-1.27043871e-5) * tdb * v3 * pa2
        + (-3.04620472e-6) * v4 * pa2
        + 0.0514507424 * d * pa2
        + (-0.00432510997) * tdb * d * pa2
        + (8.99281156e-5) * tdb2 * d * pa2
        + (-7.14663943e-7) * tdb3 * d * pa2
        + (-2.66016305e-4) * v * d * pa2
        + (2.63789586e-4) * tdb * v * d * pa2
        + (-7.01199003e-6) * tdb2 * v * d * pa2
        + (-1.06823306e-4) * v2 * d * pa2
        + (3.61341136e-6) * tdb * v2 * d * pa2
        + (2.29748967e-7) * v3 * d * pa2
        + (3.04788893e-4) * d2 * pa2
        + (-6.42070836e-5) * tdb * d2 * pa2
        + (1.16257971e-6) * tdb2 * d2 * pa2
        + (7.68023384e-6) * v * d2 * pa2
        + (-5.47446896e-7) * tdb * v * d2 * pa2
        + (-3.59937910e-8) * v2 * d2 * pa2
        + (-4.36497725e-6) * d3 * pa2
        + (1.68737969e-7) * tdb * d3 * pa2
        + (2.67489271e-8) * v * d3 * pa2
        + (3.23926897e-9) * d4 * pa2
        + (-0.0353874123) * pa3
        + (-0.221201190) * tdb * pa3
        + 0.0155126038 * tdb2 * pa3
        + (-2.63917279e-4) * tdb3 * pa3
        + 0.0453433455 * v * pa3
        + (-0.00432943862) * tdb * v * pa3
        + (1.45389826e-4) * tdb2 * v * pa3
        + (2.17508610e-4) * v2 * pa3
        + (-6.66724702e-5) * tdb * v2 * pa3
        + (3.33217140e-5) * v3 * pa3
        + (-0.00226921615) * d * pa3
        + (3.80261982e-4) * tdb * d * pa3
        + (-5.45314314e-9) * tdb2 * d * pa3
        + (-7.96355448e-4) * v * d * pa3
        + (2.53458034e-5) * tdb * v * d * pa3
        + (-6.31223658e-6) * v2 * d * pa3
        + (3.02122035e-4) * d2 * pa3
        + (-4.77403547e-6) * tdb * d2 * pa3
        + (1.73825715e-6) * v * d2 * pa3
        + (-4.09087898e-7) * d3 * pa3
        + 0.614155345 * pa4
        + (-0.0616755931) * tdb * pa4
        + 0.00133374846 * tdb2 * pa4
        + 0.00355375387 * v * pa4
        + (-5.13027851e-4) * tdb * v * pa4
        + (1.02449757e-4) * v2 * pa4
        + (-0.00148526421) * d * pa4
        + (-4.11469183e-5) * tdb * d * pa4
        + (-6.80434415e-6) * v * d * pa4
        + (-9.77675906e-6) * d2 * pa4
        + 0.0882773108 * pa5
        + (-0.00301859306) * tdb * pa5
        + 0.00104452989 * v * pa5
        + (2.47090539e-4) * d * pa5
        + 0.00148348065 * pa6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utci_basic() {
        let result = utci(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(1.0),
            Humidity::from_percent(50.0),
            Default::default(),
        );
        assert!(result.utci > 20.0 && result.utci < 30.0);
        assert_eq!(result.stress_category, StressCategory::NoThermalStress);
    }

    #[test]
    fn test_utci_cold() {
        let result = utci(
            Temperature::from_celsius(-10.0),
            Temperature::from_celsius(-10.0),
            Speed::from_meters_per_second(2.0),
            Humidity::from_percent(50.0),
            Default::default(),
        );
        assert!(result.utci < 0.0);
        assert!(matches!(
            result.stress_category,
            StressCategory::StrongColdStress | StressCategory::ModerateColdStress
        ));
    }

    #[test]
    fn test_utci_hot() {
        let result = utci(
            Temperature::from_celsius(35.0),
            Temperature::from_celsius(35.0),
            Speed::from_meters_per_second(1.0),
            Humidity::from_percent(50.0),
            Default::default(),
        );
        assert!(result.utci > 30.0);
        assert!(matches!(
            result.stress_category,
            StressCategory::ModerateHeatStress
                | StressCategory::StrongHeatStress
                | StressCategory::VeryStrongHeatStress
        ));
    }

    #[test]
    fn test_utci_limits() {
        // Test invalid tdb
        let result = utci(
            Temperature::from_celsius(-60.0),
            Temperature::from_celsius(-60.0),
            Speed::from_meters_per_second(1.0),
            Humidity::from_percent(50.0),
            Default::default(),
        );
        assert!(result.utci.is_nan());

        // Test invalid wind speed
        let result = utci(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.2),
            Humidity::from_percent(50.0),
            Default::default(),
        );
        assert!(result.utci.is_nan());

        // Test with limits off
        let options = UtciOptions {
            limit_inputs: false,
            round_output: true,
        };
        let result = utci(
            Temperature::from_celsius(-60.0),
            Temperature::from_celsius(-60.0),
            Speed::from_meters_per_second(1.0),
            Humidity::from_percent(50.0),
            options,
        );
        assert!(!result.utci.is_nan());
    }

    #[test]
    fn test_stress_categories() {
        assert_eq!(
            StressCategory::from_utci(-45.0),
            StressCategory::ExtremeColdStress
        );
        assert_eq!(
            StressCategory::from_utci(-30.0),
            StressCategory::VeryStrongColdStress
        );
        assert_eq!(
            StressCategory::from_utci(-15.0),
            StressCategory::StrongColdStress
        );
        assert_eq!(
            StressCategory::from_utci(-5.0),
            StressCategory::ModerateColdStress
        );
        assert_eq!(
            StressCategory::from_utci(5.0),
            StressCategory::SlightColdStress
        );
        assert_eq!(
            StressCategory::from_utci(20.0),
            StressCategory::NoThermalStress
        );
        assert_eq!(
            StressCategory::from_utci(30.0),
            StressCategory::ModerateHeatStress
        );
        assert_eq!(
            StressCategory::from_utci(35.0),
            StressCategory::StrongHeatStress
        );
        assert_eq!(
            StressCategory::from_utci(42.0),
            StressCategory::VeryStrongHeatStress
        );
        assert_eq!(
            StressCategory::from_utci(50.0),
            StressCategory::ExtremeHeatStress
        );
    }
}
