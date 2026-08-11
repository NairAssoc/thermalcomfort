//! # Sports Heat Stress Risk
//!
//! Calculate sports heat stress risk levels based on environmental conditions and
//! sport-specific parameters.
//!
//! This module assesses heat stress risk for athletes during outdoor sports by
//! combining environmental conditions with sport-specific metabolic rates and clothing
//! insulation. It uses the Predicted Heat Strain (PHS) model to determine threshold
//! temperatures for different risk categories (Low, Medium, High, Extreme).
//!
//! Based on the Sports Medicine Australia heat policy framework.
//!
//! ## Risk Levels
//!
//! - **1.0 - 2.0**: Low risk - Increase hydration & modify clothing
//! - **2.0 - 3.0**: Moderate risk - Increase frequency/duration of rest breaks
//! - **3.0 - 4.0**: High risk - Apply active cooling strategies
//! - **4.0 - 4.9**: Extreme risk - Consider suspending play. The level ramps from 4.0 at
//!   `t_extreme` to 4.9 at `t_extreme + 5°C`, and is capped there.
//!
//! ## References
//!
//! - Sports Medicine Australia heat policy framework
//! - ISO 7933 (PHS model used internally)

use crate::models::phs::{Iso7933Model, PhsInputs, PhsOptions, PhsPosture, phs};
use crate::numerical::brentq;
use crate::utilities::{np_maximum, py_min, round_to_exact_decimal};
use crate::{ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature};

/// The comfort inputs to [`sports_heat_stress_risk`]: pythermalcomfort requires all five
/// (no default -- Python has zero optional parameters for this model).
///
/// `tdb` and `tr` are consecutive [`Temperature`]s; naming every field forecloses a
/// silent transposition between them.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SportsHeatStressRiskInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Relative humidity
    pub rh: Humidity,
    /// Relative air speed
    pub vr: Speed,
    /// Sport-specific parameters (use a constant from [`Sports`])
    pub sport: SportsValues,
}

/// Sport-specific parameters for heat stress risk calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SportsValues {
    /// Clothing insulation
    pub clo: ClothingInsulation,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Relative air speed [m/s]
    pub vr: f64,
    /// Activity duration \[minutes\]
    pub duration: i32,
}

impl SportsValues {
    /// Create a new sport-specific parameter set.
    ///
    /// # Panics
    ///
    /// Panics if clo, met, or vr are not positive, or if duration is negative.
    pub const fn new(clo: ClothingInsulation, met: MetabolicRate, vr: f64, duration: i32) -> Self {
        Self {
            clo,
            met,
            vr,
            duration,
        }
    }
}

/// Predefined sport-specific parameters.
///
/// Each constant provides the clothing insulation, metabolic rate,
/// relative air speed (m/s), and typical activity duration (minutes) for that sport.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::sports_heat_stress_risk::{
///     Sports, SportsHeatStressRiskInputs, sports_heat_stress_risk,
/// };
/// use thermalcomfort::{Temperature, Humidity, Speed};
///
/// let result = sports_heat_stress_risk(SportsHeatStressRiskInputs {
///     tdb: Temperature::from_celsius(35.0),
///     tr: Temperature::from_celsius(35.0),
///     rh: Humidity::from_percent(40.0),
///     vr: Speed::from_meters_per_second(0.1),
///     sport: Sports::RUNNING,
/// });
/// // vr=0.1 is clamped to sport minimum (2.0 for running)
/// assert_eq!(result.risk_level_interpolated, 2.1);
/// ```
pub struct Sports;

impl Sports {
    pub const ABSEILING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.6),
        MetabolicRate::from_met(6.0),
        0.5,
        120,
    );
    pub const ARCHERY: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.75),
        MetabolicRate::from_met(4.5),
        0.5,
        180,
    );
    pub const AUSTRALIAN_FOOTBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.47),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const BASEBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.7),
        MetabolicRate::from_met(6.0),
        0.75,
        120,
    );
    pub const BASKETBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.37),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const BOWLS: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.5),
        MetabolicRate::from_met(5.0),
        0.5,
        180,
    );
    pub const CANOEING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.6),
        MetabolicRate::from_met(7.5),
        2.0,
        60,
    );
    pub const CRICKET: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.7),
        MetabolicRate::from_met(6.0),
        0.75,
        120,
    );
    pub const CROQUET: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.7),
        MetabolicRate::from_met(4.5),
        0.5,
        90,
    );
    pub const CYCLING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.4),
        MetabolicRate::from_met(7.0),
        3.0,
        60,
    );
    pub const EQUESTRIAN: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.9),
        MetabolicRate::from_met(7.4),
        3.0,
        60,
    );
    pub const FIELD_ATHLETICS: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.3),
        MetabolicRate::from_met(7.0),
        1.0,
        60,
    );
    pub const FIELD_HOCKEY: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.6),
        MetabolicRate::from_met(7.4),
        0.75,
        45,
    );
    pub const FISHING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.9),
        MetabolicRate::from_met(4.0),
        0.5,
        180,
    );
    pub const GOLF: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.5),
        MetabolicRate::from_met(5.0),
        0.5,
        180,
    );
    pub const HORSEBACK: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.9),
        MetabolicRate::from_met(7.4),
        3.0,
        60,
    );
    pub const KAYAKING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.6),
        MetabolicRate::from_met(7.5),
        2.0,
        60,
    );
    pub const RUNNING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.37),
        MetabolicRate::from_met(7.5),
        2.0,
        60,
    );
    pub const MTB: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.55),
        MetabolicRate::from_met(7.5),
        3.0,
        60,
    );
    pub const NETBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.37),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const OZTAG: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.4),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const PICKLEBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.4),
        MetabolicRate::from_met(6.5),
        0.5,
        60,
    );
    pub const CLIMBING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.6),
        MetabolicRate::from_met(7.5),
        1.0,
        45,
    );
    pub const ROWING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.4),
        MetabolicRate::from_met(7.5),
        2.0,
        60,
    );
    pub const RUGBY_LEAGUE: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.47),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const RUGBY_UNION: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.47),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const SAILING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(1.0),
        MetabolicRate::from_met(6.5),
        2.0,
        180,
    );
    pub const SHOOTING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.6),
        MetabolicRate::from_met(5.0),
        0.5,
        120,
    );
    pub const SOCCER: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.47),
        MetabolicRate::from_met(7.5),
        1.0,
        45,
    );
    pub const SOFTBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.9),
        MetabolicRate::from_met(6.1),
        1.0,
        120,
    );
    pub const TENNIS: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.4),
        MetabolicRate::from_met(7.0),
        0.75,
        60,
    );
    pub const TOUCH: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.4),
        MetabolicRate::from_met(7.5),
        0.75,
        45,
    );
    pub const VOLLEYBALL: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.37),
        MetabolicRate::from_met(6.8),
        0.75,
        60,
    );
    pub const WALKING: SportsValues = SportsValues::new(
        ClothingInsulation::from_clo(0.5),
        MetabolicRate::from_met(5.0),
        0.5,
        180,
    );
}

/// Result of sports heat stress risk calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct SportsHeatStressRisk {
    /// Interpolated risk level (1.0-4.9), truncated to one decimal place.
    /// Risk levels: 1-2 = low, 2-3 = moderate, 3-4 = high, 4 = extreme.
    pub risk_level_interpolated: f64,
    /// Temperature threshold for medium risk level
    pub t_medium: Temperature,
    /// Temperature threshold for high risk level
    pub t_high: Temperature,
    /// Temperature threshold for extreme risk level
    pub t_extreme: Temperature,
    /// Heat stress management recommendation
    pub recommendation: &'static str,
}

// Risk level threshold constants
const MAX_T_LOW: f64 = 34.5;
const MAX_T_MEDIUM: f64 = 39.0;
const MAX_T_HIGH: f64 = 43.5;
const MIN_T_LOW: f64 = 21.0;
const MIN_T_MEDIUM: f64 = 23.0;
const MIN_T_HIGH: f64 = 25.0;
const MIN_T_EXTREME: f64 = 26.0;
/// Width of the extreme band above `t_extreme` over which risk ramps 4.0 -> 4.9 [°C]
const T_UPPER_EXTREME_DELTA: f64 = 5.0;

const SWEAT_LOSS_G: f64 = 850.0; // g per hour
const T_CR_EXTREME: f64 = 40.0; // core temperature for extreme risk

/// Get recommendation text for a given risk level.
fn get_recommendation(risk_level: f64) -> &'static str {
    if risk_level < 2.0 {
        "Increase hydration & modify clothing"
    } else if risk_level < 3.0 {
        "Increase frequency and/or duration of rest breaks"
    } else if risk_level < 4.0 {
        "Apply active cooling strategies"
    } else {
        "Consider suspending play"
    }
}

/// Run PHS and return sweat loss [g]
fn phs_sweat_loss(tdb: f64, tr: f64, rh: f64, vr: f64, sport: &SportsValues) -> f64 {
    let result = phs(
        PhsInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(vr),
            rh: Humidity::from_percent(rh),
            met: sport.met,
            clo: sport.clo,
            posture: PhsPosture::Standing,
        },
        PhsOptions {
            duration: sport.duration,
            round_output: false,
            limit_inputs: false,
            acclimatized: true,
            i_mst: 0.4,
            model: Iso7933Model::Iso2023,
            ..Default::default()
        },
    );
    result.sweat_loss_g.as_grams()
}

/// Run PHS and return core temperature [°C]
fn phs_core_temp(tdb: f64, tr: f64, rh: f64, vr: f64, sport: &SportsValues) -> f64 {
    let result = phs(
        PhsInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(vr),
            rh: Humidity::from_percent(rh),
            met: sport.met,
            clo: sport.clo,
            posture: PhsPosture::Standing,
        },
        PhsOptions {
            duration: sport.duration,
            round_output: false,
            limit_inputs: false,
            acclimatized: true,
            i_mst: 0.4,
            model: Iso7933Model::Iso2023,
            ..Default::default()
        },
    );
    result.t_cr.as_celsius()
}

/// Floor-truncate to 1 decimal place toward negative infinity
fn floor1(x: f64) -> f64 {
    libm::floor(x * 10.0) / 10.0
}

/// Calculate sports heat stress risk levels based on environmental conditions and
/// sport-specific parameters.
///
/// This function assesses heat stress risk for athletes during outdoor sports by
/// combining environmental conditions with sport-specific metabolic rates and clothing
/// insulation. It uses the Predicted Heat Strain (PHS) model to determine threshold
/// temperatures for different risk categories (Low, Medium, High, Extreme). The method
/// is based on the Sports Medicine Australia heat policy framework.
///
/// # Arguments
///
/// * `inputs` - Required comfort inputs, see [`SportsHeatStressRiskInputs`]
///
/// # Returns
///
/// [`SportsHeatStressRisk`] containing:
/// - Risk level (1.0-4.9, truncated to one decimal place)
/// - Temperature thresholds for medium, high, and extreme risk
/// - Recommendation text
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::sports_heat_stress_risk::{
///     Sports, SportsHeatStressRiskInputs, sports_heat_stress_risk,
/// };
/// use thermalcomfort::{Temperature, Humidity, Speed};
///
/// // Running at 35°C, 40% RH
/// let result = sports_heat_stress_risk(SportsHeatStressRiskInputs {
///     tdb: Temperature::from_celsius(35.0),
///     tr: Temperature::from_celsius(35.0),
///     rh: Humidity::from_percent(40.0),
///     vr: Speed::from_meters_per_second(0.1),
///     sport: Sports::RUNNING,
/// });
/// assert_eq!(result.risk_level_interpolated, 2.1);
/// assert!((result.t_medium.as_celsius() - 34.5).abs() < 1e-9);
/// assert!((result.t_extreme.as_celsius() - 41.6).abs() < 1e-9);
/// assert_eq!(result.recommendation, "Increase frequency and/or duration of rest breaks");
///
/// // Soccer at moderate conditions
/// let result = sports_heat_stress_risk(SportsHeatStressRiskInputs {
///     tdb: Temperature::from_celsius(30.0),
///     tr: Temperature::from_celsius(30.0),
///     rh: Humidity::from_percent(50.0),
///     vr: Speed::from_meters_per_second(0.5),
///     sport: Sports::SOCCER,
/// });
/// assert!(result.risk_level_interpolated < 2.0); // Low risk
/// ```
///
/// # References
///
/// - Sports Medicine Australia heat policy framework
/// - ISO 7933 (PHS model used internally for threshold calculation)
pub fn sports_heat_stress_risk(inputs: SportsHeatStressRiskInputs) -> SportsHeatStressRisk {
    let SportsHeatStressRiskInputs {
        tdb,
        tr,
        rh,
        vr,
        sport,
    } = inputs;
    let tdb_c = tdb.as_celsius();
    let tr_c = tr.as_celsius();
    let rh_pct = rh.as_percent();
    // Enforce sport-specific minimum air speed.
    // `sports_heat_stress_risk.py:199` is `np.maximum(vr, sport.vr)`, which propagates a
    // NaN `vr`. The `if v < sport.vr` form previously written here happened to agree, but
    // spelling the construct out keeps the whole file on one min/max vocabulary.
    let vr_ms = np_maximum(vr.as_meters_per_second(), sport.vr);

    // Early returns for temperatures outside the threshold range
    if tdb_c < MIN_T_MEDIUM {
        return SportsHeatStressRisk {
            risk_level_interpolated: 1.0,
            t_medium: Temperature::from_celsius(MIN_T_MEDIUM),
            t_high: Temperature::from_celsius(MIN_T_HIGH),
            t_extreme: Temperature::from_celsius(MIN_T_EXTREME),
            recommendation: get_recommendation(1.0),
        };
    }

    // Find t_medium: temperature where sweat loss rate equals threshold
    let t_medium = find_threshold_water_loss(tr_c, rh_pct, vr_ms, &sport);

    // Find t_extreme: temperature where core temperature reaches t_cr_extreme
    let t_extreme = find_threshold_core_temp(tr_c, rh_pct, vr_ms, &sport);

    // t_high is the average of t_medium and t_extreme
    let mut t_high = if t_medium.is_nan() || t_extreme.is_nan() {
        f64::NAN
    } else {
        (t_medium + t_extreme) / 2.0
    };

    // Clamp thresholds to max limits
    let mut t_medium = if t_medium > MAX_T_LOW {
        MAX_T_LOW
    } else {
        t_medium
    };
    if t_high > MAX_T_MEDIUM {
        t_high = MAX_T_MEDIUM;
    }
    let mut t_extreme = if t_extreme > MAX_T_HIGH {
        MAX_T_HIGH
    } else {
        t_extreme
    };

    // Clamp thresholds to min limits
    if t_extreme < MIN_T_EXTREME {
        t_extreme = MIN_T_EXTREME;
    }
    if t_high < MIN_T_HIGH {
        t_high = MIN_T_HIGH;
    }
    if t_medium < MIN_T_MEDIUM {
        t_medium = MIN_T_MEDIUM;
    }

    // The extreme band is entered at the *rounded* t_extreme — the same value returned
    // to callers — so the reported threshold and the risk level stay consistent.
    //
    // `sports_heat_stress_risk.py:356` is the builtin `min(round(t_extreme, 1),
    // max_t_high)`, so a NaN threshold survives. That matters beyond the returned value:
    // in Python a NaN here makes every risk-band comparison false, leaving
    // `risk_level_interpolated` NaN and raising `ValueError`. `f64::min` returned
    // `MAX_T_HIGH` instead, manufacturing a finite risk level for an unsolved threshold.
    let extreme_entry_t = py_min(round_to_exact_decimal(t_extreme, 1), MAX_T_HIGH);

    // Calculate interpolated risk level (1.0-4.9 scale)
    let risk_level = if MIN_T_LOW <= tdb_c && tdb_c < t_medium {
        1.0 + (tdb_c - MIN_T_MEDIUM) / (t_medium - MIN_T_MEDIUM)
    } else if t_medium <= tdb_c && tdb_c < t_high {
        2.0 + (tdb_c - t_medium) / (t_high - t_medium)
    } else if t_high <= tdb_c && tdb_c < extreme_entry_t {
        3.0 + (tdb_c - t_high) / (extreme_entry_t - t_high)
    } else {
        // tdb >= extreme_entry_t. Scale to [4.0, 4.9] so risk reaches 4.9 exactly at
        // extreme_entry_t + T_UPPER_EXTREME_DELTA. Without the 0.9 factor the formula
        // would hit 4.9 already at +4.5°C, leaving the last 0.5°C of the range dead.
        4.0 + (tdb_c - extreme_entry_t) / T_UPPER_EXTREME_DELTA * 0.9
    };

    // Floor-truncate to one decimal place. The 1e-9 epsilon guards against the float
    // representation of 4.9 (e.g. 4.8999…) flooring to 4.8.
    // `sports_heat_stress_risk.py:377`: builtin `min(np.floor(...) / 10.0, 4.9)`, which
    // keeps a NaN risk level rather than reporting a confident 4.9.
    let risk_level_floor = py_min(floor1(risk_level + 1e-9), 4.9);

    SportsHeatStressRisk {
        risk_level_interpolated: risk_level_floor,
        t_medium: Temperature::from_celsius(round_to_exact_decimal(t_medium, 1)),
        t_high: Temperature::from_celsius(round_to_exact_decimal(t_high, 1)),
        t_extreme: Temperature::from_celsius(round_to_exact_decimal(t_extreme, 1)),
        recommendation: get_recommendation(risk_level_floor),
    }
}

/// Find temperature threshold for water loss (medium risk boundary).
fn find_threshold_water_loss(tr: f64, rh: f64, vr: f64, sport: &SportsValues) -> f64 {
    let duration = sport.duration as f64;
    let target = |x: f64| -> f64 {
        let sl = phs_sweat_loss(x, tr, rh, vr, sport);
        sl / duration * 45.0 - SWEAT_LOSS_G
    };

    // Try two bracket ranges, matching Python
    for &(min_t, max_t) in &[(0.0, 36.0), (20.0, 50.0)] {
        if let Ok(root) = brentq(target, min_t, max_t, None, None) {
            return root;
        }
    }

    // Fallback to max threshold
    MAX_T_LOW
}

/// Find temperature threshold for core temperature (extreme risk boundary).
fn find_threshold_core_temp(tr: f64, rh: f64, vr: f64, sport: &SportsValues) -> f64 {
    let target = |x: f64| -> f64 {
        let tcr = phs_core_temp(x, tr, rh, vr, sport);
        tcr - T_CR_EXTREME
    };

    // Try two bracket ranges, matching Python
    for &(min_t, max_t) in &[(0.0, 36.0), (20.0, 50.0)] {
        if let Ok(root) = brentq(target, min_t, max_t, None, None) {
            return root;
        }
    }

    // Fallback to max threshold
    MAX_T_HIGH
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(
        tdb: f64,
        tr: f64,
        rh: f64,
        vr: f64,
        sport: SportsValues,
    ) -> SportsHeatStressRiskInputs {
        SportsHeatStressRiskInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            rh: Humidity::from_percent(rh),
            vr: Speed::from_meters_per_second(vr),
            sport,
        }
    }

    /// Compares a [`Temperature`] against a Celsius literal with a tiny tolerance.
    ///
    /// `Temperature` stores kelvin internally, so `from_celsius(x).as_celsius()` can lose
    /// the last ULP on the round trip; `assert_eq!` against a literal is too strict for
    /// values produced that way (e.g. 38.2 comes back as 38.19999999999999). This is a
    /// float-representation artifact of the newtype, not a difference worth pinning
    /// exactly.
    fn assert_temp_eq(actual: Temperature, expected_celsius: f64) {
        let actual_celsius = actual.as_celsius();
        assert!(
            (actual_celsius - expected_celsius).abs() < 1e-9,
            "expected {expected_celsius}, got {actual_celsius}"
        );
    }

    #[test]
    fn test_running_vr_clamped() {
        // vr=0.1 is clamped to sport minimum (2.0 for running)
        let result = sports_heat_stress_risk(inputs(35.0, 35.0, 40.0, 0.1, Sports::RUNNING));
        assert_eq!(result.risk_level_interpolated, 2.1);
        assert_temp_eq(result.t_medium, 34.5);
        assert_temp_eq(result.t_high, 39.0);
        assert_temp_eq(result.t_extreme, 41.6);
        assert_eq!(
            result.recommendation,
            "Increase frequency and/or duration of rest breaks"
        );
    }

    #[test]
    fn test_soccer_low_risk() {
        let result = sports_heat_stress_risk(inputs(30.0, 30.0, 50.0, 0.5, Sports::SOCCER));
        assert_eq!(result.risk_level_interpolated, 1.6);
        assert_temp_eq(result.t_medium, 34.5);
        assert_temp_eq(result.t_high, 38.2);
        assert_temp_eq(result.t_extreme, 39.9);
        assert_eq!(
            result.recommendation,
            "Increase hydration & modify clothing"
        );
    }

    #[test]
    fn test_low_temperature() {
        let result = sports_heat_stress_risk(inputs(20.0, 20.0, 50.0, 0.5, Sports::WALKING));
        assert_eq!(result.risk_level_interpolated, 1.0);
        assert_temp_eq(result.t_medium, 23.0);
        assert_temp_eq(result.t_high, 25.0);
        assert_temp_eq(result.t_extreme, 26.0);
        assert_eq!(
            result.recommendation,
            "Increase hydration & modify clothing"
        );
    }

    #[test]
    fn test_very_high_temperature() {
        let result = sports_heat_stress_risk(inputs(45.0, 45.0, 30.0, 0.5, Sports::CYCLING));
        // 45°C is 1.5°C above t_extreme (43.5), so risk ramps into the extreme band:
        // 4.0 + 1.5/5.0*0.9 = 4.27 -> floored to 4.2
        assert_eq!(result.risk_level_interpolated, 4.2);
        assert_temp_eq(result.t_medium, 34.5);
        assert_temp_eq(result.t_high, 39.0);
        assert_temp_eq(result.t_extreme, 43.5);
        assert_eq!(result.recommendation, "Consider suspending play");
    }

    #[test]
    fn test_tennis_high_radiant() {
        let result = sports_heat_stress_risk(inputs(33.0, 70.0, 60.0, 0.1, Sports::TENNIS));
        // 33°C is 3.5°C above t_extreme (29.5): 4.0 + 3.5/5.0*0.9 = 4.63 -> floored to 4.6
        assert_eq!(result.risk_level_interpolated, 4.6);
        assert_temp_eq(result.t_medium, 23.0);
        assert_temp_eq(result.t_high, 25.0);
        assert_temp_eq(result.t_extreme, 29.5);
        assert_eq!(result.recommendation, "Consider suspending play");
    }

    #[test]
    fn test_croquet_preset() {
        let result = sports_heat_stress_risk(inputs(35.0, 35.0, 40.0, 0.1, Sports::CROQUET));
        assert_eq!(result.risk_level_interpolated, 2.1);
        assert_temp_eq(result.t_medium, 34.5);
        assert_temp_eq(result.t_high, 39.0);
        assert_temp_eq(result.t_extreme, 43.3);
    }

    #[test]
    fn test_extreme_band_caps_at_4_9() {
        // Far above t_extreme the risk level must clamp at 4.9, not grow without bound.
        let result = sports_heat_stress_risk(inputs(70.0, 70.0, 30.0, 0.5, Sports::CYCLING));
        assert_eq!(result.risk_level_interpolated, 4.9);
    }

    #[test]
    fn test_sports_values() {
        assert_eq!(Sports::RUNNING.clo.as_clo(), 0.37);
        assert_eq!(Sports::RUNNING.met.as_met(), 7.5);
        assert_eq!(Sports::RUNNING.vr, 2.0);
        assert_eq!(Sports::RUNNING.duration, 60);
    }
}
