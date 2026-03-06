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
//! - **0.0 - 1.0**: Low risk - Increase hydration & modify clothing
//! - **1.0 - 2.0**: Moderate risk - Increase frequency/duration of rest breaks
//! - **2.0 - 3.0**: High risk - Apply active cooling strategies
//! - **3.0**: Extreme risk - Consider suspending play
//!
//! ## References
//!
//! - Sports Medicine Australia heat policy framework
//! - ISO 7933 (PHS model used internally)

use crate::models::phs::{Iso7933Model, PhsOptions, PhsPosture, phs};
use crate::numerical::brentq;
use crate::{ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature};

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
/// use thermalcomfort::models::sports_heat_stress_risk::{Sports, sports_heat_stress_risk};
/// use thermalcomfort::{Temperature, Humidity, Speed};
///
/// let result = sports_heat_stress_risk(
///     Temperature::from_celsius(35.0),
///     Temperature::from_celsius(35.0),
///     Humidity::from_percent(40.0),
///     Speed::from_meters_per_second(0.1),
///     Sports::RUNNING,
/// );
/// assert_eq!(result.risk_level_interpolated, 3.0);
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
    /// Interpolated risk level (0.0-3.0), truncated to one decimal place.
    /// Risk levels: 0-1 = low, 1-2 = moderate, 2-3 = high, 3+ = extreme.
    pub risk_level_interpolated: f64,
    /// Temperature threshold for medium risk level [°C]
    pub t_medium: f64,
    /// Temperature threshold for high risk level [°C]
    pub t_high: f64,
    /// Temperature threshold for extreme risk level [°C]
    pub t_extreme: f64,
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

const SWEAT_LOSS_G: f64 = 850.0; // g per hour
const T_CR_EXTREME: f64 = 40.0; // core temperature for extreme risk

/// Get recommendation text for a given risk level.
fn get_recommendation(risk_level: f64) -> &'static str {
    if risk_level < 1.0 {
        "Increase hydration & modify clothing"
    } else if risk_level < 2.0 {
        "Increase frequency and/or duration of rest breaks"
    } else if risk_level < 3.0 {
        "Apply active cooling strategies"
    } else {
        "Consider suspending play"
    }
}

/// Run PHS and return sweat loss [g]
fn phs_sweat_loss(tdb: f64, tr: f64, rh: f64, vr: f64, sport: &SportsValues) -> f64 {
    let result = phs(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(vr),
        Humidity::from_percent(rh),
        sport.met,
        sport.clo,
        PhsPosture::Standing,
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
    result.sweat_loss_g
}

/// Run PHS and return core temperature [°C]
fn phs_core_temp(tdb: f64, tr: f64, rh: f64, vr: f64, sport: &SportsValues) -> f64 {
    let result = phs(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(vr),
        Humidity::from_percent(rh),
        sport.met,
        sport.clo,
        PhsPosture::Standing,
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
    result.t_cr
}

/// Round to 1 decimal place (Python-compatible rounding)
fn round1(x: f64) -> f64 {
    libm::floor(x * 10.0 + 0.5) / 10.0
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
/// * `tdb` - Dry bulb air temperature
/// * `tr` - Mean radiant temperature
/// * `rh` - Relative humidity [%]
/// * `vr` - Relative air speed [m/s]
/// * `sport` - Sport-specific parameters (use a constant from [`Sports`])
///
/// # Returns
///
/// [`SportsHeatStressRisk`] containing:
/// - Risk level (0.0-3.0, truncated to one decimal place)
/// - Temperature thresholds for medium, high, and extreme risk
/// - Recommendation text
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::sports_heat_stress_risk::{Sports, sports_heat_stress_risk};
/// use thermalcomfort::{Temperature, Humidity, Speed};
///
/// // Running at 35°C, 40% RH
/// let result = sports_heat_stress_risk(
///     Temperature::from_celsius(35.0),
///     Temperature::from_celsius(35.0),
///     Humidity::from_percent(40.0),
///     Speed::from_meters_per_second(0.1),
///     Sports::RUNNING,
/// );
/// assert_eq!(result.risk_level_interpolated, 3.0);
/// assert_eq!(result.t_medium, 23.0);
/// assert_eq!(result.t_extreme, 28.6);
/// assert_eq!(result.recommendation, "Consider suspending play");
///
/// // Soccer at moderate conditions
/// let result = sports_heat_stress_risk(
///     Temperature::from_celsius(30.0),
///     Temperature::from_celsius(30.0),
///     Humidity::from_percent(50.0),
///     Speed::from_meters_per_second(0.5),
///     Sports::SOCCER,
/// );
/// assert!(result.risk_level_interpolated < 1.0); // Low risk
/// ```
///
/// # References
///
/// - Sports Medicine Australia heat policy framework
/// - ISO 7933 (PHS model used internally for threshold calculation)
pub fn sports_heat_stress_risk(
    tdb: Temperature,
    tr: Temperature,
    rh: Humidity,
    vr: Speed,
    sport: SportsValues,
) -> SportsHeatStressRisk {
    let tdb_c = tdb.as_celsius();
    let tr_c = tr.as_celsius();
    let rh_pct = rh.as_percent();
    let vr_ms = vr.as_meters_per_second();

    // Early returns for temperatures outside the threshold range
    if tdb_c < MIN_T_MEDIUM {
        return SportsHeatStressRisk {
            risk_level_interpolated: 0.0,
            t_medium: MIN_T_MEDIUM,
            t_high: MIN_T_HIGH,
            t_extreme: MIN_T_EXTREME,
            recommendation: get_recommendation(0.0),
        };
    }

    if tdb_c > MAX_T_HIGH {
        return SportsHeatStressRisk {
            risk_level_interpolated: 3.0,
            t_medium: MAX_T_LOW,
            t_high: MAX_T_MEDIUM,
            t_extreme: MAX_T_HIGH,
            recommendation: get_recommendation(3.0),
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

    // Calculate interpolated risk level
    let risk_level = if MIN_T_LOW <= tdb_c && tdb_c < t_medium {
        (tdb_c - MIN_T_MEDIUM) / (t_medium - MIN_T_MEDIUM)
    } else if t_medium <= tdb_c && tdb_c < t_high {
        1.0 + (tdb_c - t_medium) / (t_high - t_medium)
    } else if t_high <= tdb_c && tdb_c < t_extreme {
        2.0 + (tdb_c - t_high) / (t_extreme - t_high)
    } else {
        // tdb >= t_extreme
        3.0
    };

    // Floor-truncate to one decimal place
    let risk_level_floor = floor1(risk_level);

    SportsHeatStressRisk {
        risk_level_interpolated: risk_level_floor,
        t_medium: round1(t_medium),
        t_high: round1(t_high),
        t_extreme: round1(t_extreme),
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

    #[test]
    fn test_running_extreme() {
        let result = sports_heat_stress_risk(
            Temperature::from_celsius(35.0),
            Temperature::from_celsius(35.0),
            Humidity::from_percent(40.0),
            Speed::from_meters_per_second(0.1),
            Sports::RUNNING,
        );
        assert_eq!(result.risk_level_interpolated, 3.0);
        assert_eq!(result.t_medium, 23.0);
        assert_eq!(result.t_high, 25.0);
        assert_eq!(result.t_extreme, 28.6);
        assert_eq!(result.recommendation, "Consider suspending play");
    }

    #[test]
    fn test_soccer_low_risk() {
        let result = sports_heat_stress_risk(
            Temperature::from_celsius(30.0),
            Temperature::from_celsius(30.0),
            Humidity::from_percent(50.0),
            Speed::from_meters_per_second(0.5),
            Sports::SOCCER,
        );
        assert_eq!(result.risk_level_interpolated, 0.7);
        assert_eq!(result.t_medium, 32.7);
        assert_eq!(result.t_high, 34.9);
        assert_eq!(result.t_extreme, 37.1);
        assert_eq!(
            result.recommendation,
            "Increase hydration & modify clothing"
        );
    }

    #[test]
    fn test_low_temperature() {
        let result = sports_heat_stress_risk(
            Temperature::from_celsius(20.0),
            Temperature::from_celsius(20.0),
            Humidity::from_percent(50.0),
            Speed::from_meters_per_second(0.5),
            Sports::WALKING,
        );
        assert_eq!(result.risk_level_interpolated, 0.0);
        assert_eq!(result.t_medium, 23.0);
        assert_eq!(result.t_high, 25.0);
        assert_eq!(result.t_extreme, 26.0);
        assert_eq!(
            result.recommendation,
            "Increase hydration & modify clothing"
        );
    }

    #[test]
    fn test_very_high_temperature() {
        let result = sports_heat_stress_risk(
            Temperature::from_celsius(45.0),
            Temperature::from_celsius(45.0),
            Humidity::from_percent(30.0),
            Speed::from_meters_per_second(0.5),
            Sports::CYCLING,
        );
        assert_eq!(result.risk_level_interpolated, 3.0);
        assert_eq!(result.t_medium, 34.5);
        assert_eq!(result.t_high, 39.0);
        assert_eq!(result.t_extreme, 43.5);
        assert_eq!(result.recommendation, "Consider suspending play");
    }

    #[test]
    fn test_tennis_high_radiant() {
        let result = sports_heat_stress_risk(
            Temperature::from_celsius(33.0),
            Temperature::from_celsius(70.0),
            Humidity::from_percent(60.0),
            Speed::from_meters_per_second(0.1),
            Sports::TENNIS,
        );
        assert_eq!(result.risk_level_interpolated, 3.0);
        assert_eq!(result.t_medium, 23.0);
        assert_eq!(result.t_high, 25.0);
        assert_eq!(result.t_extreme, 26.0);
        assert_eq!(result.recommendation, "Consider suspending play");
    }

    #[test]
    fn test_sports_values() {
        assert_eq!(Sports::RUNNING.clo.as_clo(), 0.37);
        assert_eq!(Sports::RUNNING.met.as_met(), 7.5);
        assert_eq!(Sports::RUNNING.vr, 2.0);
        assert_eq!(Sports::RUNNING.duration, 60);
    }
}
