//! Domain-specific measurement types for thermal comfort
//!
//! This module provides strongly-typed wrappers for thermal comfort units
//! that aren't available in the standard `measurements` crate.

use core::fmt;
use core::ops::{Add, Sub, Mul, Div};

/// Metabolic rate in met units
///
/// 1 met = 58.15 W/m² (the energy produced per unit surface area of an average person
/// seated at rest)
///
/// # Examples
///
/// ```
/// use thermalcomfort::types::Met;
///
/// let resting = Met::from_met(1.0);
/// let walking = Met::from_met(2.0);
/// let running = Met::from_met(5.0);
///
/// // Convert to W/m²
/// assert_eq!(resting.as_watts_per_m2(), 58.15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Met(f64);

impl Met {
    /// Create a Met value from met units
    #[inline]
    pub const fn from_met(met: f64) -> Self {
        Met(met)
    }

    /// Create a Met value from W/m²
    #[inline]
    pub const fn from_watts_per_m2(watts_per_m2: f64) -> Self {
        Met(watts_per_m2 / 58.15)
    }

    /// Get the value in met units
    #[inline]
    pub const fn as_met(&self) -> f64 {
        self.0
    }

    /// Get the value in W/m²
    #[inline]
    pub const fn as_watts_per_m2(&self) -> f64 {
        self.0 * 58.15
    }

}

impl fmt::Display for Met {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} met", self.0)
    }
}

/// Clothing insulation in clo units
///
/// 1 clo = 0.155 m²·K/W (the thermal insulation needed to keep a resting person
/// comfortable at 21°C in a normally ventilated room)
///
/// # Examples
///
/// ```
/// use thermalcomfort::types::Clo;
///
/// let naked = Clo::from_clo(0.0);
/// let summer = Clo::from_clo(0.5);
/// let winter = Clo::from_clo(1.0);
///
/// // Convert to m²·K/W
/// assert_eq!(summer.as_m2_k_per_w(), 0.0775);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Clo(f64);

impl Clo {
    /// Create a Clo value from clo units
    #[inline]
    pub const fn from_clo(clo: f64) -> Self {
        Clo(clo)
    }

    /// Create a Clo value from m²·K/W
    #[inline]
    pub const fn from_m2_k_per_w(m2_k_per_w: f64) -> Self {
        Clo(m2_k_per_w / 0.155)
    }

    /// Get the value in clo units
    #[inline]
    pub const fn as_clo(&self) -> f64 {
        self.0
    }

    /// Get the value in m²·K/W
    #[inline]
    pub const fn as_m2_k_per_w(&self) -> f64 {
        self.0 * 0.155
    }

}

impl fmt::Display for Clo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} clo", self.0)
    }
}

/// Predicted Mean Vote (PMV) - dimensionless thermal sensation scale
///
/// PMV ranges from -3 (cold) to +3 (hot), with 0 being neutral
///
/// # Examples
///
/// ```
/// use thermalcomfort::types::Pmv;
///
/// let neutral = Pmv::from_value(0.0);
/// let slightly_warm = Pmv::from_value(0.5);
/// let cold = Pmv::from_value(-2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pmv(f64);

impl Pmv {
    /// Create a PMV value
    #[inline]
    pub const fn from_value(value: f64) -> Self {
        Pmv(value)
    }

    /// Get the PMV value
    #[inline]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Check if the PMV indicates thermal comfort (typically -0.5 to +0.5)
    #[inline]
    pub fn is_comfortable(&self) -> bool {
        self.0 >= -0.5 && self.0 <= 0.5
    }

    /// Get thermal sensation category
    pub fn sensation(&self) -> ThermalSensation {
        ThermalSensation::from_pmv(self.0)
    }
}

impl fmt::Display for Pmv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PMV: {:.2}", self.0)
    }
}

/// Predicted Percentage of Dissatisfied (PPD) - percentage of people dissatisfied
///
/// PPD ranges from 5% (optimal comfort) to 100% (complete dissatisfaction)
///
/// # Examples
///
/// ```
/// use thermalcomfort::types::Ppd;
///
/// let optimal = Ppd::from_percentage(5.0);
/// let acceptable = Ppd::from_percentage(10.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Ppd(f64);

impl Ppd {
    /// Create a PPD value from percentage
    #[inline]
    pub const fn from_percentage(percentage: f64) -> Self {
        Ppd(percentage)
    }

    /// Get the percentage value
    #[inline]
    pub const fn as_percentage(&self) -> f64 {
        self.0
    }

    /// Check if PPD is acceptable (typically < 10%)
    #[inline]
    pub fn is_acceptable(&self) -> bool {
        self.0 < 10.0
    }
}

impl fmt::Display for Ppd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PPD: {:.1}%", self.0)
    }
}

/// Thermal sensation categories based on PMV
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for ThermalSensation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThermalSensation::Cold => write!(f, "Cold"),
            ThermalSensation::Cool => write!(f, "Cool"),
            ThermalSensation::SlightlyCool => write!(f, "Slightly Cool"),
            ThermalSensation::Neutral => write!(f, "Neutral"),
            ThermalSensation::SlightlyWarm => write!(f, "Slightly Warm"),
            ThermalSensation::Warm => write!(f, "Warm"),
            ThermalSensation::Hot => write!(f, "Hot"),
        }
    }
}

/// Relative humidity as a percentage (0-100%)
///
/// # Examples
///
/// ```
/// use thermalcomfort::types::RelativeHumidity;
///
/// let rh = RelativeHumidity::from_percentage(50.0);
/// assert_eq!(rh.as_percentage(), 50.0);
/// assert_eq!(rh.as_fraction(), 0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RelativeHumidity(f64);

impl RelativeHumidity {
    /// Create from percentage (0-100)
    #[inline]
    pub fn from_percentage(percentage: f64) -> Self {
        RelativeHumidity(percentage.clamp(0.0, 100.0))
    }

    /// Create from fraction (0.0-1.0)
    #[inline]
    pub fn from_fraction(fraction: f64) -> Self {
        RelativeHumidity((fraction * 100.0).clamp(0.0, 100.0))
    }

    /// Get as percentage (0-100)
    #[inline]
    pub const fn as_percentage(&self) -> f64 {
        self.0
    }

    /// Get as fraction (0.0-1.0)
    #[inline]
    pub const fn as_fraction(&self) -> f64 {
        self.0 / 100.0
    }
}

impl fmt::Display for RelativeHumidity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}% RH", self.0)
    }
}

// Arithmetic operations for Met
impl Add for Met {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Met(self.0 + other.0)
    }
}

impl Sub for Met {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Met(self.0 - other.0)
    }
}

impl Mul<f64> for Met {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Met(self.0 * scalar)
    }
}

impl Div<f64> for Met {
    type Output = Self;
    fn div(self, scalar: f64) -> Self {
        Met(self.0 / scalar)
    }
}

// Arithmetic operations for Clo
impl Add for Clo {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Clo(self.0 + other.0)
    }
}

impl Sub for Clo {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Clo(self.0 - other.0)
    }
}

impl Mul<f64> for Clo {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Clo(self.0 * scalar)
    }
}

impl Div<f64> for Clo {
    type Output = Self;
    fn div(self, scalar: f64) -> Self {
        Clo(self.0 / scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_met_conversions() {
        let met = Met::from_met(1.0);
        assert_eq!(met.as_met(), 1.0);
        assert!((met.as_watts_per_m2() - 58.15).abs() < 0.01);

        let watts = Met::from_watts_per_m2(116.3);
        assert!((watts.as_met() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_clo_conversions() {
        let clo = Clo::from_clo(1.0);
        assert_eq!(clo.as_clo(), 1.0);
        assert!((clo.as_m2_k_per_w() - 0.155).abs() < 0.001);

        let m2 = Clo::from_m2_k_per_w(0.31);
        assert!((m2.as_clo() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_pmv_comfort() {
        assert!(Pmv::from_value(0.0).is_comfortable());
        assert!(Pmv::from_value(0.5).is_comfortable());
        assert!(Pmv::from_value(-0.5).is_comfortable());
        assert!(!Pmv::from_value(1.0).is_comfortable());
        assert!(!Pmv::from_value(-1.0).is_comfortable());
    }

    #[test]
    fn test_ppd_acceptable() {
        assert!(Ppd::from_percentage(5.0).is_acceptable());
        assert!(Ppd::from_percentage(9.9).is_acceptable());
        assert!(!Ppd::from_percentage(10.0).is_acceptable());
        assert!(!Ppd::from_percentage(20.0).is_acceptable());
    }

    #[test]
    fn test_thermal_sensation() {
        assert_eq!(ThermalSensation::from_pmv(-3.0), ThermalSensation::Cold);
        assert_eq!(ThermalSensation::from_pmv(-2.0), ThermalSensation::Cool);
        assert_eq!(ThermalSensation::from_pmv(-1.0), ThermalSensation::SlightlyCool);
        assert_eq!(ThermalSensation::from_pmv(0.0), ThermalSensation::Neutral);
        assert_eq!(ThermalSensation::from_pmv(1.0), ThermalSensation::SlightlyWarm);
        assert_eq!(ThermalSensation::from_pmv(2.0), ThermalSensation::Warm);
        assert_eq!(ThermalSensation::from_pmv(3.0), ThermalSensation::Hot);
    }

    #[test]
    fn test_relative_humidity() {
        let rh = RelativeHumidity::from_percentage(50.0);
        assert_eq!(rh.as_percentage(), 50.0);
        assert_eq!(rh.as_fraction(), 0.5);

        let rh2 = RelativeHumidity::from_fraction(0.75);
        assert_eq!(rh2.as_percentage(), 75.0);
        assert_eq!(rh2.as_fraction(), 0.75);
    }

    #[test]
    fn test_met_arithmetic() {
        let m1 = Met::from_met(1.0);
        let m2 = Met::from_met(2.0);

        assert_eq!((m1 + m2).as_met(), 3.0);
        assert_eq!((m2 - m1).as_met(), 1.0);
        assert_eq!((m1 * 2.0).as_met(), 2.0);
        assert_eq!((m2 / 2.0).as_met(), 1.0);
    }

    #[test]
    fn test_common_values() {
        assert_eq!(met::SEATED_QUIET.as_met(), 1.0);
        assert_eq!(clo::SUMMER_INDOOR.as_clo(), 0.5);
    }
}

/// Common metabolic rate values
pub mod met {
    use super::Met;

    /// Sleeping (0.7 met)
    pub const SLEEPING: Met = Met(0.7);
    /// Reclining (0.8 met)
    pub const RECLINING: Met = Met(0.8);
    /// Seated, quiet (1.0 met)
    pub const SEATED_QUIET: Met = Met(1.0);
    /// Standing, relaxed (1.2 met)
    pub const STANDING_RELAXED: Met = Met(1.2);
    /// Walking slowly (2.0 met)
    pub const WALKING_SLOW: Met = Met(2.0);
    /// Walking normally (2.6 met)
    pub const WALKING_NORMAL: Met = Met(2.6);
    /// Walking quickly (3.8 met)
    pub const WALKING_FAST: Met = Met(3.8);
    /// Running (6.0 met)
    pub const RUNNING: Met = Met(6.0);
}

/// Common clothing insulation values
pub mod clo {
    use super::Clo;

    /// Naked (0.0 clo)
    pub const NAKED: Clo = Clo(0.0);
    /// Typical summer indoor clothing (0.5 clo)
    pub const SUMMER_INDOOR: Clo = Clo(0.5);
    /// Walking shorts, short-sleeve shirt (0.36 clo)
    pub const SUMMER_CASUAL: Clo = Clo(0.36);
    /// Trousers, long-sleeve shirt (0.61 clo)
    pub const BUSINESS_CASUAL: Clo = Clo(0.61);
    /// Typical winter indoor clothing (1.0 clo)
    pub const WINTER_INDOOR: Clo = Clo(1.0);
    /// Sweat pants, long-sleeve sweatshirt (0.74 clo)
    pub const ATHLETIC: Clo = Clo(0.74);
}
