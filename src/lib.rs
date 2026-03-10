//! # Thermal Comfort Library
//!
//! A comprehensive Rust port of the pythermalcomfort Python package for thermal comfort calculations.
//! This library is `no_std` compatible and can run in WASM environments.
//!
//! This library provides tools for calculating thermal comfort indices, heat/cold stress metrics,
//! and thermophysiological responses using multiple models including:
//!
//! - PMV/PPD (Predicted Mean Vote and Predicted Percentage Dissatisfied) - ISO 7730 & ASHRAE 55
//! - Adaptive comfort models (ASHRAE 55 and EN 16798)
//! - UTCI (Universal Thermal Climate Index)
//! - SET (Standard Effective Temperature)
//! - Heat stress indices (WBGT, Heat Index, etc.)
//! - And many more...
//!
//! ## Example
//!
//! ```
//! use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
//!
//! let tdb = Temperature::from_celsius(25.0);
//! let tr = Temperature::from_celsius(25.0);
//! let rh = Humidity::from_percent(50.0);
//! let v = Speed::from_meters_per_second(0.1);
//! let met = MetabolicRate::from_met(1.4);
//! let clo = ClothingInsulation::from_clo(0.5);
//!
//! // Calculate relative air speed
//! let vr = v_relative(v, met);
//!
//! // Calculate PMV and PPD
//! let result = pmv_ppd_iso(
//!     tdb,
//!     tr,
//!     vr,
//!     rh,
//!     met,
//!     clo,
//!     Default::default()
//! );
//! ```

#![no_std]

pub mod constants;
pub mod models;
pub mod numerical;
pub mod psychrometrics;
pub mod utilities;

// Re-export commonly used items
pub use models::pmv::{PmvPpdResult, pmv_ppd_iso};
pub use utilities::{
    CLO_INDIVIDUAL_GARMENTS, CLO_TYPICAL_ENSEMBLES, clo_individual_garment, clo_typical_ensemble,
    v_relative,
};

// Re-export measurements types for convenience
// Users should import these from thermalcomfort instead of directly from measurements
pub use measurements::{Area, Humidity, Length, Mass, Power, Pressure, Speed, Temperature};

/// Clothing insulation measurement.
///
/// Represents thermal resistance of clothing per unit body surface area.
/// 1 clo = 0.155 m²·K/W ≈ the insulation of a typical business suit.
///
/// # Constructors
///
/// - [`from_clo(0.5)`](ClothingInsulation::from_clo) — primary; clo values found in ASHRAE 55 / ISO 7730 clothing tables
/// - [`from_tog(0.775)`](ClothingInsulation::from_tog) — tog units common in bedding industry (1 clo = 1.55 tog)
/// - [`from_m2_k_per_w(0.0775)`](ClothingInsulation::from_m2_k_per_w) — SI thermal resistance (1 clo = 0.155 m²·K/W)
///
/// # Common Values (clo)
///
/// | Ensemble | clo |
/// |----------|-----|
/// | Nude | ≈ 0 |
/// | Light summer (shorts, t-shirt) | 0.3–0.5 |
/// | Typical business suit | 1.0 |
/// | Heavy winter clothing | 1.5 |
///
/// # Examples
///
/// ```
/// use thermalcomfort::ClothingInsulation;
///
/// let clo = ClothingInsulation::from_clo(1.0);
/// assert!((clo.as_m2_k_per_w() - 0.155).abs() < 1e-10);
/// assert!((clo.as_tog() - 1.55).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ClothingInsulation(f64);

impl ClothingInsulation {
    /// Create from clo units (1 clo = 0.155 m²·K/W)
    #[inline]
    pub const fn from_clo(value: f64) -> Self {
        Self(value)
    }

    /// Create from tog units (1 clo = 1.55 tog)
    #[inline]
    pub fn from_tog(value: f64) -> Self {
        Self(value / 1.55)
    }

    /// Create from SI thermal resistance (m²·K/W) (1 clo = 0.155 m²·K/W)
    #[inline]
    pub fn from_m2_k_per_w(value: f64) -> Self {
        Self(value / 0.155)
    }

    /// Get value in clo units
    #[inline]
    pub const fn as_clo(&self) -> f64 {
        self.0
    }

    /// Get value in tog units (1 clo = 1.55 tog)
    #[inline]
    pub fn as_tog(&self) -> f64 {
        self.0 * 1.55
    }

    /// Get value in SI thermal resistance (m²·K/W) (1 clo = 0.155 m²·K/W)
    #[inline]
    pub fn as_m2_k_per_w(&self) -> f64 {
        self.0 * 0.155
    }
}

impl Default for ClothingInsulation {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Metabolic rate measurement.
///
/// Represents metabolic heat production per unit body surface area.
/// 1 met = 58.15 W/m², the resting metabolic rate of a seated person.
///
/// # Constructors
///
/// - [`from_met(1.4)`](MetabolicRate::from_met) — primary; met values found in ASHRAE 55 / ISO 7730 activity tables
/// - [`from_w_per_m2(81.41)`](MetabolicRate::from_w_per_m2) — SI heat flux per body surface area (1 met = 58.15 W/m²)
/// - [`from_btu_per_h_ft2(25.76)`](MetabolicRate::from_btu_per_h_ft2) — Imperial equivalent (1 met = 18.4 Btu/(h·ft²))
///
/// # Common Values (met)
///
/// | Activity | met |
/// |----------|-----|
/// | Seated, quiet | 1.0 |
/// | Standing, relaxed | 1.2 |
/// | Walking 3.2 km/h | 2.0 |
/// | Heavy work | 3.0+ |
///
/// # Examples
///
/// ```
/// use thermalcomfort::MetabolicRate;
///
/// let met = MetabolicRate::from_met(1.0);
/// assert!((met.as_w_per_m2() - 58.15).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MetabolicRate(f64);

impl MetabolicRate {
    /// Conversion factor: 1 met = 58.15 W/m²
    pub const MET_TO_W_M2: f64 = 58.15;

    /// Conversion factor: 1 met = 18.4 Btu/(h·ft²)
    pub const MET_TO_BTU_H_FT2: f64 = 18.4;

    /// Create from met units (1 met = 58.15 W/m²)
    #[inline]
    pub const fn from_met(value: f64) -> Self {
        Self(value)
    }

    /// Create from W/m² (1 met = 58.15 W/m²)
    #[inline]
    pub fn from_w_per_m2(value: f64) -> Self {
        Self(value / Self::MET_TO_W_M2)
    }

    /// Create from Btu/(h·ft²) (1 met = 18.4 Btu/(h·ft²))
    #[inline]
    pub fn from_btu_per_h_ft2(value: f64) -> Self {
        Self(value / Self::MET_TO_BTU_H_FT2)
    }

    /// Get value in met units
    #[inline]
    pub const fn as_met(&self) -> f64 {
        self.0
    }

    /// Get value in W/m² (1 met = 58.15 W/m²)
    #[inline]
    pub fn as_w_per_m2(&self) -> f64 {
        self.0 * Self::MET_TO_W_M2
    }

    /// Get value in Btu/(h·ft²) (1 met = 18.4 Btu/(h·ft²))
    #[inline]
    pub fn as_btu_per_h_ft2(&self) -> f64 {
        self.0 * Self::MET_TO_BTU_H_FT2
    }
}

impl Default for MetabolicRate {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Biological sex for physiological calculations
///
/// Used in models that differentiate physiological responses by sex,
/// such as PET (basal metabolism) and ridge regression (body temperature prediction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sex {
    Male,
    Female,
}

impl Sex {
    /// Get numeric value (0.0 for Male, 1.0 for Female)
    ///
    /// Used internally by ridge regression and other models.
    pub fn as_value(&self) -> f64 {
        match self {
            Sex::Male => 0.0,
            Sex::Female => 1.0,
        }
    }
}
