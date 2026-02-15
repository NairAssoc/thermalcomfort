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
//! use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity};
//!
//! let tdb = Temperature::from_celsius(25.0);
//! let tr = Temperature::from_celsius(25.0);
//! let rh = Humidity::from_percent(50.0);
//! let v = Speed::from_meters_per_second(0.1);
//! let met = 1.4;  // metabolic rate [met]
//! let clo = 0.5;  // clothing insulation [clo]
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
    v_relative,
    CLO_INDIVIDUAL_GARMENTS,
    CLO_TYPICAL_ENSEMBLES,
    clo_individual_garment,
    clo_typical_ensemble,
};

// Re-export measurements types for convenience
// Users should import these from thermalcomfort instead of directly from measurements
pub use measurements::{Area, Humidity, Length, Mass, Pressure, Speed, Temperature};
