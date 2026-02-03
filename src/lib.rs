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
//! let tdb = 25.0; // dry bulb temperature [°C]
//! let tr = 25.0;  // mean radiant temperature [°C]
//! let rh = 50.0;  // relative humidity [%]
//! let v = 0.1;    // air speed [m/s]
//! let met = 1.4;  // metabolic rate [met]
//! let clo = 0.5;  // clothing insulation [clo]
//!
//! // Calculate relative air speed
//! let vr = v_relative(v, met);
//!
//! // Calculate PMV and PPD using measurement types
//! let result = pmv_ppd_iso(
//!     Temperature::from_celsius(tdb),
//!     Temperature::from_celsius(tr),
//!     Speed::from_meters_per_second(vr),
//!     Humidity::from_percent(rh),
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
pub use models::pmv::{pmv_ppd_iso, PmvPpdResult};
pub use utilities::v_relative;

// Re-export measurements types for convenience
pub use measurements::{Temperature, Speed, Area, Pressure, Humidity};
