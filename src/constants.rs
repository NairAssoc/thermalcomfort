//! Physical and thermal comfort constants
//!
//! This module provides both raw f64 constants for performance-critical calculations
//! and typed versions using the measurements crate for type-safe usage.

/// Celsius to Kelvin conversion constant
pub const C_TO_K: f64 = 273.15;

/// Specific heat of water vapor [J/(kg·K)]
pub const CP_VAPOUR: f64 = 1805.0;

/// Specific heat of water [J/(kg·K)]
pub const CP_WATER: f64 = 4186.0;

/// Specific heat of air [J/(kg·K)]
pub const CP_AIR: f64 = 1004.0;

/// Latent heat of vaporization [J/kg]
pub const H_FG: f64 = 2501000.0;

/// Gas constant for air [J/(kg·K)]
pub const R_AIR: f64 = 287.055;

/// Gravitational acceleration [m/s²]
///
/// Raw value: 9.81 m/s²
///
/// For typed version, use:
/// ```
/// use measurements::Acceleration;
/// let g = Acceleration::from_meters_per_second_per_second(9.81);
/// ```
pub const G: f64 = 9.81;

/// Conversion factor from met to W/m²
///
/// 1 met = 58.15 W/m² (metabolic heat production per unit body surface area)
pub const MET_TO_W_M2: f64 = 58.15;

/// Stefan-Boltzmann constant [W/(m²·K⁴)]
pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;

/// Thermal comfort specific units
pub mod thermal_units {
    //! Domain-specific thermal comfort units

    /// One met (metabolic equivalent) in W/m²
    pub const MET: f64 = 58.15;

    /// One clo (clothing insulation unit) in m²·K/W
    pub const CLO: f64 = 0.155;
}
