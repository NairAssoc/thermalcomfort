//! Physical and thermal comfort constants
//!
//! This module provides both raw f64 constants for performance-critical calculations
//! and typed versions using the measurements crate for type-safe usage.

/// Celsius to Kelvin conversion constant
///
/// Standard physics conversion: K = °C + 273.15
pub const C_TO_K: f64 = 273.15;

/// Specific heat of water vapor at constant pressure [J/(kg·K)]
///
/// Source: Standard thermodynamic tables (approximate value at typical indoor conditions)
pub const CP_VAPOUR: f64 = 1805.0;

/// Specific heat of water [J/(kg·K)]
///
/// Source: Standard thermodynamic property at 25°C
pub const CP_WATER: f64 = 4186.0;

/// Specific heat of air at constant pressure [J/(kg·K)]
///
/// Source: ASHRAE Fundamentals and ISO 7730 (dry air at typical indoor conditions)
pub const CP_AIR: f64 = 1004.0;

/// Latent heat of vaporization of water [J/kg]
///
/// Source: Standard psychrometric value at 0°C (2,501,000 J/kg)
pub const H_FG: f64 = 2501000.0;

/// Specific gas constant for dry air [J/(kg·K)]
///
/// Source: Standard thermodynamic property (R_universal / M_air)
pub const R_AIR: f64 = 287.055;

/// Gravitational acceleration [m/s²]
///
/// Source: Standard gravity (9.80665 m/s² rounded to 9.81)
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
/// Source: ISO 7730 and ASHRAE 55
/// 1 met = 58.15 W/m² (metabolic heat production per unit body surface area)
/// Based on resting metabolic rate of a seated person
pub const MET_TO_W_M2: f64 = 58.15;

/// Stefan-Boltzmann constant [W/(m²·K⁴)]
///
/// Source: Fundamental physical constant for thermal radiation
/// σ = 5.670374419×10⁻⁸ W/(m²·K⁴) (rounded to 5.67×10⁻⁸)
pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;

/// Thermal comfort specific units
pub mod thermal_units {
    /// One met (metabolic equivalent) in W/m²
    ///
    /// Source: ISO 7730 and ASHRAE 55
    /// 1 met = 58.15 W/m² (metabolic heat production rate)
    pub const MET: f64 = 58.15;

    /// One clo (clothing insulation unit) in m²·K/W
    ///
    /// Source: ISO 7730 and ASHRAE 55
    /// 1 clo = 0.155 m²·K/W (thermal resistance of clothing)
    /// Approximately equivalent to a business suit
    pub const CLO: f64 = 0.155;
}
