//! Physical and thermal comfort constants shared by the models, as raw `f64` values in the
//! units each one's documentation states. Crate-private: the public API carries quantities
//! as unit types.

/// Celsius to Kelvin conversion constant
///
/// Standard physics conversion: K = °C + 273.15
pub const C_TO_K: f64 = 273.15;

/// Specific heat of water vapor at constant pressure [J/(kg·K)]
///
/// Source: Standard thermodynamic tables (approximate value at typical indoor conditions)
pub const CP_VAPOUR: f64 = 1805.0;

/// Specific heat of air at constant pressure [J/(kg·K)]
///
/// Source: ASHRAE Fundamentals and ISO 7730 (dry air at typical indoor conditions)
pub const CP_AIR: f64 = 1004.0;

/// Latent heat of vaporization of water [J/kg]
///
/// Source: Standard psychrometric value at 0°C (2,501,000 J/kg)
pub const H_FG: f64 = 2501000.0;

/// Gravitational acceleration [m/s²]
///
/// Source: Standard gravity (9.80665 m/s² rounded to 9.81)
pub const G: f64 = 9.81;
