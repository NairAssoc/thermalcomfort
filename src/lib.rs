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
pub use measurements::{Angle, Area, Humidity, Length, Mass, Power, Pressure, Speed, Temperature};

/// A temperature *difference*.
///
/// Distinct from [`Temperature`], which is absolute: 25 °C and 25 °F are different
/// temperatures, but a *change* of 1 °C is a change of 1.8 °F, with no offset. Keeping
/// the two apart stops a gradient being handed to something expecting an absolute
/// reading, and makes the unit explicit at the API boundary instead of leaving it
/// implied by a bare `f64`.
///
/// # Examples
///
/// ```
/// use thermalcomfort::TemperatureDelta;
///
/// let gradient = TemperatureDelta::from_celsius(2.0);
/// assert!((gradient.as_fahrenheit() - 3.6).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TemperatureDelta(f64);

impl TemperatureDelta {
    /// Construct from a difference in degrees Celsius (equivalently, kelvin)
    pub const fn from_celsius(value: f64) -> Self {
        Self(value)
    }

    /// Construct from a difference in degrees Fahrenheit
    pub const fn from_fahrenheit(value: f64) -> Self {
        Self(value / 1.8)
    }

    /// The difference in degrees Celsius (equivalently, kelvin)
    pub const fn as_celsius(self) -> f64 {
        self.0
    }

    /// The difference in degrees Fahrenheit
    pub const fn as_fahrenheit(self) -> f64 {
        self.0 * 1.8
    }
}

/// Air permeability of clothing.
///
/// ISO 11079 expresses this in litres per square metre per second: the rate at which
/// air passes through the fabric. It is a distinct dimension from air speed, so it gets
/// its own type rather than reusing [`Speed`].
///
/// # Examples
///
/// ```
/// use thermalcomfort::AirPermeability;
///
/// let p = AirPermeability::from_l_per_m2_s(50.0);
/// assert!((p.as_l_per_m2_s() - 50.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AirPermeability(f64);

impl AirPermeability {
    /// Construct from litres per square metre per second [l/(m²·s)]
    pub const fn from_l_per_m2_s(value: f64) -> Self {
        Self(value)
    }

    /// Value in litres per square metre per second [l/(m²·s)]
    pub const fn as_l_per_m2_s(self) -> f64 {
        self.0
    }
}

/// Mechanical efficiency of external work, as a fraction of metabolic heat production.
///
/// Distinct from [`MetabolicRate`], which is the *rate* a model subtracts from the heat
/// balance. This is the dimensionless multiplier in PET's source term,
/// `h = he * (1 - wme)`, so it is only meaningful on `[0, 1]`: at 1 all metabolic energy
/// leaves as work and none as heat, and above 1 the term goes negative and the model
/// describes a body that absorbs heat by working. The range is enforced at construction
/// rather than documented, because an out-of-range value does not fail loudly — it
/// quietly yields a plausible-looking temperature from an unphysical energy balance.
///
/// # Examples
///
/// ```
/// use thermalcomfort::WorkEfficiency;
///
/// let w = WorkEfficiency::new(0.2).expect("0.2 is a valid efficiency");
/// assert!((w.as_fraction() - 0.2).abs() < 1e-12);
///
/// // Outside [0, 1] there is no valid value to construct
/// assert!(WorkEfficiency::new(1.5).is_none());
/// assert!(WorkEfficiency::new(-0.1).is_none());
/// assert!(WorkEfficiency::new(f64::NAN).is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct WorkEfficiency(f64);

impl WorkEfficiency {
    /// No external work: all metabolic energy is released as heat.
    pub const ZERO: Self = Self(0.0);

    /// Construct from a fraction in `[0, 1]`, returning `None` outside that range.
    ///
    /// NaN is rejected too: it would propagate silently through the whole heat balance.
    pub fn new(value: f64) -> Option<Self> {
        if value.is_nan() || !(0.0..=1.0).contains(&value) {
            None
        } else {
            Some(Self(value))
        }
    }

    /// The efficiency as a fraction in `[0, 1]`
    pub const fn as_fraction(self) -> f64 {
        self.0
    }
}

impl Default for WorkEfficiency {
    fn default() -> Self {
        Self::ZERO
    }
}

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

/// A density of heat flow rate: power per unit area.
///
/// Models report skin evaporation, sensible loss and respiratory loss in W/m², while
/// [`Power`] alone would be watts over the whole body. Keeping the two apart stops a
/// per-area flux being summed with a whole-body power. Signed: heat flows both ways.
///
/// # Examples
///
/// ```
/// use thermalcomfort::HeatFluxDensity;
///
/// let e_skin = HeatFluxDensity::from_watts_per_square_meter(32.2);
/// assert!((e_skin.as_watts_per_square_meter() - 32.2).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct HeatFluxDensity(f64);

impl HeatFluxDensity {
    /// Construct from watts per square metre [W/m²]
    pub const fn from_watts_per_square_meter(value: f64) -> Self {
        Self(value)
    }

    /// Value in watts per square metre [W/m²]
    pub const fn as_watts_per_square_meter(self) -> f64 {
        self.0
    }
}

/// Cardiac index: cardiac output normalised by body surface area.
///
/// JOS3 uses it to set the basal blood-flow distribution. Deliberately unbounded, because
/// pythermalcomfort does not validate it — `validate_body_parameters` checks height,
/// weight, age and body fat only, so imposing a range here would reject inputs upstream
/// accepts and break parity.
///
/// # Examples
///
/// ```
/// use thermalcomfort::CardiacIndex;
///
/// let ci = CardiacIndex::from_liters_per_minute_per_square_meter(2.59);
/// assert!((ci.as_liters_per_minute_per_square_meter() - 2.59).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CardiacIndex(f64);

impl CardiacIndex {
    /// pythermalcomfort's default, 2.59 L/(min·m²) (`Default.cardiac_index`)
    pub const DEFAULT: Self = Self(2.59);

    /// Construct from litres per minute per square metre [L/(min·m²)]
    pub const fn from_liters_per_minute_per_square_meter(value: f64) -> Self {
        Self(value)
    }

    /// Value in litres per minute per square metre [L/(min·m²)]
    pub const fn as_liters_per_minute_per_square_meter(self) -> f64 {
        self.0
    }
}

/// Physical activity ratio (PAR): metabolic rate as a multiple of basal metabolic rate.
///
/// Dimensionless, and distinct from [`MetabolicRate`], which is an absolute rate in met.
/// Unbounded for the same reason as [`CardiacIndex`]: pythermalcomfort does not validate
/// it.
///
/// # Examples
///
/// ```
/// use thermalcomfort::ActivityRatio;
///
/// let par = ActivityRatio::from_ratio(1.25);
/// assert!((par.as_ratio() - 1.25).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ActivityRatio(f64);

impl ActivityRatio {
    /// pythermalcomfort's default, 1.25 (`Default.physical_activity_ratio`)
    pub const DEFAULT: Self = Self(1.25);

    /// Construct from a dimensionless multiple of basal metabolic rate
    pub const fn from_ratio(value: f64) -> Self {
        Self(value)
    }

    /// The ratio, as a dimensionless multiple of basal metabolic rate
    pub const fn as_ratio(self) -> f64 {
        self.0
    }
}

/// Body fat, as a percentage of total body mass.
///
/// Bounded on `[1, 90]`, which is not a range chosen here: it is exactly what
/// `validate_body_parameters` enforces in pythermalcomfort's
/// `jos3_functions/construction.py`. Like [`WorkEfficiency`], an out-of-range value does
/// not fail loudly — it yields a plausible-looking body composition and a wrong heat
/// balance — so the range is enforced at construction rather than documented.
///
/// # Examples
///
/// ```
/// use thermalcomfort::BodyFat;
///
/// let fat = BodyFat::new(15.0).expect("15% is in range");
/// assert!((fat.as_percent() - 15.0).abs() < 1e-12);
///
/// // Upstream rejects these, so this crate cannot represent them
/// assert!(BodyFat::new(0.5).is_none());
/// assert!(BodyFat::new(95.0).is_none());
/// assert!(BodyFat::new(f64::NAN).is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BodyFat(f64);

impl BodyFat {
    /// Construct from a percentage in `[1, 90]`, returning `None` outside that range.
    ///
    /// NaN is rejected: it would propagate silently through the whole heat balance.
    pub fn new(value: f64) -> Option<Self> {
        if value.is_nan() || !(1.0..=90.0).contains(&value) {
            None
        } else {
            Some(Self(value))
        }
    }

    /// The body fat as a percentage of total body mass
    pub const fn as_percent(self) -> f64 {
        self.0
    }
}

impl Default for BodyFat {
    /// pythermalcomfort's default, 15% (`Default.body_fat`)
    fn default() -> Self {
        Self(15.0)
    }
}

/// Which equation estimates basal metabolic rate.
///
/// Mirrors pythermalcomfort's `bmr_equation` string, dispatched in
/// `jos3_functions/thermoregulation.py::basal_met`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BmrEquation {
    /// Harris-Benedict, revised — upstream's default (`"harris-benedict"`)
    #[default]
    HarrisBenedict,
    /// Harris-Benedict as originally published (`"harris-benedict_origin"`)
    HarrisBenedictOriginal,
    /// Ganpule's equation for a Japanese population.
    ///
    /// Upstream accepts either `"japanese"` or `"ganpule"` for this; they select the same
    /// equation, so there is one variant rather than two.
    Japanese,
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

#[cfg(test)]
mod newtype_tests {
    use super::*;

    #[test]
    fn body_fat_enforces_upstreams_range_and_nothing_wider() {
        // The bounds are pythermalcomfort's, from validate_body_parameters: [1, 90].
        assert!(BodyFat::new(1.0).is_some());
        assert!(BodyFat::new(90.0).is_some());
        assert!(BodyFat::new(0.999).is_none());
        assert!(BodyFat::new(90.001).is_none());
        assert!(BodyFat::new(f64::NAN).is_none());
        assert!(BodyFat::new(f64::INFINITY).is_none());
    }

    #[test]
    fn unvalidated_upstream_parameters_stay_unbounded() {
        // pythermalcomfort validates height, weight, age and body fat — not these two.
        // Rejecting a value upstream accepts would be a parity break, so they take any
        // finite value, including implausible ones.
        assert!((CardiacIndex::from_liters_per_minute_per_square_meter(0.0)
            .as_liters_per_minute_per_square_meter())
        .abs()
            < 1e-12);
        assert!((ActivityRatio::from_ratio(-1.0).as_ratio() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn defaults_match_pythermalcomfort() {
        // jos3_functions/parameters.py, class Default
        assert!(
            (CardiacIndex::DEFAULT.as_liters_per_minute_per_square_meter() - 2.59).abs() < 1e-12
        );
        assert!((ActivityRatio::DEFAULT.as_ratio() - 1.25).abs() < 1e-12);
        assert!((BodyFat::default().as_percent() - 15.0).abs() < 1e-12);
        assert_eq!(BmrEquation::default(), BmrEquation::HarrisBenedict);
    }

    #[test]
    fn heat_flux_density_is_signed() {
        // Heat flows both directions; a flux type that clamped at zero would lose that.
        let loss = HeatFluxDensity::from_watts_per_square_meter(-12.5);
        assert!((loss.as_watts_per_square_meter() + 12.5).abs() < 1e-12);
    }

    #[test]
    fn temperature_delta_converts_without_an_offset() {
        // A 1 °C change is a 1.8 °F change — no 32 degree offset
        assert!((TemperatureDelta::from_celsius(1.0).as_fahrenheit() - 1.8).abs() < 1e-12);
        assert!((TemperatureDelta::from_fahrenheit(1.8).as_celsius() - 1.0).abs() < 1e-12);
        assert!(TemperatureDelta::from_celsius(0.0).as_fahrenheit().abs() < 1e-12);

        let d = TemperatureDelta::from_celsius(3.5);
        assert!(
            (TemperatureDelta::from_fahrenheit(d.as_fahrenheit()).as_celsius() - 3.5).abs() < 1e-12
        );
    }

    #[test]
    fn work_efficiency_rejects_values_outside_zero_to_one() {
        // Both ends are valid: 0 is all heat, 1 is all work
        assert_eq!(
            WorkEfficiency::new(0.0).map(WorkEfficiency::as_fraction),
            Some(0.0)
        );
        assert_eq!(
            WorkEfficiency::new(1.0).map(WorkEfficiency::as_fraction),
            Some(1.0)
        );
        assert_eq!(
            WorkEfficiency::new(0.25).map(WorkEfficiency::as_fraction),
            Some(0.25)
        );

        // Above 1 the PET source term `he * (1 - wme)` goes negative, so there is no
        // valid value to construct rather than a value that quietly misbehaves.
        assert!(WorkEfficiency::new(1.000_001).is_none());
        assert!(WorkEfficiency::new(-1e-9).is_none());
        assert!(WorkEfficiency::new(f64::NAN).is_none());
        assert!(WorkEfficiency::new(f64::INFINITY).is_none());

        assert_eq!(WorkEfficiency::default(), WorkEfficiency::ZERO);
    }

    #[test]
    fn air_permeability_round_trips() {
        let p = AirPermeability::from_l_per_m2_s(50.0);
        assert!((p.as_l_per_m2_s() - 50.0).abs() < 1e-12);
    }
}
