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
//! use thermalcomfort::models::pmv::PmvPpdInputs;
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
//!     PmvPpdInputs {
//!         tdb,
//!         tr,
//!         vr,
//!         rh,
//!         met,
//!         clo,
//!     },
//!     Default::default()
//! );
//! ```

#![no_std]

/// Compiles every ```` ```rust ```` block in `README.md` as a doctest.
///
/// The README's examples had all rotted to a previous API — they still showed positional
/// calls after every model moved to named input structs, so anything copy-pasted from the
/// front page did not compile. Nothing caught it, because a fenced block in a `.md` file is
/// just text. This makes that impossible: the examples are now compiled by
/// `cargo test --doc` like any other doctest.
///
/// `#[cfg(doctest)]` means the item exists only during a doctest run, so it costs nothing
/// in a normal build and adds nothing to the public API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeExamples;

pub mod clothing;
pub(crate) mod constants;
pub mod environment;
pub mod models;
pub mod numerical;
pub mod psychrometrics;
pub mod unit_system;
pub mod utilities;

// Re-export commonly used items
pub use environment::v_relative;
pub use models::pmv::{PmvPpdResult, pmv_ppd_iso};
pub use unit_system::UnitSystem;
pub use utilities::{
    CLO_INDIVIDUAL_GARMENTS, CLO_TYPICAL_ENSEMBLES, Units, clo_individual_garment,
    clo_typical_ensemble,
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
/// Units: SI °C, equal to K (`as_si`, [`as_celsius`](Self::as_celsius)); IP (inch-pound) °F
/// (`as_ip`, [`as_fahrenheit`](Self::as_fahrenheit)), × 1.8 with no offset.
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
/// Units: SI L/(m²·s) (`as_si`, [`as_l_per_m2_s`](Self::as_l_per_m2_s)); IP (inch-pound)
/// ft³/(min·ft²) (`as_ip`).
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

/// Practical upper limit of skin wettedness, as a fraction in `(0.06, 1]`.
///
/// 0.06 is the wettedness of skin diffusion alone, with no regulatory sweating, so a cap at
/// or below it leaves the two-node model a zero or negative sweating rate at the cap, and
/// its discomfort index divides by zero. pythermalcomfort rejects those values, and NaN, so
/// they cannot be constructed here.
///
/// # Examples
///
/// ```
/// use thermalcomfort::MaxSkinWettedness;
///
/// let w = MaxSkinWettedness::new(0.5).expect("0.5 is a valid cap");
/// assert_eq!(w.as_fraction(), 0.5);
///
/// assert!(MaxSkinWettedness::new(1.0).is_some());
/// assert!(MaxSkinWettedness::new(0.06).is_none());
/// assert!(MaxSkinWettedness::new(1.01).is_none());
/// assert!(MaxSkinWettedness::new(f64::NAN).is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MaxSkinWettedness(f64);

impl MaxSkinWettedness {
    /// Construct from a fraction in `(0.06, 1]`, returning `None` outside that range.
    pub fn new(value: f64) -> Option<Self> {
        (value > 0.06 && value <= 1.0).then_some(Self(value))
    }

    /// The cap as a fraction in `(0.06, 1]`
    pub const fn as_fraction(self) -> f64 {
        self.0
    }
}

/// Clothing insulation measurement.
///
/// Represents thermal resistance of clothing per unit body surface area.
/// 1 clo = 0.155 m²·K/W ≈ the insulation of a typical business suit.
///
/// Units: SI m²·K/W (`as_si`, [`as_m2_k_per_w`](Self::as_m2_k_per_w)); IP (inch-pound)
/// ft²·h·°F/Btu (`as_ip`). [`as_clo`](Self::as_clo) and [`as_tog`](Self::as_tog) give the
/// clo and tog scales.
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
/// Units: SI W/m² (`as_si`, [`as_w_per_m2`](Self::as_w_per_m2)); IP (inch-pound)
/// Btu/(h·ft²) (`as_ip`, [`as_btu_per_h_ft2`](Self::as_btu_per_h_ft2)), by ASHRAE's
/// 1 met = 18.4 Btu/(h·ft²). [`as_met`](Self::as_met) gives the met scale.
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
/// Units: SI W/m² (`as_si`, [`as_watts_per_square_meter`](Self::as_watts_per_square_meter));
/// IP (inch-pound) Btu/(h·ft²) (`as_ip`).
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

/// Mass flow per unit area, in grams per hour per square metre.
///
/// The two-node models report regulatory sweating as a rate per unit of body surface area.
/// pythermalcomfort quotes it in mL/(h·m²), which is the same number at sweat's 1 g/mL.
/// Distinct from [`MassFlow`], which is a whole-body rate; skin blood flow is a
/// [`VolumeFluxDensity`].
///
/// Units: SI g/(h·m²) (`as_si`, [`as_grams_per_hour_per_square_meter`](Self::as_grams_per_hour_per_square_meter));
/// IP (inch-pound) lb/(h·ft²) (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{MassFluxDensity, UnitSystem};
///
/// let m_rsw = MassFluxDensity::from_grams_per_hour_per_square_meter(500.0);
/// assert!((m_rsw.as_grams_per_hour_per_square_meter() - 500.0).abs() < 1e-12);
/// // 1 lb/(h·ft²) = 4 882.427 6 g/(h·m²)
/// assert!((MassFluxDensity::from_ip(1.0).as_si() - 4_882.427_6).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MassFluxDensity(f64);

impl MassFluxDensity {
    /// Construct from grams per hour per square metre [g/(h·m²)]
    pub const fn from_grams_per_hour_per_square_meter(value: f64) -> Self {
        Self(value)
    }

    /// Value in grams per hour per square metre [g/(h·m²)]
    pub const fn as_grams_per_hour_per_square_meter(self) -> f64 {
        self.0
    }
}

/// Volume flow per unit area, in litres per hour per square metre.
///
/// Skin blood flow normalised by body surface area. pythermalcomfort quotes it in
/// kg/(h·m²), the same number at the 1 kg/L its heat balance assumes for blood. Distinct from [`VolumeFlow`],
/// which is a whole-body or per-segment rate.
///
/// Units: SI L/(h·m²) (`as_si`, [`as_liters_per_hour_per_square_meter`](Self::as_liters_per_hour_per_square_meter));
/// IP (inch-pound) gal/(h·ft²) (`as_ip`), in US gallons of 3.785 411 784 L.
///
/// # Examples
///
/// ```
/// use thermalcomfort::{UnitSystem, VolumeFluxDensity};
///
/// let skbf = VolumeFluxDensity::from_liters_per_hour_per_square_meter(6.3);
/// assert!((skbf.as_liters_per_hour_per_square_meter() - 6.3).abs() < 1e-12);
/// // 1 gal/(h·ft²) = 40.745 833 L/(h·m²)
/// assert!((VolumeFluxDensity::from_ip(1.0).as_si() - 40.745_833).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct VolumeFluxDensity(f64);

impl VolumeFluxDensity {
    /// Construct from litres per hour per square metre [L/(h·m²)]
    pub const fn from_liters_per_hour_per_square_meter(value: f64) -> Self {
        Self(value)
    }

    /// Value in litres per hour per square metre [L/(h·m²)]
    pub const fn as_liters_per_hour_per_square_meter(self) -> f64 {
        self.0
    }
}

/// Volume flow rate, in litres per hour.
///
/// JOS3 reports cardiac output and every segment's blood flow in L/h.
///
/// Units: SI L/h (`as_si`, [`as_liters_per_hour`](Self::as_liters_per_hour)); IP
/// (inch-pound) gal/h (`as_ip`), in US gallons of 3.785 411 784 L.
///
/// # Examples
///
/// ```
/// use thermalcomfort::{UnitSystem, VolumeFlow};
///
/// let cardiac_output = VolumeFlow::from_liters_per_hour(290.0);
/// assert!((cardiac_output.as_liters_per_hour() - 290.0).abs() < 1e-12);
/// // 1 gal/h = 3.785 411 784 L/h
/// assert!((VolumeFlow::from_ip(1.0).as_si() - 3.785_411_784).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct VolumeFlow(f64);

impl VolumeFlow {
    /// Construct from litres per hour [L/h]
    pub const fn from_liters_per_hour(value: f64) -> Self {
        Self(value)
    }

    /// Value in litres per hour [L/h]
    pub const fn as_liters_per_hour(self) -> f64 {
        self.0
    }
}

/// Mass flow rate, in grams per second.
///
/// JOS3 reports the body's weight loss by evaporation and respiration in g/s.
///
/// Units: SI g/s (`as_si`, [`as_grams_per_second`](Self::as_grams_per_second)); IP
/// (inch-pound) lb/h (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{MassFlow, UnitSystem};
///
/// let weight_loss = MassFlow::from_grams_per_second(0.02);
/// assert!((weight_loss.as_grams_per_second() - 0.02).abs() < 1e-12);
/// // 1 lb/h = 0.125 997 88 g/s
/// assert!((MassFlow::from_ip(1.0).as_si() - 0.125_997_88).abs() < 1e-8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MassFlow(f64);

impl MassFlow {
    /// Construct from grams per second [g/s]
    pub const fn from_grams_per_second(value: f64) -> Self {
        Self(value)
    }

    /// Value in grams per second [g/s]
    pub const fn as_grams_per_second(self) -> f64 {
        self.0
    }
}

/// Specific enthalpy: energy per unit mass, in joules per kilogram.
///
/// The psychrometric functions give moist-air enthalpy per kilogram of dry air.
///
/// Units: SI J/kg (`as_si`, [`as_joules_per_kilogram`](Self::as_joules_per_kilogram)); IP
/// (inch-pound) Btu/lb (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{SpecificEnthalpy, UnitSystem};
///
/// let h = SpecificEnthalpy::from_joules_per_kilogram(50_000.0);
/// assert!((h.as_joules_per_kilogram() - 50_000.0).abs() < 1e-9);
/// // 1 Btu/lb = 2326 J/kg
/// assert!((SpecificEnthalpy::from_ip(1.0).as_si() - 2326.0).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SpecificEnthalpy(f64);

impl SpecificEnthalpy {
    /// Construct from joules per kilogram [J/kg]
    pub const fn from_joules_per_kilogram(value: f64) -> Self {
        Self(value)
    }

    /// Value in joules per kilogram [J/kg]
    pub const fn as_joules_per_kilogram(self) -> f64 {
        self.0
    }
}

/// Humidity ratio: mass of water vapour per unit mass of dry air.
///
/// A mass ratio, so the same number in any consistent mass unit, but it is quoted in
/// several scales: kg/kg in the psychrometric equations, g/kg on charts, and grains per
/// pound in US practice.
///
/// Units: SI kg/kg (`as_si`, [`as_kilograms_per_kilogram`](Self::as_kilograms_per_kilogram));
/// IP (inch-pound) lb/lb (`as_ip`), the same number. [`as_grams_per_kilogram`](Self::as_grams_per_kilogram)
/// gives the other SI scale, g/kg, and [`as_grains_per_pound`](Self::as_grains_per_pound)
/// the US one, gr/lb (7000 gr = 1 lb).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{HumidityRatio, UnitSystem};
///
/// let w = HumidityRatio::from_grams_per_kilogram(10.0);
/// assert!((w.as_kilograms_per_kilogram() - 0.01).abs() < 1e-15);
/// assert!((w.as_si() - 0.01).abs() < 1e-15);
/// assert!((w.as_ip() - 0.01).abs() < 1e-15);
/// assert!((w.as_grains_per_pound() - 70.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct HumidityRatio(f64);

impl HumidityRatio {
    /// Grains per pound, exact (1 lb = 7000 gr)
    const GRAINS_PER_POUND: f64 = 7000.0;

    /// Construct from kilograms of water vapour per kilogram of dry air [kg/kg]
    pub const fn from_kilograms_per_kilogram(value: f64) -> Self {
        Self(value)
    }

    /// Construct from grams of water vapour per kilogram of dry air [g/kg]
    pub const fn from_grams_per_kilogram(value: f64) -> Self {
        Self(value / 1000.0)
    }

    /// Value in kilograms of water vapour per kilogram of dry air [kg/kg]
    pub const fn as_kilograms_per_kilogram(self) -> f64 {
        self.0
    }

    /// Value in grams of water vapour per kilogram of dry air [g/kg]
    pub const fn as_grams_per_kilogram(self) -> f64 {
        self.0 * 1000.0
    }

    /// Value in grains of water vapour per pound of dry air [gr/lb]
    pub const fn as_grains_per_pound(self) -> f64 {
        self.0 * Self::GRAINS_PER_POUND
    }
}

/// Heat per unit area: an energy density over a surface, in joules per square metre.
///
/// PHS accumulates its evaporative load per unit of body surface area in W·min/m², which
/// [`from_watt_minutes_per_square_meter`](Self::from_watt_minutes_per_square_meter) and
/// [`as_watt_minutes_per_square_meter`](Self::as_watt_minutes_per_square_meter) read and
/// build (1 W·min/m² = 60 J/m²).
///
/// Units: SI J/m² (`as_si`, [`as_joules_per_square_meter`](Self::as_joules_per_square_meter));
/// IP (inch-pound) Btu/ft² (`as_ip`). W·min/m² is the other SI scale, through the named
/// accessors above.
///
/// # Examples
///
/// ```
/// use thermalcomfort::{HeatPerArea, UnitSystem};
///
/// let load = HeatPerArea::from_watt_minutes_per_square_meter(2.0);
/// assert!((load.as_joules_per_square_meter() - 120.0).abs() < 1e-12);
/// assert!((load.as_si() - 120.0).abs() < 1e-12);
/// // 1 Btu/ft² = 11 356.53 J/m²
/// assert!((HeatPerArea::from_ip(1.0).as_si() - 11_356.53).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct HeatPerArea(f64);

impl HeatPerArea {
    /// Construct from joules per square metre [J/m²]
    pub const fn from_joules_per_square_meter(value: f64) -> Self {
        Self(value)
    }

    /// Construct from watt-minutes per square metre [W·min/m²] (1 W·min/m² = 60 J/m²)
    pub const fn from_watt_minutes_per_square_meter(value: f64) -> Self {
        Self(value * 60.0)
    }

    /// Value in joules per square metre [J/m²]
    pub const fn as_joules_per_square_meter(self) -> f64 {
        self.0
    }

    /// Value in watt-minutes per square metre [W·min/m²] (1 W·min/m² = 60 J/m²)
    pub const fn as_watt_minutes_per_square_meter(self) -> f64 {
        self.0 / 60.0
    }
}

/// Dry (sensible) thermal resistance per unit area, in square metre kelvins per watt.
///
/// A general resistance, such as a boundary air layer's. [`ClothingInsulation`] is the same
/// dimension built around the clo scale, for clothing ensembles.
///
/// Units: SI m²·K/W (`as_si`, [`as_square_meter_kelvins_per_watt`](Self::as_square_meter_kelvins_per_watt));
/// IP (inch-pound) ft²·h·°F/Btu (`as_ip`), the same IP unit and factor as
/// [`ClothingInsulation`].
///
/// # Examples
///
/// ```
/// use thermalcomfort::{ThermalResistance, UnitSystem};
///
/// let r_a = ThermalResistance::from_square_meter_kelvins_per_watt(0.11);
/// assert!((r_a.as_square_meter_kelvins_per_watt() - 0.11).abs() < 1e-12);
/// // 1 m²·K/W = 5.678 263 ft²·h·°F/Btu
/// assert!((ThermalResistance::from_si(1.0).as_ip() - 5.678_263).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ThermalResistance(f64);

impl ThermalResistance {
    /// Construct from square metre kelvins per watt [m²·K/W]
    pub const fn from_square_meter_kelvins_per_watt(value: f64) -> Self {
        Self(value)
    }

    /// Value in square metre kelvins per watt [m²·K/W]
    pub const fn as_square_meter_kelvins_per_watt(self) -> f64 {
        self.0
    }
}

/// Evaporative resistance per unit area, in square metre kilopascals per watt.
///
/// The resistance to latent heat transfer of clothing or a boundary layer: the
/// water-vapour pressure difference, per unit of evaporative heat flux, across it.
///
/// Units: SI m²·kPa/W (`as_si`, [`as_square_meter_kilopascals_per_watt`](Self::as_square_meter_kilopascals_per_watt));
/// IP (inch-pound) ft²·h·psi/Btu (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{EvaporativeResistance, UnitSystem};
///
/// let r_e = EvaporativeResistance::from_square_meter_kilopascals_per_watt(0.015);
/// assert!((r_e.as_square_meter_kilopascals_per_watt() - 0.015).abs() < 1e-12);
/// // 1 ft²·h·psi/Btu = 2.185 627 m²·kPa/W
/// assert!((EvaporativeResistance::from_ip(1.0).as_si() - 2.185_627).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct EvaporativeResistance(f64);

impl EvaporativeResistance {
    /// Construct from square metre kilopascals per watt [m²·kPa/W]
    pub const fn from_square_meter_kilopascals_per_watt(value: f64) -> Self {
        Self(value)
    }

    /// Value in square metre kilopascals per watt [m²·kPa/W]
    pub const fn as_square_meter_kilopascals_per_watt(self) -> f64 {
        self.0
    }
}

/// Rate of change of a power, in watts per second.
///
/// JOS3 can cap how fast shivering thermogenesis rises or falls (Asaka, 2016: 0.0077 W/s).
///
/// Units: SI W/s (`as_si`, [`as_watts_per_second`](Self::as_watts_per_second)); IP
/// (inch-pound) Btu/(h·s) (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{PowerRate, UnitSystem};
///
/// let limit = PowerRate::from_watts_per_second(0.0077);
/// assert!((limit.as_watts_per_second() - 0.0077).abs() < 1e-15);
/// // 1 W/s = 3.412 14 Btu/(h·s)
/// assert!((PowerRate::from_si(1.0).as_ip() - 3.412_14).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PowerRate(f64);

impl PowerRate {
    /// Construct from watts per second [W/s]
    pub const fn from_watts_per_second(value: f64) -> Self {
        Self(value)
    }

    /// Value in watts per second [W/s]
    pub const fn as_watts_per_second(self) -> f64 {
        self.0
    }
}

/// Driving coefficient of regulatory sweating, in grams per square metre per hour per kelvin.
///
/// The two-node models' `c_sw`: the sweat rate, in g/(m²·h), that each kelvin of mean body
/// temperature above its set point drives (`reg_sw = c_sw · warm_b · exp(warm_sk / 10.7)`).
///
/// Units: SI g/(m²·h·K) (`as_si`, [`as_grams_per_square_meter_per_hour_per_kelvin`](Self::as_grams_per_square_meter_per_hour_per_kelvin));
/// IP (inch-pound) lb/(ft²·h·°F) (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{SweatingCoefficient, UnitSystem};
///
/// let c_sw = SweatingCoefficient::from_grams_per_square_meter_per_hour_per_kelvin(170.0);
/// assert!((c_sw.as_grams_per_square_meter_per_hour_per_kelvin() - 170.0).abs() < 1e-12);
/// // 1 lb/(ft²·h·°F) = 8788.37 g/(m²·h·K)
/// assert!((SweatingCoefficient::from_ip(1.0).as_si() - 8788.37).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SweatingCoefficient(f64);

impl SweatingCoefficient {
    /// Construct from grams per square metre per hour per kelvin [g/(m²·h·K)]
    pub const fn from_grams_per_square_meter_per_hour_per_kelvin(value: f64) -> Self {
        Self(value)
    }

    /// Value in grams per square metre per hour per kelvin [g/(m²·h·K)]
    pub const fn as_grams_per_square_meter_per_hour_per_kelvin(self) -> f64 {
        self.0
    }
}

/// Driving coefficient of vasodilation, in litres per square metre per hour per kelvin.
///
/// The two-node models' `c_dil`: the skin blood flow, in L/(m²·h), that each kelvin of core
/// temperature above its set point adds (`skbf = (skbf_n + c_dil · warm_c) / (1 + c_str · cold_s)`).
///
/// Units: SI L/(m²·h·K) (`as_si`, [`as_liters_per_square_meter_per_hour_per_kelvin`](Self::as_liters_per_square_meter_per_hour_per_kelvin));
/// IP (inch-pound) gal/(ft²·h·°F) (`as_ip`), in US gallons of 3.785 411 784 L.
///
/// # Examples
///
/// ```
/// use thermalcomfort::{UnitSystem, VasodilationCoefficient};
///
/// let c_dil = VasodilationCoefficient::from_liters_per_square_meter_per_hour_per_kelvin(120.0);
/// assert!((c_dil.as_liters_per_square_meter_per_hour_per_kelvin() - 120.0).abs() < 1e-12);
/// // 1 gal/(ft²·h·°F) = 73.3425 L/(m²·h·K)
/// assert!((VasodilationCoefficient::from_ip(1.0).as_si() - 73.3425).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct VasodilationCoefficient(f64);

impl VasodilationCoefficient {
    /// Construct from litres per square metre per hour per kelvin [L/(m²·h·K)]
    pub const fn from_liters_per_square_meter_per_hour_per_kelvin(value: f64) -> Self {
        Self(value)
    }

    /// Value in litres per square metre per hour per kelvin [L/(m²·h·K)]
    pub const fn as_liters_per_square_meter_per_hour_per_kelvin(self) -> f64 {
        self.0
    }
}

/// Driving coefficient of vasoconstriction, per kelvin.
///
/// The two-node models' `c_str`: how much each kelvin of skin temperature below its set
/// point adds to the divisor of skin blood flow (the `1 + c_str · cold_s` denominator).
///
/// Units: SI 1/K (`as_si`, [`as_per_kelvin`](Self::as_per_kelvin)); IP (inch-pound) 1/°F
/// (`as_ip`).
///
/// # Examples
///
/// ```
/// use thermalcomfort::{UnitSystem, VasoconstrictionCoefficient};
///
/// let c_str = VasoconstrictionCoefficient::from_per_kelvin(0.5);
/// assert!((c_str.as_per_kelvin() - 0.5).abs() < 1e-15);
/// // 0.5 per kelvin is 0.5 / 1.8 per °F
/// assert!((c_str.as_ip() - 0.5 / 1.8).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct VasoconstrictionCoefficient(f64);

impl VasoconstrictionCoefficient {
    /// Construct from a coefficient per kelvin [1/K]
    pub const fn from_per_kelvin(value: f64) -> Self {
        Self(value)
    }

    /// Value per kelvin [1/K]
    pub const fn as_per_kelvin(self) -> f64 {
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

/// Age, in years.
///
/// PET, the ridge-regression predictor and JOS3 all take an age; a shared type stops a
/// height in centimetres or a weight in kilograms being handed to the age slot of a
/// long run of body parameters. Unbounded, because the three models validate it
/// differently upstream (JOS3's `validate_body_parameters` accepts 5–100; the others do
/// not check), and the model, not the type, is the place to reproduce that.
///
/// # Examples
///
/// ```
/// use thermalcomfort::Age;
///
/// let age = Age::from_years(23.0);
/// assert!((age.as_years() - 23.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Age(f64);

impl Age {
    /// Construct from an age in years
    pub const fn from_years(value: f64) -> Self {
        Self(value)
    }

    /// The age in years
    pub const fn as_years(self) -> f64 {
        self.0
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
        assert!(
            (CardiacIndex::from_liters_per_minute_per_square_meter(0.0)
                .as_liters_per_minute_per_square_meter())
            .abs()
                < 1e-12
        );
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
