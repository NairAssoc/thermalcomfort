//! The JOS3 simulation loop and public surface.
//!
//! Mirrors:
//! `pythermalcomfort/models/jos3.py`
//! (pythermalcomfort 4.4.0, 1651 lines) — specifically the `JOS3` class's constructor,
//! `simulate`/`_run`/`_reset_setpt`, and the property surface.
//!
//! # API shape
//!
//! Python's `JOS3` is a mutable-property class: construct it, then assign to
//! `model.tdb`, `model.v`, etc. and call `model.simulate(times, dtime)` repeatedly. This
//! port instead uses a builder for construction ([`Jos3Builder`]) and an explicit
//! step/advance call ([`Jos3Model::advance`]) that takes the whole environment snapshot
//! for that call ([`Jos3Conditions`]) plus the loop count and time step, rather than
//! mutating fields one at a time and calling a bare `simulate(times, dtime)`. This is a
//! deliberate, repo-owner-decided departure from mirroring Python's mutable-property
//! shape; everything else — constructor arguments and their defaults, what is
//! settable, and what `_run` computes and returns each step — matches Python exactly.
//!
//! # Ported
//!
//! - `JOS3.__init__` -> [`Jos3Builder`]/[`Jos3Model`] construction (which runs
//!   `_reset_setpt` internally, as Python's constructor does).
//! - `JOS3._calculate_operative_temp_when_pmv_is_zero` -> `operative_temp_for_pmv_zero`.
//! - `JOS3._reset_setpt` -> `Jos3Model::reset_setpoint`.
//! - `JOS3.simulate`/`JOS3._run` -> [`Jos3Model::advance`]/`Jos3Model::run_step`.
//! - `JOS3.results`/`JOS3.dict_results` -> [`Jos3Model::results`], which returns a
//!   [`Jos3Results`] of `alloc::Vec` trajectories (one entry appended per step), the same
//!   shape settled on for `two_nodes_gagge_sleep`'s per-minute result. `to_csv` is not
//!   ported: it is an I/O convenience (`std::fs`, `csv` writer) with nothing to port to in
//!   `no_std`.
//! - The property getters that only reindex the existing body-temperature state
//!   (`t_skin`, `t_core`, `t_cb`, `t_artery`, `t_vein`, `t_superficial_vein`, `t_muscle`,
//!   `t_fat`, `t_skin_mean`, `bsa`, `bmr`, `body_names`) -> methods of the same name on
//!   [`Jos3Model`].
//!
//! # Not ported
//!
//! - `JOS3.to_csv`: see above.
//! - `JOS3._set_ex_q` / `self.ex_q`: a leading-underscore private helper (never called
//!   elsewhere in `jos3.py`) that lets a caller inject an extra per-tissue heat gain.
//!   Nothing in the public, documented surface of the class (its class-level `Attributes`
//!   docstring, or the API shape this port was asked to build) exposes it, so `arr_q` here
//!   never has anything added to it from outside the step itself — equivalent to Python's
//!   `ex_q` always being all zero, which is what every caller of the public API gets
//!   anyway (nothing else in `jos3.py` ever calls `_set_ex_q`).
//! - Manual `hc`/`hr` override (Python's `self._hc`/`self._hr`, checked in `_run` but
//!   set only via direct attribute assignment, `model._hc = ...`): there is no public
//!   setter for either in Python's property list, so no caller of the public API can
//!   reach this branch. `Jos3Model::run_step` always computes `hc`/`hr` from
//!   [`threg::conv_coef`]/[`threg::rad_coef`], which is what every public-API caller
//!   observes in Python too.
//! - `JOS3.to`/`JOS3.r_t`/`JOS3.r_et`/`JOS3.w`/`JOS3.w_mean` instantaneous getters
//!   (properties that recompute from the *current* state without advancing the
//!   simulation): every one of these values is already produced, computed the same way,
//!   and available per step through [`Jos3Model::results`] after any [`Jos3Model::advance`]
//!   call — nothing observable is missing, only a zero-argument shortcut to recompute the
//!   same numbers again outside a step.
//!
//! # `NUM_NODES`-sized matrices
//!
//! The 85x85 linear solve that used to be `np.linalg.inv(arr_a).dot(arr)` is a direct
//! `nalgebra` LU solve here (`arr_a.lu().solve(&arr)`) rather than an explicit matrix
//! inverse. See `Jos3Model::run_step` for the note on where this can differ from
//! Python in the last few digits.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::fabs;
use nalgebra::{DMatrix, DVector};

use super::construction::{self, BodyParameterError, BodyPartsInputError};
use super::matrix::{self, IDICT, NUM_NODES};
use super::parameters::{BODY_PART_NAMES, NUM_BODY_PARTS, defaults};
use super::thermoregulation::{self as threg, Posture, ShiveringOptions, ThermoregulationError};
use core::time::Duration;

use crate::models::pmv::PmvPpdIsoOptions;
use crate::utilities::{BsaFormula, antoine, round_to};
use crate::{
    ActivityRatio, BmrEquation, BodyFat, CardiacIndex, Length, Mass, MetabolicRate, Sex,
    Temperature,
};

/// Number of body segments that carry a superficial-vein node: the 12 limb segments
/// (shoulders, arms, hands, thighs, legs, feet). Python: `len(VINDEX["sfvein"])`.
pub const NUM_SFVEIN_PARTS: usize = 12;

/// Number of body segments with a muscle (and fat) layer: head and pelvis only.
/// Python: `len(VINDEX["muscle"])` (== `len(VINDEX["fat"])`).
pub const NUM_MUSCLE_FAT_PARTS: usize = 2;

// ---------------------------------------------------------------------------
// PerBodyPart / Jos3Conditions
// ---------------------------------------------------------------------------

/// A value for all 17 body segments, given in one of the three forms Python's
/// `to_array_body_parts` dispatches on at runtime: a single value broadcast to every
/// segment, an explicit value per segment, or a name/value mapping. Rust's type system
/// makes the choice static instead, mirroring how [`construction`] already splits
/// `to_array_body_parts` into named functions per case (see
/// [`construction::to_array_body_parts_scalar`], [`construction::to_array_body_parts_by_name`]).
///
/// [`BySegment`](PerBodyPart::BySegment) carries an owned `[f64; 17]` rather than
/// borrowing a slice: a fixed-size array already statically has the length Python's
/// runtime check enforces, so there is nothing left to validate — no
/// `to_array_body_parts_from_slice` function exists in [`construction`] for this case,
/// unlike the other two. Owning the array also keeps [`Jos3Conditions`] free of
/// self-referential borrows of a [`Jos3Model`] it was built from (see
/// [`Jos3Model::conditions`]).
#[derive(Debug, Clone, Copy)]
pub enum PerBodyPart<'a> {
    /// The same value for every body segment. Python: `to_array_body_parts(inp)` for
    /// `inp: int | float`.
    Uniform(f64),
    /// One value per body segment, in [`BODY_PART_NAMES`] order. Python:
    /// `to_array_body_parts(inp)` for `inp: list | np.ndarray`.
    BySegment([f64; NUM_BODY_PARTS]),
    /// Name/value pairs, order-independent, one for every entry in [`BODY_PART_NAMES`].
    /// Python: `to_array_body_parts(inp)` for `inp: dict`.
    ByName(&'a [(&'a str, f64)]),
}

impl<'a> PerBodyPart<'a> {
    /// Resolve to a `[f64; 17]` in [`BODY_PART_NAMES`] order.
    ///
    /// # Errors
    ///
    /// Returns [`BodyPartsInputError::MissingKey`] if a [`PerBodyPart::ByName`] input is
    /// missing one of the 17 body-part names.
    pub fn resolve(&self) -> Result<[f64; NUM_BODY_PARTS], BodyPartsInputError> {
        match self {
            PerBodyPart::Uniform(value) => Ok(construction::to_array_body_parts_scalar(*value)),
            PerBodyPart::BySegment(values) => Ok(*values),
            PerBodyPart::ByName(pairs) => construction::to_array_body_parts_by_name(pairs),
        }
    }
}

/// The settable environment for one [`Jos3Model::advance`] call.
///
/// Mirrors the settable properties of Python's `JOS3`: `tdb`, `tr`, `rh`, `v`, `clo`,
/// `par`, `posture`, and `to`. Every per-segment field accepts a uniform value or a
/// per-segment one via [`PerBodyPart`], matching Python's `to_array_body_parts`
/// broadcasting for each of these properties' setters.
///
/// `to` mirrors Python's `to` *setter* specifically (not the `to` getter, which is
/// `Jos3Model::to`): Python's `@to.setter` assigns the same input to both `_tdb` and
/// `_tr`. Here, `to: Some(_)` does the same and takes priority over `tdb`/`tr` when
/// applied; `to: None` leaves `tdb`/`tr` as given.
#[derive(Debug, Clone, Copy)]
pub struct Jos3Conditions<'a> {
    /// Dry bulb air temperature, °C. Python: `JOS3.tdb`.
    pub tdb: PerBodyPart<'a>,
    /// Mean radiant temperature, °C. Python: `JOS3.tr`.
    pub tr: PerBodyPart<'a>,
    /// Relative humidity, %. Python: `JOS3.rh`.
    pub rh: PerBodyPart<'a>,
    /// Air velocity, m/s. Python: `JOS3.v`.
    pub v: PerBodyPart<'a>,
    /// Clothing insulation, clo. Python: `JOS3.clo`.
    pub clo: PerBodyPart<'a>,
    /// Physical activity ratio. Python: `JOS3.par`.
    pub par: ActivityRatio,
    /// Body posture. Python: `JOS3.posture`.
    pub posture: Posture,
    /// Operative temperature override; when `Some`, replaces both `tdb` and `tr`.
    /// Python: `JOS3.to` setter.
    pub to: Option<PerBodyPart<'a>>,
}

impl Default for Jos3Conditions<'_> {
    /// Python's `Default` values for `tdb`, `tr`, `rh`, `v`, `clo`, and `par`, standing
    /// posture, and no `to` override — i.e. the environment a freshly-constructed
    /// `JOS3()` has *before* `_reset_setpt` overwrites it. To reproduce a model's
    /// current environment instead (e.g. to change one field and leave the rest),
    /// use [`Jos3Model::conditions`].
    fn default() -> Self {
        Self {
            tdb: PerBodyPart::Uniform(defaults::DRY_BULB_AIR_TEMPERATURE),
            tr: PerBodyPart::Uniform(defaults::MEAN_RADIANT_TEMPERATURE),
            rh: PerBodyPart::Uniform(defaults::RELATIVE_HUMIDITY),
            v: PerBodyPart::Uniform(defaults::AIR_SPEED),
            clo: PerBodyPart::Uniform(defaults::CLOTHING_INSULATION),
            par: ActivityRatio::from_ratio(defaults::PHYSICAL_ACTIVITY_RATIO),
            posture: Posture::Standing,
            to: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Jos3Options
// ---------------------------------------------------------------------------

/// Thermoregulation behavior toggles. Python: the `self.options` dict, a plain mutable
/// instance attribute (not a property) that callers may write directly
/// (`model.options["cold_acclimated"] = True`) — mirrored here as a public field on
/// [`Jos3Model`] rather than a setter method, for the same reason.
///
/// Python's dict also has a `"shivering"` key; it is read nowhere in `jos3.py` or
/// `thermoregulation.py` (confirmed by grepping both), matching Python's own comment
/// `# TODO shivering is not used in the model`. It is omitted here rather than carried
/// forward as an inert field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jos3Options {
    /// Whether to include non-shivering thermogenesis. Python:
    /// `options["nonshivering_thermogenesis"]`.
    pub nonshivering_thermogenesis: bool,
    /// Whether the subject is cold-acclimated (affects non-shivering thermogenesis).
    /// Python: `options["cold_acclimated"]`.
    pub cold_acclimated: bool,
    /// Whether to apply the Asaka (2016) shivering-onset threshold. Python:
    /// `options["shivering_threshold"]`.
    pub shivering_threshold: bool,
    /// Rate limit on shivering signal change, \[W/s\]. `None` disables the limit
    /// (Python: `False`); `Some(0.0077)` is Python's default-rate sentinel (`True`);
    /// `Some(rate)` is a custom rate. Python: `options["limit_dshiv/dt"]`.
    pub limit_dshiv_dt: Option<f64>,
    /// Whether brown adipose tissue is BAT-positive (affects non-shivering
    /// thermogenesis). Python: `options["bat_positive"]`.
    pub bat_positive: bool,
    /// Whether to force AVA (arteriovenous anastomoses) blood flow to zero during
    /// passive steps. Python: `options["ava_zero"]`.
    pub ava_zero: bool,
}

impl Default for Jos3Options {
    fn default() -> Self {
        Self {
            nonshivering_thermogenesis: true,
            cold_acclimated: false,
            shivering_threshold: false,
            limit_dshiv_dt: None,
            bat_positive: false,
            ava_zero: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A [`Jos3Model::advance`] call could not be completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jos3Error {
    /// A [`Jos3Conditions`] field could not be resolved to a `[f64; 17]`.
    BodyParts(BodyPartsInputError),
    /// A thermoregulation precondition was violated (e.g. `par < 1`, or a negative
    /// convective/radiative coefficient reached `dry_r`/`wet_r`).
    Thermoregulation(ThermoregulationError),
    /// The 85x85 system matrix was singular and could not be solved.
    ///
    /// Not expected in practice: the matrix is `-(conductance + blood flow)` off the
    /// diagonal and `row_sum + 1` on it, which is diagonally dominant (hence
    /// nonsingular) for any physically sane conductance/blood-flow/capacity input. This
    /// variant exists so a pathological input is reported rather than panicking.
    SingularSystem,
}

impl core::fmt::Display for Jos3Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Jos3Error::BodyParts(e) => write!(f, "{e}"),
            Jos3Error::Thermoregulation(e) => write!(f, "{e}"),
            Jos3Error::SingularSystem => {
                write!(
                    f,
                    "the JOS3 system matrix was singular and could not be solved"
                )
            }
        }
    }
}

impl From<BodyPartsInputError> for Jos3Error {
    fn from(e: BodyPartsInputError) -> Self {
        Jos3Error::BodyParts(e)
    }
}

impl From<ThermoregulationError> for Jos3Error {
    fn from(e: ThermoregulationError) -> Self {
        Jos3Error::Thermoregulation(e)
    }
}

/// A [`Jos3Builder::build`] call could not construct a [`Jos3Model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jos3BuildError {
    /// A body parameter (height, weight, age, or body fat) was out of range. Python:
    /// `validate_body_parameters` raises `ValueError`.
    Validation(BodyParameterError),
    /// Computing the initial steady-state set-point temperatures
    /// (`Jos3Model::reset_setpoint`, Python's `_reset_setpt`) failed.
    Simulation(Jos3Error),
}

impl core::fmt::Display for Jos3BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Jos3BuildError::Validation(e) => write!(f, "{e}"),
            Jos3BuildError::Simulation(e) => write!(f, "{e}"),
        }
    }
}

impl From<BodyParameterError> for Jos3BuildError {
    fn from(e: BodyParameterError) -> Self {
        Jos3BuildError::Validation(e)
    }
}

impl From<Jos3Error> for Jos3BuildError {
    fn from(e: Jos3Error) -> Self {
        Jos3BuildError::Simulation(e)
    }
}

// ---------------------------------------------------------------------------
// Jos3Builder
// ---------------------------------------------------------------------------

/// Builds a [`Jos3Model`]. Python: `JOS3.__init__`'s keyword arguments.
#[derive(Debug, Clone, Copy)]
pub struct Jos3Builder {
    height: Length,
    weight: Mass,
    age: i32,
    fat: BodyFat,
    sex: Sex,
    ci: CardiacIndex,
    bmr_equation: BmrEquation,
    bsa_equation: BsaFormula,
}

impl Default for Jos3Builder {
    fn default() -> Self {
        Self {
            height: Length::from_meters(defaults::HEIGHT),
            weight: Mass::from_kilograms(defaults::WEIGHT),
            age: defaults::AGE,
            fat: BodyFat::default(),
            sex: Sex::Male,
            ci: CardiacIndex::DEFAULT,
            bmr_equation: BmrEquation::default(),
            bsa_equation: BsaFormula::default(),
        }
    }
}

impl Jos3Builder {
    /// Start from pythermalcomfort's defaults: height 1.72 m, weight 74.43 kg, age 20,
    /// 15% body fat, male, cardiac index 2.59, Harris-Benedict BMR, DuBois BSA.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Body height. Python: `height`.
    #[must_use]
    pub const fn height(mut self, height: Length) -> Self {
        self.height = height;
        self
    }

    /// Body weight. Python: `weight`.
    #[must_use]
    pub const fn weight(mut self, weight: Mass) -> Self {
        self.weight = weight;
        self
    }

    /// Age, in years. Python: `age`.
    #[must_use]
    pub const fn age(mut self, age: i32) -> Self {
        self.age = age;
        self
    }

    /// Body fat percentage. Python: `fat`.
    #[must_use]
    pub const fn fat(mut self, fat: BodyFat) -> Self {
        self.fat = fat;
        self
    }

    /// Biological sex. Python: `sex`.
    #[must_use]
    pub const fn sex(mut self, sex: Sex) -> Self {
        self.sex = sex;
        self
    }

    /// Cardiac index. Python: `ci`.
    #[must_use]
    pub const fn cardiac_index(mut self, ci: CardiacIndex) -> Self {
        self.ci = ci;
        self
    }

    /// Basal metabolic rate equation. Python: `bmr_equation`.
    #[must_use]
    pub const fn bmr_equation(mut self, bmr_equation: BmrEquation) -> Self {
        self.bmr_equation = bmr_equation;
        self
    }

    /// Body surface area equation. Python: `bsa_equation`.
    #[must_use]
    pub const fn bsa_equation(mut self, bsa_equation: BsaFormula) -> Self {
        self.bsa_equation = bsa_equation;
        self
    }

    /// Construct the [`Jos3Model`].
    ///
    /// Runs `validate_body_parameters` (Python: called at the top of `__init__`) and
    /// then the same steady-state set-point calculation Python's `__init__` runs via
    /// `_reset_setpt` — the reference-environment PMV=0 search followed by 10 passive
    /// steps — so the returned model starts in the same state a freshly-constructed
    /// Python `JOS3()` does, history included.
    ///
    /// # Errors
    ///
    /// Returns [`Jos3BuildError::Validation`] if height, weight, age, or body fat is out
    /// of range, or [`Jos3BuildError::Simulation`] if the steady-state calculation fails
    /// (not expected for any parameter combination that passes validation).
    pub fn build(self) -> Result<Jos3Model, Jos3BuildError> {
        let height_m = self.height.as_meters();
        let weight_kg = self.weight.as_kilograms();
        let fat_percent = self.fat.as_percent();
        construction::validate_body_parameters(height_m, weight_kg, self.age, fat_percent)?;

        let ci = self.ci.as_liters_per_minute_per_square_meter();
        let bsa = construction::local_bsa(height_m, weight_kg, self.bsa_equation);
        let cdt = construction::conductance(height_m, weight_kg, self.bsa_equation, fat_percent);
        let cap = construction::capacity(height_m, weight_kg, self.bsa_equation, self.age, ci);

        let mut model = Jos3Model {
            height_m,
            weight_kg,
            fat_percent,
            sex: self.sex,
            age: self.age,
            ci,
            bmr_equation: self.bmr_equation,
            bsa_equation: self.bsa_equation,

            bsa,
            cdt,
            cap,

            cr_set_point: [defaults::CORE_TEMPERATURE; NUM_BODY_PARTS],
            sk_set_point: [defaults::SKIN_TEMPERATURE; NUM_BODY_PARTS],
            t_body: vec![defaults::OTHER_BODY_TEMPERATURE; NUM_NODES],

            tdb: [defaults::DRY_BULB_AIR_TEMPERATURE; NUM_BODY_PARTS],
            tr: [defaults::MEAN_RADIANT_TEMPERATURE; NUM_BODY_PARTS],
            rh: [defaults::RELATIVE_HUMIDITY; NUM_BODY_PARTS],
            v: [defaults::AIR_SPEED; NUM_BODY_PARTS],
            clo: [defaults::CLOTHING_INSULATION; NUM_BODY_PARTS],
            par: defaults::PHYSICAL_ACTIVITY_RATIO,
            posture: Posture::Standing,

            pre_shiv: 0.0,
            elapsed_seconds: 0.0,

            options: Jos3Options::default(),
            history: Jos3Results::with_capacity(1),
        };

        model.reset_setpoint()?;
        Ok(model)
    }
}

// ---------------------------------------------------------------------------
// operative_temp_for_pmv_zero
// ---------------------------------------------------------------------------

/// Find the operative temperature at which PMV (ISO 7730) is zero, for a subject at
/// the given air speed, relative humidity, metabolic rate, and clothing insulation.
///
/// Python: `JOS3._calculate_operative_temp_when_pmv_is_zero`. This is a free function
/// here rather than a method: Python's version does not read or write `self` either.
///
/// Ported faithfully, including the inner retry loop's behavior when the initial PMV
/// calculation is NaN: Python resets `to` to `initial_to` at the top of every retry
/// iteration before recomputing PMV from the *same* `v`/`rh`/`met`/`clo`, so every retry
/// reproduces the same (NaN) result and the loop cannot actually converge — it always
/// falls through to `return to` with `to` left NaN from the last iteration. That is
/// upstream's behavior for this edge case, not a defect introduced here.
#[must_use]
fn operative_temp_for_pmv_zero(v: f64, rh: f64, met: f64, clo: f64) -> f64 {
    let initial_to = 28.0;
    let tolerance = 0.001;
    let max_iterations = 100;
    let mut adjustment_factor = 3.0;
    let retry_adjustment_factor = adjustment_factor * 200.0;
    let retry_attempts = 100;

    let mut to = initial_to;
    for _ in 0..max_iterations {
        let pmv_value = pmv_value_at(to, v, rh, met, clo);

        if pmv_value.is_nan() {
            for _ in 0..retry_attempts {
                adjustment_factor = retry_adjustment_factor;
                to = initial_to;
                let pmv_value = pmv_value_at(to, v, rh, met, clo);
                if fabs(pmv_value) < tolerance {
                    return to;
                }
                to -= pmv_value / adjustment_factor;
            }
            return to;
        }

        if fabs(pmv_value) < tolerance {
            return to;
        }
        to -= pmv_value / adjustment_factor;
    }
    to
}

/// One `pmv_ppd_iso(to, to, v, rh, met, clo)` evaluation. Python: the `pmv_ppd_iso(...)`
/// call inside `_calculate_operative_temp_when_pmv_is_zero`.
fn pmv_value_at(to: f64, v: f64, rh: f64, met: f64, clo: f64) -> f64 {
    // Deliberately the plain-f64 entry point, NOT the newtype one. `Temperature` stores
    // kelvin, so `from_celsius(x).as_celsius()` loses the last ULP -- and this search walks
    // `to` right onto the 30 degrees C ISO limit and relies on crossing it to trigger the
    // NaN-and-retry branch below. Round-tripping through the newtype rounded the crossing
    // away, so the retry never fired and the set points came out ~2 degrees C wrong.
    crate::models::pmv::pmv_ppd_iso_celsius(to, to, v, rh, met, clo, PmvPpdIsoOptions::default())
        .pmv
}

// ---------------------------------------------------------------------------
// Jos3Model
// ---------------------------------------------------------------------------

/// A JOS3 simulation. Python: an instance of the `JOS3` class.
///
/// Construct with [`Jos3Builder`], advance with [`Jos3Model::advance`], and read
/// accumulated per-step trajectories with [`Jos3Model::results`].
#[derive(Debug)]
pub struct Jos3Model {
    // Body parameters, fixed for the model's lifetime (Python has no setters for
    // these after `__init__`).
    height_m: f64,
    weight_kg: f64,
    fat_percent: f64,
    sex: Sex,
    age: i32,
    ci: f64,
    bmr_equation: BmrEquation,
    bsa_equation: BsaFormula,

    // Derived physiological constants, computed once at construction.
    bsa: [f64; NUM_BODY_PARTS],
    /// 85x85 thermal conductance matrix, row-major. Python: `self._cdt`.
    cdt: Vec<f64>,
    /// 85-length thermal capacity vector. Python: `self._cap`.
    cap: Vec<f64>,

    // Thermoregulation set points, established by `reset_setpoint` and otherwise fixed.
    cr_set_point: [f64; NUM_BODY_PARTS],
    sk_set_point: [f64; NUM_BODY_PARTS],

    // The 85-node body temperature state. Python: `self._t_body`.
    t_body: Vec<f64>,

    // Settable environment (see `Jos3Conditions`).
    tdb: [f64; NUM_BODY_PARTS],
    tr: [f64; NUM_BODY_PARTS],
    rh: [f64; NUM_BODY_PARTS],
    v: [f64; NUM_BODY_PARTS],
    clo: [f64; NUM_BODY_PARTS],
    par: f64,
    posture: Posture,

    // Thermoregulation state threaded between steps.
    pre_shiv: f64,
    elapsed_seconds: f64,

    /// Thermoregulation behavior toggles. Python: `self.options`, a mutable dict
    /// attribute; mirrored here as a mutable public field for the same reason.
    pub options: Jos3Options,

    history: Jos3Results,
}

impl Jos3Model {
    // -- Body parameter getters (Python: read-only properties with no setter) --

    /// Body height. Python: no direct getter, but the value is available via
    /// [`Jos3Results::height`].
    #[must_use]
    pub fn height(&self) -> Length {
        Length::from_meters(self.height_m)
    }

    /// Body weight.
    #[must_use]
    pub fn weight(&self) -> Mass {
        Mass::from_kilograms(self.weight_kg)
    }

    /// Body fat percentage.
    #[must_use]
    pub fn fat(&self) -> BodyFat {
        BodyFat::new(self.fat_percent).expect("validated at build time")
    }

    /// Biological sex.
    #[must_use]
    pub fn sex(&self) -> Sex {
        self.sex
    }

    /// Age, in years.
    #[must_use]
    pub fn age(&self) -> i32 {
        self.age
    }

    /// Cardiac index.
    #[must_use]
    pub fn cardiac_index(&self) -> CardiacIndex {
        CardiacIndex::from_liters_per_minute_per_square_meter(self.ci)
    }

    /// Names of the 17 body segments, in the order every per-segment array here uses.
    /// Python: `JOS3.body_names`.
    #[must_use]
    pub fn body_names(&self) -> [&'static str; NUM_BODY_PARTS] {
        BODY_PART_NAMES
    }

    /// Body surface area of each of the 17 body segments, \[m2\]. Python: `JOS3.bsa`.
    #[must_use]
    pub fn bsa(&self) -> [f64; NUM_BODY_PARTS] {
        self.bsa
    }

    /// Basal metabolic rate, \[W/m2\]. Python: `JOS3.bmr`.
    #[must_use]
    pub fn bmr(&self) -> f64 {
        threg::basal_met(
            self.height_m,
            self.weight_kg,
            self.age,
            self.sex,
            self.bmr_equation,
        ) / self.bsa.iter().sum::<f64>()
    }

    // -- Environment getters (Python: read/write properties) --

    /// Current dry bulb air temperature, °C, one per body segment. Python: `JOS3.tdb`.
    #[must_use]
    pub fn tdb(&self) -> [f64; NUM_BODY_PARTS] {
        self.tdb
    }

    /// Current mean radiant temperature, °C, one per body segment. Python: `JOS3.tr`.
    #[must_use]
    pub fn tr(&self) -> [f64; NUM_BODY_PARTS] {
        self.tr
    }

    /// Current relative humidity, %, one per body segment. Python: `JOS3.rh`.
    #[must_use]
    pub fn rh(&self) -> [f64; NUM_BODY_PARTS] {
        self.rh
    }

    /// Current air velocity, m/s, one per body segment. Python: `JOS3.v`.
    #[must_use]
    pub fn v(&self) -> [f64; NUM_BODY_PARTS] {
        self.v
    }

    /// Current clothing insulation, clo, one per body segment. Python: `JOS3.clo`.
    #[must_use]
    pub fn clo(&self) -> [f64; NUM_BODY_PARTS] {
        self.clo
    }

    /// Current physical activity ratio. Python: `JOS3.par`.
    #[must_use]
    pub fn par(&self) -> f64 {
        self.par
    }

    /// Current posture. Python: `JOS3.posture`.
    #[must_use]
    pub fn posture(&self) -> Posture {
        self.posture
    }

    /// The model's current environment, as a [`Jos3Conditions`] that can be modified
    /// and passed back to [`Jos3Model::advance`] — the equivalent of Python setting only
    /// one property (e.g. `model.v = 0.5`) and leaving the rest as previously set.
    ///
    /// Every field resolves through [`PerBodyPart::BySegment`], which owns its data, so
    /// the result does not borrow from `self` and does not block a later `&mut self`
    /// call such as `advance`.
    #[must_use]
    pub fn conditions(&self) -> Jos3Conditions<'static> {
        Jos3Conditions {
            tdb: PerBodyPart::BySegment(self.tdb),
            tr: PerBodyPart::BySegment(self.tr),
            rh: PerBodyPart::BySegment(self.rh),
            v: PerBodyPart::BySegment(self.v),
            clo: PerBodyPart::BySegment(self.clo),
            par: ActivityRatio::from_ratio(self.par),
            posture: self.posture,
            to: None,
        }
    }

    // -- Body temperature state getters (Python: read-only properties) --

    /// All 85 node temperatures, °C. Python: `JOS3.t_body`.
    #[must_use]
    pub fn t_body(&self) -> &[f64] {
        &self.t_body
    }

    /// Skin temperature of each of the 17 body segments, °C. Python: `JOS3.t_skin`.
    #[must_use]
    pub fn t_skin(&self) -> [f64; NUM_BODY_PARTS] {
        self.extract_full_layer(|li| li.skin)
    }

    /// Core temperature of each of the 17 body segments, °C. Python: `JOS3.t_core`.
    #[must_use]
    pub fn t_core(&self) -> [f64; NUM_BODY_PARTS] {
        self.extract_full_layer(|li| li.core)
    }

    /// Arterial temperature of each of the 17 body segments, °C. Python: `JOS3.t_artery`.
    #[must_use]
    pub fn t_artery(&self) -> [f64; NUM_BODY_PARTS] {
        self.extract_full_layer(|li| li.artery)
    }

    /// Venous temperature of each of the 17 body segments, °C. Python: `JOS3.t_vein`.
    #[must_use]
    pub fn t_vein(&self) -> [f64; NUM_BODY_PARTS] {
        self.extract_full_layer(|li| li.vein)
    }

    /// Central blood pool temperature, °C. Python: `JOS3.t_cb`.
    #[must_use]
    pub fn t_cb(&self) -> f64 {
        self.t_body[matrix::CB]
    }

    /// Superficial vein temperature of the 12 limb segments that have one, °C, in
    /// [`BODY_PART_NAMES`] order restricted to those segments. Python:
    /// `JOS3.t_superficial_vein`.
    #[must_use]
    pub fn t_superficial_vein(&self) -> [f64; NUM_SFVEIN_PARTS] {
        self.extract_nodes(matrix::index_by_layer("sfvein"))
    }

    /// Muscle temperature of head and pelvis (the only segments with a muscle layer),
    /// °C. Python: `JOS3.t_muscle`.
    #[must_use]
    pub fn t_muscle(&self) -> [f64; NUM_MUSCLE_FAT_PARTS] {
        self.extract_nodes(matrix::index_by_layer("muscle"))
    }

    /// Fat temperature of head and pelvis (the only segments with a fat layer), °C.
    /// Python: `JOS3.t_fat`.
    #[must_use]
    pub fn t_fat(&self) -> [f64; NUM_MUSCLE_FAT_PARTS] {
        self.extract_nodes(matrix::index_by_layer("fat"))
    }

    /// Mean skin temperature, weighted by local body surface area, °C. Python:
    /// `JOS3.t_skin_mean`.
    #[must_use]
    pub fn t_skin_mean(&self) -> f64 {
        threg::weighted_average(&self.t_skin(), &defaults::LOCAL_BSA)
    }

    /// Accumulated per-step results, one entry per call to [`Jos3Model::advance`] (the
    /// zeroth entry is the reference steady state established at construction). Python:
    /// `JOS3.results()`/`JOS3.dict_results()`.
    #[must_use]
    pub fn results(&self) -> &Jos3Results {
        &self.history
    }

    fn extract_full_layer(
        &self,
        pick: impl Fn(&matrix::LayerIndex) -> Option<usize>,
    ) -> [f64; NUM_BODY_PARTS] {
        let mut out = [0.0; NUM_BODY_PARTS];
        for (i, li) in IDICT.iter().enumerate() {
            out[i] = self.t_body[pick(li).expect("every body part has this layer")];
        }
        out
    }

    fn extract_nodes<const N: usize>(&self, node_indices: Vec<usize>) -> [f64; N] {
        let mut out = [0.0; N];
        for (k, &node) in node_indices.iter().enumerate() {
            out[k] = self.t_body[node];
        }
        out
    }

    // -- Simulation --

    /// Set the environment for this call and advance the simulation `times` steps of
    /// `dtime` seconds each, appending one [`Jos3Results`] entry per step. Python:
    /// setting `tdb`/`tr`/`rh`/`v`/`clo`/`par`/`posture`/`to` and then calling
    /// `simulate(times, dtime)`.
    ///
    /// # Errors
    ///
    /// Returns [`Jos3Error::BodyParts`] if a [`Jos3Conditions`] field cannot be resolved
    /// (e.g. a [`PerBodyPart::ByName`] missing a body-part name), or
    /// [`Jos3Error::Thermoregulation`] if `conditions.par < 1`.
    ///
    /// # Examples
    ///
    /// Two phases, changing the environment in between — the reason this is an explicit
    /// `advance` rather than a one-shot function.
    ///
    /// ```
    /// use thermalcomfort::models::jos3::Jos3Builder;
    /// use core::time::Duration;
    ///
    /// let mut sim = Jos3Builder::new().build().expect("defaults are valid");
    ///
    /// let mut conditions = sim.conditions();
    /// sim.advance(&conditions, 30, Duration::from_secs(60))?;
    ///
    /// // Step the air temperature down and carry straight on from the current state.
    /// conditions.tdb = thermalcomfort::models::jos3::PerBodyPart::Uniform(20.0);
    /// sim.advance(&conditions, 30, Duration::from_secs(60))?;
    ///
    /// assert_eq!(sim.results().t_skin_mean.len(), 61); // initial row + 60 steps
    /// # Ok::<(), thermalcomfort::models::jos3::Jos3Error>(())
    /// ```
    pub fn advance(
        &mut self,
        conditions: &Jos3Conditions<'_>,
        times: u32,
        dtime: Duration,
    ) -> Result<(), Jos3Error> {
        self.apply_conditions(conditions)?;
        let dtime_secs = dtime.as_secs_f64();
        for _ in 0..times {
            self.elapsed_seconds += dtime_secs;
            let row = self.run_step(dtime_secs, false)?;
            self.history.push(row);
        }
        Ok(())
    }

    fn apply_conditions(
        &mut self,
        conditions: &Jos3Conditions<'_>,
    ) -> Result<(), BodyPartsInputError> {
        self.tdb = conditions.tdb.resolve()?;
        self.tr = conditions.tr.resolve()?;
        self.rh = conditions.rh.resolve()?;
        self.v = conditions.v.resolve()?;
        self.clo = conditions.clo.resolve()?;
        self.par = conditions.par.as_ratio();
        self.posture = conditions.posture;
        if let Some(to) = &conditions.to {
            let resolved = to.resolve()?;
            self.tdb = resolved;
            self.tr = resolved;
        }
        Ok(())
    }

    /// Calculate the reference-environment steady-state set-point temperatures and seed
    /// the history with the resulting state. Python: `JOS3._reset_setpt`.
    fn reset_setpoint(&mut self) -> Result<(), Jos3Error> {
        let par = defaults::PHYSICAL_ACTIVITY_RATIO;
        let met = self.bmr() * par / MetabolicRate::MET_TO_W_M2;
        let rh = defaults::RELATIVE_HUMIDITY;
        let v = defaults::AIR_SPEED;
        let clo = 0.0;

        let to = operative_temp_for_pmv_zero(v, rh, met, clo);
        self.tdb = [to; NUM_BODY_PARTS];
        self.tr = [to; NUM_BODY_PARTS];
        self.rh = [rh; NUM_BODY_PARTS];
        self.v = [v; NUM_BODY_PARTS];
        self.clo = [clo; NUM_BODY_PARTS];
        self.par = par;

        self.options.ava_zero = true;
        let mut last_row = None;
        for _ in 0..10 {
            last_row = Some(self.run_step(60_000.0, true)?);
        }
        self.options.ava_zero = false;

        self.cr_set_point = self.t_core();
        self.sk_set_point = self.t_skin();

        // Python appends this last passive-step row as history[0] even though the
        // set points above were only just derived from it — see the module docs'
        // "why is the first element for a naked person" note, ported as-is.
        self.history
            .push(last_row.expect("the loop above runs 10 times"));
        Ok(())
    }

    /// Run one simulation step. Python: `JOS3._run`.
    ///
    /// `dtime` is in seconds; `passive` mirrors Python's `passive` flag (set-point
    /// temperatures are the current body temperatures rather than `cr_set_point`/
    /// `sk_set_point`, and `ava_zero` applies if enabled).
    fn run_step(&mut self, dtime: f64, passive: bool) -> Result<Jos3Row, Jos3Error> {
        let tcr = self.t_core();
        let tsk = self.t_skin();

        let hc = threg::fixed_hc(
            threg::conv_coef(self.posture, self.v, self.tdb, tsk),
            self.v,
        );
        let hr = threg::fixed_hr(threg::rad_coef(self.posture));

        let to = threg::operative_temp(self.tdb, self.tr, hc, hr);
        let r_t = threg::dry_r(hc, hr, self.clo)?;
        let i_clo = [defaults::CLOTHING_VAPOR_PERMEATION_EFFICIENCY; NUM_BODY_PARTS];
        let r_et = threg::wet_r(hc, self.clo, i_clo, defaults::LEWIS_RATE)?;

        let (setpt_cr, setpt_sk) = if passive {
            (tcr, tsk)
        } else {
            (self.cr_set_point, self.sk_set_point)
        };

        let mut err_cr = [0.0; NUM_BODY_PARTS];
        let mut err_sk = [0.0; NUM_BODY_PARTS];
        for i in 0..NUM_BODY_PARTS {
            err_cr[i] = tcr[i] - setpt_cr[i];
            err_sk[i] = tsk[i] - setpt_sk[i];
        }

        let (wet, e_sk, e_max, e_sweat) = threg::evaporation(
            err_cr,
            err_sk,
            tsk,
            self.tdb,
            self.rh,
            r_et,
            self.height_m,
            self.weight_kg,
            self.bsa_equation,
            self.age,
        );

        let bf_skin = threg::skin_blood_flow(
            err_cr,
            err_sk,
            self.height_m,
            self.weight_kg,
            self.bsa_equation,
            self.age,
            self.ci,
        );

        let (mut bf_ava_hand, mut bf_ava_foot) = threg::ava_blood_flow(
            err_cr,
            err_sk,
            self.height_m,
            self.weight_kg,
            self.bsa_equation,
            self.age,
            self.ci,
        );
        if self.options.ava_zero && passive {
            bf_ava_hand = 0.0;
            bf_ava_foot = 0.0;
        }

        // Python's `_run` always passes its (non-None) `self.options` dict, which makes
        // `shivering`'s `if options:` branches always taken regardless of the two
        // booleans inside it — see `ShiveringOptions`'s doc for how `new_pre_shiv` is
        // affected purely by `Some(_)` vs `None`. `Some` here, always, is the faithful
        // port of that "always truthy" call site.
        let shiv_options = Some(ShiveringOptions {
            shivering_threshold: self.options.shivering_threshold,
            limit_dshiv_dt: self.options.limit_dshiv_dt,
        });
        let (q_shiv, new_pre_shiv) = threg::shivering(
            err_cr,
            err_sk,
            tcr,
            tsk,
            self.height_m,
            self.weight_kg,
            self.bsa_equation,
            self.age,
            self.sex,
            dtime,
            shiv_options,
            self.pre_shiv,
        );
        self.pre_shiv = new_pre_shiv;

        let q_nst = if self.options.nonshivering_thermogenesis {
            threg::nonshivering(
                err_sk,
                self.height_m,
                self.weight_kg,
                self.bsa_equation,
                self.age,
                self.options.cold_acclimated,
                self.options.bat_positive,
            )
        } else {
            [0.0; NUM_BODY_PARTS]
        };

        let (mbase_cr, mbase_ms, mbase_fat, mbase_sk) = threg::local_mbase(
            self.height_m,
            self.weight_kg,
            self.age,
            self.sex,
            self.bmr_equation,
        );
        let q_bmr_total: f64 = mbase_cr.iter().sum::<f64>()
            + mbase_ms.iter().sum::<f64>()
            + mbase_fat.iter().sum::<f64>()
            + mbase_sk.iter().sum::<f64>();

        let q_work = threg::local_q_work(q_bmr_total, self.par)?;

        let (
            q_thermogenesis_core,
            q_thermogenesis_muscle,
            q_thermogenesis_fat,
            q_thermogenesis_skin,
        ) = threg::sum_m(
            (mbase_cr, mbase_ms, mbase_fat, mbase_sk),
            q_work,
            q_shiv,
            q_nst,
        );
        let q_thermogenesis_total: f64 = q_thermogenesis_core.iter().sum::<f64>()
            + q_thermogenesis_muscle.iter().sum::<f64>()
            + q_thermogenesis_fat.iter().sum::<f64>()
            + q_thermogenesis_skin.iter().sum::<f64>();

        let (bf_core, bf_muscle, bf_fat) = threg::cr_ms_fat_blood_flow(
            q_work,
            q_shiv,
            self.height_m,
            self.weight_kg,
            self.bsa_equation,
            self.age,
            self.ci,
        );

        // Respiratory heat loss uses the head segment's tdb/vapor pressure (index 0),
        // exactly as Python's `self._tdb[0]`/`p_a[0]` does.
        let p_a0 = antoine(Temperature::from_celsius(self.tdb[0])) * self.rh[0] / 100.0;
        let (res_sh, res_lh) = threg::resp_heat_loss(self.tdb[0], p_a0, q_thermogenesis_total);

        let mut shl_sk = [0.0; NUM_BODY_PARTS];
        for i in 0..NUM_BODY_PARTS {
            shl_sk[i] = (tsk[i] - to[i]) / r_t[i] * self.bsa[i];
        }

        let co = threg::sum_bf(
            bf_core,
            bf_muscle,
            bf_fat,
            bf_skin,
            bf_ava_hand,
            bf_ava_foot,
        );

        let mut wlesk_sum = 0.0;
        for i in 0..NUM_BODY_PARTS {
            wlesk_sum += (e_sweat[i] + 0.06 * e_max[i]) / 2418.0;
        }
        let wleres = res_lh / 2418.0;

        // ------------------------------------------------------------------
        // Matrix solve
        // ------------------------------------------------------------------
        let (bf_art, bf_vein) = matrix::vessel_blood_flow(
            &bf_core,
            &bf_muscle,
            &bf_fat,
            &bf_skin,
            bf_ava_hand,
            bf_ava_foot,
        );
        let mut arr_bf = matrix::local_arr(
            &bf_core,
            &bf_muscle,
            &bf_fat,
            &bf_skin,
            bf_ava_hand,
            bf_ava_foot,
        );
        arr_bf += matrix::whole_body(&bf_art, &bf_vein, bf_ava_hand, bf_ava_foot);
        for r in 0..NUM_NODES {
            let scale = dtime / self.cap[r];
            for c in 0..NUM_NODES {
                arr_bf[(r, c)] *= scale;
            }
        }

        let mut arr_cdt = DMatrix::from_row_slice(NUM_NODES, NUM_NODES, &self.cdt);
        for r in 0..NUM_NODES {
            let scale = dtime / self.cap[r];
            for c in 0..NUM_NODES {
                arr_cdt[(r, c)] *= scale;
            }
        }

        let mut arr_b = vec![0.0_f64; NUM_NODES];
        for i in 0..NUM_BODY_PARTS {
            let skin = IDICT[i].skin.expect("every body part has a skin node");
            arr_b[skin] += (1.0 / r_t[i]) * self.bsa[i];
        }
        for (b, &cap) in arr_b.iter_mut().zip(self.cap.iter()) {
            *b = *b / cap * dtime;
        }

        let mut arr_q = vec![0.0_f64; NUM_NODES];
        for i in 0..NUM_BODY_PARTS {
            let core = IDICT[i].core.expect("every body part has a core node");
            arr_q[core] += q_thermogenesis_core[i];
        }
        for i in 0..NUM_BODY_PARTS {
            if let Some(muscle) = IDICT[i].muscle {
                arr_q[muscle] += q_thermogenesis_muscle[i];
            }
        }
        for i in 0..NUM_BODY_PARTS {
            if let Some(fat) = IDICT[i].fat {
                arr_q[fat] += q_thermogenesis_fat[i];
            }
        }
        for i in 0..NUM_BODY_PARTS {
            let skin = IDICT[i].skin.expect("every body part has a skin node");
            arr_q[skin] += q_thermogenesis_skin[i];
        }
        // Respiratory: chest (body-part index 2) core node.
        let chest_core = IDICT[2].core.expect("chest has a core node");
        arr_q[chest_core] -= res_sh + res_lh;
        // Sweating.
        for i in 0..NUM_BODY_PARTS {
            let skin = IDICT[i].skin.expect("every body part has a skin node");
            arr_q[skin] -= e_sk[i];
        }
        // No extra heat gain term: see the module docs for why `ex_q` is not ported.
        for (q, &cap) in arr_q.iter_mut().zip(self.cap.iter()) {
            *q = *q / cap * dtime;
        }

        let mut arr_to = vec![0.0_f64; NUM_NODES];
        for i in 0..NUM_BODY_PARTS {
            let skin = IDICT[i].skin.expect("every body part has a skin node");
            arr_to[skin] += to[i];
        }

        let mut arr = vec![0.0_f64; NUM_NODES];
        for i in 0..NUM_NODES {
            arr[i] = self.t_body[i] + arr_b[i] * arr_to[i] + arr_q[i];
        }

        // arr_a_tria + arr_a_dia, built directly on the diagonal rather than as two
        // separate matrices summed afterward (see the type-level docs for why this is
        // equivalent): `combined = arr_cdt + arr_bf`; off-diagonal entries of `arr_a`
        // are `-combined[r, c]`, and the diagonal additionally gets `row_sum(combined) +
        // arr_b + 1` (the `+1` is Python's `np.eye(NUM_NODES)`).
        let mut combined = arr_cdt;
        combined += &arr_bf;
        let mut arr_a = -combined.clone();
        for i in 0..NUM_NODES {
            let row_sum: f64 = (0..NUM_NODES).map(|c| combined[(i, c)]).sum();
            arr_a[(i, i)] += row_sum + arr_b[i] + 1.0;
        }

        // Python computes an explicit inverse (`np.linalg.inv(arr_a)` then
        // `.dot(arr)`); this is a direct LU solve instead, which is more numerically
        // accurate and can differ from Python in the last few digits of `t_body`. In
        // cross-checks against pythermalcomfort 4.4.0 (see the tests below) the
        // difference was below the 2-decimal rounding applied to every output field, so
        // it never showed up in any reported value.
        let arr_vec = DVector::from_row_slice(&arr);
        let solved = arr_a
            .lu()
            .solve(&arr_vec)
            .ok_or(Jos3Error::SingularSystem)?;
        self.t_body = solved.iter().copied().collect();

        // ------------------------------------------------------------------
        // Output row
        // ------------------------------------------------------------------
        let new_t_skin = self.t_skin();
        let new_t_core = self.t_core();
        let muscle_positions = matrix::valid_index_by_layer("muscle");
        let fat_positions = matrix::valid_index_by_layer("fat");

        Ok(Jos3Row {
            simulation_time: self.elapsed_seconds,
            dt: dtime,
            t_skin_mean: round_to(
                threg::weighted_average(&new_t_skin, &defaults::LOCAL_BSA),
                2,
            ),
            t_skin: round_arr(new_t_skin, 2),
            t_core: round_arr(new_t_core, 2),
            w_mean: round_to(threg::weighted_average(&wet, &defaults::LOCAL_BSA), 2),
            w: round_arr(wet, 2),
            weight_loss_by_evap_and_res: round_to(wlesk_sum + wleres, 5),
            cardiac_output: round_to(co, 1),
            q_thermogenesis_total: round_to(q_thermogenesis_total, 2),
            q_res: round_to(res_sh + res_lh, 2),
            q_skin2env: round_arr(add_arr(shl_sk, e_sk), 2),
            height: self.height_m,
            weight: self.weight_kg,
            bsa: round_arr(self.bsa, 2),
            fat: self.fat_percent,
            sex: self.sex,
            age: self.age,
            t_core_set: round_arr(setpt_cr, 2),
            t_skin_set: round_arr(setpt_sk, 2),
            t_cb: round_to(self.t_body[matrix::CB], 2),
            t_artery: round_arr(self.t_artery(), 2),
            t_vein: round_arr(self.t_vein(), 2),
            t_superficial_vein: round_arr(self.t_superficial_vein(), 2),
            t_muscle: round_arr(pick2(&self.t_body, &matrix::index_by_layer("muscle")), 2),
            t_fat: round_arr(pick2(&self.t_body, &matrix::index_by_layer("fat")), 2),
            to: round_arr(to, 2),
            r_t: round_arr(r_t, 3),
            r_et: round_arr(r_et, 3),
            tdb: round_arr(self.tdb, 2),
            tr: round_arr(self.tr, 2),
            rh: round_arr(self.rh, 2),
            v: round_arr(self.v, 2),
            par: self.par,
            clo: round_arr(self.clo, 2),
            e_skin: round_arr(e_sk, 2),
            e_max: round_arr(e_max, 2),
            e_sweat: round_arr(e_sweat, 2),
            bf_core: round_arr(bf_core, 2),
            bf_muscle: round_arr(pick2_by_position(&bf_muscle, &muscle_positions), 2),
            bf_fat: round_arr(pick2_by_position(&bf_fat, &fat_positions), 2),
            bf_skin: round_arr(bf_skin, 2),
            bf_ava_hand: round_to(bf_ava_hand, 2),
            bf_ava_foot: round_to(bf_ava_foot, 2),
            q_bmr_core: round_arr(mbase_cr, 2),
            q_bmr_muscle: round_arr(pick2_by_position(&mbase_ms, &muscle_positions), 2),
            q_bmr_fat: round_arr(pick2_by_position(&mbase_fat, &fat_positions), 2),
            q_bmr_skin: round_arr(mbase_sk, 2),
            q_work: round_arr(q_work, 2),
            q_shiv: round_arr(q_shiv, 2),
            q_nst: round_arr(q_nst, 2),
            q_thermogenesis_core: round_arr(q_thermogenesis_core, 2),
            q_thermogenesis_muscle: round_arr(
                pick2_by_position(&q_thermogenesis_muscle, &muscle_positions),
                2,
            ),
            q_thermogenesis_fat: round_arr(
                pick2_by_position(&q_thermogenesis_fat, &fat_positions),
                2,
            ),
            q_thermogenesis_skin: round_arr(q_thermogenesis_skin, 2),
            q_skin2env_sensible: round_arr(shl_sk, 2),
            q_skin2env_latent: round_arr(e_sk, 2),
            q_res_sensible: round_to(res_sh, 2),
            q_res_latent: round_to(res_lh, 2),
        })
    }
}

fn round_arr<const N: usize>(values: [f64; N], decimals: i32) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = round_to(values[i], decimals);
    }
    out
}

fn add_arr(a: [f64; NUM_BODY_PARTS], b: [f64; NUM_BODY_PARTS]) -> [f64; NUM_BODY_PARTS] {
    let mut out = [0.0; NUM_BODY_PARTS];
    for i in 0..NUM_BODY_PARTS {
        out[i] = a[i] + b[i];
    }
    out
}

/// Pick the two node values (head, pelvis) named by `node_indices` (as returned by
/// [`matrix::index_by_layer`] for `"muscle"`/`"fat"`) out of the 85-length body state.
fn pick2(t_body: &[f64], node_indices: &[usize]) -> [f64; NUM_MUSCLE_FAT_PARTS] {
    [t_body[node_indices[0]], t_body[node_indices[1]]]
}

/// Pick the two body-part-indexed values (head, pelvis) named by `positions` (as
/// returned by [`matrix::valid_index_by_layer`] for `"muscle"`/`"fat"`) out of a
/// 17-length body-part array.
fn pick2_by_position(
    values: &[f64; NUM_BODY_PARTS],
    positions: &[usize],
) -> [f64; NUM_MUSCLE_FAT_PARTS] {
    [values[positions[0]], values[positions[1]]]
}

// ---------------------------------------------------------------------------
// Jos3Row / Jos3Results
// ---------------------------------------------------------------------------

/// One simulation step's output. Python: one `JOS3Output` instance, as returned by
/// `JOS3._run`. Field names and units match `JOS3Output` exactly; see
/// [`Jos3Results`] for the aggregated (multi-step) form and the type-mapping notes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Jos3Row {
    simulation_time: f64,
    dt: f64,
    t_skin_mean: f64,
    t_skin: [f64; NUM_BODY_PARTS],
    t_core: [f64; NUM_BODY_PARTS],
    w_mean: f64,
    w: [f64; NUM_BODY_PARTS],
    weight_loss_by_evap_and_res: f64,
    cardiac_output: f64,
    q_thermogenesis_total: f64,
    q_res: f64,
    q_skin2env: [f64; NUM_BODY_PARTS],
    height: f64,
    weight: f64,
    bsa: [f64; NUM_BODY_PARTS],
    fat: f64,
    sex: Sex,
    age: i32,
    t_core_set: [f64; NUM_BODY_PARTS],
    t_skin_set: [f64; NUM_BODY_PARTS],
    t_cb: f64,
    t_artery: [f64; NUM_BODY_PARTS],
    t_vein: [f64; NUM_BODY_PARTS],
    t_superficial_vein: [f64; NUM_SFVEIN_PARTS],
    t_muscle: [f64; NUM_MUSCLE_FAT_PARTS],
    t_fat: [f64; NUM_MUSCLE_FAT_PARTS],
    to: [f64; NUM_BODY_PARTS],
    r_t: [f64; NUM_BODY_PARTS],
    r_et: [f64; NUM_BODY_PARTS],
    tdb: [f64; NUM_BODY_PARTS],
    tr: [f64; NUM_BODY_PARTS],
    rh: [f64; NUM_BODY_PARTS],
    v: [f64; NUM_BODY_PARTS],
    par: f64,
    clo: [f64; NUM_BODY_PARTS],
    e_skin: [f64; NUM_BODY_PARTS],
    e_max: [f64; NUM_BODY_PARTS],
    e_sweat: [f64; NUM_BODY_PARTS],
    bf_core: [f64; NUM_BODY_PARTS],
    bf_muscle: [f64; NUM_MUSCLE_FAT_PARTS],
    bf_fat: [f64; NUM_MUSCLE_FAT_PARTS],
    bf_skin: [f64; NUM_BODY_PARTS],
    bf_ava_hand: f64,
    bf_ava_foot: f64,
    q_bmr_core: [f64; NUM_BODY_PARTS],
    q_bmr_muscle: [f64; NUM_MUSCLE_FAT_PARTS],
    q_bmr_fat: [f64; NUM_MUSCLE_FAT_PARTS],
    q_bmr_skin: [f64; NUM_BODY_PARTS],
    q_work: [f64; NUM_BODY_PARTS],
    q_shiv: [f64; NUM_BODY_PARTS],
    q_nst: [f64; NUM_BODY_PARTS],
    q_thermogenesis_core: [f64; NUM_BODY_PARTS],
    q_thermogenesis_muscle: [f64; NUM_MUSCLE_FAT_PARTS],
    q_thermogenesis_fat: [f64; NUM_MUSCLE_FAT_PARTS],
    q_thermogenesis_skin: [f64; NUM_BODY_PARTS],
    q_skin2env_sensible: [f64; NUM_BODY_PARTS],
    q_skin2env_latent: [f64; NUM_BODY_PARTS],
    q_res_sensible: f64,
    q_res_latent: f64,
}

/// Accumulated per-step JOS3 simulation results: one entry per call to
/// [`Jos3Model::advance`] (plus the initial reference-steady-state entry from
/// construction), read afterward via [`Jos3Model::results`].
///
/// Mirrors Python's `JOS3.results()`/`JOS3.dict_results()`: a struct of `alloc::Vec`
/// trajectories rather than a `Vec` of per-step structs, the same shape used for
/// [`crate::models::two_nodes_gagge_sleep::GaggeTwoNodesSleepResult`]. Field names and
/// units match `JOS3Output` exactly, with two systematic type differences:
///
/// - `simulation_time` is elapsed seconds (`f64`) rather than Python's
///   `datetime.timedelta`, since there is no timedelta type in `no_std`.
/// - The four fields that only exist for head and pelvis (`t_muscle`, `t_fat`,
///   `bf_muscle`, `bf_fat`, and the three `q_bmr_muscle`/`q_bmr_fat`/
///   `q_thermogenesis_muscle`/`q_thermogenesis_fat` pairs) are `[f64; 2]` — the head and
///   pelvis values Python actually computes — rather than a 17-slot structure with 15
///   unused/`None` entries (Python's `pass_values_to_jos3_body_parts(..., body_parts=
///   ["head", "pelvis"])`, which only ever sets those two of `JOS3BodyParts`'s 17
///   fields).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Jos3Results {
    pub simulation_time: Vec<f64>,
    pub dt: Vec<f64>,
    pub t_skin_mean: Vec<f64>,
    pub t_skin: Vec<[f64; NUM_BODY_PARTS]>,
    pub t_core: Vec<[f64; NUM_BODY_PARTS]>,
    pub w_mean: Vec<f64>,
    pub w: Vec<[f64; NUM_BODY_PARTS]>,
    pub weight_loss_by_evap_and_res: Vec<f64>,
    pub cardiac_output: Vec<f64>,
    pub q_thermogenesis_total: Vec<f64>,
    pub q_res: Vec<f64>,
    pub q_skin2env: Vec<[f64; NUM_BODY_PARTS]>,
    pub height: Vec<f64>,
    pub weight: Vec<f64>,
    pub bsa: Vec<[f64; NUM_BODY_PARTS]>,
    pub fat: Vec<f64>,
    pub sex: Vec<Sex>,
    pub age: Vec<i32>,
    pub t_core_set: Vec<[f64; NUM_BODY_PARTS]>,
    pub t_skin_set: Vec<[f64; NUM_BODY_PARTS]>,
    pub t_cb: Vec<f64>,
    pub t_artery: Vec<[f64; NUM_BODY_PARTS]>,
    pub t_vein: Vec<[f64; NUM_BODY_PARTS]>,
    pub t_superficial_vein: Vec<[f64; NUM_SFVEIN_PARTS]>,
    pub t_muscle: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub t_fat: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub to: Vec<[f64; NUM_BODY_PARTS]>,
    pub r_t: Vec<[f64; NUM_BODY_PARTS]>,
    pub r_et: Vec<[f64; NUM_BODY_PARTS]>,
    pub tdb: Vec<[f64; NUM_BODY_PARTS]>,
    pub tr: Vec<[f64; NUM_BODY_PARTS]>,
    pub rh: Vec<[f64; NUM_BODY_PARTS]>,
    pub v: Vec<[f64; NUM_BODY_PARTS]>,
    pub par: Vec<f64>,
    pub clo: Vec<[f64; NUM_BODY_PARTS]>,
    pub e_skin: Vec<[f64; NUM_BODY_PARTS]>,
    pub e_max: Vec<[f64; NUM_BODY_PARTS]>,
    pub e_sweat: Vec<[f64; NUM_BODY_PARTS]>,
    pub bf_core: Vec<[f64; NUM_BODY_PARTS]>,
    pub bf_muscle: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub bf_fat: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub bf_skin: Vec<[f64; NUM_BODY_PARTS]>,
    pub bf_ava_hand: Vec<f64>,
    pub bf_ava_foot: Vec<f64>,
    pub q_bmr_core: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_bmr_muscle: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub q_bmr_fat: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub q_bmr_skin: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_work: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_shiv: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_nst: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_thermogenesis_core: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_thermogenesis_muscle: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub q_thermogenesis_fat: Vec<[f64; NUM_MUSCLE_FAT_PARTS]>,
    pub q_thermogenesis_skin: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_skin2env_sensible: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_skin2env_latent: Vec<[f64; NUM_BODY_PARTS]>,
    pub q_res_sensible: Vec<f64>,
    pub q_res_latent: Vec<f64>,
}

impl Jos3Results {
    fn with_capacity(cap: usize) -> Self {
        Self {
            simulation_time: Vec::with_capacity(cap),
            dt: Vec::with_capacity(cap),
            t_skin_mean: Vec::with_capacity(cap),
            t_skin: Vec::with_capacity(cap),
            t_core: Vec::with_capacity(cap),
            w_mean: Vec::with_capacity(cap),
            w: Vec::with_capacity(cap),
            weight_loss_by_evap_and_res: Vec::with_capacity(cap),
            cardiac_output: Vec::with_capacity(cap),
            q_thermogenesis_total: Vec::with_capacity(cap),
            q_res: Vec::with_capacity(cap),
            q_skin2env: Vec::with_capacity(cap),
            height: Vec::with_capacity(cap),
            weight: Vec::with_capacity(cap),
            bsa: Vec::with_capacity(cap),
            fat: Vec::with_capacity(cap),
            sex: Vec::with_capacity(cap),
            age: Vec::with_capacity(cap),
            t_core_set: Vec::with_capacity(cap),
            t_skin_set: Vec::with_capacity(cap),
            t_cb: Vec::with_capacity(cap),
            t_artery: Vec::with_capacity(cap),
            t_vein: Vec::with_capacity(cap),
            t_superficial_vein: Vec::with_capacity(cap),
            t_muscle: Vec::with_capacity(cap),
            t_fat: Vec::with_capacity(cap),
            to: Vec::with_capacity(cap),
            r_t: Vec::with_capacity(cap),
            r_et: Vec::with_capacity(cap),
            tdb: Vec::with_capacity(cap),
            tr: Vec::with_capacity(cap),
            rh: Vec::with_capacity(cap),
            v: Vec::with_capacity(cap),
            par: Vec::with_capacity(cap),
            clo: Vec::with_capacity(cap),
            e_skin: Vec::with_capacity(cap),
            e_max: Vec::with_capacity(cap),
            e_sweat: Vec::with_capacity(cap),
            bf_core: Vec::with_capacity(cap),
            bf_muscle: Vec::with_capacity(cap),
            bf_fat: Vec::with_capacity(cap),
            bf_skin: Vec::with_capacity(cap),
            bf_ava_hand: Vec::with_capacity(cap),
            bf_ava_foot: Vec::with_capacity(cap),
            q_bmr_core: Vec::with_capacity(cap),
            q_bmr_muscle: Vec::with_capacity(cap),
            q_bmr_fat: Vec::with_capacity(cap),
            q_bmr_skin: Vec::with_capacity(cap),
            q_work: Vec::with_capacity(cap),
            q_shiv: Vec::with_capacity(cap),
            q_nst: Vec::with_capacity(cap),
            q_thermogenesis_core: Vec::with_capacity(cap),
            q_thermogenesis_muscle: Vec::with_capacity(cap),
            q_thermogenesis_fat: Vec::with_capacity(cap),
            q_thermogenesis_skin: Vec::with_capacity(cap),
            q_skin2env_sensible: Vec::with_capacity(cap),
            q_skin2env_latent: Vec::with_capacity(cap),
            q_res_sensible: Vec::with_capacity(cap),
            q_res_latent: Vec::with_capacity(cap),
        }
    }

    fn push(&mut self, row: Jos3Row) {
        self.simulation_time.push(row.simulation_time);
        self.dt.push(row.dt);
        self.t_skin_mean.push(row.t_skin_mean);
        self.t_skin.push(row.t_skin);
        self.t_core.push(row.t_core);
        self.w_mean.push(row.w_mean);
        self.w.push(row.w);
        self.weight_loss_by_evap_and_res
            .push(row.weight_loss_by_evap_and_res);
        self.cardiac_output.push(row.cardiac_output);
        self.q_thermogenesis_total.push(row.q_thermogenesis_total);
        self.q_res.push(row.q_res);
        self.q_skin2env.push(row.q_skin2env);
        self.height.push(row.height);
        self.weight.push(row.weight);
        self.bsa.push(row.bsa);
        self.fat.push(row.fat);
        self.sex.push(row.sex);
        self.age.push(row.age);
        self.t_core_set.push(row.t_core_set);
        self.t_skin_set.push(row.t_skin_set);
        self.t_cb.push(row.t_cb);
        self.t_artery.push(row.t_artery);
        self.t_vein.push(row.t_vein);
        self.t_superficial_vein.push(row.t_superficial_vein);
        self.t_muscle.push(row.t_muscle);
        self.t_fat.push(row.t_fat);
        self.to.push(row.to);
        self.r_t.push(row.r_t);
        self.r_et.push(row.r_et);
        self.tdb.push(row.tdb);
        self.tr.push(row.tr);
        self.rh.push(row.rh);
        self.v.push(row.v);
        self.par.push(row.par);
        self.clo.push(row.clo);
        self.e_skin.push(row.e_skin);
        self.e_max.push(row.e_max);
        self.e_sweat.push(row.e_sweat);
        self.bf_core.push(row.bf_core);
        self.bf_muscle.push(row.bf_muscle);
        self.bf_fat.push(row.bf_fat);
        self.bf_skin.push(row.bf_skin);
        self.bf_ava_hand.push(row.bf_ava_hand);
        self.bf_ava_foot.push(row.bf_ava_foot);
        self.q_bmr_core.push(row.q_bmr_core);
        self.q_bmr_muscle.push(row.q_bmr_muscle);
        self.q_bmr_fat.push(row.q_bmr_fat);
        self.q_bmr_skin.push(row.q_bmr_skin);
        self.q_work.push(row.q_work);
        self.q_shiv.push(row.q_shiv);
        self.q_nst.push(row.q_nst);
        self.q_thermogenesis_core.push(row.q_thermogenesis_core);
        self.q_thermogenesis_muscle.push(row.q_thermogenesis_muscle);
        self.q_thermogenesis_fat.push(row.q_thermogenesis_fat);
        self.q_thermogenesis_skin.push(row.q_thermogenesis_skin);
        self.q_skin2env_sensible.push(row.q_skin2env_sensible);
        self.q_skin2env_latent.push(row.q_skin2env_latent);
        self.q_res_sensible.push(row.q_res_sensible);
        self.q_res_latent.push(row.q_res_latent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        fabs(a - b) < tol
    }

    fn approx_eq_arr<const N: usize>(a: [f64; N], b: [f64; N], tol: f64) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    #[test]
    fn default_builder_matches_python_defaults() {
        let model = Jos3Builder::new()
            .build()
            .expect("default parameters are valid");
        assert!(approx_eq(model.height().as_meters(), 1.72, 1e-12));
        assert!(approx_eq(model.weight().as_kilograms(), 74.43, 1e-12));
        assert_eq!(model.age(), 20);
        assert!(approx_eq(model.fat().as_percent(), 15.0, 1e-12));
        assert_eq!(model.sex(), Sex::Male);
    }

    #[test]
    fn build_rejects_out_of_range_height() {
        let err = Jos3Builder::new()
            .height(Length::from_meters(0.4))
            .build()
            .unwrap_err();
        assert_eq!(err, Jos3BuildError::Validation(BodyParameterError::Height));
    }

    /// Cross-checked against pythermalcomfort 4.4.0, `JOS3()` (all defaults):
    /// `model.bsa.sum() == 1.8689703065022139`, `model.bmr == 47.062749886388204`.
    #[test]
    fn bsa_and_bmr_match_python() {
        let model = Jos3Builder::new()
            .build()
            .expect("default parameters are valid");
        let bsa_sum: f64 = model.bsa().iter().sum();
        assert!(approx_eq(bsa_sum, 1.868_970_306_502_213_9, 1e-9));
        assert!(approx_eq(model.bmr(), 47.062_749_886_388_204, 1e-6));
    }

    /// Cross-checked against pythermalcomfort 4.4.0, `JOS3()` immediately after
    /// construction (`model._history[0]`, i.e. before any `simulate` call):
    /// `t_skin_mean == 34.46`, and `t_skin`/`t_core` rounded to 2 decimals as below
    /// (`BODY_PART_NAMES` order: head, neck, chest, back, pelvis, left_shoulder,
    /// left_arm, left_hand, right_shoulder, right_arm, right_hand, left_thigh,
    /// left_leg, left_foot, right_thigh, right_leg, right_foot).
    /// Regression test for a boundary bug that cost ~2 °C on the set points.
    ///
    /// `_calculate_operative_temp_when_pmv_is_zero` is a damped fixed-point search that
    /// walks `to` upward and *relies on crossing* ISO 7730's 30 °C applicability limit to
    /// get a NaN back and fall into its retry branch. This subject drives `to` to
    /// 30.00000000000001, one ULP outside the limit. Python NaNs there and retries,
    /// landing on to = 28.0023...; the Rust port used to construct a `Temperature` from
    /// that value, and since `Temperature` stores kelvin the round trip snapped it back to
    /// exactly 30.0, so the limit check passed, the retry never ran, and every set point
    /// was wrong for the life of the model.
    ///
    /// Cross-checked against pythermalcomfort 4.4.0:
    /// ```python
    /// m = JOS3(height=1.55, weight=74.43, age=20, fat=20, sex="female",
    ///          ci=2.59, bmr_equation="japanese", bsa_equation="dubois")
    /// m.simulate(times=60, dtime=60)
    /// # t_skin_mean == 32.93, cardiac_output == 290.3
    /// ```
    #[test]
    fn operative_temp_search_crosses_the_iso_limit_like_python() {
        let mut model = Jos3Builder::new()
            .height(Length::from_meters(1.55))
            .weight(Mass::from_kilograms(74.43))
            .age(20)
            .fat(BodyFat::new(20.0).expect("20% is in range"))
            .sex(Sex::Female)
            .cardiac_index(CardiacIndex::from_liters_per_minute_per_square_meter(2.59))
            .bmr_equation(BmrEquation::Japanese)
            .bsa_equation(BsaFormula::DuBois)
            .build()
            .expect("valid subject");

        let conditions = model.conditions();
        model
            .advance(&conditions, 60, Duration::from_secs(60))
            .expect("valid conditions");

        let results = model.results();
        let last = results.t_skin_mean.len() - 1;
        assert!(
            approx_eq(results.t_skin_mean[last], 32.93, 1e-9),
            "t_skin_mean = {}, pythermalcomfort gives 32.93 (35.22 means the retry \
             branch never fired)",
            results.t_skin_mean[last]
        );
        assert!(
            approx_eq(results.cardiac_output[last], 290.3, 1e-9),
            "cardiac_output = {}, pythermalcomfort gives 290.3",
            results.cardiac_output[last]
        );
    }

    #[test]
    fn initial_reset_setpoint_state_matches_python() {
        let model = Jos3Builder::new()
            .build()
            .expect("default parameters are valid");
        let first = model.results();
        assert_eq!(first.t_skin.len(), 1);
        assert!(approx_eq(first.t_skin_mean[0], 34.46, 1e-9));

        let want_t_skin = [
            34.99, 35.11, 34.62, 34.52, 35.84, 34.25, 33.86, 34.31, 34.25, 33.86, 34.31, 34.15,
            33.95, 34.12, 34.15, 33.95, 34.12,
        ];
        let want_t_core = [
            37.28, 36.82, 37.05, 37.11, 37.26, 36.25, 35.72, 34.9, 36.25, 35.72, 34.9, 36.68,
            36.32, 34.6, 36.68, 36.32, 34.6,
        ];
        assert!(approx_eq_arr(first.t_skin[0], want_t_skin, 1e-9));
        assert!(approx_eq_arr(first.t_core[0], want_t_core, 1e-9));
        assert!(approx_eq(first.dt[0], 60_000.0, 1e-9));
        assert!(approx_eq(first.simulation_time[0], 0.0, 1e-9));
    }

    /// Cross-checked against pythermalcomfort 4.4.0:
    /// ```python
    /// model = JOS3()
    /// model.simulate(times=60, dtime=60)
    /// ```
    /// with no environment properties changed in between (so the simulation runs at
    /// the reference environment `_reset_setpt` established), giving after 60 steps:
    /// `t_skin_mean == 34.69`, `t_cb == 36.73`, `cardiac_output == 316.2`,
    /// `q_thermogenesis_total == 110.42`, `q_res == 8.26`, `w_mean == 0.06`, and
    /// `t_skin`/`t_core` rounded to 2 decimals as below.
    #[test]
    fn sixty_steps_at_reference_environment_matches_python() {
        let mut model = Jos3Builder::new()
            .build()
            .expect("default parameters are valid");
        // Reproduce Python's `model.simulate(60, 60)` with no property changes: reuse
        // the environment `reset_setpoint` already established rather than Python's
        // *original* `Default` environment (which `_reset_setpt` overwrites during
        // construction and never restores).
        let conditions = model.conditions();
        model
            .advance(&conditions, 60, Duration::from_secs(60))
            .expect("valid conditions and par >= 1");

        let results = model.results();
        assert_eq!(results.t_skin.len(), 61); // initial reset-setpoint row + 60 steps
        let last = results.t_skin.len() - 1;

        assert!(approx_eq(results.t_skin_mean[last], 34.69, 1e-9));
        assert!(approx_eq(results.t_cb[last], 36.73, 1e-9));
        assert!(approx_eq(results.cardiac_output[last], 316.2, 1e-6));
        assert!(approx_eq(results.q_thermogenesis_total[last], 110.42, 1e-6));
        assert!(approx_eq(results.q_res[last], 8.26, 1e-6));
        assert!(approx_eq(results.w_mean[last], 0.06, 1e-9));
        assert!(approx_eq(results.dt[last], 60.0, 1e-9));
        assert!(approx_eq(results.simulation_time[last], 3600.0, 1e-9));

        let want_t_skin = [
            34.86, 34.97, 34.49, 34.4, 35.73, 34.44, 34.76, 35.49, 34.44, 34.76, 35.49, 34.16,
            34.42, 35.28, 34.16, 34.42, 35.28,
        ];
        let want_t_core = [
            37.11, 36.65, 36.88, 36.94, 37.11, 36.29, 36.08, 35.89, 36.29, 36.08, 35.89, 36.62,
            36.44, 35.67, 36.62, 36.44, 35.67,
        ];
        assert!(approx_eq_arr(results.t_skin[last], want_t_skin, 1e-9));
        assert!(approx_eq_arr(results.t_core[last], want_t_core, 1e-9));
    }

    #[test]
    fn conditions_round_trip_does_not_borrow_model() {
        let mut model = Jos3Builder::new()
            .build()
            .expect("default parameters are valid");
        let mut conditions = model.conditions();
        conditions.v = PerBodyPart::Uniform(0.5);
        // This would not compile if `conditions` still borrowed `model`.
        model
            .advance(&conditions, 1, Duration::from_secs(60))
            .expect("valid conditions");
        assert!(approx_eq(model.v()[0], 0.5, 1e-12));
    }

    #[test]
    fn advance_rejects_par_below_one() {
        let mut model = Jos3Builder::new()
            .build()
            .expect("default parameters are valid");
        let mut conditions = model.conditions();
        conditions.par = ActivityRatio::from_ratio(0.5);
        let err = model
            .advance(&conditions, 1, Duration::from_secs(60))
            .unwrap_err();
        assert_eq!(
            err,
            Jos3Error::Thermoregulation(ThermoregulationError::ParTooSmall)
        );
    }

    #[test]
    fn per_body_part_by_name_resolves_and_rejects_missing_key() {
        let pairs: Vec<(&str, f64)> = BODY_PART_NAMES.iter().map(|name| (*name, 1.0)).collect();
        let resolved = PerBodyPart::ByName(&pairs)
            .resolve()
            .expect("all names present");
        assert_eq!(resolved, [1.0; NUM_BODY_PARTS]);

        let missing: Vec<(&str, f64)> = BODY_PART_NAMES[1..]
            .iter()
            .map(|name| (*name, 1.0))
            .collect();
        assert_eq!(
            PerBodyPart::ByName(&missing).resolve(),
            Err(BodyPartsInputError::MissingKey("head"))
        );
    }
}
