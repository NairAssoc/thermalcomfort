//! # PHS (Predicted Heat Strain)
//!
//! Calculate the Predicted Heat Strain according to ISO 7933:2004 or ISO 7933:2023.
//!
//! The PHS provides a method for the analytical evaluation and interpretation of the thermal
//! stress experienced by a subject in a hot environment. It predicts sweat rate and internal
//! core temperature that the human body will develop in response to working conditions.
//!
//! ## Algorithm
//!
//! The PHS uses a **time-stepping simulation** approach that simulates human thermoregulatory
//! response over a duration (default 480 minutes) with 1-minute timesteps.
//!
//! Each timestep:
//! 1. Updates core temperature equilibrium
//! 2. Calculates skin temperature
//! 3. Solves for clothing surface temperature (iterative)
//! 4. Calculates heat flows (convection, radiation, evaporation)
//! 5. Solves for new core temperature (iterative)
//! 6. Updates rectal temperature
//! 7. Updates regulatory sweat rate
//! 8. Accumulates sweat loss and checks exposure limits
//!
//! ## ISO Standards
//!
//! Supports both ISO 7933:2004 and ISO 7933:2023 versions with key differences:
//! - Emissivity (f_r): 0.97 (2004) vs 0.42 (2023)
//! - Maximum sweat rate calculations
//! - Convective heat transfer coefficient basis
//! - Clothing area factor formula
//!
//! ## References
//!
//! - ISO 7933:2004 - Ergonomics of the thermal environment
//! - ISO 7933:2023 - Ergonomics of the thermal environment

#![allow(clippy::excessive_precision)]
#![allow(clippy::too_many_arguments)]

use crate::utilities::{body_surface_area_dubois, p_sat, py_max, py_min};
use crate::{
    ClothingInsulation, HeatFluxDensity, Humidity, Length, Mass, MetabolicRate, Speed, Temperature,
};
use libm::{cos, exp, pow, sqrt};

/// The comfort inputs to [`phs`]: pythermalcomfort requires all seven (no default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhsInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Air speed
    pub v: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
    /// Body posture
    pub posture: PhsPosture,
}

/// Result of PHS calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhsResult {
    /// Rectal temperature
    pub t_re: Temperature,
    /// Skin temperature
    pub t_sk: Temperature,
    /// Core temperature
    pub t_cr: Temperature,
    /// Core temperature as a function of metabolic rate
    pub t_cr_eq: Temperature,
    /// Fraction of body mass at skin temperature \[dimensionless\]
    pub t_sk_t_cr_wg: f64,
    /// Maximum allowable exposure time for 50% worker (dehydration) \[minutes\]
    pub d_lim_loss_50: f64,
    /// Maximum allowable exposure time for 95% worker (dehydration) \[minutes\]
    pub d_lim_loss_95: f64,
    /// Maximum allowable exposure time for heat storage \[minutes\]
    pub d_lim_t_re: f64,
    /// Cumulative sweat loss for whole person
    pub sweat_loss_g: Mass,
    /// Instantaneous evaporative heat flux at skin
    pub sweat_rate_watt: HeatFluxDensity,
    /// Accumulated evaporative load [W·min/m²]. NOT a [`HeatFluxDensity`]: this is a
    /// time-integrated accumulator (watt-minutes per square metre), not an instantaneous
    /// flux, so wrapping it in the same newtype as `sweat_rate_watt` would misstate its
    /// unit.
    pub evap_load_wm2_min: f64,
}

/// Posture for PHS calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhsPosture {
    /// Standing posture
    Standing,
    /// Sitting posture
    Sitting,
    /// Crouching posture
    Crouching,
}

/// ISO 7933 model version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iso7933Model {
    /// ISO 7933:2004 version
    Iso2004,
    /// ISO 7933:2023 version (default)
    Iso2023,
}

/// Options for PHS calculation
#[derive(Debug, Clone, Copy)]
pub struct PhsOptions {
    /// External work. Default: 0 met
    pub wme: MetabolicRate,
    /// Round output values. Default: true
    pub round_output: bool,
    /// ISO 7933 model version. Default: Iso2023
    pub model: Iso7933Model,
    /// Limit inputs to standard applicability. Default: true
    pub limit_inputs: bool,
    /// Static moisture permeability index \[dimensionless\]. Default: 0.38
    pub i_mst: f64,
    /// Fraction of body covered by reflective clothing \[dimensionless\]. Default: 0.54
    pub a_p: f64,
    /// Whether workers can drink freely. Default: true
    pub drink: bool,
    /// Body weight. Default: 75.0 kg
    pub weight: Mass,
    /// Height. Default: 1.8 m
    pub height: Length,
    /// Walking speed. Default: 0.0 m/s
    pub walk_sp: Speed,
    /// Angle between walking and wind direction \[degrees\]. Default: 0.0
    pub theta: f64,
    /// Whether worker is acclimatized. Default: true
    pub acclimatized: bool,
    /// Duration of work sequence \[minutes\]. Default: 480
    pub duration: i32,
    /// Emissivity of reflective clothing \[dimensionless\]. Default: depends on model
    pub f_r: Option<f64>,
    /// Initial mean skin temperature. Default: 34.1°C
    pub t_sk: Temperature,
    /// Initial mean core temperature. Default: 36.8°C
    pub t_cr: Temperature,
    /// Initial rectal temperature. Default: depends on model
    pub t_re: Option<Temperature>,
    /// Initial core temp equilibrium. Default: depends on model
    pub t_cr_eq: Option<Temperature>,
    /// Initial skin/core weighting fraction. Default: 0.3
    pub t_sk_t_cr_wg: f64,
    /// Initial sweat rate [W/m²]. Default: 0.0
    pub sweat_rate_watt: f64,
    /// Initial evaporative load [W·min/m²]. Default: 0.0
    pub evap_load_wm2_min: f64,
}

impl Default for PhsOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            round_output: true,
            model: Iso7933Model::Iso2023,
            limit_inputs: true,
            i_mst: 0.38,
            a_p: 0.54,
            drink: true,
            weight: Mass::from_kilograms(75.0),
            height: Length::from_meters(1.8),
            walk_sp: Speed::from_meters_per_second(0.0),
            theta: 0.0,
            acclimatized: true,
            duration: 480,
            f_r: None,
            t_sk: Temperature::from_celsius(34.1),
            t_cr: Temperature::from_celsius(36.8),
            t_re: None,
            t_cr_eq: None,
            t_sk_t_cr_wg: 0.3,
            sweat_rate_watt: 0.0,
            evap_load_wm2_min: 0.0,
        }
    }
}

// Constants for exponential averaging
const CONST_T_EQ: f64 = 0.9048374180359595; // exp(-1/10)
const CONST_T_SK: f64 = 0.7165313105737893; // exp(-1/3)
const CONST_SW: f64 = 0.9048374180359595; // exp(-1/10)

// MET to W/m² conversion (same as in PET)
const MET_TO_W_M2: f64 = 58.15;

// Static boundary layer insulation
const I_A_ST: f64 = 0.111; // m²·K/W

/// Required skin wettedness and sweat rate for one PHS timestep, replicating
/// `phs.py:729-752`.
///
/// `phs.py:731-732` reads `if e_max == 0: e_max = 0.001` — an *exact* equality check,
/// not a "near zero" one — and that assignment mutates `e_max` itself, so the
/// substituted value is what `w_req` divides by *and* what every downstream use of
/// `e_max` (including this timestep's `e_p`, computed by the caller) sees for the rest
/// of the step. A prior version of this port instead computed `w_req` from
/// `e_max.max(1e-6)` without touching `e_max`: at `e_max == 0.0` that takes the
/// `elif e_max <= 0` branch below (since the real `e_max` is still 0), where Python —
/// having already substituted `e_max = 0.001` — takes the `else` branch instead. The
/// two diverge in `sw_req` whenever `e_req` is small enough that `w_req = e_req /
/// 0.001` stays under the `1.7` cutoff; see `phs_e_max_zero_pins_pythons_branch` in the
/// tests below.
///
/// `e_req` is mutated the same way, to zero, whenever it is non-positive.
///
/// Returns `(e_req, e_max, sw_req)`: the possibly-mutated `e_req`/`e_max`, which the
/// caller must use in place of the pre-call values for the rest of the timestep.
fn phs_required_sweat_rate(mut e_req: f64, mut e_max: f64, sw_max: f64) -> (f64, f64, f64) {
    if e_max == 0.0 {
        e_max = 0.001;
    }
    let w_req = e_req / e_max;

    let sw_req = if e_req <= 0.0 {
        e_req = 0.0;
        0.0
    } else if e_max <= 0.0 {
        e_max = 0.0;
        sw_max
    } else if w_req >= 1.7 {
        sw_max
    } else {
        let e_v_eff = if w_req > 1.0 {
            pow(2.0 - w_req, 2.0) / 2.0
        } else {
            1.0 - pow(w_req, 2.0) / 2.0
        };
        // `phs.py:749` is `max(0.05, e_v_eff)` — arguments in that order, so a NaN
        // `e_v_eff` loses and 0.05 wins, which is what `f64::max` already does here.
        let e_v_eff = e_v_eff.max(0.05);
        // `phs.py:752`: builtin `min(sw_req, sw_max)`, NaN-propagating in `sw_req`.
        py_min(e_req / e_v_eff, sw_max)
    };

    (e_req, e_max, sw_req)
}

/// Calculate PHS (Predicted Heat Strain)
///
/// Predicts physiological strain in hot environments according to ISO 7933.
///
/// # Arguments
///
/// * `inputs` - Required comfort inputs, see [`PhsInputs`]
/// * `options` - Additional parameters, see [`PhsOptions`]
///
/// # Returns
///
/// `PhsResult` containing:
/// - Rectal, skin, and core temperatures
/// - Maximum exposure times (dehydration and heat storage)
/// - Cumulative sweat loss
/// - Current sweat rate
///
/// # Standard Applicability Limits (ISO 7933)
///
/// When `limit_inputs` is true (Annex A, Table A.1):
/// - Temperature: 15-50°C (tdb), 0-60°C on the difference `tr - tdb` (not on `tr`)
/// - Air speed: 0-3 m/s
/// - Metabolic rate, standard-specific: 1.7-7.5 met (100-450 W/m²) for
///   [`Iso7933Model::Iso2004`], 0.96-4.3 met (56-250 W/m²) for
///   [`Iso7933Model::Iso2023`]
/// - Clothing: 0.1-1.0 clo
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::{phs, PhsInputs, PhsOptions, PhsPosture};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let result = phs(
///     PhsInputs {
///         tdb: Temperature::from_celsius(40.0),
///         tr: Temperature::from_celsius(40.0),
///         v: Speed::from_meters_per_second(0.3),
///         rh: Humidity::from_percent(33.85),
///         met: MetabolicRate::from_met(2.5),
///         clo: ClothingInsulation::from_clo(0.5),
///         posture: PhsPosture::Standing,
///     },
///     PhsOptions::default(),
/// );
///
/// println!("Rectal temperature: {:.1}°C", result.t_re.as_celsius());
/// println!("Max exposure (50%): {:.0} min", result.d_lim_loss_50);
/// ```
///
/// # References
///
/// - ISO 7933:2004 - Ergonomics of the thermal environment
/// - ISO 7933:2023 - Ergonomics of the thermal environment
pub fn phs(inputs: PhsInputs, options: PhsOptions) -> PhsResult {
    let PhsInputs {
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        posture,
    } = inputs;
    let tdb = tdb.as_celsius();
    let tr = tr.as_celsius();
    let v = v.as_meters_per_second();
    let rh = rh.as_percent();
    let met = met.as_met();
    let clo = clo.as_clo();

    // Water vapour partial pressure [kPa], computed per the selected edition. ISO 7933
    // additionally bounds it, which the port previously did not check at all: outside
    // the range pythermalcomfort masks every output to NaN.
    let p_a = match options.model {
        Iso7933Model::Iso2023 => 0.6105 * exp(17.27 * tdb / (tdb + 237.3)) * rh / 100.0,
        Iso7933Model::Iso2004 => {
            p_sat(Temperature::from_celsius(tdb)).as_pascals() / 1000.0 * rh / 100.0
        }
    };
    // The 2004 edition has no lower bound on p_a; the 2023 edition sets it at 0.5 kPa.
    let p_a_lower = match options.model {
        Iso7933Model::Iso2023 => 0.5,
        Iso7933Model::Iso2004 => 0.0,
    };

    // ISO 7933 Annex A, Table A.1 metabolic rate range, in W/m². Standard-specific:
    // the 2023 revision widens the range downward and narrows it at the top.
    let met_range = match options.model {
        Iso7933Model::Iso2023 => 56.0..=250.0,
        Iso7933Model::Iso2004 => 100.0..=450.0,
    };

    // Input validation. Table A.1 bounds the air-to-radiant *difference*, not `tr`
    // alone; an earlier version of this port checked raw `tr` against (0, 60), as
    // pythermalcomfort itself did before 4.4.1.
    if options.limit_inputs
        && (!(15.0..=50.0).contains(&tdb)
            || !(0.0..=60.0).contains(&(tr - tdb))
            || !(0.0..=3.0).contains(&v)
            || !met_range.contains(&(met * MET_TO_W_M2))
            || !(0.1..=1.0).contains(&clo)
            || !(p_a_lower..=4.5).contains(&p_a))
    {
        return PhsResult {
            t_re: Temperature::from_celsius(f64::NAN),
            t_sk: Temperature::from_celsius(f64::NAN),
            t_cr: Temperature::from_celsius(f64::NAN),
            t_cr_eq: Temperature::from_celsius(f64::NAN),
            t_sk_t_cr_wg: f64::NAN,
            d_lim_loss_50: f64::NAN,
            d_lim_loss_95: f64::NAN,
            d_lim_t_re: f64::NAN,
            sweat_loss_g: Mass::from_grams(f64::NAN),
            sweat_rate_watt: HeatFluxDensity::from_watts_per_square_meter(f64::NAN),
            evap_load_wm2_min: f64::NAN,
        };
    }

    // Model-specific defaults
    let f_r = options.f_r.unwrap_or(match options.model {
        Iso7933Model::Iso2004 => 0.97,
        Iso7933Model::Iso2023 => 0.42,
    });

    let opt_t_sk = options.t_sk.as_celsius();
    let opt_t_cr = options.t_cr.as_celsius();
    let opt_weight = options.weight.as_kilograms();
    let opt_walk_sp = options.walk_sp.as_meters_per_second();

    let t_re_init = options
        .t_re
        .map(|t| t.as_celsius())
        .unwrap_or(match options.model {
            Iso7933Model::Iso2004 => opt_t_cr,
            Iso7933Model::Iso2023 => 36.8,
        });

    let t_cr_eq_init = options
        .t_cr_eq
        .map(|t: Temperature| t.as_celsius())
        .unwrap_or(match options.model {
            Iso7933Model::Iso2004 => opt_t_cr,
            Iso7933Model::Iso2023 => 36.8,
        });

    // Body properties
    let a_dubois = body_surface_area_dubois(options.weight, options.height).as_square_meters();

    let sp_heat = MET_TO_W_M2 * opt_weight / a_dubois;

    // Radiating area ratio
    let a_r_du = match posture {
        PhsPosture::Standing => 0.77,
        PhsPosture::Sitting => 0.70,
        PhsPosture::Crouching => 0.67,
    };

    // Clothing properties
    let i_cl_st = clo * 0.155;
    let fcl = match options.model {
        Iso7933Model::Iso2004 => 1.0 + 0.3 * clo,
        Iso7933Model::Iso2023 => 1.0 + 0.28 * clo,
    };
    let i_tot_st = i_cl_st + I_A_ST / fcl;

    // Maximum sweat rate
    let sw_max = match options.model {
        Iso7933Model::Iso2004 => {
            // ISO 7933 expresses this in W/m2, not met. With `met` left in met units the
            // bracket was always negative, so sw_max clamped to its 250 floor (312.5
            // when acclimatised) instead of tracking metabolic rate - which then bound
            // the required sweat rate and shifted t_re by ~1 degC.
            let mut sw = (met * MET_TO_W_M2 - 32.0) * a_dubois;
            sw = sw.clamp(250.0, 400.0);
            if options.acclimatized { sw * 1.25 } else { sw }
        }
        Iso7933Model::Iso2023 => {
            if !options.acclimatized {
                400.0
            } else {
                500.0
            }
        }
    };

    // Vapor pressure [kPa]
    let p_a = match options.model {
        Iso7933Model::Iso2004 => {
            p_sat(Temperature::from_celsius(tdb)).as_pascals() / 1000.0 * rh / 100.0
        }
        Iso7933Model::Iso2023 => 0.6105 * exp(17.27 * tdb / (tdb + 237.3)) * rh / 100.0,
    };

    // Walking speed
    let mut walk_sp = opt_walk_sp;
    let walking = walk_sp > 0.0;
    if !walking {
        // ISO 7933 uses a literal 58 here, not the 58.15 met->W/m2 factor.
        walk_sp = 0.0052 * (met * MET_TO_W_M2 - 58.0);
        // `phs.py:622` is the builtin `min(walk_sp, 0.7)`: NaN in the first argument
        // propagates, so `f64::min` (which would return 0.7) is the wrong helper.
        walk_sp = py_min(walk_sp, 0.7);
    }

    // Relative air velocity.
    //
    // ISO 7933 distinguishes unidirectional walking (theta != 0, where the walking
    // vector is projected onto the air-speed axis) from omni-directional walking
    // (theta == 0, where the faster of the two simply wins). The port previously applied
    // the unidirectional formula unconditionally and then took `v.max(v_diff)` rather
    // than the projection itself, which changed sweat_loss_g by ~47 g at walk_sp ~1 m/s.
    let v_r = if walking {
        if options.theta != 0.0 {
            // Unidirectional walking
            // ISO 7933 (and pythermalcomfort) use a literal 3.14159 here, not PI. The
            // truncation is part of the standard's arithmetic, so matching it is
            // required for parity rather than an oversight to be "corrected".
            #[allow(clippy::approx_constant)]
            const ISO_PI: f64 = 3.14159;
            let theta_rad = options.theta * ISO_PI / 180.0;
            (v - walk_sp * cos(theta_rad)).abs()
        } else if v < walk_sp {
            walk_sp
        } else {
            v
        }
    } else {
        v
    };

    // Dynamic insulation corrections.
    //
    // `phs.py:625-629` writes these two as `v_ux = v_r; if v_r > 3: v_ux = 3`, which keeps
    // a NaN `v_r` (the `if` is false) and is therefore the builtin-`min` semantics, not
    // `f64::min`'s.
    let v_ux = py_min(v_r, 3.0);
    let w_a_ux = py_min(walk_sp, 1.5);

    let corr_cl = 1.044 * exp((0.066 * v_ux - 0.398) * v_ux + (0.094 * w_a_ux - 0.378) * w_a_ux);
    // `phs.py:636` / `:639`: builtin `min(corr_*, 1)`, NaN-propagating in the first
    // argument.
    let corr_cl = py_min(corr_cl, 1.0);

    let corr_ia = exp((0.047 * v_r - 0.472) * v_r + (0.117 * w_a_ux - 0.342) * w_a_ux);
    let corr_ia = py_min(corr_ia, 1.0);

    let corr_tot = if clo <= 0.6 {
        ((0.6 - clo) * corr_ia + clo * corr_cl) / 0.6
    } else {
        corr_cl
    };

    let i_tot_dyn = i_tot_st * corr_tot;
    let i_a_dyn = corr_ia * I_A_ST;
    let i_cl_dyn = i_tot_dyn - i_a_dyn / fcl;

    let corr_e = (2.6 * corr_tot - 6.5) * corr_tot + 4.9;
    // `phs.py:651`: builtin `min(im_dyn, 0.9)`.
    let im_dyn = py_min(options.i_mst * corr_e, 0.9);
    let r_t_dyn = i_tot_dyn / im_dyn / 16.7;

    // Respiratory heat loss
    let t_exp = 28.56 + 0.115 * tdb + 0.641 * p_a;
    let c_res = 0.001516 * met * MET_TO_W_M2 * (t_exp - tdb);
    let e_res = 0.00127 * met * MET_TO_W_M2 * (59.34 + 0.53 * tdb - 11.63 * p_a);

    // Convective heat transfer
    let z = if v_r > 1.0 {
        8.7 * pow(v_r, 0.6)
    } else {
        3.5 + 5.2 * v_r
    };

    // Radiation coefficient
    let aux_r = 5.67e-8 * a_r_du;
    let f_cl_r = match options.model {
        Iso7933Model::Iso2004 => (1.0 - options.a_p) * 0.97 + options.a_p * f_r,
        Iso7933Model::Iso2023 => (1.0 - options.a_p) * 0.97 + options.a_p * (1.0 - f_r),
    };

    // Pre-calculate skin temperature equilibrium base values
    let t_sk_eq_cl_base = 12.165 + 0.02017 * tdb + 0.04361 * tr + 0.19354 * p_a - 0.25315 * v
        + 0.005346 * met * MET_TO_W_M2;
    let t_sk_eq_nu_base = 7.191 + 0.064 * tdb + 0.061 * tr + 0.198 * p_a - 0.348 * v;

    // Maximum water loss limits
    let (d_max_50, d_max_95) = match options.model {
        Iso7933Model::Iso2004 => (0.075 * opt_weight * 1000.0, 0.05 * opt_weight * 1000.0),
        Iso7933Model::Iso2023 => {
            let max_loss = if options.drink {
                0.05 * opt_weight * 1000.0
            } else {
                0.03 * opt_weight * 1000.0
            };
            (max_loss, max_loss)
        }
    };

    // Maximum skin wettedness (based on acclimatization, matching Python reference)
    let w_max = if !options.acclimatized { 0.85 } else { 1.0 };

    // Compute hc_dyn ONCE before the time loop (matching Python reference)
    let hc_dyn = {
        let hc_dyn_base = match options.model {
            Iso7933Model::Iso2004 => 2.38 * pow((opt_t_sk - tdb).abs(), 0.25),
            Iso7933Model::Iso2023 => {
                let t_cl_init = tr + 0.1;
                2.38 * pow((t_cl_init - tdb).abs(), 0.25)
            }
        };
        // `phs.py:669`: builtin `max(hc_dyn, z)`. A NaN `t_sk`/`tdb`/`tr` makes
        // `hc_dyn_base` NaN and must stay NaN; `f64::max` would return `z`.
        py_max(hc_dyn_base, z)
    };

    // Initialize state
    let mut t_sk = opt_t_sk;
    let mut t_cr = opt_t_cr;
    let mut t_re = t_re_init;
    let mut t_cr_eq = t_cr_eq_init;
    let mut t_sk_t_cr_wg = options.t_sk_t_cr_wg;
    let mut sweat_rate_watt = options.sweat_rate_watt;
    let mut evap_load_wm2_min = options.evap_load_wm2_min;

    let mut d_lim_loss_50 = 0.0;
    let mut d_lim_loss_95 = 0.0;
    let mut d_lim_t_re = 0.0;

    // Time-stepping simulation
    for time in 1..=options.duration {
        let time_f = time as f64;

        // Save previous values
        let t_sk0 = t_sk;
        let t_cr0 = t_cr;
        let t_re0 = t_re;
        let t_cr_eq0 = t_cr_eq;
        let t_sk_t_cr_wg0 = t_sk_t_cr_wg;

        // Core temperature equilibrium
        let t_cr_eq_m = 0.0036 * met * MET_TO_W_M2 + 36.6;
        t_cr_eq = t_cr_eq0 * CONST_T_EQ + t_cr_eq_m * (1.0 - CONST_T_EQ);
        let d_stored_eq = sp_heat * (t_cr_eq - t_cr_eq0) * (1.0 - t_sk_t_cr_wg0);

        // Skin temperature equilibrium
        let t_sk_eq_cl = t_sk_eq_cl_base + 0.51274 * t_re;
        let t_sk_eq_nu = t_sk_eq_nu_base + 0.616 * t_re;

        let t_sk_eq = if clo >= 0.6 {
            t_sk_eq_cl
        } else if clo <= 0.2 {
            t_sk_eq_nu
        } else {
            t_sk_eq_nu + 2.5 * (t_sk_eq_cl - t_sk_eq_nu) * (clo - 0.2)
        };

        t_sk = t_sk0 * CONST_T_SK + t_sk_eq * (1.0 - CONST_T_SK);
        if time == 1 && options.model == Iso7933Model::Iso2023 {
            // ISO 7933:2023 Annex E forces the skin temperature to its equilibrium
            // value on the first minute, removing the exponential lag for that step.
            // This special case is not present in the 2004 Annex E reference code.
            t_sk = t_sk_eq;
        }

        // Clothing surface temperature (iterative)
        let p_sk = 0.6105 * exp(17.27 * t_sk / (t_sk + 237.3));
        let mut t_cl = tr + 0.1;

        for _ in 0..100 {
            // Radiative heat transfer coefficient
            let h_r =
                f_cl_r * aux_r * (pow(t_cl + 273.0, 4.0) - pow(tr + 273.0, 4.0)) / (t_cl - tr);

            let t_cl_new = (fcl * (hc_dyn * tdb + h_r * tr) + t_sk / i_cl_dyn)
                / (fcl * (hc_dyn + h_r) + 1.0 / i_cl_dyn);

            if (t_cl - t_cl_new).abs() <= 0.001 {
                break;
            }
            t_cl = (t_cl + t_cl_new) / 2.0;
        }

        // Final h_r with converged t_cl
        let h_r = f_cl_r * aux_r * (pow(t_cl + 273.0, 4.0) - pow(tr + 273.0, 4.0)) / (t_cl - tr);

        // Heat flows
        let convection = fcl * hc_dyn * (t_cl - tdb);
        let radiation = fcl * h_r * (t_cl - tr);
        let e_max_raw = (p_sk - p_a) / r_t_dyn;
        let e_req_raw = met * MET_TO_W_M2
            - d_stored_eq
            - options.wme.as_met() * MET_TO_W_M2
            - c_res
            - e_res
            - convection
            - radiation;

        // Required sweat rate. `phs_required_sweat_rate` also returns the possibly
        // mutated `e_req`/`e_max`, which are used downstream (d_storage uses e_req, e_p
        // uses e_max) exactly as Python's mutated locals are.
        let (e_req, e_max, sw_req) = phs_required_sweat_rate(e_req_raw, e_max_raw, sw_max);

        sweat_rate_watt = sweat_rate_watt * CONST_SW + sw_req * (1.0 - CONST_SW);

        // Predicted evaporation
        let e_p = if sweat_rate_watt <= 0.0 {
            0.0
        } else {
            // `phs.py:762`: builtin `max(sweat_rate_watt, EPSILON)`. A NaN sweat rate
            // takes this branch (`NaN <= 0` is false) and must stay NaN; `f64::max` would
            // silently substitute 1e-6.
            let k = e_max / py_max(sweat_rate_watt, 1e-6);
            let wp = if k >= 0.5 {
                -k + sqrt(k * k + 2.0)
            } else {
                1.0
            };
            // `phs.py:766`: builtin `min(wp, w_max)`.
            let wp = py_min(wp, w_max);
            wp * e_max
        };

        // Core temperature (iterative)
        let d_storage = e_req - e_p + d_stored_eq;
        let mut t_cr_new = t_cr0;

        for _ in 0..100 {
            let mut t_sk_t_cr_wg_new = 0.3 - 0.09 * (t_cr_new - 36.8);
            t_sk_t_cr_wg_new = t_sk_t_cr_wg_new.clamp(0.1, 0.3);

            let t_cr_calc = (d_storage / sp_heat + t_sk0 * t_sk_t_cr_wg0 / 2.0
                - t_sk * t_sk_t_cr_wg_new / 2.0
                + t_cr0 * (1.0 - t_sk_t_cr_wg0 / 2.0))
                / (1.0 - t_sk_t_cr_wg_new / 2.0);

            if (t_cr_calc - t_cr_new).abs() <= 0.001 {
                t_cr = t_cr_calc;
                t_sk_t_cr_wg = t_sk_t_cr_wg_new;
                break;
            }
            t_cr_new = (t_cr_new + t_cr_calc) / 2.0;
        }

        // Rectal temperature
        t_re = t_re0 + (2.0 * t_cr - 1.962 * t_re0 - 1.31) / 9.0;

        // Check rectal temperature limit
        if d_lim_t_re == 0.0 && t_re >= 38.0 {
            d_lim_t_re = time_f;
        }

        // Accumulate evaporative load
        evap_load_wm2_min += sweat_rate_watt + e_res;

        // Convert to total sweat loss (grams)
        let sw_tot_g = evap_load_wm2_min * 2.67 * a_dubois / 1.8 / 60.0;

        // Check sweat loss limits
        if d_lim_loss_50 == 0.0 && sw_tot_g >= d_max_50 {
            d_lim_loss_50 = time_f;
        }
        if d_lim_loss_95 == 0.0 && sw_tot_g >= d_max_95 {
            d_lim_loss_95 = time_f;
        }
    }

    // Post-simulation adjustments
    if !options.drink {
        d_lim_loss_95 *= 0.6;
        d_lim_loss_50 = d_lim_loss_95;
    }

    if d_lim_loss_50 == 0.0 {
        d_lim_loss_50 = options.duration as f64;
    }
    if d_lim_loss_95 == 0.0 {
        d_lim_loss_95 = options.duration as f64;
    }
    if d_lim_t_re == 0.0 {
        d_lim_t_re = options.duration as f64;
    }

    // Calculate final sweat loss
    let sweat_loss_g = evap_load_wm2_min * 2.67 * a_dubois / 1.8 / 60.0;

    // Round output if requested
    let round_1 = |x: f64| {
        if options.round_output {
            let scaled = x * 10.0;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5) as i64 as f64
            } else {
                (scaled - 0.5) as i64 as f64
            };
            rounded / 10.0
        } else {
            x
        }
    };

    // Round t_sk_t_cr_wg to 4 decimal places
    let t_sk_t_cr_wg_rounded = if options.round_output {
        // pythermalcomfort rounds this to 2 decimals, not 4
        let scaled = t_sk_t_cr_wg * 100.0;
        let rounded = if scaled >= 0.0 {
            (scaled + 0.5) as i64 as f64
        } else {
            (scaled - 0.5) as i64 as f64
        };
        rounded / 100.0
    } else {
        t_sk_t_cr_wg
    };

    PhsResult {
        t_re: Temperature::from_celsius(round_1(t_re)),
        t_sk: Temperature::from_celsius(round_1(t_sk)),
        t_cr: Temperature::from_celsius(round_1(t_cr)),
        t_cr_eq: Temperature::from_celsius(round_1(t_cr_eq)),
        t_sk_t_cr_wg: t_sk_t_cr_wg_rounded,
        d_lim_loss_50: round_1(d_lim_loss_50),
        d_lim_loss_95: round_1(d_lim_loss_95),
        d_lim_t_re: round_1(d_lim_t_re),
        sweat_loss_g: Mass::from_grams(round_1(sweat_loss_g)),
        sweat_rate_watt: HeatFluxDensity::from_watts_per_square_meter(round_1(sweat_rate_watt)),
        evap_load_wm2_min: round_1(evap_load_wm2_min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(
        tdb: f64,
        tr: f64,
        v: f64,
        rh: f64,
        met: f64,
        clo: f64,
        posture: PhsPosture,
    ) -> PhsInputs {
        PhsInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(v),
            rh: Humidity::from_percent(rh),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
            posture,
        }
    }

    #[test]
    fn test_phs_basic() {
        let result = phs(
            inputs(40.0, 40.0, 0.3, 33.85, 2.5, 0.5, PhsPosture::Standing),
            PhsOptions::default(),
        );

        // Should not be NaN
        assert!(!result.t_re.as_celsius().is_nan());
        assert!(!result.t_sk.as_celsius().is_nan());
        assert!(!result.t_cr.as_celsius().is_nan());

        // Reasonable ranges
        let t_re = result.t_re.as_celsius();
        let t_sk = result.t_sk.as_celsius();
        let t_cr = result.t_cr.as_celsius();
        assert!(t_re > 36.0 && t_re < 40.0);
        assert!(t_sk > 33.0 && t_sk < 38.0);
        assert!(t_cr > 36.0 && t_cr < 39.0);
    }

    #[test]
    fn test_phs_out_of_range() {
        let result = phs(
            inputs(10.0, 40.0, 0.3, 50.0, 2.5, 0.5, PhsPosture::Standing), // tdb too cold
            PhsOptions::default(),
        );

        assert!(result.t_re.as_celsius().is_nan());
    }

    #[test]
    fn test_phs_iso_2004() {
        let options = PhsOptions {
            model: Iso7933Model::Iso2004,
            ..Default::default()
        };

        let result = phs(
            inputs(35.0, 35.0, 0.5, 50.0, 2.0, 0.5, PhsPosture::Standing),
            options,
        );

        assert!(!result.t_re.as_celsius().is_nan());
        assert!(result.t_re.as_celsius() > 36.0);
    }

    // --- e_max == 0.0 bug pin (see `phs_required_sweat_rate`'s doc comment) --------

    #[test]
    fn phs_e_max_zero_pins_pythons_branch() {
        // At e_max == 0.0 exactly, Python (`phs.py:731-736`) substitutes e_max = 0.001
        // and *keeps that substitution* for the rest of the branch chain, so a small
        // positive e_req lands in the `else` branch (computing sw_req from e_v_eff)
        // rather than the `elif e_max <= 0` branch (sw_req = sw_max). The previous
        // Rust port took the `elif e_max <= 0` branch here instead, because it clamped
        // e_max only inside the w_req division, not in the variable itself.
        let sw_max = 500.0;
        let e_req = 0.001;

        let (e_req_out, e_max_out, sw_req) = phs_required_sweat_rate(e_req, 0.0, sw_max);

        // e_max is left at the substituted 0.001, not reset to 0.0 by the `elif`
        // branch -- that branch must not be taken.
        assert!(
            (e_max_out - 0.001).abs() < 1e-12,
            "e_max_out = {e_max_out}, expected the substituted 0.001"
        );
        assert_eq!(e_req_out, e_req);

        // w_req = e_req / e_max = 0.001 / 0.001 = 1.0 -> e_v_eff = 1 - w_req^2/2 = 0.5
        // sw_req = e_req / e_v_eff = 0.001 / 0.5 = 0.002
        assert!(
            (sw_req - 0.002).abs() < 1e-9,
            "sw_req = {sw_req}, expected ~0.002 (not sw_max = {sw_max})"
        );
        assert_ne!(sw_req, sw_max);
    }

    #[test]
    fn phs_e_max_zero_with_large_e_req_still_saturates() {
        // For a large enough e_req, both the buggy and fixed control flow land on
        // sw_max, since w_req = e_req / 0.001 easily clears the 1.7 cutoff. This pins
        // that the fix does not change that ordinary case.
        let sw_max = 500.0;
        let (_, _, sw_req) = phs_required_sweat_rate(10.0, 0.0, sw_max);
        assert_eq!(sw_req, sw_max);
    }

    #[test]
    fn phs_negative_e_req_zeroes_it() {
        let (e_req_out, _, sw_req) = phs_required_sweat_rate(-5.0, 100.0, 500.0);
        assert_eq!(e_req_out, 0.0);
        assert_eq!(sw_req, 0.0);
    }

    #[test]
    fn phs_negative_e_max_saturates_and_zeroes_e_max() {
        let (_, e_max_out, sw_req) = phs_required_sweat_rate(5.0, -2.0, 500.0);
        assert_eq!(e_max_out, 0.0);
        assert_eq!(sw_req, 500.0);
    }
}
