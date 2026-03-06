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

use crate::utilities::{body_surface_area_dubois, p_sat};
use crate::{ClothingInsulation, Humidity, Length, Mass, MetabolicRate, Speed, Temperature};
use libm::{cos, exp, pow, sqrt};

/// Result of PHS calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhsResult {
    /// Rectal temperature [°C]
    pub t_re: f64,
    /// Skin temperature [°C]
    pub t_sk: f64,
    /// Core temperature [°C]
    pub t_cr: f64,
    /// Core temperature as a function of metabolic rate [°C]
    pub t_cr_eq: f64,
    /// Fraction of body mass at skin temperature [dimensionless]
    pub t_sk_t_cr_wg: f64,
    /// Maximum allowable exposure time for 50% worker (dehydration) [minutes]
    pub d_lim_loss_50: f64,
    /// Maximum allowable exposure time for 95% worker (dehydration) [minutes]
    pub d_lim_loss_95: f64,
    /// Maximum allowable exposure time for heat storage [minutes]
    pub d_lim_t_re: f64,
    /// Cumulative sweat loss for whole person [grams]
    pub sweat_loss_g: f64,
    /// Instantaneous evaporative heat flux at skin [W/m²]
    pub sweat_rate_watt: f64,
    /// Accumulated evaporative load [W·min/m²]
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
    /// Static moisture permeability index [dimensionless]. Default: 0.38
    pub i_mst: f64,
    /// Fraction of body covered by reflective clothing [dimensionless]. Default: 0.54
    pub a_p: f64,
    /// Whether workers can drink freely. Default: true
    pub drink: bool,
    /// Body weight. Default: 75.0 kg
    pub weight: Mass,
    /// Height. Default: 1.8 m
    pub height: Length,
    /// Walking speed. Default: 0.0 m/s
    pub walk_sp: Speed,
    /// Angle between walking and wind direction [degrees]. Default: 0.0
    pub theta: f64,
    /// Whether worker is acclimatized. Default: true
    pub acclimatized: bool,
    /// Duration of work sequence [minutes]. Default: 480
    pub duration: i32,
    /// Emissivity of reflective clothing [dimensionless]. Default: depends on model
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

/// Calculate PHS (Predicted Heat Strain)
///
/// Predicts physiological strain in hot environments according to ISO 7933.
///
/// # Arguments
///
/// * `tdb` - Dry bulb air temperature [°C]
/// * `tr` - Mean radiant temperature [°C]
/// * `v` - Air speed [m/s]
/// * `rh` - Relative humidity [%]
/// * `met` - Metabolic rate
/// * `clo` - Clothing insulation
/// * `posture` - Body posture (standing, sitting, crouching)
/// * `options` - Additional parameters
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
/// When `limit_inputs` is true:
/// - Temperature: 15-50°C (tdb), 0-60°C (tr)
/// - Air speed: 0-3 m/s
/// - Metabolic rate: 1.7-7.5 met
/// - Clothing: 0.1-1.0 clo
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::{phs, PhsOptions, PhsPosture};
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let result = phs(
///     Temperature::from_celsius(40.0),
///     Temperature::from_celsius(40.0),
///     Speed::from_meters_per_second(0.3),
///     Humidity::from_percent(33.85),
///     MetabolicRate::from_met(2.5),
///     ClothingInsulation::from_clo(0.5),
///     PhsPosture::Standing,
///     PhsOptions::default(),
/// );
///
/// println!("Rectal temperature: {:.1}°C", result.t_re);
/// println!("Max exposure (50%): {:.0} min", result.d_lim_loss_50);
/// ```
///
/// # References
///
/// - ISO 7933:2004 - Ergonomics of the thermal environment
/// - ISO 7933:2023 - Ergonomics of the thermal environment
pub fn phs(
    tdb: Temperature,
    tr: Temperature,
    v: Speed,
    rh: Humidity,
    met: MetabolicRate,
    clo: ClothingInsulation,
    posture: PhsPosture,
    options: PhsOptions,
) -> PhsResult {
    let tdb = tdb.as_celsius();
    let tr = tr.as_celsius();
    let v = v.as_meters_per_second();
    let rh = rh.as_percent();
    let met = met.as_met();
    let clo = clo.as_clo();

    // Input validation
    if options.limit_inputs
        && (!(15.0..=50.0).contains(&tdb)
            || !(0.0..=60.0).contains(&tr)
            || !(0.0..=3.0).contains(&v)
            || !(1.7..=7.5).contains(&met)
            || !(0.1..=1.0).contains(&clo))
    {
        return PhsResult {
            t_re: f64::NAN,
            t_sk: f64::NAN,
            t_cr: f64::NAN,
            t_cr_eq: f64::NAN,
            t_sk_t_cr_wg: f64::NAN,
            d_lim_loss_50: f64::NAN,
            d_lim_loss_95: f64::NAN,
            d_lim_t_re: f64::NAN,
            sweat_loss_g: f64::NAN,
            sweat_rate_watt: f64::NAN,
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

    let t_re_init = options.t_re.map(|t| t.as_celsius()).unwrap_or(match options.model {
        Iso7933Model::Iso2004 => opt_t_cr,
        Iso7933Model::Iso2023 => 36.8,
    });

    let t_cr_eq_init = options.t_cr_eq.map(|t: Temperature| t.as_celsius()).unwrap_or(match options.model {
        Iso7933Model::Iso2004 => opt_t_cr,
        Iso7933Model::Iso2023 => 36.8,
    });

    // Body properties
    let a_dubois = body_surface_area_dubois(options.weight, options.height)
        .as_square_meters();

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
            let mut sw = (met - 32.0) * a_dubois;
            sw = sw.clamp(250.0, 400.0);
            if options.acclimatized {
                sw * 1.25
            } else {
                sw
            }
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
        walk_sp = 0.0052 * (met * MET_TO_W_M2 - MET_TO_W_M2);
        walk_sp = walk_sp.min(0.7);
    }

    // Relative air velocity
    let v_r = if walking {
        let theta_rad = options.theta * core::f64::consts::PI / 180.0;
        let v_diff = (v - walk_sp * cos(theta_rad)).abs();
        v.max(v_diff)
    } else {
        v
    };

    // Dynamic insulation corrections
    let v_ux = v_r.min(3.0);
    let w_a_ux = walk_sp.min(1.5);

    let corr_cl = 1.044 * exp((0.066 * v_ux - 0.398) * v_ux + (0.094 * w_a_ux - 0.378) * w_a_ux);
    let corr_cl = corr_cl.min(1.0);

    let corr_ia = exp((0.047 * v_r - 0.472) * v_r + (0.117 * w_a_ux - 0.342) * w_a_ux);
    let corr_ia = corr_ia.min(1.0);

    let corr_tot = if clo <= 0.6 {
        ((0.6 - clo) * corr_ia + clo * corr_cl) / 0.6
    } else {
        corr_cl
    };

    let i_tot_dyn = i_tot_st * corr_tot;
    let i_a_dyn = corr_ia * I_A_ST;
    let i_cl_dyn = i_tot_dyn - i_a_dyn / fcl;

    let corr_e = (2.6 * corr_tot - 6.5) * corr_tot + 4.9;
    let im_dyn = (options.i_mst * corr_e).min(0.9);
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
        Iso7933Model::Iso2004 => (
            0.075 * opt_weight * 1000.0,
            0.05 * opt_weight * 1000.0,
        ),
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
        hc_dyn_base.max(z)
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

        // Clothing surface temperature (iterative)
        let p_sk = 0.6105 * exp(17.27 * t_sk / (t_sk + 237.3));
        let mut t_cl = tr + 0.1;

        for _ in 0..100 {
            // Radiative heat transfer coefficient
            let h_r = f_cl_r * aux_r * (pow(t_cl + 273.0, 4.0) - pow(tr + 273.0, 4.0))
                / (t_cl - tr);

            let t_cl_new = (fcl * (hc_dyn * tdb + h_r * tr) + t_sk / i_cl_dyn)
                / (fcl * (hc_dyn + h_r) + 1.0 / i_cl_dyn);

            if (t_cl - t_cl_new).abs() <= 0.001 {
                break;
            }
            t_cl = (t_cl + t_cl_new) / 2.0;
        }

        // Final h_r with converged t_cl
        let h_r = f_cl_r * aux_r * (pow(t_cl + 273.0, 4.0) - pow(tr + 273.0, 4.0))
            / (t_cl - tr);

        // Heat flows
        let convection = fcl * hc_dyn * (t_cl - tdb);
        let radiation = fcl * h_r * (t_cl - tr);
        let e_max = (p_sk - p_a) / r_t_dyn;
        let e_req = met * MET_TO_W_M2
            - d_stored_eq
            - options.wme.as_met() * MET_TO_W_M2
            - c_res
            - e_res
            - convection
            - radiation;
        let w_req = e_req / e_max.max(1e-6);

        // Required sweat rate
        let sw_req = if e_req <= 0.0 {
            0.0
        } else if e_max <= 0.0 || w_req >= 1.7 {
            sw_max
        } else {
            let e_v_eff = if w_req > 1.0 {
                pow(2.0 - w_req, 2.0) / 2.0
            } else {
                1.0 - pow(w_req, 2.0) / 2.0
            };
            let e_v_eff = e_v_eff.max(0.05);
            (e_req / e_v_eff).min(sw_max)
        };

        sweat_rate_watt = sweat_rate_watt * CONST_SW + sw_req * (1.0 - CONST_SW);

        // Predicted evaporation
        let e_p = if sweat_rate_watt <= 0.0 {
            0.0
        } else {
            let k = e_max / sweat_rate_watt.max(1e-6);
            let wp = if k >= 0.5 {
                -k + sqrt(k * k + 2.0)
            } else {
                1.0
            };
            let wp = wp.min(w_max);
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
    let round_0 = |x: f64| {
        if options.round_output {
            if x >= 0.0 {
                (x + 0.5) as i64 as f64
            } else {
                (x - 0.5) as i64 as f64
            }
        } else {
            x
        }
    };

    // Round t_sk_t_cr_wg to 4 decimal places
    let t_sk_t_cr_wg_rounded = if options.round_output {
        let scaled = t_sk_t_cr_wg * 10000.0;
        let rounded = if scaled >= 0.0 {
            (scaled + 0.5) as i64 as f64
        } else {
            (scaled - 0.5) as i64 as f64
        };
        rounded / 10000.0
    } else {
        t_sk_t_cr_wg
    };

    PhsResult {
        t_re: round_1(t_re),
        t_sk: round_1(t_sk),
        t_cr: round_1(t_cr),
        t_cr_eq: round_1(t_cr_eq),
        t_sk_t_cr_wg: t_sk_t_cr_wg_rounded,
        d_lim_loss_50: round_0(d_lim_loss_50),
        d_lim_loss_95: round_0(d_lim_loss_95),
        d_lim_t_re: round_0(d_lim_t_re),
        sweat_loss_g: round_0(sweat_loss_g),
        sweat_rate_watt: round_1(sweat_rate_watt),
        evap_load_wm2_min: round_1(evap_load_wm2_min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phs_basic() {
        let result = phs(
            Temperature::from_celsius(40.0),
            Temperature::from_celsius(40.0),
            Speed::from_meters_per_second(0.3),
            Humidity::from_percent(33.85),
            MetabolicRate::from_met(2.5),
            ClothingInsulation::from_clo(0.5),
            PhsPosture::Standing,
            PhsOptions::default(),
        );

        // Should not be NaN
        assert!(!result.t_re.is_nan());
        assert!(!result.t_sk.is_nan());
        assert!(!result.t_cr.is_nan());

        // Reasonable ranges
        assert!(result.t_re > 36.0 && result.t_re < 40.0);
        assert!(result.t_sk > 33.0 && result.t_sk < 38.0);
        assert!(result.t_cr > 36.0 && result.t_cr < 39.0);
    }

    #[test]
    fn test_phs_out_of_range() {
        let result = phs(
            Temperature::from_celsius(10.0), // Too cold
            Temperature::from_celsius(40.0),
            Speed::from_meters_per_second(0.3),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(2.5),
            ClothingInsulation::from_clo(0.5),
            PhsPosture::Standing,
            PhsOptions::default(),
        );

        assert!(result.t_re.is_nan());
    }

    #[test]
    fn test_phs_iso_2004() {
        let options = PhsOptions {
            model: Iso7933Model::Iso2004,
            ..Default::default()
        };

        let result = phs(
            Temperature::from_celsius(35.0),
            Temperature::from_celsius(35.0),
            Speed::from_meters_per_second(0.5),
            Humidity::from_percent(50.0),
            MetabolicRate::from_met(2.0),
            ClothingInsulation::from_clo(0.5),
            PhsPosture::Standing,
            options,
        );

        assert!(!result.t_re.is_nan());
        assert!(result.t_re > 36.0);
    }
}
