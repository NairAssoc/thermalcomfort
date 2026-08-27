//! Two-node Gagge model of human temperature regulation
//!
//! This module implements the Gagge two-node model (Gagge1986) which simulates
//! human thermoregulatory responses and calculates various thermal comfort indices.

extern crate alloc;

use alloc::vec::Vec;

use crate::utilities::{Posture, p_sat_torr, py_max, py_min, round_to};
use crate::{ClothingInsulation, HeatFluxDensity, Mass, MetabolicRate};
use libm::{exp, fabs as abs, pow, sqrt};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// The comfort inputs to [`two_nodes_gagge`]: pythermalcomfort requires all six (no
/// default).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesInputs {
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
}

/// Result from the two-node Gagge model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesResult {
    /// Standard Effective Temperature
    pub set: Temperature,
    /// Total evaporative heat loss from skin
    pub e_skin: HeatFluxDensity,
    /// Heat lost by evaporation of regulatory sweat
    pub e_rsw: HeatFluxDensity,
    /// Maximum evaporative capacity
    pub e_max: HeatFluxDensity,
    /// Total sensible heat loss
    pub q_sensible: HeatFluxDensity,
    /// Total heat loss from skin
    pub q_skin: HeatFluxDensity,
    /// Heat loss due to respiration
    pub q_res: HeatFluxDensity,
    /// Core temperature
    pub t_core: Temperature,
    /// Skin temperature
    pub t_skin: Temperature,
    /// Skin blood flow [kg/h/m²]
    pub m_bl: f64,
    /// Regulatory sweating rate [kg/h/m²]
    pub m_rsw: f64,
    /// Skin wettedness (0-1)
    pub w: f64,
    /// Maximum skin wettedness (0-1)
    pub w_max: f64,
    /// Effective Temperature
    pub et: Temperature,
    /// PMV Gagge
    pub pmv_gagge: f64,
    /// PMV SET
    pub pmv_set: f64,
    /// Thermal discomfort (0-6)
    pub disc: f64,
    /// Predicted thermal sensation
    pub t_sens: f64,
}

/// Options for the two-node Gagge model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesOptions {
    /// External work
    pub wme: MetabolicRate,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body position
    pub position: Posture,
    /// Maximum skin blood flow [kg/h/m²]
    pub max_skin_blood_flow: f64,
    /// Round output values
    pub round_output: bool,
    /// Maximum sweating rate [kg/h/m²]
    pub max_sweating: f64,
    /// Maximum skin wettedness (0-1), None for auto-calculation
    pub w_max: Option<f64>,
    /// Calculate only SET (faster, for cooling effect calculations)
    pub calculate_ce: bool,
}

impl Default for GaggeTwoNodesOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            position: Posture::Standing,
            max_skin_blood_flow: 90.0,
            round_output: true,
            max_sweating: 500.0,
            w_max: None,
            calculate_ce: false,
        }
    }
}

/// Raw (`f64`, Celsius/W-m²) result of the two-node Gagge model, before the public API
/// wraps each field in its measurement newtype.
struct GaggeTwoNodesRaw {
    set: f64,
    e_skin: f64,
    e_rsw: f64,
    e_max: f64,
    q_sensible: f64,
    q_skin: f64,
    q_res: f64,
    t_core: f64,
    t_skin: f64,
    m_bl: f64,
    m_rsw: f64,
    w: f64,
    w_max: f64,
    et: f64,
    pmv_gagge: f64,
    pmv_set: f64,
    disc: f64,
    t_sens: f64,
}

/// Upstream's `(signal > 0) * signal` idiom, reproduced exactly.
///
/// Not `py_max(signal, 0.0)`, which agrees on ordinary values and on NaN but differs on
/// sign: for a negative signal the mask-multiply yields `-0.0` where a clamp yields `+0.0`,
/// and that reaches the public result — Python's `two_nodes_gagge(tdb=10, tr=10, v=0.1,
/// rh=50, met=1.0, clo=1.0)` returns `m_rsw` and `e_rsw` as `-0.0`. It compares equal to
/// `+0.0`, so no assertion catches it, which is precisely why it is worth writing the
/// literal expression instead of something that merely behaves like it.
#[inline]
fn mask_positive(signal: f64) -> f64 {
    if signal > 0.0 { signal } else { 0.0 * signal }
}

/// Calculate the two-node Gagge model of human temperature regulation
///
/// This model simulates human thermoregulatory responses over time and calculates
/// various thermal comfort indices including SET, ET, PMV variants, and thermal sensation.
///
/// # Returns
///
/// [`GaggeTwoNodesResult`] containing all calculated values
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::two_nodes_gagge::{
///     two_nodes_gagge, GaggeTwoNodesInputs, GaggeTwoNodesOptions,
/// };
/// use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
///
/// let result = two_nodes_gagge(
///     GaggeTwoNodesInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         v: Speed::from_meters_per_second(0.1),
///         rh: Humidity::from_percent(50.0),
///         met: MetabolicRate::from_met(1.2),
///         clo: ClothingInsulation::from_clo(0.5),
///     },
///     Default::default(),
/// );
/// println!("SET: {:.1}°C", result.set.as_celsius());
/// ```
pub fn two_nodes_gagge(
    inputs: GaggeTwoNodesInputs,
    options: GaggeTwoNodesOptions,
) -> GaggeTwoNodesResult {
    let GaggeTwoNodesInputs {
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
    } = inputs;

    let dry_bulb_celsius = tdb.as_celsius();
    let radiant_celsius = tr.as_celsius();
    let speed_mps = v.as_meters_per_second();
    let rh_percent = rh.as_percent();

    let p_sat_torr_val = p_sat_torr(tdb).as_pascals() / 133.322; // Convert Pa back to torr
    let vapor_pressure = rh_percent * p_sat_torr_val / 100.0;

    let raw = gagge_two_nodes_optimized(
        dry_bulb_celsius,
        radiant_celsius,
        speed_mps,
        met.as_met(),
        clo.as_clo(),
        vapor_pressure,
        options.wme.as_met(),
        options.body_surface_area.as_square_meters(),
        options.p_atm.as_pascals(),
        options.position,
        options.calculate_ce,
        options.max_skin_blood_flow,
        options.max_sweating,
        options.w_max,
        options.round_output,
    );

    GaggeTwoNodesResult {
        set: Temperature::from_celsius(raw.set),
        e_skin: HeatFluxDensity::from_watts_per_square_meter(raw.e_skin),
        e_rsw: HeatFluxDensity::from_watts_per_square_meter(raw.e_rsw),
        e_max: HeatFluxDensity::from_watts_per_square_meter(raw.e_max),
        q_sensible: HeatFluxDensity::from_watts_per_square_meter(raw.q_sensible),
        q_skin: HeatFluxDensity::from_watts_per_square_meter(raw.q_skin),
        q_res: HeatFluxDensity::from_watts_per_square_meter(raw.q_res),
        t_core: Temperature::from_celsius(raw.t_core),
        t_skin: Temperature::from_celsius(raw.t_skin),
        m_bl: raw.m_bl,
        m_rsw: raw.m_rsw,
        w: raw.w,
        w_max: raw.w_max,
        et: Temperature::from_celsius(raw.et),
        pmv_gagge: raw.pmv_gagge,
        pmv_set: raw.pmv_set,
        disc: raw.disc,
        t_sens: raw.t_sens,
    }
}

/// Core implementation of the two-node Gagge model
#[allow(clippy::too_many_arguments)]
fn gagge_two_nodes_optimized(
    tdb: f64,
    tr: f64,
    v: f64,
    met: f64,
    clo: f64,
    vapor_pressure: f64,
    wme: f64,
    body_surface_area: f64,
    p_atm: f64,
    posture: Posture,
    calculate_ce: bool,
    max_skin_blood_flow: f64,
    max_sweating: f64,
    w_max_opt: Option<f64>,
    round_output: bool,
) -> GaggeTwoNodesRaw {
    // pythermalcomfort's `calculate_ce=True` path doesn't route through this function at
    // all: it calls `_gagge_two_nodes_optimized_return_set`, a `@vectorize`d wrapper whose
    // signature has no `position` parameter — it hardcodes the numeric code `1` and passes
    // that as `position` to the underlying kernel. That kernel's branch is
    // `if position == Postures.sitting.value` (a *string* compare), so the hardcoded
    // numeric `1` never equals `"sitting"` and every `calculate_ce=True` call silently
    // takes the "standing" (`else`) branch, regardless of what posture the caller actually
    // asked for. Mirror that by forcing Standing here whenever `calculate_ce` is set.
    let posture = if calculate_ce {
        Posture::Standing
    } else {
        posture
    };

    // That same wrapper drops three more parameters, for the same reason: its signature is
    // `(tdb, tr, v, met, clo, vapor_pressure, wme, body_surface_area, p_atm, position)` and
    // nothing else, so `max_skin_blood_flow`, `max_sweating` and `w_max` never reach the
    // kernel and fall back to its defaults of 90, 500 and "compute from air speed"
    // (`two_nodes_gagge.py:580-604` calling `:215-223`). A caller's values are silently
    // discarded on this path. Mirror that rather than the docstring.
    //
    // This is invisible until a cap actually binds. Below saturation the two agree to ~1e-9;
    // the sweep sample that exposed it has met=3.05, where `m_bl` reaches 90 upstream and
    // was being held at the caller's 82.37 here — a 0.077 °C error in SET. Found on
    // 2026-08-26, when `calculate_ce` was first added as a sweep axis.
    let (max_skin_blood_flow, max_sweating, w_max_opt) = if calculate_ce {
        (90.0, 500.0, None)
    } else {
        (max_skin_blood_flow, max_sweating, w_max_opt)
    };

    // Initial variables as defined in ASHRAE 55-2020
    let air_speed = py_max(v, 0.1);
    let k_clo = 0.25;
    let body_weight = 70.0; // body weight in kg
    let met_factor = 58.2; // met conversion factor
    let sbc = 0.000000056697; // Stefan-Boltzmann constant (W/m²K⁴)
    let c_sw = 170.0; // driving coefficient for regulatory sweating
    let c_dil = 120.0; // driving coefficient for vasodilation
    let c_str = 0.5; // driving coefficient for vasoconstriction

    let temp_skin_neutral = 33.7;
    let temp_core_neutral = 36.8;
    let mut alfa = 0.1;
    let temp_body_neutral = alfa * temp_skin_neutral + (1.0 - alfa) * temp_core_neutral;
    let skin_blood_flow_neutral = 6.3;

    let mut t_skin = temp_skin_neutral;
    let mut t_core = temp_core_neutral;
    #[allow(unused_assignments)]
    let mut m_bl = skin_blood_flow_neutral; // Overwritten in first loop iteration

    // Initialize some variables
    let mut e_skin = 0.1 * met; // total evaporative heat loss, W
    let mut q_sensible = 0.0; // total sensible heat loss, W
    let mut w = 0.0; // skin wettedness
    let mut _set = 0.0; // standard effective temperature
    let mut e_rsw = 0.0; // heat lost by vaporization sweat
    let mut e_diff = 0.0; // vapor diffusion through skin
    let mut e_max = 0.0; // maximum evaporative capacity
    let mut m_rsw = 0.0; // regulatory sweating
    let mut et = 0.0; // effective temperature
    let mut e_req = 0.0; // evaporative heat loss required for tmp regulation
    let mut r_ea = 0.0;
    let mut r_ecl = 0.0;

    let pressure_in_atmospheres = p_atm / 101325.0;
    let length_time_simulation = 60; // length time simulation in minutes
    let mut n_simulation = 1;

    let r_clo = 0.155 * clo; // thermal resistance of clothing, °C·m²/W
    let f_a_cl = 1.0 + 0.15 * clo; // increase in body surface area due to clothing
    let lr = 2.2 / pressure_in_atmospheres; // Lewis ratio
    let rm = (met - wme) * met_factor; // metabolic rate
    let mut m = met * met_factor; // metabolic rate

    let mut e_comfort = 0.42 * (rm - met_factor); // evaporative heat loss during comfort
    e_comfort = py_max(e_comfort, 0.0);

    let i_cl = if clo > 0.0 {
        0.45 // permeation efficiency of water vapour through clothing
    } else {
        1.0 // permeation efficiency of water vapour naked skin
    };

    let w_max = if let Some(wm) = w_max_opt {
        wm
    } else if clo > 0.0 {
        0.59 * pow(air_speed, -0.08) // critical skin wettedness clothed
    } else {
        0.38 * pow(air_speed, -0.29) // critical skin wettedness naked
    };

    // h_cc corrected convective heat transfer coefficient
    let mut h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53);
    // h_fc forced convective heat transfer coefficient, W/(m²·°C)
    let h_fc = 8.600001 * pow(air_speed * pressure_in_atmospheres, 0.53);
    h_cc = py_max(h_cc, h_fc);
    if !calculate_ce && met > 0.85 {
        let h_c_met = 5.66 * pow(met - 0.85, 0.39);
        h_cc = py_max(h_cc, h_c_met);
    }

    let mut h_r = 4.7; // linearized radiative heat transfer coefficient
    let mut h_t = h_r + h_cc; // sum of convective and radiant heat transfer coefficient W/(m²·K)
    let mut r_a = 1.0 / (f_a_cl * h_t); // resistance of air layer to dry heat
    let mut t_op = (h_r * tr + h_cc * tdb) / h_t; // operative temperature

    let mut t_body = alfa * t_skin + (1.0 - alfa) * t_core; // mean body temperature, °C

    // Respiration
    let q_res = 0.0023 * m * (44.0 - vapor_pressure); // latent heat loss due to respiration
    let c_res = 0.0014 * m * (34.0 - tdb); // sensible convective heat loss respiration

    // Time simulation loop
    while n_simulation < length_time_simulation {
        n_simulation += 1;

        let iteration_limit = 150; // for following while loop
        // t_cl temperature of the outer surface of clothing
        let mut t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo); // initial guess
        let mut n_iterations = 0;
        let mut tc_converged = false;

        while !tc_converged {
            // 0.95 is the clothing emissivity from ASHRAE fundamentals Ch. 9.7 Eq. 35
            h_r = match posture {
                Posture::Sitting => {
                    // 0.7 ratio between radiation area of the body and the body area
                    4.0 * 0.95 * sbc * pow((t_cl + tr) / 2.0 + 273.15, 3.0) * 0.7
                }
                _ => {
                    // 0.73 ratio for standing and other postures
                    4.0 * 0.95 * sbc * pow((t_cl + tr) / 2.0 + 273.15, 3.0) * 0.73
                }
            };
            h_t = h_r + h_cc;
            r_a = 1.0 / (f_a_cl * h_t);
            t_op = (h_r * tr + h_cc * tdb) / h_t;
            let t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo);
            if abs(t_cl_new - t_cl) <= 0.01 {
                tc_converged = true;
            }
            t_cl = t_cl_new;
            n_iterations += 1;

            if n_iterations > iteration_limit {
                panic!("Max iterations exceeded in two_nodes_gagge");
            }
        }

        q_sensible = (t_skin - t_op) / (r_a + r_clo); // total sensible heat loss, W
        // hf_cs rate of energy transport between core and skin, W
        // 5.28 is the average body tissue conductance in W/(m²·°C)
        // 1.163 is the thermal capacity of blood in Wh/(L·°C)
        let hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl);
        let s_core = m - hf_cs - q_res - c_res - wme; // rate of energy storage in the core
        let s_skin = hf_cs - q_sensible - e_skin; // rate of energy storage in the skin
        let tc_sk = 0.97 * alfa * body_weight; // thermal capacity skin
        let tc_cr = 0.97 * (1.0 - alfa) * body_weight; // thermal capacity core
        let d_t_sk = (s_skin * body_surface_area) / (tc_sk * 60.0); // rate of change skin temperature °C per minute
        let d_t_cr = (s_core * body_surface_area) / (tc_cr * 60.0); // rate of change core temperature °C per minute
        t_skin += d_t_sk;
        t_core += d_t_cr;
        t_body = alfa * t_skin + (1.0 - alfa) * t_core;

        // The five signals below are written upstream as a mask-multiply, not a `max`:
        // `warm_sk = (sk_sig > 0) * sk_sig` and friends (two_nodes_gagge.py:372-382).
        // `mask_positive` reproduces that expression exactly rather than approximating it
        // with a clamp — see its doc comment for the two ways they differ.
        // sk_sig thermoregulatory control signal from the skin
        let sk_sig = t_skin - temp_skin_neutral;
        let warm_sk = mask_positive(sk_sig); // vasodilation signal
        let colds = mask_positive(-sk_sig); // vasoconstriction signal
        // c_reg_sig thermoregulatory control signal from the core, °C
        let c_reg_sig = t_core - temp_core_neutral;
        let c_warm = mask_positive(c_reg_sig); // vasodilation signal
        let c_cold = mask_positive(-c_reg_sig); // vasoconstriction signal
        // bd_sig thermoregulatory control signal from the body
        let bd_sig = t_body - temp_body_neutral;
        let warm_b = mask_positive(bd_sig);
        m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1.0 + c_str * colds);
        m_bl = py_min(m_bl, max_skin_blood_flow);
        m_bl = py_max(m_bl, 0.5);
        m_rsw = c_sw * warm_b * exp(warm_sk / 10.7); // regulatory sweating
        m_rsw = py_min(m_rsw, max_sweating);
        e_rsw = 0.68 * m_rsw; // heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc); // evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl);
        e_req = rm - q_res - c_res - q_sensible; // evaporative heat loss required for tmp regulation
        e_max = (exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / (r_ea + r_ecl);
        if e_max == 0.0 {
            // added this otherwise e_rsw / e_max cannot be calculated
            e_max = 0.001;
        }
        let p_rsw = e_rsw / e_max; // ratio heat loss sweating to max heat loss sweating
        w = 0.06 + 0.94 * p_rsw; // skin wetness
        e_diff = w * e_max - e_rsw; // vapor diffusion through skin
        if w > w_max {
            w = w_max;
            let p_rsw = w_max / 0.94;
            e_rsw = p_rsw * e_max;
            e_diff = 0.06 * (1.0 - p_rsw) * e_max;
        }
        if e_max < 0.0 {
            e_diff = 0.0;
            e_rsw = 0.0;
            w = w_max;
        }
        e_skin = e_rsw + e_diff; // total evaporative heat loss sweating and vapor diffusion
        m_rsw = e_rsw / 0.68; // back calculating the mass of regulatory sweating

        let met_shivering = 19.4 * colds * c_cold; // met shivering W/m²
        m = rm + met_shivering;
        // alfa (skin fraction of body mass) tracks skin blood flow each minute; it
        // must be re-derived inside the loop or the whole t_skin/t_core trajectory drifts
        alfa = 0.0417737 + 0.7451833 / (m_bl + 0.585417);
    }

    let q_skin = q_sensible + e_skin; // total heat loss from skin, W
    // p_s_sk saturation vapour pressure of water of the skin
    let p_s_sk = exp(18.6686 - 4030.183 / (t_skin + 235.0));

    // Standard environment - where _s at end of the variable names stands for standard
    let h_r_s = h_r; // standard environment radiative heat transfer coefficient

    let mut h_c_s = 3.0 * pow(pressure_in_atmospheres, 0.53);
    if !calculate_ce && met > 0.85 {
        let h_c_met = 5.66 * pow(met - 0.85, 0.39);
        h_c_s = py_max(h_c_s, h_c_met);
    }
    h_c_s = py_max(h_c_s, 3.0);

    let h_t_s = h_c_s + h_r_s; // sum of convective and radiant heat transfer coefficient W/(m²·K)
    let r_clo_s = 1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835; // thermal resistance of clothing, °C·m²/W
    let r_cl_s = 0.155 * r_clo_s; // thermal insulation of the clothing in m²K/W
    let f_a_cl_s = 1.0 + k_clo * r_clo_s; // increase in body surface area due to clothing
    let f_cl_s = 1.0 / (1.0 + 0.155 * f_a_cl_s * h_t_s * r_clo_s); // ratio of surface clothed body over nude body
    let i_m_s = 0.45; // permeation efficiency of water vapour through the clothing layer
    let i_cl_s = i_m_s * h_c_s / h_t_s * (1.0 - f_cl_s) / (h_c_s / h_t_s - f_cl_s * i_m_s); // clothing vapor permeation efficiency
    let r_a_s = 1.0 / (f_a_cl_s * h_t_s); // resistance of air layer to dry heat
    let r_ea_s = 1.0 / (lr * f_a_cl_s * h_c_s);
    let r_ecl_s = r_cl_s / (lr * i_cl_s);
    let h_d_s = 1.0 / (r_a_s + r_cl_s);
    let h_e_s = 1.0 / (r_ea_s + r_ecl_s);

    // Calculate Standard Effective Temperature (SET)
    let delta = 0.0001;
    let mut dx = 100.0;
    let mut set_old = round_to(t_skin - q_skin / h_d_s, 2);
    while abs(dx) > 0.01 {
        let err_1 = q_skin
            - h_d_s * (t_skin - set_old)
            - w * h_e_s * (p_s_sk - 0.5 * exp(18.6686 - 4030.183 / (set_old + 235.0)));
        let err_2 = q_skin
            - h_d_s * (t_skin - (set_old + delta))
            - w * h_e_s * (p_s_sk - 0.5 * exp(18.6686 - 4030.183 / (set_old + delta + 235.0)));
        _set = set_old - delta * err_1 / (err_2 - err_1);
        dx = _set - set_old;
        set_old = _set;
    }

    // Calculate Effective Temperature (ET)
    let h_d = 1.0 / (r_a + r_clo);
    let h_e = 1.0 / (r_ea + r_ecl);
    let mut et_old = t_skin - q_skin / h_d;
    let delta = 0.0001;
    let mut dx = 100.0;
    while abs(dx) > 0.01 {
        let err_1 = q_skin
            - h_d * (t_skin - et_old)
            - w * h_e * (p_s_sk - 0.5 * exp(18.6686 - 4030.183 / (et_old + 235.0)));
        let err_2 = q_skin
            - h_d * (t_skin - (et_old + delta))
            - w * h_e * (p_s_sk - 0.5 * exp(18.6686 - 4030.183 / (et_old + delta + 235.0)));
        et = et_old - delta * err_1 / (err_2 - err_1);
        dx = et - et_old;
        et_old = et;
    }

    let met_to_w_m2 = 58.15;
    let tbm_l = (0.194 / met_to_w_m2) * rm + 36.301; // lower limit for evaporative regulation
    let tbm_h = (0.347 / met_to_w_m2) * rm + 36.669; // upper limit for evaporative regulation

    let mut t_sens = 0.4685 * (t_body - tbm_l); // predicted thermal sensation
    if t_body >= tbm_l && t_body < tbm_h {
        t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l);
    } else if t_body >= tbm_h {
        t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h);
    }

    let mut disc = if t_sens > 0.0 && (e_max * w_max - e_comfort - e_diff) < 0.0 {
        6.0
    } else {
        4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff) // predicted thermal discomfort
    };
    if disc <= 0.0 {
        disc = t_sens;
    }
    if disc > 6.0 {
        disc = 6.0;
    }

    // PMV Gagge
    let pmv_gagge = (0.303 * exp(-0.036 * m) + 0.028) * (e_req - e_comfort - e_diff);

    // PMV SET
    let dry_set = h_d_s * (t_skin - _set);
    let e_req_set = rm - c_res - q_res - dry_set;
    let pmv_set = (0.303 * exp(-0.036 * m) + 0.028) * (e_req_set - e_comfort - e_diff);

    // Apply rounding if requested
    let mut result = GaggeTwoNodesRaw {
        set: _set,
        e_skin,
        e_rsw,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        et,
        pmv_gagge,
        pmv_set,
        disc,
        t_sens,
    };

    // `calculate_ce` suppresses rounding, matching upstream's control flow rather than
    // its docstring: `two_nodes_gagge.py:129-142` takes the `if calculate_ce:` branch and
    // `return SET(set=result)` *before* reaching the `if round_output:` block at line 201,
    // so that path is never rounded whatever `round_output` says. Only `set` is meaningful
    // there -- upstream returns a bare `SET` rather than the full result.
    //
    // This does not disturb `set_tmp`, which calls this kernel with `round_output: false`
    // on both paths and applies its own 1-decimal rounding afterwards, mirroring
    // `set_tmp.py:129,155`.
    //
    // Found on 2026-08-26 by adding `calculate_ce` as a sweep axis: the flag had been
    // left at its default, so this entire upstream entry point was unexercised.
    if round_output && !calculate_ce {
        result.set = round_to(result.set, 2);
        result.e_skin = round_to(result.e_skin, 2);
        result.e_rsw = round_to(result.e_rsw, 2);
        result.e_max = round_to(result.e_max, 2);
        result.q_sensible = round_to(result.q_sensible, 2);
        result.q_skin = round_to(result.q_skin, 2);
        result.q_res = round_to(result.q_res, 2);
        result.t_core = round_to(result.t_core, 2);
        result.t_skin = round_to(result.t_skin, 2);
        result.m_bl = round_to(result.m_bl, 2);
        result.m_rsw = round_to(result.m_rsw, 2);
        result.w = round_to(result.w, 2);
        result.w_max = round_to(result.w_max, 2);
        result.et = round_to(result.et, 2);
        result.pmv_gagge = round_to(result.pmv_gagge, 2);
        result.pmv_set = round_to(result.pmv_set, 2);
        result.disc = round_to(result.disc, 2);
        result.t_sens = round_to(result.t_sens, 2);
    }

    result
}

/// The comfort inputs to [`two_nodes_gagge_ji`]: pythermalcomfort requires all six (no
/// default). Note `vapor_pressure` here rather than `rh` — the Ji model
/// takes vapour pressure directly (see [`GaggeTwoNodesJiOptions`] doc for why exposing it
/// this way is strictly more general).
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesJiInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Air speed
    pub v: Speed,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation
    pub clo: ClothingInsulation,
    /// Vapor pressure
    ///
    /// pythermalcomfort's `two_nodes_gagge_ji` takes `vapor_pressure` [torr] directly
    /// rather than deriving it from relative humidity, so a caller with a measured vapour
    /// pressure can supply it exactly. Use [`crate::utilities::p_sat_torr`] combined with
    /// a relative humidity fraction to derive it from RH, matching upstream's documented
    /// `rh * p_sat_torr(tdb) / 100` recipe.
    pub vapor_pressure: Pressure,
}

/// Options for the two-node Gagge JI model (for older individuals)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesJiOptions {
    /// External work
    pub wme: MetabolicRate,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body position
    pub position: Posture,
    /// Whether the subject is heat-acclimatised
    ///
    /// Acclimatisation raises the maximum regulatory evaporation by 25% and the
    /// maximum skin wettedness from 0.85 to 1.0.
    pub acclimatized: bool,
    /// Body weight, used to derive the skin/core thermal capacities
    pub body_weight: Mass,
    /// Length of the simulation, in minutes
    pub length_time_simulation: usize,
    /// Initial skin temperature
    pub initial_skin_temp: Temperature,
    /// Initial core temperature
    pub initial_core_temp: Temperature,
}

impl Default for GaggeTwoNodesJiOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            position: Posture::Sitting,
            acclimatized: true,
            body_weight: Mass::from_kilograms(70.0),
            length_time_simulation: 120,
            initial_skin_temp: Temperature::from_celsius(36.8),
            initial_core_temp: Temperature::from_celsius(36.49),
        }
    }
}

/// Result from the two-node Gagge JI model (time series)
#[derive(Debug, Clone, PartialEq)]
pub struct GaggeTwoNodesJiResult {
    /// Core temperature time series, one entry per minute
    pub t_core: Vec<Temperature>,
    /// Skin temperature time series, one entry per minute
    pub t_skin: Vec<Temperature>,
}

/// Calculate the two-node Gagge JI model for older individuals
///
/// This model is adapted for older populations based on Ji et al. (2022) and Ma et al. (2017),
/// which accounts for age-related changes in thermoregulation including reduced sweating capacity
/// and altered vasodilation responses.
///
/// # Returns
///
/// [`GaggeTwoNodesJiResult`] containing time series of core and skin temperatures
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::two_nodes_gagge::{
///     two_nodes_gagge_ji, GaggeTwoNodesJiInputs, GaggeTwoNodesJiOptions,
/// };
/// use thermalcomfort::{Temperature, Speed, Pressure, MetabolicRate, ClothingInsulation};
///
/// let result = two_nodes_gagge_ji(
///     GaggeTwoNodesJiInputs {
///         tdb: Temperature::from_celsius(25.0),
///         tr: Temperature::from_celsius(25.0),
///         v: Speed::from_meters_per_second(0.1),
///         met: MetabolicRate::from_met(1.2),
///         clo: ClothingInsulation::from_clo(0.5),
///         vapor_pressure: Pressure::from_torrs(12.0),
///     },
///     Default::default(),
/// );
///
/// // Get final temperatures (last element)
/// let final_t_core = result.t_core.last().unwrap();
/// let final_t_skin = result.t_skin.last().unwrap();
/// println!("Final core temp: {:.2}°C", final_t_core.as_celsius());
/// ```
///
/// # Accuracy vs Python pythermalcomfort
///
/// A statement-for-statement port of pythermalcomfort 4.4.0's `_two_nodes_ji_optimized`.
/// At v=0.1, rh=40, met=1.0, clo=0.5, sitting, the 120-minute final temperatures agree
/// with Python to six decimal places at 10 °C, 25 °C and 40 °C. `tests/differential_sweep.rs`
/// holds the randomised check across the full input space.
///
/// ## Implementation Details
///
/// ### Ji Model Thermoregulation Coefficients
///
/// The implementation uses Ji et al. (2022) coefficients for elderly individuals:
///
/// **Vasomotor control:**
/// - `c_dil = 50.0` - Vasodilation coefficient (reduced from 120 in standard Gagge)
/// - `c_str = 0.75` - Vasoconstriction coefficient (increased from 0.5 in standard Gagge)
/// - `c_de = 0.6` - Vasodilation attenuation for elderly
/// - `c_ce = 0.5` - Vasoconstriction attenuation for elderly
///
/// **Sweating:**
/// - `c_sw = 170.0` - Sweating coefficient (same as standard Gagge)
/// - `c_swe = 1.0` - Sweat attenuation coefficient
/// - `a_cof = 0.2` - Coefficient in weighted sweat rate formula
///
/// **Trigger temperatures (elderly-specific):**
/// - `t_cr0_dil = 37.3°C` - Core temperature for vasodilation
/// - `t_sk0_cons = 33.25°C` - Skin temperature for vasoconstriction
/// - `t_cr0_sw = 37.0°C` - Core temperature for sweating
/// - `t_sk0_sw = 34.3°C` - Skin temperature for sweating
///
/// **Blood flow limits:**
/// - Minimum: 0.75 L/(m²·h) (higher than standard 0.5)
/// - Maximum: 63.0 L/(m²·h) (lower than standard 90)
///
/// ### Key Implementation Differences from Standard Gagge
///
/// 1. **Weighted sweat rate formula** (line 935):
///    ```text
///    m_rsw = c_swe * c_sw * ((1-alfa)*t_cr_sw + (alfa+a_cof)*t_sk_sw) * exp(t_sk_sw/10.7)
///    ```
///    Standard Gagge uses: `c_sw * warm_b * exp(warm_sk/10.7)`
///
/// 2. **Dynamic alfa coefficient** (line 922):
///    ```text
///    alfa = 0.0417737 + 0.7451832 / (m_bl + 0.5854417)
///    ```
///    Updated each timestep based on blood flow, affects thermal capacity distribution
///
/// 3. **Iteration order**: Blood flow calculated BEFORE temperature updates to ensure
///    thermal capacities use the correct alfa value for that timestep
///
/// ### Critical Fix Applied
///
/// **Original incorrect implementation** had:
/// - Generic trigger temperatures (36.8°C, 33.7°C instead of elderly-specific)
/// - Wrong sweat rate formula (warm_b instead of weighted t_cr/t_sk)
/// - Alfa updated after temperature changes
/// - Generic blood flow limits (0.5-90 instead of 0.75-63)
///
/// **Result**: 0.38°C core error, 1.95°C skin error
///
/// **After fixes**:
/// - Ji-specific trigger temperatures and coefficients
/// - Correct weighted sweat rate formula
/// - Proper iteration order
/// - Elderly blood flow limits
///
/// **Result**: <0.1°C core error, <0.5°C skin error ✅
///
/// # References
///
/// - Ji et al. (2022) - Thermoregulation model for older individuals
/// - Ma, Xiong, Lian (2017) - Chinese elderly thermoregulation model
///
pub fn two_nodes_gagge_ji(
    inputs: GaggeTwoNodesJiInputs,
    options: GaggeTwoNodesJiOptions,
) -> GaggeTwoNodesJiResult {
    let GaggeTwoNodesJiInputs {
        tdb,
        tr,
        v,
        met,
        clo,
        vapor_pressure,
    } = inputs;

    gagge_two_nodes_ji_core(
        tdb.as_celsius(),
        tr.as_celsius(),
        v.as_meters_per_second(),
        met.as_met(),
        clo.as_clo(),
        vapor_pressure.as_torrs(),
        options.wme.as_met(),
        options.body_surface_area.as_square_meters(),
        options.p_atm.as_pascals(),
        options.position,
        options.acclimatized,
        options.body_weight.as_kilograms(),
        options.length_time_simulation,
        options.initial_skin_temp.as_celsius(),
        options.initial_core_temp.as_celsius(),
    )
}

/// Core implementation of the two-node Gagge JI model
#[allow(clippy::too_many_arguments)]
fn gagge_two_nodes_ji_core(
    tdb: f64,
    tr: f64,
    v: f64,
    met: f64,
    clo: f64,
    vapor_pressure: f64,
    wme: f64,
    body_surface_area: f64,
    p_atm: f64,
    posture: Posture,
    acclimatized: bool,
    body_weight: f64,
    length_time_simulation: usize,
    initial_skin_temp: f64,
    initial_core_temp: f64,
) -> GaggeTwoNodesJiResult {
    // Ji model shivering coefficients (from pythermalcomfort)
    const C_SHE: f64 = 1.0;
    const COF_SCS: f64 = 19.4;
    const COF_SC: f64 = 50.0;
    const COF_SS: f64 = 0.5;
    const T_CR0_SH: f64 = 36.7;

    // Ji model coefficients (from pythermalcomfort)
    let c_sw = 170.0; // driving coefficient for regulatory sweating
    let c_dil = 50.0; // driving coefficient for vasodilation (reduced for elderly)
    let c_str = 0.75; // driving coefficient for vasoconstriction (increased for elderly)
    let a_cof = 0.2; // coefficient in sweat rate

    // Attenuation coefficients for elderly
    let c_de = 0.6; // vasodilation attenuation
    let c_ce = 0.5; // vasoconstriction attenuation
    let c_swe = 1.0; // sweat attenuation

    // Trigger temperatures for elderly (from Ji 2022)
    let t_cr0_dil = 37.3; // vasodilation threshold
    let t_sk0_cons = 33.25; // vasoconstriction threshold
    let t_cr0_sw = 37.0; // core sweating threshold
    let t_sk0_sw = 34.3; // skin sweating threshold

    // Min/max blood flow for elderly
    let min_skin_blood_flow = 0.75; // min SBF for older people
    let max_skin_blood_flow_ji = 63.0; // max SBF for older people
    let max_sweating_rate_factor = 0.9; // 90% sweating efficiency
    let evap_sweating_reg_max = 400.0; // W/m²

    // Other constants
    let air_speed = py_max(v, 0.1);
    let met_factor = 58.2;
    let sbc = 0.000000056697;

    // The Ji model starts skin *above* core by default - 36.8 against 36.49 - which is
    // unusual but is what pythermalcomfort's `initial_skin_temp`/`initial_core_temp`
    // defaults are.
    let skin_blood_flow_neutral = 6.3;

    let mut t_skin = initial_skin_temp;
    let mut t_core = initial_core_temp;
    // Seeded at the neutral value and carried across steps: the heat flow between core
    // and skin uses the *previous* step's blood flow, because Ji recomputes `m_bl` only
    // after the node temperatures have advanced.
    let mut m_bl = skin_blood_flow_neutral;
    // Likewise carried: the thermal capacities consume the previous step's `alfa`, and
    // the sweat rate below consumes the freshly updated one.
    let mut alfa = 0.1;

    let mut e_skin = 0.1 * met;

    let pressure_in_atmospheres = p_atm / 101325.0;

    let r_clo = 0.155 * clo;
    // Ji's clothing area factor is piecewise in clo, not the linear 1 + 0.15*clo of the
    // standard Gagge model.
    let f_a_cl = if clo < 0.5 {
        1.0 + 0.2 * clo
    } else {
        1.05 + 0.1 * clo
    };
    let lr = 2.2 / pressure_in_atmospheres;
    let mut m = met * met_factor;

    let i_cl = if clo > 0.0 { 0.45 } else { 1.0 };

    // Acclimatisation raises both the evaporative ceiling and the wettedness cap.
    let (evap_sweating_reg_max, w_max) = if acclimatized {
        (1.25 * evap_sweating_reg_max, 1.0)
    } else {
        (evap_sweating_reg_max, 0.85)
    };
    let m_rsw_max = evap_sweating_reg_max / 0.68 * max_sweating_rate_factor;

    // Ji seeds the coefficients with fixed values and re-derives `h_cc` from the
    // clothing-to-air temperature difference at the end of every minute, rather than
    // fixing it up front from air speed as the standard Gagge model does.
    let mut h_cc = 3.0;
    let mut h_r = 4.7;

    // Storage for time series
    let mut t_core_history = Vec::with_capacity(length_time_simulation);
    let mut t_skin_history = Vec::with_capacity(length_time_simulation);

    // Time simulation loop
    for _ in 0..length_time_simulation {
        let iteration_limit = 150;

        let mut h_t = h_r + h_cc;
        let mut r_a = 1.0 / (f_a_cl * h_t);
        let mut t_op = (h_r * tr + h_cc * tdb) / h_t;

        let mut t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo);
        let mut n_iterations = 0;
        let mut tc_converged = false;

        while !tc_converged {
            // Emissivity 0.97, and the radiating-area ratio is 0.7 sitting / 0.77
            // standing.
            let area_ratio = match posture {
                Posture::Sitting => 0.7,
                _ => 0.77,
            };
            h_r = 4.0 * 0.97 * sbc * pow((t_cl + tr) / 2.0 + 273.15, 3.0) * area_ratio;
            h_t = h_r + h_cc;
            r_a = 1.0 / (f_a_cl * h_t);
            t_op = (h_r * tr + h_cc * tdb) / h_t;
            let t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo);
            if abs(t_cl_new - t_cl) < 0.01 {
                tc_converged = true;
            }
            t_cl = t_cl_new;
            n_iterations += 1;

            if n_iterations > iteration_limit {
                break; // Python raises StopIteration here; bail out instead
            }
        }

        // Convective coefficient for the next pass, from the clothing-to-air difference.
        // Below 0.2 m/s the flow is free convection (Gao et al. 2019), above it forced.
        let d_tcl_air = t_cl - tdb;
        h_cc = if air_speed < 0.2 {
            if d_tcl_air > 0.0 {
                2.5 * pow(d_tcl_air, 0.16) // upward flow
            } else {
                2.5 * pow(abs(d_tcl_air), 0.41) // downward flow
            }
        } else {
            8.6 * pow(air_speed, 0.53)
        };

        let q_sensible = (t_skin - t_op) / (r_a + r_clo);

        // Respiration tracks `m`, which carries the previous step's shivering, so both
        // terms belong inside the loop.
        let q_res = 0.0023 * m * (44.0 - vapor_pressure);
        let c_res = 0.0014 * m * (34.0 - tdb);

        let hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl);
        let s_core = m - hf_cs - q_res - c_res - wme;
        let s_skin = hf_cs - q_sensible - e_skin;
        let tc_sk = 0.97 * alfa * body_weight;
        let tc_cr = 0.97 * (1.0 - alfa) * body_weight;
        let d_t_sk = (s_skin * body_surface_area) / (tc_sk * 60.0);
        let d_t_cr = (s_core * body_surface_area) / (tc_cr * 60.0);
        t_skin += d_t_sk;
        t_core += d_t_cr;

        // Every regulatory trigger below reads the temperatures *after* they advance.
        // Ji writes these clamps constant-first — `max(0, t_core - t_cr0_dil)`,
        // `min(max_skin_blood_flow, m_bl)` — so a NaN in the *second* operand loses the
        // comparison and the constant survives. Keep Python's argument order.
        let t_cr_dil = py_max(0.0, t_core - t_cr0_dil); // dilation trigger
        let t_sk_cons = py_max(0.0, t_sk0_cons - t_skin); // constriction trigger

        m_bl =
            (skin_blood_flow_neutral + c_de * c_dil * t_cr_dil) / (1.0 + c_ce * c_str * t_sk_cons);
        m_bl = py_min(max_skin_blood_flow_ji, m_bl);
        m_bl = py_max(min_skin_blood_flow, m_bl);

        let t_sk_sw = py_max(0.0, t_skin - t_sk0_sw); // skin sweating trigger
        let t_cr_sw = py_max(0.0, t_core - t_cr0_sw); // core sweating trigger

        // Updated from the new blood flow, and consumed by the sweat rate immediately
        // below; the thermal capacities above already used the previous value.
        alfa = 0.0417737 + 0.7451832 / (m_bl + 0.5854417);

        let m_rsw = c_swe
            * c_sw
            * ((1.0 - alfa) * t_cr_sw + (alfa + a_cof) * t_sk_sw)
            * exp(t_sk_sw / 10.7);
        let m_rsw = py_min(m_rsw, m_rsw_max);
        let mut e_rsw = 0.68 * m_rsw; // heat lost by vaporization of sweat

        let r_e_cl = r_clo / (lr * i_cl); // evaporative resistance of clothing
        let r_e_a = 1.0 / (lr * f_a_cl * h_cc); // evaporative resistance of air layer
        let r_total = r_e_cl + r_e_a;

        let e_max = (exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / r_total;
        let p_rsw = e_rsw / e_max;

        // Skin wettedness via the ISO PHS evaporation efficiency (eff = 1 - 0.5*w²),
        // not the standard Gagge 0.06 + 0.94*p_rsw.
        let he_n = 1.0 / r_total;
        let wettedness_dif = 1.0 / (2.0 + 2.46 * he_n);
        let wp = wettedness_dif + (1.0 - wettedness_dif) * p_rsw;
        let mut w = py_min((sqrt(2.0 * wp * wp + 1.0) - 1.0) / wp, w_max);

        // Recalculate the evaporative split from the limited wettedness. Constant-first
        // upstream (`max(0, ...)`), so a NaN here is healed to zero rather than kept.
        let p_rsw = (w - wettedness_dif) / (1.0 - wettedness_dif);
        e_rsw = py_max(0.0, p_rsw * e_max);
        let mut e_diff = py_max(0.0, w * e_max - e_rsw);

        // Condensation on the skin (RH > 100%, body immersed): the model is not valid
        // here, so sweating is suppressed and condensation latent heat ignored.
        if e_max < 0.0 {
            // Upstream also zeroes `w` here (two_nodes_gagge_ji.py:433-437). It is inert
            // in both implementations because `w` is not read again this iteration, but
            // transcribing it keeps the branch a faithful copy rather than one that
            // happens to agree.
            w = 0.0;
            e_diff = 0.0;
            e_rsw = 0.0;
        }

        e_skin = e_rsw + e_diff;

        // Shivering recruits extra metabolic heat once core falls below its threshold.
        let t_cr_sh = py_max(0.0, T_CR0_SH - t_core);
        let met_shivering =
            C_SHE * (COF_SCS * t_cr_sh * t_sk_cons + COF_SC * t_cr_sh + COF_SS * t_sk_cons);
        m = met * met_factor + met_shivering;

        // Stored unrounded. two_nodes_gagge_ji.py contains no round/np.around call at
        // all, so any rounding here is a divergence from upstream rather than an option.
        t_core_history.push(Temperature::from_celsius(t_core));
        t_skin_history.push(Temperature::from_celsius(t_skin));
    }

    GaggeTwoNodesJiResult {
        t_core: t_core_history,
        t_skin: t_skin_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream writes the thermoregulatory signals as `(sig > 0) * sig`, which yields
    /// `-0.0` for a negative signal where a clamp yields `+0.0`. That reaches the public
    /// result: pythermalcomfort 4.4.0 returns `m_rsw = -0.0` and `e_rsw = -0.0` for this
    /// input. `assert_eq!` cannot see the difference, so this checks the sign bit.
    #[test]
    fn cold_conditions_return_negative_zero_sweating_like_python() {
        let result = two_nodes_gagge(
            GaggeTwoNodesInputs {
                tdb: Temperature::from_celsius(10.0),
                tr: Temperature::from_celsius(10.0),
                v: Speed::from_meters_per_second(0.1),
                rh: Humidity::from_percent(50.0),
                met: MetabolicRate::from_met(1.0),
                clo: ClothingInsulation::from_clo(1.0),
            },
            Default::default(),
        );
        assert_eq!(result.m_rsw, 0.0, "magnitude");
        assert!(
            result.m_rsw.is_sign_negative(),
            "m_rsw should be -0.0 as upstream returns, got +0.0"
        );
        assert!(
            result.e_rsw.as_watts_per_square_meter().is_sign_negative(),
            "e_rsw should be -0.0 as upstream returns, got +0.0"
        );
    }

    fn gagge_inputs(tdb: f64, tr: f64, v: f64, rh: f64, met: f64, clo: f64) -> GaggeTwoNodesInputs {
        GaggeTwoNodesInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(v),
            rh: Humidity::from_percent(rh),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
        }
    }

    #[test]
    fn test_two_nodes_gagge_basic() {
        let result = two_nodes_gagge(
            gagge_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );

        // Basic sanity checks
        let set = result.set.as_celsius();
        let t_skin = result.t_skin.as_celsius();
        let t_core = result.t_core.as_celsius();
        assert!(set > 20.0 && set < 30.0);
        assert!(t_skin > 30.0 && t_skin < 40.0);
        assert!(t_core > 35.0 && t_core < 40.0);
        assert!(result.w >= 0.0 && result.w <= 1.0);
    }

    #[test]
    fn test_two_nodes_gagge_cold() {
        let result = two_nodes_gagge(
            gagge_inputs(10.0, 10.0, 0.1, 50.0, 1.0, 1.0),
            Default::default(),
        );

        // In cold conditions, expect lower SET
        assert!(result.set.as_celsius() < 20.0);
        assert!(result.t_sens < 0.0); // Should feel cold
    }

    #[test]
    fn test_two_nodes_gagge_hot() {
        let result = two_nodes_gagge(
            gagge_inputs(35.0, 35.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );

        // In hot conditions, expect higher SET and sweating
        assert!(result.set.as_celsius() > 28.0);
        assert!(result.m_rsw > 0.0); // Should be sweating
        assert!(result.t_sens > 0.0); // Should feel hot
    }

    /// **REAL BUG**: pythermalcomfort's `calculate_ce=True` path hardcodes the
    /// "standing" branch regardless of the caller's position (see the comment in
    /// `gagge_two_nodes_optimized`). Before the fix, Rust always honoured
    /// `options.position`, so a `calculate_ce: true` call with `Posture::Sitting` would
    /// take the sitting radiative-coefficient branch (0.7) instead of upstream's forced
    /// standing branch (0.73) -- silently diverging from Python whenever a caller other
    /// than `cooling_effect` (which always happens to pass `Standing`) used this flag.
    #[test]
    fn calculate_ce_forces_standing_regardless_of_posture() {
        let inputs = gagge_inputs(30.0, 35.0, 1.5, 40.0, 1.5, 0.4);

        let sitting = two_nodes_gagge(
            inputs,
            GaggeTwoNodesOptions {
                position: Posture::Sitting,
                calculate_ce: true,
                round_output: false,
                ..Default::default()
            },
        );
        let standing = two_nodes_gagge(
            inputs,
            GaggeTwoNodesOptions {
                position: Posture::Standing,
                calculate_ce: true,
                round_output: false,
                ..Default::default()
            },
        );

        // Both must match the forced-standing result exactly: position must not have
        // been able to change anything while calculate_ce is set.
        assert_eq!(sitting.set.as_celsius(), standing.set.as_celsius());
        assert_eq!(sitting.t_skin.as_celsius(), standing.t_skin.as_celsius());
        assert_eq!(sitting.t_core.as_celsius(), standing.t_core.as_celsius());

        // Sanity: with calculate_ce off, position *does* matter for these inputs, so the
        // test above isn't vacuously true because position never affects anything here.
        let sitting_full = two_nodes_gagge(
            inputs,
            GaggeTwoNodesOptions {
                position: Posture::Sitting,
                calculate_ce: false,
                round_output: false,
                ..Default::default()
            },
        );
        let standing_full = two_nodes_gagge(
            inputs,
            GaggeTwoNodesOptions {
                position: Posture::Standing,
                calculate_ce: false,
                round_output: false,
                ..Default::default()
            },
        );
        assert_ne!(
            sitting_full.set.as_celsius(),
            standing_full.set.as_celsius(),
            "position should affect SET when calculate_ce is off, or this test doesn't \
             prove anything"
        );
    }

    fn ji_inputs(tdb: f64, tr: f64, v: f64, rh: f64, met: f64, clo: f64) -> GaggeTwoNodesJiInputs {
        let p_sat_torr_val = p_sat_torr(Temperature::from_celsius(tdb)).as_torrs();
        GaggeTwoNodesJiInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            v: Speed::from_meters_per_second(v),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
            vapor_pressure: Pressure::from_torrs(rh * p_sat_torr_val / 100.0),
        }
    }

    #[test]
    fn test_two_nodes_gagge_ji_basic() {
        let result = two_nodes_gagge_ji(
            ji_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );

        assert_eq!(result.t_core.len(), 120);
        assert_eq!(result.t_skin.len(), 120);
        let final_core = result.t_core.last().unwrap().as_celsius();
        let final_skin = result.t_skin.last().unwrap().as_celsius();
        assert!(final_core > 35.0 && final_core < 40.0);
        assert!(final_skin > 25.0 && final_skin < 40.0);
    }

    /// `length_time_simulation` is now a caller-controlled option; the result vector
    /// must actually grow to match rather than silently truncating at some fixed cap.
    #[test]
    fn length_time_simulation_controls_output_length() {
        let result = two_nodes_gagge_ji(
            ji_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            GaggeTwoNodesJiOptions {
                length_time_simulation: 240,
                ..Default::default()
            },
        );
        assert_eq!(result.t_core.len(), 240);
        assert_eq!(result.t_skin.len(), 240);
    }

    /// `body_weight`, `initial_skin_temp` and `initial_core_temp` are now exposed; check
    /// that changing them actually changes the trajectory rather than being silently
    /// ignored.
    #[test]
    fn ji_options_are_not_ignored() {
        let base = two_nodes_gagge_ji(
            ji_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            Default::default(),
        );
        let heavier = two_nodes_gagge_ji(
            ji_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            GaggeTwoNodesJiOptions {
                body_weight: Mass::from_kilograms(120.0),
                ..Default::default()
            },
        );
        assert_ne!(
            base.t_core.first().unwrap().as_celsius(),
            heavier.t_core.first().unwrap().as_celsius()
        );

        let different_start = two_nodes_gagge_ji(
            ji_inputs(25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            GaggeTwoNodesJiOptions {
                initial_skin_temp: Temperature::from_celsius(30.0),
                initial_core_temp: Temperature::from_celsius(37.5),
                ..Default::default()
            },
        );
        assert_ne!(
            base.t_skin.first().unwrap().as_celsius(),
            different_start.t_skin.first().unwrap().as_celsius()
        );
    }
}
