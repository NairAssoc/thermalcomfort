//! Two-node Gagge model of human temperature regulation
//!
//! This module implements the Gagge two-node model [Gagge1986] which simulates
//! human thermoregulatory responses and calculates various thermal comfort indices.

use crate::utilities::{Posture, p_sat_torr};
use libm::{exp, fabs as abs, pow};
use measurements::{Area, Humidity, Pressure, Speed, Temperature};

/// Result from the two-node Gagge model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesResult {
    /// Standard Effective Temperature [°C]
    pub set: f64,
    /// Total evaporative heat loss from skin [W/m²]
    pub e_skin: f64,
    /// Heat lost by evaporation of regulatory sweat [W/m²]
    pub e_rsw: f64,
    /// Maximum evaporative capacity [W/m²]
    pub e_max: f64,
    /// Total sensible heat loss [W]
    pub q_sensible: f64,
    /// Total heat loss from skin [W]
    pub q_skin: f64,
    /// Heat loss due to respiration [W]
    pub q_res: f64,
    /// Core temperature [°C]
    pub t_core: f64,
    /// Skin temperature [°C]
    pub t_skin: f64,
    /// Skin blood flow [kg/h/m²]
    pub m_bl: f64,
    /// Regulatory sweating rate [kg/h/m²]
    pub m_rsw: f64,
    /// Skin wettedness (0-1)
    pub w: f64,
    /// Maximum skin wettedness (0-1)
    pub w_max: f64,
    /// Effective Temperature [°C]
    pub et: f64,
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
    /// External work [met]
    pub wme: f64,
    /// Body surface area
    pub body_surface_area: Area,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Body posture
    pub posture: Posture,
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
            wme: 0.0,
            body_surface_area: Area::from_square_meters(1.8258),
            p_atm: Pressure::from_pascals(101325.0),
            posture: Posture::Standing,
            max_skin_blood_flow: 90.0,
            round_output: true,
            max_sweating: 500.0,
            w_max: None,
            calculate_ce: false,
        }
    }
}

#[inline]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[inline]
fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

#[inline]
fn round_to(value: f64, decimals: u32) -> f64 {
    let multiplier = pow(10.0, decimals as f64);
    libm::round(value * multiplier) / multiplier
}

/// Calculate the two-node Gagge model of human temperature regulation
///
/// This model simulates human thermoregulatory responses over time and calculates
/// various thermal comfort indices including SET, ET, PMV variants, and thermal sensation.
///
/// # Arguments
///
/// * `dry_bulb_temp` - Dry bulb air temperature (use `Temperature::from_celsius()` or similar)
/// * `mean_radiant_temp` - Mean radiant temperature (use `Temperature::from_celsius()` or similar)
/// * `air_speed` - Air speed (use `Speed::from_meters_per_second()` or similar)
/// * `relative_humidity` - Relative humidity (use `Humidity::from_percent()` for RH%)
/// * `metabolic_rate` - Metabolic rate [met]
/// * `clothing_insulation` - Clothing insulation [clo]
/// * `options` - Model options
///
/// # Returns
///
/// GaggeTwoNodesResult containing all calculated values
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::two_nodes_gagge::{two_nodes_gagge, GaggeTwoNodesOptions};
/// use thermalcomfort::{Temperature, Speed, Humidity};
///
/// let result = two_nodes_gagge(
///     Temperature::from_celsius(25.0),
///     Temperature::from_celsius(25.0),
///     Speed::from_meters_per_second(0.1),
///     Humidity::from_percent(50.0),
///     1.2,
///     0.5,
///     Default::default()
/// );
/// println!("SET: {:.1}°C", result.set);
/// ```
pub fn two_nodes_gagge(
    dry_bulb_temp: Temperature,
    mean_radiant_temp: Temperature,
    air_speed: Speed,
    relative_humidity: Humidity,
    metabolic_rate: f64,
    clothing_insulation: f64,
    options: GaggeTwoNodesOptions,
) -> GaggeTwoNodesResult {
    let dry_bulb_celsius = dry_bulb_temp.as_celsius();
    let radiant_celsius = mean_radiant_temp.as_celsius();
    let speed_mps = air_speed.as_meters_per_second();
    let rh_percent = relative_humidity.as_percent();

    let p_sat_torr_val = p_sat_torr(dry_bulb_temp).as_pascals() / 133.322; // Convert Pa back to torr
    let vapor_pressure = rh_percent * p_sat_torr_val / 100.0;

    gagge_two_nodes_optimized(
        dry_bulb_celsius,
        radiant_celsius,
        speed_mps,
        metabolic_rate,
        clothing_insulation,
        vapor_pressure,
        options.wme,
        options.body_surface_area.as_square_meters(),
        options.p_atm.as_pascals(),
        options.posture,
        options.calculate_ce,
        options.max_skin_blood_flow,
        options.max_sweating,
        options.w_max,
        options.round_output,
    )
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
) -> GaggeTwoNodesResult {
    // Initial variables as defined in ASHRAE 55-2020
    let air_speed = fmax(v, 0.1);
    let k_clo = 0.25;
    let body_weight = 70.0; // body weight in kg
    let met_factor = 58.2; // met conversion factor
    let sbc = 0.000000056697; // Stefan-Boltzmann constant (W/m²K⁴)
    let c_sw = 170.0; // driving coefficient for regulatory sweating
    let c_dil = 120.0; // driving coefficient for vasodilation
    let c_str = 0.5; // driving coefficient for vasoconstriction

    let temp_skin_neutral = 33.7;
    let temp_core_neutral = 36.8;
    let alfa = 0.1;
    let temp_body_neutral = alfa * temp_skin_neutral + (1.0 - alfa) * temp_core_neutral;
    let skin_blood_flow_neutral = 6.3;

    let mut t_skin = temp_skin_neutral;
    let mut t_core = temp_core_neutral;
    let mut m_bl = skin_blood_flow_neutral;

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
    let q_res; // heat loss due to respiration
    let c_res; // convective heat loss respiration

    let pressure_in_atmospheres = p_atm / 101325.0;
    let length_time_simulation = 60; // length time simulation in minutes
    let mut n_simulation = 1;

    let r_clo = 0.155 * clo; // thermal resistance of clothing, °C·m²/W
    let f_a_cl = 1.0 + 0.15 * clo; // increase in body surface area due to clothing
    let lr = 2.2 / pressure_in_atmospheres; // Lewis ratio
    let rm = (met - wme) * met_factor; // metabolic rate
    let m = met * met_factor; // metabolic rate

    let mut e_comfort = 0.42 * (rm - met_factor); // evaporative heat loss during comfort
    e_comfort = fmax(e_comfort, 0.0);

    let i_cl = if clo > 0.0 {
        0.45 // permeation efficiency of water vapour through clothing
    } else {
        1.0 // permeation efficiency of water vapour naked skin
    };

    let w_max = if let Some(wm) = w_max_opt {
        wm
    } else {
        if clo > 0.0 {
            0.59 * pow(air_speed, -0.08) // critical skin wettedness clothed
        } else {
            0.38 * pow(air_speed, -0.29) // critical skin wettedness naked
        }
    };

    // h_cc corrected convective heat transfer coefficient
    let mut h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53);
    // h_fc forced convective heat transfer coefficient, W/(m²·°C)
    let h_fc = 8.600001 * pow(air_speed * pressure_in_atmospheres, 0.53);
    h_cc = fmax(h_cc, h_fc);
    if !calculate_ce && met > 0.85 {
        let h_c_met = 5.66 * pow(met - 0.85, 0.39);
        h_cc = fmax(h_cc, h_c_met);
    }

    let mut h_r = 4.7; // linearized radiative heat transfer coefficient
    let mut h_t = h_r + h_cc; // sum of convective and radiant heat transfer coefficient W/(m²·K)
    let mut r_a = 1.0 / (f_a_cl * h_t); // resistance of air layer to dry heat
    let mut t_op = (h_r * tr + h_cc * tdb) / h_t; // operative temperature

    let mut t_body = alfa * t_skin + (1.0 - alfa) * t_core; // mean body temperature, °C

    // Respiration
    q_res = 0.0023 * m * (44.0 - vapor_pressure); // latent heat loss due to respiration
    c_res = 0.0014 * m * (34.0 - tdb); // sensible convective heat loss respiration

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
        t_skin = t_skin + d_t_sk;
        t_core = t_core + d_t_cr;
        t_body = alfa * t_skin + (1.0 - alfa) * t_core;

        // sk_sig thermoregulatory control signal from the skin
        let sk_sig = t_skin - temp_skin_neutral;
        let warm_sk = fmax(sk_sig, 0.0); // vasodilation signal
        let colds = fmax(-sk_sig, 0.0); // vasoconstriction signal
        // c_reg_sig thermoregulatory control signal from the core, °C
        let c_reg_sig = t_core - temp_core_neutral;
        let c_warm = fmax(c_reg_sig, 0.0); // vasodilation signal
        let _c_cold = fmax(-c_reg_sig, 0.0); // vasoconstriction signal (unused but kept for clarity)
        // bd_sig thermoregulatory control signal from the body
        let bd_sig = t_body - temp_body_neutral;
        let warm_b = fmax(bd_sig, 0.0);
        m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1.0 + c_str * colds);
        m_bl = fmin(m_bl, max_skin_blood_flow);
        m_bl = fmax(m_bl, 0.5);
        m_rsw = c_sw * warm_b * exp(warm_sk / 10.7); // regulatory sweating
        m_rsw = fmin(m_rsw, max_sweating);
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
    }

    let q_skin = q_sensible + e_skin; // total heat loss from skin, W
    // p_s_sk saturation vapour pressure of water of the skin
    let p_s_sk = exp(18.6686 - 4030.183 / (t_skin + 235.0));

    // Standard environment - where _s at end of the variable names stands for standard
    let h_r_s = h_r; // standard environment radiative heat transfer coefficient

    let mut h_c_s = 3.0 * pow(pressure_in_atmospheres, 0.53);
    if !calculate_ce && met > 0.85 {
        let h_c_met = 5.66 * pow(met - 0.85, 0.39);
        h_c_s = fmax(h_c_s, h_c_met);
    }
    h_c_s = fmax(h_c_s, 3.0);

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

    let mut disc = if t_sens > 0.0 && (e_max * w_max - e_comfort - e_diff) <= 0.0 {
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
    let mut result = GaggeTwoNodesResult {
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

    if round_output {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_nodes_gagge_basic() {
        let result = two_nodes_gagge(
            Temperature::from_celsius(25.0),
            Temperature::from_celsius(25.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            Default::default(),
        );

        // Basic sanity checks
        assert!(result.set > 20.0 && result.set < 30.0);
        assert!(result.t_skin > 30.0 && result.t_skin < 40.0);
        assert!(result.t_core > 35.0 && result.t_core < 40.0);
        assert!(result.w >= 0.0 && result.w <= 1.0);
    }

    #[test]
    fn test_two_nodes_gagge_cold() {
        let result = two_nodes_gagge(
            Temperature::from_celsius(10.0),
            Temperature::from_celsius(10.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.0,
            1.0,
            Default::default(),
        );

        // In cold conditions, expect lower SET
        assert!(result.set < 20.0);
        assert!(result.t_sens < 0.0); // Should feel cold
    }

    #[test]
    fn test_two_nodes_gagge_hot() {
        let result = two_nodes_gagge(
            Temperature::from_celsius(35.0),
            Temperature::from_celsius(35.0),
            Speed::from_meters_per_second(0.1),
            Humidity::from_percent(50.0),
            1.2,
            0.5,
            Default::default(),
        );

        // In hot conditions, expect higher SET and sweating
        assert!(result.set > 28.0);
        assert!(result.m_rsw > 0.0); // Should be sweating
        assert!(result.t_sens > 0.0); // Should feel hot
    }
}
