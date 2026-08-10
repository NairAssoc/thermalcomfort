//! Two-node Gagge model adapted for sleep, after Yan et al. (2022)
//!
//! Mirrors `pythermalcomfort/models/two_nodes_gagge_sleep.py`. Unlike the standard Gagge
//! model this is a per-minute simulation: metabolic rate follows the paper's polynomial
//! and core temperature is *prescribed* by a quadratic, while skin temperature, skin
//! wettedness, blood flow and shivering carry over from the previous minute.
//!
//! The environment is therefore a schedule, not a single condition — one value per minute
//! for each of the six driving variables, exactly as upstream defines it. Their common
//! length is the duration of the night.

extern crate alloc;

use alloc::vec::Vec;

use crate::{ClothingInsulation, HeatFluxDensity, MetabolicRate};
use libm::{exp, fabs as abs, pow, sqrt};
use measurements::{Humidity, Length, Mass, Pressure, Speed, Temperature};

/// Per-minute environmental schedule for the sleep model.
///
/// Every field is one value per minute of the simulation, and all six must be the same
/// length — that length is the duration. This mirrors upstream, where `tdb`, `tr`, `v`,
/// `rh`, `clo` and `thickness_quilt` are arrays and their common length sets how long the
/// night runs.
///
/// Borrowed rather than owned, so a caller holding the schedule in any contiguous
/// container passes it without allocating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SleepInputs<'a> {
    /// Dry bulb air temperature, per minute
    pub tdb: &'a [Temperature],
    /// Mean radiant temperature, per minute
    pub tr: &'a [Temperature],
    /// Air speed, per minute
    pub v: &'a [Speed],
    /// Relative humidity, per minute
    pub rh: &'a [Humidity],
    /// Clothing insulation, per minute
    pub clo: &'a [ClothingInsulation],
    /// Quilt thickness, per minute
    pub thickness_quilt: &'a [Length],
}

/// The six schedules did not all have the same length, so the duration is ambiguous.
///
/// Upstream raises `ValueError` for this; there is no sensible answer to fall back on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MismatchedScheduleLengths {
    /// Lengths of `tdb`, `tr`, `v`, `rh`, `clo` and `thickness_quilt`, in that order
    pub lengths: [usize; 6],
}

impl core::fmt::Display for MismatchedScheduleLengths {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "tdb, tr, v, rh, clo and thickness_quilt must have the same length; got {:?}",
            self.lengths
        )
    }
}

/// Tuning coefficients and initial physiological state for the sleep model.
///
/// These are upstream's `**kwargs`, with upstream's defaults. The last five describe the
/// state the sleeper starts the night in and are carried forward minute to minute
/// thereafter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaggeTwoNodesSleepOptions {
    /// External work
    pub wme: MetabolicRate,
    /// Atmospheric pressure
    pub p_atm: Pressure,
    /// Number of time steps per minute of simulation
    pub ltime: u32,
    /// Body height
    pub height: Length,
    /// Body weight
    pub weight: Mass,
    /// Driving coefficient for regulatory sweating
    pub c_sw: f64,
    /// Driving coefficient for vasodilation
    pub c_dil: f64,
    /// Driving coefficient for vasoconstriction
    pub c_str: f64,
    /// Skin temperature at neutral conditions
    pub temp_skin_neutral: Temperature,
    /// Core temperature at neutral conditions.
    ///
    /// Accepted for parity with upstream's kwarg of the same name, but never read there:
    /// it seeds a dictionary entry that the simulation loop overwrites before anything
    /// looks at it, because each minute's neutral reference is the prescribed core
    /// temperature rather than a fixed value. Kept so the two APIs take the same
    /// arguments; changing it has no effect in either implementation.
    pub temp_core_neutral: Temperature,
    /// Initial evaporative heat loss from skin
    pub e_skin: HeatFluxDensity,
    /// Initial fraction of body mass at skin temperature
    pub alfa: f64,
    /// Initial skin blood flow [L/(h·m²)]
    pub skin_blood_flow: f64,
    /// Initial shivering thermogenesis
    pub met_shivering: HeatFluxDensity,
}

impl Default for GaggeTwoNodesSleepOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            p_atm: Pressure::from_pascals(101325.0),
            ltime: 1,
            height: Length::from_centimeters(171.0),
            weight: Mass::from_kilograms(70.0),
            c_sw: 170.0,
            c_dil: 120.0,
            c_str: 0.5,
            temp_skin_neutral: Temperature::from_celsius(33.7),
            temp_core_neutral: Temperature::from_celsius(36.8),
            e_skin: HeatFluxDensity::from_watts_per_square_meter(0.094),
            alfa: 0.1,
            skin_blood_flow: 6.3,
            met_shivering: HeatFluxDensity::from_watts_per_square_meter(0.0),
        }
    }
}

/// Per-minute results from the sleep model.
///
/// Field names and units mirror upstream's `GaggeTwoNodesSleep` dataclass. Every vector
/// has one entry per minute and they are all the same length as the input schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct GaggeTwoNodesSleepResult {
    /// Standard Effective Temperature
    pub set: Vec<Temperature>,
    /// Core temperature
    pub t_core: Vec<Temperature>,
    /// Skin temperature
    pub t_skin: Vec<Temperature>,
    /// Skin wettedness (0-1)
    pub wet: Vec<f64>,
    /// Predicted thermal sensation
    pub t_sens: Vec<f64>,
    /// Thermal discomfort
    pub disc: Vec<f64>,
    /// Total evaporative heat loss from skin
    pub e_skin: Vec<HeatFluxDensity>,
    /// Shivering thermogenesis
    pub met_shivering: Vec<HeatFluxDensity>,
    /// Fraction of body mass at skin temperature
    pub alfa: Vec<f64>,
    /// Skin blood flow [L/(h·m²)]
    pub skin_blood_flow: Vec<f64>,
}

/// Saturation vapour pressure in torr, mirroring upstream's `_fnsvp`.
///
/// Deliberately local rather than reusing [`crate::utilities::p_sat_torr`], which returns
/// a [`Pressure`]: reaching torr back out of it means multiplying by 133.322 and dividing
/// again, and that round trip is not the identity in floating point. The sleep model feeds
/// this straight into a Newton solve, so the last bits matter for parity.
fn svp_torr(t: f64) -> f64 {
    exp(18.6686 - 4030.183 / (t + 235.0))
}

/// Error term for the SET Newton solve, mirroring upstream's `_fnerre` and `_fnerrs`.
///
/// Upstream defines the two separately, but they are the same expression evaluated with
/// the standard-environment heat-transfer coefficients rather than the actual ones.
#[allow(clippy::too_many_arguments)]
fn set_error(x: f64, hsk: f64, hd: f64, tsk: f64, w: f64, he: f64, pssk: f64) -> f64 {
    hsk - hd * (tsk - x) - w * he * (pssk - 0.5 * svp_torr(x))
}

/// State carried from one simulated minute to the next.
struct SleepState {
    t_skin: f64,
    e_skin: f64,
    alfa: f64,
    skin_blood_flow: f64,
    met_shivering: f64,
}

/// One minute of the sleep simulation, mirroring upstream's `_sleep_set`.
///
/// `t_core` is prescribed by the caller from the Yan quadratic rather than integrated
/// freely, which is what distinguishes this from the standard two-node model.
#[allow(clippy::too_many_arguments)]
fn sleep_set(
    tdb: f64,
    tr: f64,
    v: f64,
    rh: f64,
    clo: f64,
    thickness: f64,
    met: f64,
    t_core_prescribed: f64,
    state: &SleepState,
    options: &GaggeTwoNodesSleepOptions,
) -> (SleepMinute, SleepState) {
    let height_cm = options.height.as_centimeters();
    let weight_kg = options.weight.as_kilograms();
    let wme = options.wme.as_met();

    let mut m = met * 58.2;
    let w = wme * 58.2;
    let k_clo = 0.25;
    let temp_body_neutral = 36.49;
    let skin_blood_flow_neutral = 6.3;
    let sbc = 5.6697e-8;
    let sa = sqrt((height_cm * weight_kg) / 3600.0);

    let v = if v > 0.1 { v } else { 0.1 };
    let mut t_skin = state.t_skin;
    let mut t_core = t_core_prescribed;
    let rmm = m;

    let mut e_skin = state.e_skin;
    let mut alfa = state.alfa;
    let mut skin_blood_flow = state.skin_blood_flow;
    let mut met_shivering = state.met_shivering;

    let mut e_rsw = 0.0;
    let mut e_diff = 0.0;
    let mut e_max = 0.0;
    let mut dry = 0.0;
    let mut r_ea = 0.0;
    let mut r_ecl = 0.0;
    let mut p_wet = 0.0;
    let mut t_body = 0.0;

    let pressure_in_atmospheres = options.p_atm.as_pascals() / 101325.0;
    let r_clo = 0.155 * clo;
    // This is where the quilt enters the model, and where the previous implementation
    // computed the factor and then discarded it.
    let f_a_cl = 0.0308 * thickness + 0.7695;
    let lr = 2.2 / pressure_in_atmospheres;
    let pa = rh * svp_torr(tdb) / 100.0;

    let (w_max, i_cl) = if clo <= 0.0 {
        (0.38 * pow(v, -0.29), 1.0)
    } else {
        (0.59 * pow(v, -0.08), 0.45)
    };

    let temp_diff = abs(t_skin - tdb);
    let chc = 4.4854 * pow(temp_diff, 0.2740) * pow(v, 0.1242);
    let mut h_r = 3.235;
    let mut ctc = h_r + chc;
    let mut r_a = 1.0 / (f_a_cl * ctc);
    let mut t_op = (h_r * tr + chc * tdb) / ctc;
    let mut t_cl = t_op + (t_skin - t_op) / (ctc * (r_a + r_clo));
    let mut t_cl_old = t_cl;
    let mut flag = false;

    for _ in 0..options.ltime {
        if flag {
            t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo);
            if abs(t_cl - t_cl_old) > 0.01 {
                flag = false;
                t_cl_old = t_cl;
            } else {
                flag = true;
            }
        }

        let max_iter = 100;
        let mut iter_cnt = 0;
        while !flag && iter_cnt < max_iter {
            h_r = 4.0 * sbc * pow((t_cl + tr) / 2.0 + 273.15, 3.0) * 0.72;
            ctc = h_r + chc;
            r_a = 1.0 / (f_a_cl * ctc);
            t_op = (h_r * tr + chc * tdb) / ctc;
            t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo);
            if abs(t_cl - t_cl_old) > 0.01 {
                flag = false;
                t_cl_old = t_cl;
            } else {
                flag = true;
            }
            iter_cnt += 1;
        }

        dry = (t_skin - t_op) / (r_a + r_clo);
        let hf_cs = (t_core - t_skin) * (5.28 + 1.163 * skin_blood_flow);
        let q_res = 0.0023 * m * (44.0 - pa);
        let c_res = 0.0014 * m * (34.0 - tdb);
        let s_core = m - hf_cs - q_res - c_res - w;
        let s_skin = hf_cs - dry - e_skin;
        let tc_sk = 0.97 * alfa * weight_kg;
        let tc_cr = 0.97 * (1.0 - alfa) * weight_kg;
        let d_t_sk = (s_skin * sa) / tc_sk / 60.0;
        let d_t_cr = (s_core * sa) / tc_cr / 60.0;
        t_skin += d_t_sk;
        t_core += d_t_cr;
        t_body = alfa * t_skin + (1.0 - alfa) * t_core;

        let warm_sk = (t_skin - 33.7).max(0.0);
        let cold_s = (33.7 - t_skin).max(0.0);
        // Measured against the *prescribed* core temperature for this minute, not against
        // a fixed neutral value: upstream's driver passes the Yan quadratic in as
        // `temp_core_neutral`, so the reference moves with the trajectory. Using a fixed
        // 36.8 here leaves minute 0 correct — nothing in a single minute reads the updated
        // blood flow — and corrupts every minute after it.
        let warm_c = (t_core - t_core_prescribed).max(0.0);
        let cold_c = (t_core_prescribed - t_core).max(0.0);
        let warm_b = (t_body - temp_body_neutral).max(0.0);

        skin_blood_flow = (skin_blood_flow_neutral + options.c_dil * warm_c)
            / (1.0 + options.c_str * cold_s);
        skin_blood_flow = skin_blood_flow.max(0.5).min(90.0);
        let mut reg_sw = options.c_sw * warm_b * exp(warm_sk / 10.7);
        reg_sw = reg_sw.min(500.0);
        e_rsw = 0.68 * reg_sw;
        r_ea = 1.0 / (lr * f_a_cl * chc);
        r_ecl = r_clo / (lr * i_cl);
        e_max = (svp_torr(t_skin) - pa) / (r_ea + r_ecl);
        let mut p_rsw = e_rsw / e_max;
        p_wet = 0.06 + 0.94 * p_rsw;
        e_diff = p_wet * e_max - e_rsw;
        if p_wet > w_max {
            p_wet = w_max;
            p_rsw = w_max / 0.94;
            e_rsw = p_rsw * e_max;
            e_diff = 0.06 * (1.0 - p_rsw) * e_max;
        }
        e_skin = e_rsw + e_diff;
        met_shivering = 19.4 * cold_s * cold_c;
        m = rmm + met_shivering;
        alfa = 0.0417737 + 0.7451833 / (skin_blood_flow + 0.585417);
    }

    let q_skin = dry + e_skin;
    let rn = m - w;
    let e_comfort = (0.42 * (rn - 58.2)).max(0.0);
    e_max *= w_max;
    let h_d = 1.0 / (r_a + r_clo);
    let h_e = 1.0 / (r_ea + r_ecl);
    let wet = p_wet;
    let p_s_sk = svp_torr(t_skin);

    // Standard-environment coefficients for the SET solve
    let chrs = h_r;
    let chcs = if met >= 0.85 {
        (5.66 * pow(met - 0.85, 0.39)).max(3.0)
    } else {
        3.0
    };
    let ctcs = chcs + chrs;
    let r_clo_s = 1.52 / ((met - wme) + 0.6944) - 0.1835;
    let r_cl_s = 0.155 * r_clo_s;
    let f_a_cl_s = 1.0 + k_clo * r_clo_s;
    let f_cl_s = 1.0 / (1.0 + 0.155 * f_a_cl_s * ctcs * r_clo_s);
    let i_m_s = 0.45;
    let i_cl_s = i_m_s * chcs / ctcs * (1.0 - f_cl_s) / (chcs / ctcs - f_cl_s * i_m_s);
    let r_a_s = 1.0 / (f_a_cl_s * ctcs);
    let r_ea_s = 1.0 / (lr * f_a_cl_s * chcs);
    let r_ecl_s = r_cl_s / (lr * i_cl_s);
    let h_d_s = 1.0 / (r_a_s + r_cl_s);
    let h_e_s = 1.0 / (r_ea_s + r_ecl_s);
    let delta = 1e-4;

    // Two secant solves. Upstream's loops are unbounded except for a small-denominator
    // guard on the first; here both are capped, because a non-converging solve would hang
    // rather than raise the way Python would.
    let solve = |h_d: f64, h_e: f64| -> f64 {
        let mut xold = t_skin - q_skin / h_d;
        let mut x = xold;
        for _ in 0..MAX_SECANT_ITERATIONS {
            let err1 = set_error(xold, q_skin, h_d, t_skin, wet, h_e, p_s_sk);
            let err2 = set_error(xold + delta, q_skin, h_d, t_skin, wet, h_e, p_s_sk);
            let err_diff = err2 - err1;
            if abs(err_diff) < 1e-10 {
                break;
            }
            x = xold - delta * err1 / err_diff;
            if abs(x - xold) > 0.01 {
                xold = x;
            } else {
                return x;
            }
        }
        x
    };

    let _ = solve(h_d, h_e);
    let set_temp = solve(h_d_s, h_e_s);

    let tbm_l = (0.194 / 58.15) * rn + 36.301;
    let tbm_h = (0.347 / 58.15) * rn + 36.669;
    let t_sens = if t_body < tbm_l {
        0.4685 * (t_body - tbm_l)
    } else if t_body < tbm_h {
        w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l)
    } else {
        w_max * 4.7 + 0.4685 * (t_body - tbm_h)
    };

    let mut disc = 4.7 * (e_rsw - e_comfort) / (e_max - e_comfort - e_diff);
    if disc < 0.0 {
        disc = t_sens;
    }

    (
        SleepMinute {
            set: set_temp,
            t_core,
            t_skin,
            wet,
            t_sens,
            disc,
            e_skin,
            met_shivering,
            alfa,
            skin_blood_flow,
        },
        SleepState {
            t_skin,
            e_skin,
            alfa,
            skin_blood_flow,
            met_shivering,
        },
    )
}

/// Iteration cap for the SET secant solves.
const MAX_SECANT_ITERATIONS: usize = 150;

/// One minute of results, before they are transposed into the per-field vectors upstream
/// returns.
struct SleepMinute {
    set: f64,
    t_core: f64,
    t_skin: f64,
    wet: f64,
    t_sens: f64,
    disc: f64,
    e_skin: f64,
    met_shivering: f64,
    alfa: f64,
    skin_blood_flow: f64,
}

/// Two-node Gagge model adapted for the sleep thermal environment.
///
/// Simulates a night minute by minute. Metabolic rate follows the fifth-order polynomial
/// of Yan et al. (2022) and core temperature is prescribed by their quadratic, both in
/// elapsed hours; skin temperature, evaporative loss, blood flow, shivering and `alfa`
/// carry over between minutes.
///
/// # Errors
///
/// Returns [`MismatchedScheduleLengths`] unless all six schedules are the same length,
/// since that length is the duration and there is no meaningful way to reconcile a
/// disagreement.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::two_nodes_gagge_sleep::{
///     two_nodes_gagge_sleep, SleepInputs,
/// };
/// use thermalcomfort::ClothingInsulation;
/// use measurements::{Humidity, Length, Speed, Temperature};
///
/// // A flat two-minute night
/// let tdb = [Temperature::from_celsius(25.0); 2];
/// let tr = [Temperature::from_celsius(25.0); 2];
/// let v = [Speed::from_meters_per_second(0.1); 2];
/// let rh = [Humidity::from_percent(50.0); 2];
/// let clo = [ClothingInsulation::from_clo(0.5); 2];
/// let quilt = [Length::from_centimeters(1.0); 2];
///
/// let result = two_nodes_gagge_sleep(
///     SleepInputs { tdb: &tdb, tr: &tr, v: &v, rh: &rh, clo: &clo, thickness_quilt: &quilt },
///     Default::default(),
/// )
/// .expect("all six schedules are two minutes long");
///
/// assert_eq!(result.set.len(), 2);
/// ```
///
/// # References
///
/// - Yan, S., Xiong, J., Kim, J. and de Dear, R. (2022)
pub fn two_nodes_gagge_sleep(
    inputs: SleepInputs<'_>,
    options: GaggeTwoNodesSleepOptions,
) -> Result<GaggeTwoNodesSleepResult, MismatchedScheduleLengths> {
    let lengths = [
        inputs.tdb.len(),
        inputs.tr.len(),
        inputs.v.len(),
        inputs.rh.len(),
        inputs.clo.len(),
        inputs.thickness_quilt.len(),
    ];
    if lengths.iter().any(|&l| l != lengths[0]) {
        return Err(MismatchedScheduleLengths { lengths });
    }
    let duration = lengths[0];

    let mut state = SleepState {
        t_skin: options.temp_skin_neutral.as_celsius(),
        e_skin: options.e_skin.as_watts_per_square_meter(),
        alfa: options.alfa,
        skin_blood_flow: options.skin_blood_flow,
        met_shivering: options.met_shivering.as_watts_per_square_meter(),
    };

    let mut result = GaggeTwoNodesSleepResult {
        set: Vec::with_capacity(duration),
        t_core: Vec::with_capacity(duration),
        t_skin: Vec::with_capacity(duration),
        wet: Vec::with_capacity(duration),
        t_sens: Vec::with_capacity(duration),
        disc: Vec::with_capacity(duration),
        e_skin: Vec::with_capacity(duration),
        met_shivering: Vec::with_capacity(duration),
        alfa: Vec::with_capacity(duration),
        skin_blood_flow: Vec::with_capacity(duration),
    };

    for i in 0..duration {
        // Elapsed hours. Upstream indexes from zero but offsets by one minute, so the
        // first sample sits slightly before the start of the night; kept as-is because
        // the polynomial coefficients were fitted against exactly this.
        let x = (i as f64 - 1.0) / 60.0;

        let met = -0.000000000000575 * pow(x, 5.0)
            + 0.000000000785521 * pow(x, 4.0)
            - 0.00000039173563 * pow(x, 3.0)
            + 0.000087620232151 * pow(x, 2.0)
            - 0.008801558913211 * x
            + 1.09952538864493;

        let t_core = 0.022234 * pow(x, 2.0) - 0.27677 * x + 37.02;

        let (minute, next) = sleep_set(
            inputs.tdb[i].as_celsius(),
            inputs.tr[i].as_celsius(),
            inputs.v[i].as_meters_per_second(),
            inputs.rh[i].as_percent(),
            inputs.clo[i].as_clo(),
            inputs.thickness_quilt[i].as_centimeters(),
            met,
            t_core,
            &state,
            &options,
        );
        state = next;

        result.set.push(Temperature::from_celsius(minute.set));
        result.t_core.push(Temperature::from_celsius(minute.t_core));
        result.t_skin.push(Temperature::from_celsius(minute.t_skin));
        result.wet.push(minute.wet);
        result.t_sens.push(minute.t_sens);
        result.disc.push(minute.disc);
        result
            .e_skin
            .push(HeatFluxDensity::from_watts_per_square_meter(minute.e_skin));
        result
            .met_shivering
            .push(HeatFluxDensity::from_watts_per_square_meter(
                minute.met_shivering,
            ));
        result.alfa.push(minute.alfa);
        result.skin_blood_flow.push(minute.skin_blood_flow);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat `n`-minute schedule at one condition.
    fn flat(n: usize, thickness_cm: f64) -> ([Temperature; 3], [Length; 3]) {
        assert_eq!(n, 3);
        (
            [Temperature::from_celsius(25.0); 3],
            [Length::from_centimeters(thickness_cm); 3],
        )
    }

    fn run(thickness_cm: f64) -> GaggeTwoNodesSleepResult {
        let (temps, quilt) = flat(3, thickness_cm);
        two_nodes_gagge_sleep(
            SleepInputs {
                tdb: &temps,
                tr: &temps,
                v: &[Speed::from_meters_per_second(0.1); 3],
                rh: &[Humidity::from_percent(50.0); 3],
                clo: &[ClothingInsulation::from_clo(0.5); 3],
                thickness_quilt: &quilt,
            },
            Default::default(),
        )
        .expect("all schedules are three minutes long")
    }

    fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < 1e-9,
                "{what}[{i}]: got {a}, pythermalcomfort gives {e}"
            );
        }
    }

    /// Values from pythermalcomfort 4.4.0 at tdb=tr=25, v=0.1, rh=50, clo=0.5.
    #[test]
    fn matches_pythermalcomfort_with_a_thin_quilt() {
        let r = run(1.0);
        let set: alloc::vec::Vec<f64> = r.set.iter().map(|t| t.as_celsius()).collect();
        assert_close(
            &set,
            &[23.050663767883, 23.492344495693, 23.697580886555],
            "set",
        );
        let t_skin: alloc::vec::Vec<f64> = r.t_skin.iter().map(|t| t.as_celsius()).collect();
        assert_close(
            &t_skin,
            &[33.688565315028, 33.587690218900, 33.539457174202],
            "t_skin",
        );
        let e_skin: alloc::vec::Vec<f64> = r
            .e_skin
            .iter()
            .map(|q| q.as_watts_per_square_meter())
            .collect();
        assert_close(
            &e_skin,
            &[32.215267771310, 16.634921939684, 12.873788294211],
            "e_skin",
        );
        assert_close(
            &r.disc,
            &[0.992594988518, 0.241281532790, 0.057817877469],
            "disc",
        );
    }

    /// The regression that motivated the rewrite: the previous implementation computed the
    /// bedding area factor and discarded it, so a 9 cm quilt gave the same answer as a
    /// 1 cm one. It must now shift SET by more than two degrees.
    #[test]
    fn a_thick_quilt_changes_the_answer() {
        let r = run(9.0);
        let set: alloc::vec::Vec<f64> = r.set.iter().map(|t| t.as_celsius()).collect();
        assert_close(
            &set,
            &[20.879850455180, 21.638055616336, 21.979868417769],
            "set",
        );
        assert_close(
            &r.disc,
            &[0.851824786603, 0.149980354571, -0.006139024967],
            "disc",
        );

        let thin = run(1.0);
        assert!(
            thin.set[0].as_celsius() - r.set[0].as_celsius() > 2.0,
            "a thicker quilt must lower SET; got {} vs {}",
            r.set[0].as_celsius(),
            thin.set[0].as_celsius()
        );
        // disc sign-flips in the third minute under the thick quilt; the old code could
        // not produce a negative value here at all.
        assert!(r.disc[2] < 0.0);
    }

    /// Relocated from `two_nodes_gagge::tests::test_two_nodes_gagge_sleep`, which tested
    /// the pre-rewrite single-value signature. Same assertions, applied to every minute.
    #[test]
    fn sleep_conditions_produce_plausible_physiology() {
        let r = run(0.1);
        for (i, set) in r.set.iter().enumerate() {
            assert!(
                set.as_celsius() > 20.0 && set.as_celsius() < 30.0,
                "set[{i}] = {}",
                set.as_celsius()
            );
        }
        for (i, t) in r.t_core.iter().enumerate() {
            assert!(
                t.as_celsius() > 35.0 && t.as_celsius() < 40.0,
                "t_core[{i}] = {}",
                t.as_celsius()
            );
        }
        for (i, t) in r.t_skin.iter().enumerate() {
            assert!(
                t.as_celsius() > 30.0 && t.as_celsius() < 40.0,
                "t_skin[{i}] = {}",
                t.as_celsius()
            );
        }
    }

    #[test]
    fn schedules_of_different_lengths_are_rejected() {
        let two = [Temperature::from_celsius(25.0); 2];
        let three = [Temperature::from_celsius(25.0); 3];
        let err = two_nodes_gagge_sleep(
            SleepInputs {
                tdb: &three,
                tr: &two,
                v: &[Speed::from_meters_per_second(0.1); 3],
                rh: &[Humidity::from_percent(50.0); 3],
                clo: &[ClothingInsulation::from_clo(0.5); 3],
                thickness_quilt: &[Length::from_centimeters(1.0); 3],
            },
            Default::default(),
        )
        .expect_err("tr is shorter than the rest, so the duration is ambiguous");
        assert_eq!(err.lengths, [3, 2, 3, 3, 3, 3]);
    }
}
