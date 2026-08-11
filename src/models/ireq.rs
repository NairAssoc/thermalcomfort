//! Required Clothing Insulation (IREQ) and Duration Limited Exposure (DLE)
//!
//! Implements the ISO 11079 cold-stress model. The model estimates the clothing
//! insulation required to maintain thermal equilibrium in cold environments, and the
//! Duration Limited Exposure when the clothing actually available is insufficient.
//!
//! Two physiological criteria are evaluated:
//!
//! - **minimal** — the lowest insulation tolerable, accepting some body cooling
//! - **neutral** — the insulation giving thermal neutrality
//!
//! # References
//!
//! - ISO 11079:2007 - Ergonomics of the thermal environment

use crate::constants::MET_TO_W_M2;
use crate::utilities::{np_minimum, round_to};
use crate::{AirPermeability, ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature};
use libm::{exp, fabs, log, pow};

/// Conversion between clo and m²·K/W
const CLO_TO_M2K_W: f64 = 0.155;
/// Effective radiating area of the body relative to DuBois area
const AR_ADU: f64 = 0.77;
/// Stefan-Boltzmann constant [W/(m²·K⁴)]
const SIGMA: f64 = 5.67e-8;
/// Emissivity of clothing
const EMISSIVITY: f64 = 0.95;
/// Maximum iterations for each bisection solve
const MAX_ITER: usize = 150;
/// Convergence tolerance on the heat balance [W/m²]
const BALANCE_TOL: f64 = 0.01;
/// Body heat storage used to define the exposure limit [W·h/m²]
const STORAGE_LIMIT: f64 = -40.0;
/// Ceiling above which exposure is reported as unlimited [h]
const DLE_CEILING_H: f64 = 8.0;

/// Duration Limited Exposure.
///
/// Mirrors pythermalcomfort's mixed-type `dle` field, which is either a number of hours
/// or the string `"more than 8"`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DurationLimitedExposure {
    /// Exposure is limited to this many hours
    Hours(f64),
    /// Exposure exceeds the 8 hour reporting ceiling (Python: `"more than 8"`)
    MoreThanEight,
    /// Inputs fell outside the applicability limits, or the result was non-physical
    /// (Python: `nan`)
    NotApplicable,
}

impl core::fmt::Display for DurationLimitedExposure {
    /// Renders the same way pythermalcomfort's mixed-type `dle` field prints.
    ///
    /// Replaces an `as_str` whose contract was inverted: it returned `None` for
    /// `Hours` (a real value) and `Some("nan")` for `NotApplicable` (the absent case).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DurationLimitedExposure::Hours(h) => write!(f, "{h}"),
            DurationLimitedExposure::MoreThanEight => f.write_str("more than 8"),
            DurationLimitedExposure::NotApplicable => f.write_str("nan"),
        }
    }
}

/// The comfort inputs to [`ireq`]: pythermalcomfort requires all eight (no default).
///
/// `tdb`/`tr` and `vr`/`walk_sp` are each a pair of adjacent same-typed values; naming
/// every field forecloses a silent transposition within either pair.
///
/// Deliberately has no `Default`: every field must be given explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IreqInputs {
    /// Dry bulb air temperature
    pub tdb: Temperature,
    /// Mean radiant temperature
    pub tr: Temperature,
    /// Relative air speed
    pub vr: Speed,
    /// Relative humidity
    pub rh: Humidity,
    /// Metabolic rate
    pub met: MetabolicRate,
    /// Clothing insulation actually available
    pub clo: ClothingInsulation,
    /// Air permeability of clothing
    pub p: AirPermeability,
    /// Walking speed
    pub walk_sp: Speed,
}

/// Optional parameters for [`ireq`]
#[derive(Debug, Clone, Copy)]
pub struct IreqOptions {
    /// External work, default 0 met
    pub wme: MetabolicRate,
    /// Set results outside the ISO 11079 applicability limits to NaN
    pub limit_inputs: bool,
    /// Round numeric outputs to one decimal place
    pub round_output: bool,
}

impl Default for IreqOptions {
    fn default() -> Self {
        Self {
            wme: MetabolicRate::from_met(0.0),
            limit_inputs: true,
            round_output: true,
        }
    }
}

/// Result of the ISO 11079 IREQ calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IreqResult {
    /// Required clothing insulation, minimal criterion
    pub ireq_min: ClothingInsulation,
    /// Required clothing insulation, neutral criterion
    pub ireq_neutral: ClothingInsulation,
    /// Intrinsic clothing insulation, minimal criterion
    pub icl_min: ClothingInsulation,
    /// Intrinsic clothing insulation, neutral criterion
    pub icl_neutral: ClothingInsulation,
    /// Duration limited exposure, minimal criterion
    pub dle_min: DurationLimitedExposure,
    /// Duration limited exposure, neutral criterion
    pub dle_neutral: DurationLimitedExposure,
}

/// Calculate Required Clothing Insulation (IREQ) and Duration Limited Exposure (DLE)
///
/// # Arguments
///
/// * `inputs` - Required comfort inputs, see [`IreqInputs`]
/// * `options` - Optional parameters, see [`IreqOptions`]
///
/// # Returns
///
/// [`IreqResult`] with required and intrinsic insulation plus exposure duration for the
/// minimal and neutral criteria.
///
/// # Applicability Limits
///
/// When `options.limit_inputs` is true, results are NaN outside:
/// - 58 <= met [W/m²] <= 400
/// - tdb <= 10 °C
/// - 0.4 <= vr [m/s] <= 18
/// - `min_walk` <= walk_sp [m/s] <= 1.2, where `min_walk = min(0.0052 * (met - 58), 1.2)`
///
/// Non-physical (negative) insulation results are also set to NaN.
///
/// # Examples
///
/// ```
/// use thermalcomfort::models::ireq::{ireq, IreqInputs, IreqOptions, DurationLimitedExposure};
/// use thermalcomfort::{
///     AirPermeability, ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature,
/// };
///
/// let result = ireq(
///     IreqInputs {
///         tdb: Temperature::from_celsius(-15.0),
///         tr: Temperature::from_celsius(-15.0),
///         vr: Speed::from_meters_per_second(2.0),
///         rh: Humidity::from_percent(55.0),
///         met: MetabolicRate::from_met(175.0 / 58.15),
///         clo: ClothingInsulation::from_clo(2.8),
///         p: AirPermeability::from_l_per_m2_s(50.0),
///         walk_sp: Speed::from_meters_per_second(1.1),
///     },
///     IreqOptions::default(),
/// );
/// assert!((result.ireq_min.as_clo() - 1.6).abs() < 0.05);
/// assert_eq!(result.dle_min, DurationLimitedExposure::MoreThanEight);
/// ```
///
/// # References
///
/// - ISO 11079:2007
pub fn ireq(inputs: IreqInputs, options: IreqOptions) -> IreqResult {
    let IreqInputs {
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        p,
        walk_sp,
    } = inputs;
    let tdb_c = tdb.as_celsius();
    let tr_c = tr.as_celsius();
    let vr_ms = vr.as_meters_per_second();
    let rh_pct = rh.as_percent();
    let walk_sp_ms = walk_sp.as_meters_per_second();
    // ISO 11079 works in W/m², not met
    let met_w = met.as_met() * MET_TO_W_M2;
    let wme_w = options.wme.as_met() * MET_TO_W_M2;
    let clo_m2c_w = clo.as_clo() * CLO_TO_M2K_W;
    let p = p.as_l_per_m2_s();

    let valid = valid_iso_11079_inputs(met_w, tdb_c, vr_ms, walk_sp_ms);

    // (skin temperature, skin wettedness) for the minimal and neutral criteria
    let criteria = [
        (33.34 - 0.0354 * met_w, 0.06),
        (35.7 - 0.0285 * met_w, 0.001 * met_w),
    ];

    let mut out = [(f64::NAN, f64::NAN, DurationLimitedExposure::NotApplicable); 2];

    for (i, &(skin_temperature, wetness)) in criteria.iter().enumerate() {
        let solved = solve_criterion(
            tdb_c,
            tr_c,
            met_w,
            wme_w,
            vr_ms,
            walk_sp_ms,
            p,
            clo_m2c_w,
            rh_pct,
            skin_temperature,
            wetness,
        );

        let mut ireq_out = solved.ireq / CLO_TO_M2K_W;
        let mut icl_out = solved.icl / CLO_TO_M2K_W;

        // Evaluated before rounding, matching the Python ordering
        let non_physical = ireq_out < 0.0 || icl_out < 0.0;

        if options.round_output {
            ireq_out = round_to(ireq_out, 1);
            icl_out = round_to(icl_out, 1);
        }

        let mut dle_out = format_dle(solved.dle, options.round_output);

        if options.limit_inputs && (non_physical || !valid) {
            ireq_out = f64::NAN;
            icl_out = f64::NAN;
            dle_out = DurationLimitedExposure::NotApplicable;
        }

        out[i] = (ireq_out, icl_out, dle_out);
    }

    IreqResult {
        ireq_min: ClothingInsulation::from_clo(out[0].0),
        ireq_neutral: ClothingInsulation::from_clo(out[1].0),
        icl_min: ClothingInsulation::from_clo(out[0].1),
        icl_neutral: ClothingInsulation::from_clo(out[1].1),
        dle_min: out[0].2,
        dle_neutral: out[1].2,
    }
}

/// Solution of the ISO 11079 Annex A balance for one physiological criterion.
struct CriterionSolution {
    /// Required clothing insulation [m²·K/W]
    ireq: f64,
    /// Intrinsic clothing insulation [m²·K/W]
    icl: f64,
    /// Duration limited exposure [h]
    dle: f64,
}

/// ISO 11079 Annex A wind/permeability correction factor, shared by the total and
/// resultant clothing insulation calculations.
fn clothing_constant_part(p: f64, vr: f64, walk_sp: f64) -> f64 {
    0.54 * exp(-0.15 * vr - 0.22 * walk_sp) * pow(p, 0.075) - 0.06 * log(p) + 0.5
}

/// Saturated vapour pressure over water [kPa] using the ISO 11079 Annex A formulation.
fn vapor_pressure(t: f64) -> f64 {
    0.1333 * exp(18.6686 - 4030.183 / (t + 235.0))
}

/// Linearised radiative heat transfer coefficient [W/(m²·K)].
///
/// Falls back to the analytic derivative when clothing and radiant temperatures coincide,
/// where the difference quotient would otherwise divide by zero.
fn radiation_coefficient(t_cl: f64, tr: f64) -> f64 {
    let delta_t = t_cl - tr;
    if fabs(delta_t) < 1e-4 {
        SIGMA * EMISSIVITY * AR_ADU * 4.0 * pow(273.0 + (t_cl + tr) / 2.0, 3.0)
    } else {
        let t_cl_k = 273.0 + t_cl;
        let tr_k = 273.0 + tr;
        SIGMA * EMISSIVITY * AR_ADU * (pow(t_cl_k, 4.0) - pow(tr_k, 4.0)) / delta_t
    }
}

/// Solve the ISO 11079 Annex A thermal balance for one physiological criterion,
/// returning IREQ and ICL in m²·K/W and DLE in hours.
#[allow(clippy::too_many_arguments)]
fn solve_criterion(
    tdb: f64,
    tr: f64,
    met: f64,
    wme: f64,
    vr: f64,
    walk_sp: f64,
    p: f64,
    clo_m2c_w: f64,
    rh: f64,
    skin_temperature: f64,
    wetness: f64,
) -> CriterionSolution {
    let air_insulation = 0.092 * exp(-0.15 * vr - 0.22 * walk_sp) - 0.0045;
    let constant_part = clothing_constant_part(p, vr, walk_sp);

    let expired_air_temperature = 29.0 + 0.2 * tdb;
    let expired_air_vapor_pressure = vapor_pressure(expired_air_temperature);
    let ambient_vapor_pressure = (rh / 100.0) * vapor_pressure(tdb);
    let skin_saturated_pressure = vapor_pressure(skin_temperature);

    // Respiratory heat loss does not depend on the insulation being solved for
    let respiratory_heat_loss =
        1.73e-02 * met * (expired_air_vapor_pressure - ambient_vapor_pressure)
            + 1.4e-03 * met * (expired_air_temperature - tdb);

    // --- Bisection on the required clothing insulation --------------------------------
    let mut ireq_clo = 0.5;
    let mut factor = 0.5;
    let mut balance = 1.0;
    let mut clothing_temperature = 0.0;
    let mut radiation_heat_loss = 0.0;
    let mut convective_heat_loss = 0.0;

    for _ in 0..MAX_ITER {
        if fabs(balance) <= BALANCE_TOL {
            break;
        }

        let fcl = 1.0 + 1.197 * ireq_clo;
        let total_evaporative_resistance = (0.06 / 0.38) * (air_insulation + ireq_clo);
        let evaporative_heat_loss = wetness * (skin_saturated_pressure - ambient_vapor_pressure)
            / total_evaporative_resistance;

        clothing_temperature = skin_temperature
            - ireq_clo * (met - wme - evaporative_heat_loss - respiratory_heat_loss);

        let h_r = radiation_coefficient(clothing_temperature, tr);
        let h_c = 1.0 / air_insulation - h_r;
        radiation_heat_loss = fcl * h_r * (clothing_temperature - tr);
        convective_heat_loss = fcl * h_c * (clothing_temperature - tdb);

        balance = met
            - wme
            - evaporative_heat_loss
            - respiratory_heat_loss
            - radiation_heat_loss
            - convective_heat_loss;

        if balance > 0.0 {
            ireq_clo -= factor;
            factor /= 2.0;
        } else {
            ireq_clo += factor;
        }
    }

    let ireq_final =
        (skin_temperature - clothing_temperature) / (radiation_heat_loss + convective_heat_loss);

    // --- Bisection on body heat storage, giving the exposure limit --------------------
    let mut storage = STORAGE_LIMIT;
    let mut storage_factor = 500.0;
    let mut resultant_clothing_insulation = clo_m2c_w;
    let mut storage_balance = 1.0;

    for _ in 0..MAX_ITER {
        if fabs(storage_balance) <= BALANCE_TOL {
            break;
        }

        // fcl uses the insulation from the previous iteration, then the insulation is
        // updated - this ordering is load-bearing and matches ISO 11079 / upstream.
        let fcl_storage = 1.0 + 1.197 * resultant_clothing_insulation;
        resultant_clothing_insulation =
            (clo_m2c_w + 0.085 / fcl_storage) * constant_part - air_insulation / fcl_storage;

        let total_evaporative_resistance =
            (0.06 / 0.38) * (air_insulation + resultant_clothing_insulation);
        let evaporative_heat_loss = wetness * (skin_saturated_pressure - ambient_vapor_pressure)
            / total_evaporative_resistance;

        let clothing_temperature_storage = skin_temperature
            - resultant_clothing_insulation
                * (met - wme - evaporative_heat_loss - respiratory_heat_loss - storage);

        let h_r = radiation_coefficient(clothing_temperature_storage, tr);
        let h_c = 1.0 / air_insulation - h_r;
        let radiation_heat_loss_storage = fcl_storage * h_r * (clothing_temperature_storage - tr);
        let convective_heat_loss_storage = fcl_storage * h_c * (clothing_temperature_storage - tdb);

        storage_balance = met
            - wme
            - evaporative_heat_loss
            - respiratory_heat_loss
            - radiation_heat_loss_storage
            - convective_heat_loss_storage
            - storage;

        if storage_balance > 0.0 {
            storage += storage_factor;
            storage_factor /= 2.0;
        } else {
            storage -= storage_factor;
        }
    }

    let dle = if storage == 0.0 {
        f64::INFINITY
    } else {
        STORAGE_LIMIT / storage
    };

    let fcl_final = 1.0 + 1.197 * ireq_final;
    let icl_raw = (ireq_final + air_insulation / fcl_final) / constant_part - (0.085 / fcl_final);

    CriterionSolution {
        ireq: ireq_final,
        icl: icl_raw,
        dle,
    }
}

/// Boolean mask of inputs within the ISO 11079 applicability limits.
fn valid_iso_11079_inputs(met: f64, tdb: f64, vr: f64, walk_sp: f64) -> bool {
    // `ireq.py:395` is `np.minimum(0.0052 * (met - 58.0), 1.2)`, which propagates NaN;
    // `f64::min` would return 1.2 instead. Only the `met` range check below currently
    // keeps a NaN `met` from reaching a comparison against a healed 1.2, and relying on
    // the order of the `&&` chain for that is exactly the fragility this replaces.
    let minimum_walking_speed = np_minimum(0.0052 * (met - 58.0), 1.2);

    (58.0..=400.0).contains(&met)
        && tdb <= 10.0
        && (0.4..=18.0).contains(&vr)
        && walk_sp >= minimum_walking_speed
        && walk_sp <= 1.2
}

/// Convert raw exposure hours to the ISO 11079 reporting format, replacing values above
/// the 8 h ceiling (and negative values, which indicate no limit) with "more than 8".
fn format_dle(dle: f64, round_output: bool) -> DurationLimitedExposure {
    if !(0.0..=DLE_CEILING_H).contains(&dle) {
        DurationLimitedExposure::MoreThanEight
    } else if round_output {
        DurationLimitedExposure::Hours(round_to(dle, 1))
    } else {
        DurationLimitedExposure::Hours(dle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_inputs(
        tdb: f64,
        tr: f64,
        vr: f64,
        rh: f64,
        met: f64,
        clo: f64,
        p: f64,
        walk_sp: f64,
    ) -> IreqInputs {
        IreqInputs {
            tdb: Temperature::from_celsius(tdb),
            tr: Temperature::from_celsius(tr),
            vr: Speed::from_meters_per_second(vr),
            rh: Humidity::from_percent(rh),
            met: MetabolicRate::from_met(met),
            clo: ClothingInsulation::from_clo(clo),
            p: AirPermeability::from_l_per_m2_s(p),
            walk_sp: Speed::from_meters_per_second(walk_sp),
        }
    }

    fn default_case(tdb: f64, clo_val: f64) -> IreqResult {
        ireq(
            make_inputs(tdb, tdb, 2.0, 55.0, 175.0 / 58.15, clo_val, 50.0, 1.1),
            IreqOptions::default(),
        )
    }

    #[test]
    fn test_ireq_reference_case() {
        // Reference values from pythermalcomfort 4.4.0
        let result = default_case(-15.0, 2.8);
        assert!(
            (result.ireq_min.as_clo() - 1.6).abs() < 0.05,
            "ireq_min = {}",
            result.ireq_min.as_clo()
        );
        assert_eq!(result.dle_min, DurationLimitedExposure::MoreThanEight);
    }

    #[test]
    fn test_ireq_matches_python_reference_values() {
        // Reference values generated with pythermalcomfort 4.4.0.
        // (tdb, tr, vr, rh, met, clo, p, walk_sp) -> (ireq_min, ireq_neutral, icl_min, icl_neutral)
        #[allow(clippy::type_complexity)]
        let cases: [(
            (f64, f64, f64, f64, f64, f64, f64, f64),
            (f64, f64, f64, f64),
        ); 5] = [
            (
                (-15.0, -15.0, 2.0, 55.0, 175.0 / 58.15, 2.8, 50.0, 1.1),
                (1.6, 1.9, 2.2, 2.6),
            ),
            (
                (-5.0, -5.0, 0.5, 80.0, 2.0, 1.5, 50.0, 0.5),
                (1.9, 2.2, 2.1, 2.5),
            ),
            (
                (-30.0, -30.0, 5.0, 40.0, 3.0, 4.0, 100.0, 1.2),
                (2.3, 2.6, 4.5, 5.1),
            ),
            (
                (0.0, 0.0, 1.0, 60.0, 1.5, 2.0, 20.0, 0.3),
                (2.3, 2.6, 2.6, 3.0),
            ),
            (
                (10.0, 10.0, 0.4, 50.0, 2.5, 1.0, 50.0, 0.6),
                (0.5, 0.8, 0.6, 1.0),
            ),
        ];

        for ((tdb, tr, vr, rh, met, clo, p, walk_sp), expected) in cases {
            let result = ireq(
                make_inputs(tdb, tr, vr, rh, met, clo, p, walk_sp),
                IreqOptions::default(),
            );
            let got = (
                result.ireq_min.as_clo(),
                result.ireq_neutral.as_clo(),
                result.icl_min.as_clo(),
                result.icl_neutral.as_clo(),
            );
            for (label, g, e) in [
                ("ireq_min", got.0, expected.0),
                ("ireq_neutral", got.1, expected.1),
                ("icl_min", got.2, expected.2),
                ("icl_neutral", got.3, expected.3),
            ] {
                assert!(
                    (g - e).abs() < 0.05,
                    "tdb={tdb} vr={vr}: {label} = {g}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn test_ireq_dle_reporting() {
        // Reference values from pythermalcomfort 4.4.0
        let unlimited = default_case(-15.0, 2.8);
        assert_eq!(unlimited.dle_min, DurationLimitedExposure::MoreThanEight);
        assert_eq!(
            unlimited.dle_neutral,
            DurationLimitedExposure::MoreThanEight
        );

        let limited = ireq(
            make_inputs(-5.0, -5.0, 0.5, 80.0, 2.0, 1.5, 50.0, 0.5),
            IreqOptions::default(),
        );
        assert_eq!(limited.dle_min, DurationLimitedExposure::Hours(1.2));
        assert_eq!(limited.dle_neutral, DurationLimitedExposure::Hours(0.8));
    }

    #[test]
    fn test_ireq_outside_applicability_is_nan() {
        // tdb above the 10 °C ISO 11079 limit
        let result = default_case(20.0, 2.8);
        assert!(result.ireq_min.as_clo().is_nan());
        assert!(result.ireq_neutral.as_clo().is_nan());
        assert!(result.icl_min.as_clo().is_nan());
        assert_eq!(result.dle_min, DurationLimitedExposure::NotApplicable);
    }

    #[test]
    fn test_ireq_limits_can_be_disabled() {
        let result = ireq(
            make_inputs(20.0, 20.0, 2.0, 55.0, 175.0 / 58.15, 2.8, 50.0, 1.1),
            IreqOptions {
                limit_inputs: false,
                ..Default::default()
            },
        );
        assert!(!result.ireq_min.as_clo().is_nan());
    }
}
