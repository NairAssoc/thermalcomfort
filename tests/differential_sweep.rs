//! Randomised differential sweep against pythermalcomfort.
//!
//! Every model is driven through `SWEEP_N` pseudo-random input vectors covering its
//! optional parameters as well as its physical inputs, and every output field is
//! compared against Python. This exists because the hand-written parity tests leave 49
//! of 103 optional model parameters pinned at their defaults, and three real bugs were
//! found in that residue on 2026-08-09.
//!
//! Reproduce a failure with the command printed in the panic message.

mod support;

use core::time::Duration;
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAnyMethods, PyBool, PyDict, PyModule, PyTuple};
use support::compare::{FieldCmp, NanPolicy, compare_field};
use support::domain::{Domain, Sample};
use support::sweep::{import_reference, run_sweep};
use thermalcomfort::models::jos3::{Jos3Builder, Jos3Results, PerBodyPart};
use thermalcomfort::models::pmv::{
    Iso7730Model, PmvAInputs, PmvAOptions, PmvAthbInputs, PmvAthbOptions, PmvEInputs, PmvEOptions,
    PmvPpdAshraeOptions, PmvPpdInputs, PmvPpdIsoOptions,
};
use thermalcomfort::models::specialty::{
    AnkleDraftInputs, AnkleDraftOptions, FSvvInputs, VerticalTmpGradPpdInputs,
    VerticalTmpGradPpdOptions,
};
use thermalcomfort::models::two_nodes_gagge::{
    GaggeTwoNodesInputs, GaggeTwoNodesJiInputs, GaggeTwoNodesJiOptions, two_nodes_gagge_ji,
};
use thermalcomfort::models::{
    AdaptiveInputs, AdaptiveOptions, AtInputs, AtOptions, CoolingEffectInputs,
    CoolingEffectOptions, DiscomfortIndexInputs, DurationLimitedExposure, EsiInputs, EsiOptions,
    GaggeTwoNodesOptions, GaggeTwoNodesSleepOptions, HeatIndexLuInputs, HeatIndexLuOptions,
    HeatIndexRothfuszInputs, HeatIndexRothfuszOptions, HeatIndexSchoenInputs,
    HeatIndexSchoenOptions, HumidexInputs, HumidexModel, HumidexOptions, IreqInputs, IreqOptions,
    Iso7933Model, NetInputs, NetOptions, PetInputs, PetOptions, PetPosture, PhsInputs, PhsOptions,
    PhsPosture, RidgeRegressionInputs, RidgeRegressionOptions, SetInputs, SetOptions, SleepInputs,
    SolarGainInputs, SolarGainOptions, Sports, SportsHeatStressRiskInputs, SportsValues, ThiInputs,
    ThiOptions, UseFansHeatwavesInputs, UseFansHeatwavesOptions, UtciInputs, UtciOptions,
    WbgtInputs, WbgtOptions, WciInputs, WciOptions, WindChillTemperatureInputs,
    WindChillTemperatureOptions, WorkCapacityIntensityOptions, WorkIntensity, adaptive_ashrae,
    adaptive_en, ankle_draft, at, cooling_effect, discomfort_index, esi, heat_index_lu,
    heat_index_rothfusz, heat_index_schoen, humidex, ireq, net, pet_steady, phs, pmv_a, pmv_athb,
    pmv_e, pmv_ppd_ashrae, pmv_ppd_iso, ridge_regression_predict_t_re_t_sk, set_tmp, solar_gain,
    sports_heat_stress_risk, thi, two_nodes_gagge, two_nodes_gagge_sleep, use_fans_heatwaves, utci,
    vertical_tmp_grad_ppd, wbgt, wci, wind_chill_temperature, work_capacity_dunne,
    work_capacity_hothaps, work_capacity_iso, work_capacity_niosh,
};
use thermalcomfort::models::{f_svv, transpose_sharp_altitude};
use thermalcomfort::psychrometrics::{
    MeanRadiantTemperatureInputs, MeanRadiantTemperatureOptions, OperativeTemperatureInputs,
    OperativeTemperatureOptions, PsyTaRhInputs, PsyTaRhOptions, dew_point_temperature,
    enthalpy_air, mean_radiant_temperature, operative_temperature, psy_ta_rh, wet_bulb_temperature,
};
use thermalcomfort::utilities::{
    Ashrae55Model, BodySurfaceAreaInputs, BodySurfaceAreaOptions, BsaFormula,
    CloCorrectionFactorEnvironmentInputs, CloDynamicAshraeInputs, CloDynamicAshraeOptions,
    CloDynamicIsoInputs, CloDynamicIsoOptions, CloInsulationAirLayerInputs,
    CloTotalInsulationInputs, Iso9920Model, Posture, RunningMeanOutdoorTemperatureOptions, Units,
    antoine, body_surface_area, clo_area_factor, clo_correction_factor_environment,
    clo_dynamic_ashrae, clo_dynamic_iso, clo_individual_garment, clo_insulation_air_layer,
    clo_intrinsic_insulation_ensemble, clo_total_insulation, clo_tout, clo_typical_ensemble,
    hr_to_rh, p_sat, p_sat_antoine, p_sat_torr, running_mean_outdoor_temperature, v_relative,
};
use thermalcomfort::{
    ActivityRatio, AirPermeability, Angle, Area, BmrEquation, BodyFat, CardiacIndex,
    ClothingInsulation, HeatFluxDensity, Humidity, Length, Mass, MetabolicRate, Power, Pressure,
    Sex, Speed, Temperature, TemperatureDelta, WorkEfficiency,
};

/// Read a numeric field from a Python result, tolerating the 0-d numpy arrays the 4.x
/// `NumericInput` retyping produces.
fn py_float(obj: &Bound<'_, PyAny>, field: &str) -> Result<f64, String> {
    let attr = obj
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python result: {e}"))?;
    if let Ok(v) = attr.extract::<f64>() {
        return Ok(v);
    }
    attr.call_method0("item")
        .and_then(|i| i.extract::<f64>())
        .map_err(|e| format!("{field}: could not read as a number: {e}"))
}

/// The PMV/PPD input space shared by the ISO and ASHRAE variants.
fn pmv_domain() -> Domain {
    Domain::new()
        .real("tdb", 5.0, 45.0)
        .real("tr", 5.0, 45.0)
        .real("vr", 0.0, 2.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.6, 5.0)
        .real("clo", 0.0, 2.5)
        .real("wme", 0.0, 1.0)
        .flag("limit_inputs")
        .flag("round_output")
}

#[test]
fn sweep_pmv_ppd_iso() {
    // `model` selects between "7730-2005" and "7730-2025"; both are formula-identical
    // (see `Iso7730Model`), but the axis is swept so a future divergence would be
    // caught rather than silently passing because Rust never exercised it.
    let domain = pmv_domain().enumerated("model", 2);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = [FieldCmp::new("pmv", 0.01), FieldCmp::new("ppd", 0.11)];

        run_sweep("sweep_pmv_ppd_iso", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, wme) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
            );
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");
            let (model, py_model) = match s.index("model") {
                0 => (Iso7730Model::Iso77302005, "7730-2005"),
                _ => (Iso7730Model::Iso77302025, "7730-2025"),
            };

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                ("model", py_model.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("pmv_ppd_iso")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_ppd_iso(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                PmvPpdIsoOptions {
                    wme: MetabolicRate::from_met(wme),
                    model,
                    limit_inputs,
                    round_output,
                },
            );

            for (field, rust_value) in fields.iter().zip([rust.pmv, rust.ppd]) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
            }
            Ok(())
        });
    });
}

#[test]
fn sweep_pmv_ppd_ashrae() {
    // `model` has only one legal value today ("55-2023"), but is still swept for
    // symmetry with `sweep_pmv_ppd_iso` and so a future edition is exercised as soon as
    // it is added. `airspeed_control` gates ASHRAE 55's §7.2.1.2 cross-variable rules
    // (see `check_ashrae55_compliance`), previously unreachable from Rust.
    let domain = pmv_domain().enumerated("model", 1).flag("airspeed_control");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = [FieldCmp::new("pmv", 0.01), FieldCmp::new("ppd", 0.11)];

        run_sweep("sweep_pmv_ppd_ashrae", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, wme) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
            );
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");
            let airspeed_control = s.flag("airspeed_control");
            // Only one legal value today (see `Ashrae55Model`); the axis is drawn
            // anyway so this sweep keeps the same shape as `sweep_pmv_ppd_iso`.
            let _ = s.index("model");
            let (model, py_model) = (Ashrae55Model::Ashrae552023, "55-2023");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                ("model", py_model.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "airspeed_control",
                    PyBool::new(py, airspeed_control).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            // §7.2.1.2's cross-variable rules (airspeed_control=false) warn via
            // `warnings.warn` on the Python side rather than raising; nothing to catch.
            let py_result = models
                .getattr("pmv_ppd_ashrae")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_ppd_ashrae(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                PmvPpdAshraeOptions {
                    wme: MetabolicRate::from_met(wme),
                    model,
                    limit_inputs,
                    airspeed_control,
                    round_output,
                },
            );

            for (field, rust_value) in fields.iter().zip([rust.pmv, rust.ppd]) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
            }
            Ok(())
        });
    });
}

/// Field set shared by the Gagge two-node model and its variants.
///
/// Tolerances are loose enough to absorb a single rounding step (Python rounds to one
/// decimal when `round_output` is set) but no looser — a real divergence in these models
/// is typically far larger, as the 2026-08-09 `alfa` bug showed.
fn gagge_fields() -> Vec<FieldCmp> {
    vec![
        FieldCmp::new("e_skin", 0.11),
        FieldCmp::new("e_rsw", 0.11),
        FieldCmp::new("e_max", 0.11),
        FieldCmp::new("q_sensible", 0.11),
        FieldCmp::new("q_skin", 0.11),
        FieldCmp::new("q_res", 0.11),
        FieldCmp::new("t_core", 0.06),
        FieldCmp::new("t_skin", 0.06),
        FieldCmp::new("m_bl", 0.11),
        FieldCmp::new("m_rsw", 0.11),
        FieldCmp::new("w", 0.06),
        FieldCmp::new("w_max", 0.06),
        FieldCmp::new("set", 0.06),
        FieldCmp::new("et", 0.06),
        FieldCmp::new("pmv_gagge", 0.06),
        FieldCmp::new("pmv_set", 0.06),
        FieldCmp::new("disc", 0.06),
        FieldCmp::new("t_sens", 0.06),
    ]
}

#[test]
fn sweep_two_nodes_gagge() {
    // Every optional parameter is an axis. These are exactly the ones the hand-written
    // tests pin at their defaults, and where the double-rounding bug hid.
    let domain = Domain::new()
        .real("tdb", 10.0, 45.0)
        .real("tr", 10.0, 45.0)
        .real("v", 0.0, 4.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.8, 4.0)
        .real("clo", 0.0, 2.0)
        .real("wme", 0.0, 1.0)
        .real("body_surface_area", 1.5, 2.2)
        .real("p_atm", 80_000.0, 105_000.0)
        .real("max_skin_blood_flow", 40.0, 110.0)
        .real("max_sweating", 200.0, 700.0)
        .enumerated("posture", 2)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = gagge_fields();

        run_sweep("sweep_two_nodes_gagge", &domain, |s: &Sample| {
            let (tdb, tr, v, rh, met, clo, wme, bsa, p_atm, msbf, msw) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("v"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("body_surface_area"),
                s.real("p_atm"),
                s.real("max_skin_blood_flow"),
                s.real("max_sweating"),
            );
            let (posture, py_posture) = match s.index("posture") {
                0 => (Posture::Standing, "standing"),
                _ => (Posture::Sitting, "sitting"),
            };
            let round_output = s.flag("round_output");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "body_surface_area",
                    bsa.into_pyobject(py).unwrap().into_any(),
                ),
                ("p_atm", p_atm.into_pyobject(py).unwrap().into_any()),
                ("position", py_posture.into_pyobject(py).unwrap().into_any()),
                (
                    "max_skin_blood_flow",
                    msbf.into_pyobject(py).unwrap().into_any(),
                ),
                ("max_sweating", msw.into_pyobject(py).unwrap().into_any()),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("two_nodes_gagge")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = two_nodes_gagge(
                GaggeTwoNodesInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                GaggeTwoNodesOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    position: posture,
                    max_skin_blood_flow: msbf,
                    max_sweating: msw,
                    round_output,
                    ..Default::default()
                },
            );

            let rust_values = [
                rust.e_skin.as_watts_per_square_meter(),
                rust.e_rsw.as_watts_per_square_meter(),
                rust.e_max.as_watts_per_square_meter(),
                rust.q_sensible.as_watts_per_square_meter(),
                rust.q_skin.as_watts_per_square_meter(),
                rust.q_res.as_watts_per_square_meter(),
                rust.t_core.as_celsius(),
                rust.t_skin.as_celsius(),
                rust.m_bl,
                rust.m_rsw,
                rust.w,
                rust.w_max,
                rust.set.as_celsius(),
                rust.et.as_celsius(),
                rust.pmv_gagge,
                rust.pmv_set,
                rust.disc,
                rust.t_sens,
            ];

            for (field, rust_value) in fields.iter().zip(rust_values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
            }
            Ok(())
        });
    });
}

#[test]
fn sweep_set_tmp() {
    let domain = Domain::new()
        .real("tdb", 10.0, 45.0)
        .real("tr", 10.0, 45.0)
        .real("v", 0.0, 4.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.8, 4.0)
        .real("clo", 0.0, 2.0)
        .real("wme", 0.0, 1.0)
        .real("body_surface_area", 1.5, 2.2)
        .real("p_atm", 80_000.0, 105_000.0)
        .enumerated("posture", 2)
        .flag("limit_inputs")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("set", 0.06);

        run_sweep("sweep_set_tmp", &domain, |s: &Sample| {
            let (tdb, tr, v, rh, met, clo, wme, bsa, p_atm) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("v"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("body_surface_area"),
                s.real("p_atm"),
            );
            let (posture, py_posture) = match s.index("posture") {
                0 => (Posture::Standing, "standing"),
                _ => (Posture::Sitting, "sitting"),
            };
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "body_surface_area",
                    bsa.into_pyobject(py).unwrap().into_any(),
                ),
                ("p_atm", p_atm.into_pyobject(py).unwrap().into_any()),
                ("position", py_posture.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("set_tmp")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = set_tmp(
                SetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                SetOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    position: posture,
                    limit_inputs,
                    round_output,
                    calculate_ce: false,
                },
            )
            .as_celsius();

            compare_field(&field, rust, py_float(&py_result, "set")?)
        });
    });
}

#[test]
fn sweep_utci() {
    // Deliberately spans well beyond the UTCI applicability box so the limit_inputs
    // masking is exercised in both states rather than only inside the valid region.
    let domain = Domain::new()
        .real("tdb", -60.0, 60.0)
        .real("tr", -60.0, 80.0)
        .real("v", 0.0, 20.0)
        .real("rh", 0.0, 100.0)
        .flag("limit_inputs")
        .flag("round_output")
        .enumerated("units", 2);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field_si = FieldCmp::new("utci", 0.06);
        // A genuine SI->IP temperature conversion of the result (utci.py:126-130), so a
        // Celsius-scale tolerance becomes 9/5 as large once expressed in Fahrenheit.
        let field_ip = FieldCmp::new("utci", 0.06 * 9.0 / 5.0);

        run_sweep("sweep_utci", &domain, |s: &Sample| {
            let (tdb, tr, v, rh) = (s.real("tdb"), s.real("tr"), s.real("v"), s.real("rh"));
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");
            let (units, py_units) = match s.index("units") {
                0 => (Units::SI, "SI"),
                _ => (Units::IP, "IP"),
            };

            let kwargs = [
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
                ("units", py_units.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();

            // Python's `units="IP"` reinterprets its raw tdb/tr/v floats as
            // Fahrenheit/fps and converts them to SI before doing anything else (see
            // `sweep_cooling_effect`, which solved this the same way): feed it the IP
            // equivalents of the same SI environment Rust uses.
            let (py_tdb, py_tr, py_v) = if units == Units::IP {
                (tdb * 9.0 / 5.0 + 32.0, tr * 9.0 / 5.0 + 32.0, v * 3.281)
            } else {
                (tdb, tr, v)
            };

            let py_result = models
                .getattr("utci")
                .unwrap()
                .call((py_tdb, py_tr, py_v, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = utci(
                UtciInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                },
                UtciOptions {
                    units,
                    limit_inputs,
                    round_output,
                },
            );

            let field = match units {
                Units::SI => &field_si,
                Units::IP => &field_ip,
            };
            compare_field(field, rust.utci, py_float(&py_result, "utci")?)
        });
    });
}

#[test]
fn sweep_cooling_effect() {
    // vr below the still-air threshold short-circuits to 0, so the range spans both
    // sides of it. The objective is non-monotonic in places, which is why the port
    // needs scipy's exact brentq rather than merely *a* correct root-finder.
    let domain = Domain::new()
        .real("tdb", 15.0, 40.0)
        .real("tr", 15.0, 40.0)
        .real("vr", 0.0, 2.0)
        .real("rh", 5.0, 95.0)
        .real("met", 1.0, 4.0)
        .real("clo", 0.0, 1.5)
        .real("wme", 0.0, 1.0)
        .enumerated("units", 2);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field_si = FieldCmp::new("ce", 0.011);
        // Upstream's IP branch rescales the Celsius value by 3.28/1.8 (~1.822), so an
        // equivalent-precision absolute tolerance scales by the same factor.
        let field_ip = FieldCmp::new("ce", 0.011 * 3.28 / 1.8);

        run_sweep("sweep_cooling_effect", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, wme) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
            );
            let (units, py_units) = match s.index("units") {
                0 => (Units::SI, "SI"),
                _ => (Units::IP, "IP"),
            };

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                ("units", py_units.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();
            // Python's `units="IP"` reinterprets its raw tdb/tr/vr floats as
            // Fahrenheit/fps and converts them to SI before doing anything else, so
            // feeding it the same Celsius/m/s numbers Rust uses would silently change
            // the physical inputs. Convert to IP here (upstream's own
            // `units_converter` formulas) so both sides compute from the same
            // underlying environment; only cooling_effect's *output* rescale is under
            // test.
            let (py_tdb, py_tr, py_vr) = if units == Units::IP {
                (tdb * 9.0 / 5.0 + 32.0, tr * 9.0 / 5.0 + 32.0, vr * 3.281)
            } else {
                (tdb, tr, vr)
            };
            let py_result = models
                .getattr("cooling_effect")
                .unwrap()
                .call((py_tdb, py_tr, py_vr, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = cooling_effect(
                CoolingEffectInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                CoolingEffectOptions {
                    wme: MetabolicRate::from_met(wme),
                    units,
                },
            );

            let (field, rust_value) = match units {
                Units::SI => (&field_si, rust.as_celsius()),
                Units::IP => (&field_ip, rust.as_fahrenheit()),
            };

            compare_field(field, rust_value, py_float(&py_result, "ce")?)
        });
    });
}

#[test]
fn sweep_use_fans_heatwaves() {
    let domain = Domain::new()
        .real("tdb", 25.0, 50.0)
        .real("tr", 25.0, 50.0)
        .real("v", 0.1, 4.5)
        .real("rh", 5.0, 95.0)
        .real("met", 0.8, 2.5)
        .real("clo", 0.0, 1.0)
        .real("wme", 0.0, 1.0)
        .real("body_surface_area", 1.5, 2.2)
        .real("p_atm", 80_000.0, 105_000.0)
        .real("max_skin_blood_flow", 40.0, 110.0)
        .real("max_sweating", 200.0, 700.0)
        .enumerated("posture", 2)
        .flag("limit_inputs")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let numeric = [
            FieldCmp::new("e_skin", 0.11),
            FieldCmp::new("e_rsw", 0.11),
            FieldCmp::new("e_max", 0.11),
            FieldCmp::new("q_sensible", 0.11),
            FieldCmp::new("q_skin", 0.11),
            FieldCmp::new("q_res", 0.11),
            FieldCmp::new("t_core", 0.06),
            FieldCmp::new("t_skin", 0.06),
            FieldCmp::new("m_bl", 0.11),
            FieldCmp::new("m_rsw", 0.11),
            FieldCmp::new("w", 0.06),
            FieldCmp::new("w_max", 0.06),
        ];

        run_sweep("sweep_use_fans_heatwaves", &domain, |s: &Sample| {
            let (tdb, tr, v, rh, met, clo, wme, bsa, p_atm, msbf, msw) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("v"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("body_surface_area"),
                s.real("p_atm"),
                s.real("max_skin_blood_flow"),
                s.real("max_sweating"),
            );
            let (posture, py_posture) = match s.index("posture") {
                0 => (Posture::Standing, "standing"),
                _ => (Posture::Sitting, "sitting"),
            };

            // pythermalcomfort 4.4.0 raises UFuncTypeError for this combination: with
            // limit_inputs=False it skips the masking step that would have coerced the
            // boolean heat-strain fields, then np.around() tries to round them. An
            // upstream defect, not a parity concession - there is no Rust behaviour that
            // could match a crash, so the combination is excluded rather than papered
            // over. The other three combinations are all exercised.
            if !s.flag("limit_inputs") && s.flag("round_output") {
                return Ok(());
            }

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "body_surface_area",
                    bsa.into_pyobject(py).unwrap().into_any(),
                ),
                ("p_atm", p_atm.into_pyobject(py).unwrap().into_any()),
                ("position", py_posture.into_pyobject(py).unwrap().into_any()),
                (
                    "max_skin_blood_flow",
                    msbf.into_pyobject(py).unwrap().into_any(),
                ),
                ("max_sweating", msw.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, s.flag("limit_inputs"))
                        .to_owned()
                        .into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, s.flag("round_output"))
                        .to_owned()
                        .into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("use_fans_heatwaves")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = use_fans_heatwaves(
                UseFansHeatwavesInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                UseFansHeatwavesOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    position: posture,
                    max_skin_blood_flow: msbf,
                    max_sweating: msw,
                    limit_inputs: s.flag("limit_inputs"),
                    round_output: s.flag("round_output"),
                },
            );

            let values = [
                rust.e_skin.as_watts_per_square_meter(),
                rust.e_rsw.as_watts_per_square_meter(),
                rust.e_max.as_watts_per_square_meter(),
                rust.q_sensible.as_watts_per_square_meter(),
                rust.q_skin.as_watts_per_square_meter(),
                rust.q_res.as_watts_per_square_meter(),
                rust.t_core.as_celsius(),
                rust.t_skin.as_celsius(),
                rust.m_bl,
                rust.m_rsw,
                rust.w,
                rust.w_max,
            ];
            for (field, rust_value) in numeric.iter().zip(values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
            }

            // The heat-strain verdicts are the model's actual output; Python reports
            // them as float64 0.0/1.0 rather than bool.
            for (name, rust_flag) in [
                ("heat_strain", rust.heat_strain),
                ("heat_strain_blood_flow", rust.heat_strain_blood_flow),
                ("heat_strain_w", rust.heat_strain_w),
                ("heat_strain_sweating", rust.heat_strain_sweating),
            ] {
                // Python masks these to NaN outside the applicability limits, which
                // maps to None on the Rust side rather than to `false`.
                let py_raw = py_float(&py_result, name)?;
                let py_flag = if py_raw.is_nan() {
                    None
                } else {
                    Some(py_raw != 0.0)
                };
                if py_flag != rust_flag {
                    return Err(format!("{name}: Rust {rust_flag:?}, Python {py_flag:?}"));
                }
            }
            Ok(())
        });
    });
}

#[test]
fn sweep_phs() {
    // ISO 7933 has the largest option surface in the crate, and essentially none of it
    // is varied by the hand-written tests: posture, i_mst, a_p, drink, weight, height,
    // walk_sp, theta, acclimatized and the 2004/2023 model edition.
    let domain = Domain::new()
        .real("tdb", 15.0, 50.0)
        .real("tr", 15.0, 60.0)
        .real("v", 0.0, 3.0)
        .real("rh", 5.0, 95.0)
        .real("met", 1.0, 5.0)
        .real("clo", 0.1, 1.5)
        .real("wme", 0.0, 1.0)
        .real("i_mst", 0.2, 0.6)
        .real("a_p", 0.0, 1.0)
        .real("weight", 50.0, 110.0)
        .real("height", 1.5, 2.0)
        .real("walk_sp", 0.0, 1.5)
        .real("theta", 0.0, 180.0)
        .enumerated("posture", 3)
        .enumerated("model", 2)
        .flag("drink")
        .flag("acclimatized")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = [
            FieldCmp::new("t_re", 0.06),
            FieldCmp::new("t_sk", 0.06),
            FieldCmp::new("t_cr", 0.06),
            FieldCmp::new("t_cr_eq", 0.06),
            FieldCmp::new("t_sk_t_cr_wg", 0.06),
            FieldCmp::new("d_lim_loss_50", 0.6),
            FieldCmp::new("d_lim_loss_95", 0.6),
            FieldCmp::new("d_lim_t_re", 0.6),
            FieldCmp::new("sweat_loss_g", 1.1).rel(1e-3),
            FieldCmp::new("sweat_rate_watt", 0.6).rel(1e-3),
            FieldCmp::new("evap_load_wm2_min", 0.6).rel(1e-3),
        ];

        run_sweep("sweep_phs", &domain, |s: &Sample| {
            let (tdb, tr, v, rh, met, clo, wme, i_mst, a_p, weight, height, walk_sp, theta) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("v"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("i_mst"),
                s.real("a_p"),
                s.real("weight"),
                s.real("height"),
                s.real("walk_sp"),
                s.real("theta"),
            );
            let (posture, py_posture) = match s.index("posture") {
                0 => (PhsPosture::Sitting, "sitting"),
                1 => (PhsPosture::Standing, "standing"),
                _ => (PhsPosture::Crouching, "crouching"),
            };
            let (model, py_model) = match s.index("model") {
                0 => (Iso7933Model::Iso2004, "7933-2004"),
                _ => (Iso7933Model::Iso2023, "7933-2023"),
            };
            let drink = s.flag("drink");
            let acclimatized = s.flag("acclimatized");
            let round_output = s.flag("round_output");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
                ("model", py_model.into_pyobject(py).unwrap().into_any()),
                ("i_mst", i_mst.into_pyobject(py).unwrap().into_any()),
                ("a_p", a_p.into_pyobject(py).unwrap().into_any()),
                (
                    "drink",
                    (if drink { 1_i64 } else { 0 })
                        .into_pyobject(py)
                        .unwrap()
                        .into_any(),
                ),
                ("weight", weight.into_pyobject(py).unwrap().into_any()),
                ("height", height.into_pyobject(py).unwrap().into_any()),
                ("walk_sp", walk_sp.into_pyobject(py).unwrap().into_any()),
                ("theta", theta.into_pyobject(py).unwrap().into_any()),
                (
                    "acclimatized",
                    (if acclimatized { 100_i64 } else { 0 })
                        .into_pyobject(py)
                        .unwrap()
                        .into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("phs")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo, py_posture), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = phs(
                PhsInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    posture,
                },
                PhsOptions {
                    wme: MetabolicRate::from_met(wme),
                    round_output,
                    model,
                    i_mst,
                    a_p,
                    drink,
                    weight: Mass::from_kilograms(weight),
                    height: Length::from_meters(height),
                    walk_sp: Speed::from_meters_per_second(walk_sp),
                    theta,
                    acclimatized,
                    ..Default::default()
                },
            );

            let values = [
                rust.t_re.as_celsius(),
                rust.t_sk.as_celsius(),
                rust.t_cr.as_celsius(),
                rust.t_cr_eq.as_celsius(),
                rust.t_sk_t_cr_wg,
                rust.d_lim_loss_50,
                rust.d_lim_loss_95,
                rust.d_lim_t_re,
                rust.sweat_loss_g.as_grams(),
                rust.sweat_rate_watt.as_watts_per_square_meter(),
                rust.evap_load_wm2_min,
            ];
            for (field, rust_value) in fields.iter().zip(values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
            }
            Ok(())
        });
    });
}

/// Pull a Python sequence of floats (numpy array or list) into a `Vec`.
fn py_float_seq(obj: &Bound<'_, PyAny>, field: &str) -> Result<Vec<f64>, String> {
    let attr = obj
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python result: {e}"))?;
    attr.extract::<Vec<f64>>()
        .map_err(|e| format!("{field}: could not read as a float sequence: {e}"))
}

#[test]
fn sweep_two_nodes_gagge_ji() {
    // The Ji model takes vapour pressure where the Rust wrapper takes relative humidity,
    // so the sweep samples `rh` and lets each side derive the vapour pressure its own
    // way. That deliberately puts `p_sat_torr` inside the comparison.
    //
    // `body_weight`, `length_time_simulation`, `initial_skin_temp` and
    // `initial_core_temp` are newly-exposed options (previously hardcoded inside the
    // Rust port, silently ignoring anything Python did with them) so they get their own
    // axes here rather than being left pinned at their defaults.
    let domain = Domain::new()
        .real("tdb", 5.0, 45.0)
        .real("tr", 5.0, 45.0)
        .real("v", 0.0, 2.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.7, 2.5)
        .real("clo", 0.0, 2.0)
        .real("wme", 0.0, 1.0)
        .real("body_surface_area", 1.5, 2.2)
        .real("p_atm", 80_000.0, 105_000.0)
        .real("body_weight", 40.0, 150.0)
        .real("initial_skin_temp", 30.0, 38.0)
        .real("initial_core_temp", 35.0, 39.0)
        .enumerated("length_time_simulation", 3)
        .enumerated("position", 3)
        .flag("acclimatized");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let utilities = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_two_nodes_gagge_ji", &domain, |s: &Sample| {
            let (tdb, tr, v, rh, met, clo, wme, bsa, p_atm, body_weight, init_skin, init_core) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("v"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("body_surface_area"),
                s.real("p_atm"),
                s.real("body_weight"),
                s.real("initial_skin_temp"),
                s.real("initial_core_temp"),
            );
            // Python's Ji validator accepts exactly these three. Only `sitting` takes
            // the 0.7 radiating-area branch; both standing forms share 0.77, which the
            // Rust `Posture::Standing` covers -- the crate has no separate
            // forced-convection variant and the model is insensitive to the difference.
            let (posture, py_posture) = match s.index("position") {
                0 => (Posture::Sitting, "sitting"),
                1 => (Posture::Standing, "standing"),
                _ => (Posture::Standing, "standing, forced convection"),
            };
            let acclimatized = s.flag("acclimatized");
            let length_time_simulation = match s.index("length_time_simulation") {
                0 => 60_usize,
                1 => 120,
                _ => 180,
            };

            let p_sat = utilities
                .getattr("p_sat_torr")
                .unwrap()
                .call1((tdb,))
                .map_err(|e| format!("p_sat_torr raised: {e}"))?;
            let p_sat: f64 = p_sat
                .extract()
                .or_else(|_| p_sat.call_method0("item").and_then(|i| i.extract()))
                .map_err(|e| format!("p_sat_torr did not return a number: {e}"))?;
            let vapor_pressure = rh * p_sat / 100.0;

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "body_surface_area",
                    bsa.into_pyobject(py).unwrap().into_any(),
                ),
                ("p_atm", p_atm.into_pyobject(py).unwrap().into_any()),
                ("position", py_posture.into_pyobject(py).unwrap().into_any()),
                (
                    "acclimatized",
                    PyBool::new(py, acclimatized).to_owned().into_any(),
                ),
                (
                    "body_weight",
                    body_weight.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "length_time_simulation",
                    length_time_simulation.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "initial_skin_temp",
                    init_skin.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "initial_core_temp",
                    init_core.into_pyobject(py).unwrap().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("two_nodes_gagge_ji")
                .unwrap()
                .call((tdb, tr, v, met, clo, vapor_pressure), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = two_nodes_gagge_ji(
                GaggeTwoNodesJiInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    vapor_pressure: Pressure::from_torrs(vapor_pressure),
                },
                GaggeTwoNodesJiOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    position: posture,
                    acclimatized,
                    body_weight: Mass::from_kilograms(body_weight),
                    length_time_simulation,
                    initial_skin_temp: Temperature::from_celsius(init_skin),
                    initial_core_temp: Temperature::from_celsius(init_core),
                },
            );

            // Compared exactly. This sweep used to round *Python's* trajectory to two
            // decimals before comparing, whenever Rust's round_output was set -- making
            // the reference a value pythermalcomfort never produces, so the check could
            // not fail for the right reason. two_nodes_gagge_ji.py contains no round or
            // np.around call anywhere; the Rust flag was invented, and it and this
            // accommodation are both gone.

            for (name, rust_series) in [("t_core", &rust.t_core), ("t_skin", &rust.t_skin)] {
                let py_series = py_float_seq(&py_result, name)?;
                if py_series.len() != rust_series.len() {
                    return Err(format!(
                        "{name}: Rust returned {} minutes, Python {}",
                        rust_series.len(),
                        py_series.len()
                    ));
                }
                let cmp = FieldCmp::new(name, 1e-9);
                for (minute, (rust_value, py_value)) in
                    rust_series.iter().zip(py_series.iter()).enumerate()
                {
                    compare_field(&cmp, rust_value.as_celsius(), *py_value)
                        .map_err(|e| format!("minute {minute}: {e}"))?;
                }
            }
            Ok(())
        });
    });
}

#[test]
fn sweep_two_nodes_gagge_sleep() {
    // The sleep model's inputs are per-minute schedules, so the sweep samples a base value
    // and a drift for each variable and builds a *varying* night out of them. A flat
    // schedule would barely exercise the state carried between minutes, and that carry-over
    // is precisely where this port's one real defect lived: an error there leaves minute 0
    // exactly right and corrupts every minute after it.
    let domain = Domain::new()
        .real("tdb", 10.0, 35.0)
        .real("tr", 10.0, 35.0)
        .real("v", 0.0, 1.5)
        .real("rh", 5.0, 95.0)
        .real("clo", 0.0, 2.0)
        .real("thickness", 0.0, 15.0)
        .real("tdb_drift", -0.15, 0.15)
        .real("tr_drift", -0.15, 0.15)
        .real("rh_drift", -0.4, 0.4)
        .real("clo_drift", -0.01, 0.01)
        .real("thickness_drift", -0.12, 0.12)
        .real("wme", 0.0, 0.5)
        .real("p_atm", 80_000.0, 105_000.0)
        .enumerated("duration", 4);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        run_sweep("sweep_two_nodes_gagge_sleep", &domain, |s: &Sample| {
            // Upstream returns scalars rather than arrays for a one-minute night, so the
            // shortest schedule swept is two minutes.
            let n = match s.index("duration") {
                0 => 2usize,
                1 => 15,
                2 => 60,
                _ => 120,
            };
            let (wme, p_atm) = (s.real("wme"), s.real("p_atm"));

            // Clamped to the ranges pythermalcomfort's validator accepts, so a drift never
            // walks an input out of bounds and turns a parity check into an exception.
            let ramp = |base: f64, drift: f64, lo: f64, hi: f64| -> Vec<f64> {
                (0..n)
                    .map(|i| (base + drift * i as f64).clamp(lo, hi))
                    .collect()
            };

            let tdb = ramp(s.real("tdb"), s.real("tdb_drift"), 5.0, 45.0);
            let tr = ramp(s.real("tr"), s.real("tr_drift"), 5.0, 45.0);
            let v = vec![s.real("v"); n];
            let rh = ramp(s.real("rh"), s.real("rh_drift"), 0.0, 100.0);
            let clo = ramp(s.real("clo"), s.real("clo_drift"), 0.0, 3.0);
            let thickness = ramp(s.real("thickness"), s.real("thickness_drift"), 0.0, 30.0);

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                ("p_atm", p_atm.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();

            let call = models.getattr("two_nodes_gagge_sleep").unwrap().call(
                (
                    tdb.clone(),
                    tr.clone(),
                    v.clone(),
                    rh.clone(),
                    clo.clone(),
                    thickness.clone(),
                ),
                Some(&kwargs),
            );
            let py_result = match call {
                Ok(result) => result,
                // In a hot, humid, heavily quilted corner the model's own exponentials
                // overflow and CPython raises. There is no reference value to compare
                // against, so the sample is skipped rather than counted as a divergence.
                // Worth knowing: Rust does not raise here, it produces an infinity — a
                // genuine behavioural difference, but one confined to inputs where
                // upstream declines to answer at all.
                Err(e) if e.is_instance_of::<PyOverflowError>(py) => return Ok(()),
                Err(e) => return Err(format!("python raised: {e}")),
            };

            let rust_tdb: Vec<Temperature> =
                tdb.iter().copied().map(Temperature::from_celsius).collect();
            let rust_tr: Vec<Temperature> =
                tr.iter().copied().map(Temperature::from_celsius).collect();
            let rust_v: Vec<Speed> = v
                .iter()
                .copied()
                .map(Speed::from_meters_per_second)
                .collect();
            let rust_rh: Vec<Humidity> = rh.iter().copied().map(Humidity::from_percent).collect();
            let rust_clo: Vec<ClothingInsulation> = clo
                .iter()
                .copied()
                .map(ClothingInsulation::from_clo)
                .collect();
            let rust_quilt: Vec<Length> = thickness
                .iter()
                .copied()
                .map(Length::from_centimeters)
                .collect();

            let rust = two_nodes_gagge_sleep(
                SleepInputs {
                    tdb: &rust_tdb,
                    tr: &rust_tr,
                    v: &rust_v,
                    rh: &rust_rh,
                    clo: &rust_clo,
                    thickness_quilt: &rust_quilt,
                },
                GaggeTwoNodesSleepOptions {
                    wme: MetabolicRate::from_met(wme),
                    p_atm: Pressure::from_pascals(p_atm),
                    ..Default::default()
                },
            )
            .map_err(|e| format!("rust rejected the schedule: {e}"))?;

            // Upstream does not round this model, so an absolute 1e-9 is the primary
            // bound and it holds for the great majority of samples. The relative bound
            // covers two things an absolute one measures badly over a 120-minute run:
            //
            //   - each minute's state feeds the next, so `math.exp` and `libm::exp`
            //     differing by an ulp compounds in proportion to the result, exactly as
            //     it does in the PHS sweep's 480-minute integration;
            //   - in a low-airspeed, thick-quilt corner upstream's own SET secant solve
            //     runs away to around 5e7 °C, where 1e-9 absolute is a tolerance on the
            //     ulp rather than on the physics.
            //
            // 1e-9 relative is nine significant figures and cannot absorb a transcription
            // error; the worst observed here is 2.7e-12, on `disc` at minute 41. The exact
            // 1e-9 absolute check on short runs lives in the module's unit tests.
            let series: [(&str, Vec<f64>); 10] = [
                ("set", rust.set.iter().map(|t| t.as_celsius()).collect()),
                (
                    "t_core",
                    rust.t_core.iter().map(|t| t.as_celsius()).collect(),
                ),
                (
                    "t_skin",
                    rust.t_skin.iter().map(|t| t.as_celsius()).collect(),
                ),
                ("wet", rust.wet.clone()),
                ("t_sens", rust.t_sens.clone()),
                ("disc", rust.disc.clone()),
                (
                    "e_skin",
                    rust.e_skin
                        .iter()
                        .map(|q| q.as_watts_per_square_meter())
                        .collect(),
                ),
                (
                    "met_shivering",
                    rust.met_shivering
                        .iter()
                        .map(|q| q.as_watts_per_square_meter())
                        .collect(),
                ),
                ("alfa", rust.alfa.clone()),
                ("skin_blood_flow", rust.skin_blood_flow.clone()),
            ];

            for (name, rust_series) in series {
                let py_series = py_float_seq(&py_result, name)?;
                if py_series.len() != rust_series.len() {
                    return Err(format!(
                        "{name}: Rust returned {} minutes, Python {}",
                        rust_series.len(),
                        py_series.len()
                    ));
                }
                let cmp = FieldCmp::new(name, 1e-9).rel(1e-9);
                for (minute, (rust_value, py_value)) in
                    rust_series.iter().zip(py_series.iter()).enumerate()
                {
                    compare_field(&cmp, *rust_value, *py_value)
                        .map_err(|e| format!("minute {minute}: {e}"))?;
                }
            }
            Ok(())
        });
    });
}

// ---------------------------------------------------------------------------
// JOS3
// ---------------------------------------------------------------------------

/// The 17 JOS3 body segment names, canonical order. Duplicated from
/// `tests/python_comparison.rs` rather than shared: each parity-test binary is
/// self-contained (see that file's own copy and `py_float_seq` above for precedent).
const JOS3_BODY_NAMES: [&str; 17] = [
    "head",
    "neck",
    "chest",
    "back",
    "pelvis",
    "left_shoulder",
    "left_arm",
    "left_hand",
    "right_shoulder",
    "right_arm",
    "right_hand",
    "left_thigh",
    "left_leg",
    "left_foot",
    "right_thigh",
    "right_leg",
    "right_foot",
];

fn jos3_f64_series(py_output: &Bound<'_, PyAny>, field: &str) -> Result<Vec<f64>, String> {
    py_output
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python JOS3Output: {e}"))?
        .call_method0("tolist")
        .map_err(|e| format!("{field}: tolist() raised: {e}"))?
        .extract()
        .map_err(|e| format!("{field}: could not extract as Vec<f64>: {e}"))
}

fn jos3_body_part_series(
    py_output: &Bound<'_, PyAny>,
    field: &str,
) -> Result<[Vec<f64>; 17], String> {
    let body = py_output
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python JOS3Output: {e}"))?;
    let mut out: [Vec<f64>; 17] = Default::default();
    for (i, part) in JOS3_BODY_NAMES.iter().enumerate() {
        out[i] = body
            .getattr(*part)
            .map_err(|e| format!("{field}.{part}: {e}"))?
            .call_method0("tolist")
            .map_err(|e| format!("{field}.{part}: tolist() raised: {e}"))?
            .extract()
            .map_err(|e| format!("{field}.{part}: could not extract as Vec<f64>: {e}"))?;
    }
    Ok(out)
}

fn jos3_head_pelvis_series(
    py_output: &Bound<'_, PyAny>,
    field: &str,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let body = py_output
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python JOS3Output: {e}"))?;
    let get = |name: &str| -> Result<Vec<f64>, String> {
        body.getattr(name)
            .map_err(|e| format!("{field}.{name}: {e}"))?
            .call_method0("tolist")
            .map_err(|e| format!("{field}.{name}: tolist() raised: {e}"))?
            .extract()
            .map_err(|e| format!("{field}.{name}: could not extract as Vec<f64>: {e}"))
    };
    Ok((get("head")?, get("pelvis")?))
}

/// `t_superficial_vein`'s 12 raw values, positionally -- see the identical helper (and
/// its long comment on upstream's `pass_values_to_jos3_body_parts` mislabeling) in
/// `tests/python_comparison.rs`. Reading the first 12 canonical body-part names in order
/// recovers the 12 raw values in the same order the Rust port uses.
fn jos3_superficial_vein_series(py_output: &Bound<'_, PyAny>) -> Result<[Vec<f64>; 12], String> {
    let body = py_output
        .getattr("t_superficial_vein")
        .map_err(|e| format!("t_superficial_vein: missing on Python JOS3Output: {e}"))?;
    let mut out: [Vec<f64>; 12] = Default::default();
    for (i, part) in JOS3_BODY_NAMES.iter().take(12).enumerate() {
        out[i] = body
            .getattr(*part)
            .map_err(|e| format!("t_superficial_vein.{part}: {e}"))?
            .call_method0("tolist")
            .map_err(|e| format!("t_superficial_vein.{part}: tolist() raised: {e}"))?
            .extract()
            .map_err(|e| {
                format!("t_superficial_vein.{part}: could not extract as Vec<f64>: {e}")
            })?;
    }
    Ok(out)
}

/// One field's Python series, named the way `jos3_close` reports it: `"t_cb"`,
/// `"t_skin.head"`, or `"t_superficial_vein[3]"`. Used by the conditioning probe below
/// to re-read a single failing field out of a second Python run.
fn jos3_named_series(py_output: &Bound<'_, PyAny>, spec: &str) -> Result<Vec<f64>, String> {
    if let Some(rest) = spec.strip_prefix("t_superficial_vein[") {
        let k: usize = rest
            .trim_end_matches(']')
            .parse()
            .map_err(|_| format!("{spec}: not a t_superficial_vein index"))?;
        let all = jos3_superficial_vein_series(py_output)?;
        return all
            .into_iter()
            .nth(k)
            .ok_or_else(|| format!("{spec}: index out of range"));
    }
    let Some((field, part)) = spec.split_once('.') else {
        return jos3_f64_series(py_output, spec);
    };
    py_output
        .getattr(field)
        .map_err(|e| format!("{spec}: missing on Python JOS3Output: {e}"))?
        .getattr(part)
        .map_err(|e| format!("{spec}: {e}"))?
        .call_method0("tolist")
        .map_err(|e| format!("{spec}: tolist() raised: {e}"))?
        .extract()
        .map_err(|e| format!("{spec}: could not extract as Vec<f64>: {e}"))
}

/// How far apart one JOS3 field is allowed to be, in the units upstream reports it in.
///
/// pythermalcomfort rounds nearly every JOS3 output before returning it (`models/jos3.py`
/// lines 1054-1135), so a comparison of two rounded numbers cannot resolve anything below
/// one rounding step. The Rust port solves the same 85x85 system by LU factorisation
/// where Python multiplies by an explicitly inverted matrix; that is a legitimate ~1e-13
/// relative difference in the *unrounded* value, and where the unrounded value happens to
/// sit that close to a rounding boundary the two land on opposite sides and the reported
/// values differ by exactly one step. `sweep_pet_steady` permits 0.0101 on its own
/// 2-decimal output for the same reason and with the same reasoning.
///
/// This is a real loss of resolution, not a free pass: on a 2-decimal field the sweep can
/// no longer see a genuine drift smaller than 0.01, and it is forced on the sweep by
/// upstream rounding its own outputs -- there is no unrounded reference to compare
/// against. The per-field steps below are therefore the *smallest* number that admits a
/// boundary straddle for that field, not a blanket loose tolerance.
///
/// Each step is one unit in the field's last reported decimal, plus 1% slack so that the
/// float nearest to (say) 0.02964 - 0.02963 = 1.0000000000000026e-5 compares as one step
/// rather than one-and-a-bit.
fn jos3_tolerance(field: &str) -> f64 {
    // Strip the `.head` / `[3]` suffix the per-body-part comparisons append.
    let base = field.split(['.', '[']).next().unwrap_or(field);
    match base {
        // Not rounded upstream at all: `dt` is the caller's own time step and
        // height/weight/fat/par are echoed back off `self._height` and friends
        // (jos3.py:1064-1070) with no rounding in between. Nothing can straddle a
        // boundary that is never crossed, so these keep exact agreement.
        "simulation_time" | "dt" | "height" | "weight" | "fat" | "par" => 1e-9,
        // `round(co, 1)` -- jos3.py:1061. One step is 0.1.
        "cardiac_output" => 0.101,
        // `pass_values_to_jos3_body_parts(r_t, 3)` -- jos3.py:1085-1086. One step is 0.001.
        "r_t" | "r_et" => 0.001_01,
        // `round(wlesk.sum() + wleres, 5)` -- jos3.py:1060. One step is 1e-5.
        "weight_loss_by_evap_and_res" => 1.01e-5,
        // Everything else is 2 decimals, either `np.round(x, 2)` directly or via
        // `pass_values_to_jos3_body_parts`, whose `round_digits` defaults to 2.
        _ => 0.0101,
    }
}

/// A divergence found by [`jos3_results_match`].
enum Jos3Mismatch {
    /// Structural: a series is the wrong length, or a field is missing on the Python
    /// side. Never a rounding artefact, so it is reported as-is.
    Shape(String),
    /// One field at one step disagreed by more than [`jos3_tolerance`]. Kept structured
    /// rather than pre-formatted so `sweep_jos3` can re-measure that exact field and step
    /// against the reference model's own last-bit sensitivity before calling it a defect.
    Value {
        field: String,
        step: usize,
        rust: f64,
        py: f64,
    },
}

impl Jos3Mismatch {
    fn describe(&self) -> String {
        match self {
            Self::Shape(msg) => msg.clone(),
            Self::Value {
                field,
                step,
                rust,
                py,
            } => format!(
                "{field}[{step}]: rust={rust}, python={py}, diff={:.3e} (tolerance {:.3e})",
                (rust - py).abs(),
                jos3_tolerance(field)
            ),
        }
    }
}

impl From<String> for Jos3Mismatch {
    fn from(msg: String) -> Self {
        Self::Shape(msg)
    }
}

fn jos3_close(field: &str, step: usize, rust: f64, py: f64) -> Result<(), Jos3Mismatch> {
    if (rust.is_nan() && py.is_nan()) || (rust - py).abs() <= jos3_tolerance(field) {
        Ok(())
    } else {
        Err(Jos3Mismatch::Value {
            field: field.to_string(),
            step,
            rust,
            py,
        })
    }
}

/// Does pythermalcomfort itself still produce a stable value for `field` up to `step`?
///
/// `base` and `perturbed` are two Python runs of the same sample that differ only in the
/// last bit of `tdb`. JOS3 clips skin wettedness at saturation
/// (`jos3_functions/thermoregulation.py:673`, `wet = np.minimum(wet, 1)`), and a body
/// segment sitting exactly on that clip takes a different branch from one step to the
/// next depending on bits far below any physical significance. Once a sample is in that
/// regime the trajectory has a positive Lyapunov exponent: a measured example grew a
/// 1-ULP input change into a 1.9e-4 output change -- roughly nineteen rounding steps of
/// `weight_loss_by_evap_and_res` -- within seventeen steps, and moved 13 of that field's
/// 200 rounded values.
///
/// Where that has happened there is no reference value to compare against: upstream's own
/// answer is not a function of the inputs to any resolution the sweep can see, and *no*
/// correct implementation would reproduce it. Such a sample is skipped, exactly as
/// `sweep_pet_steady` skips the samples scipy's fsolve disowns.
///
/// The comparison is exact equality of the two *rounded* series, and it looks at every
/// step up to and including the failing one: once the reference has visibly moved at an
/// earlier step, nothing later in that field carries information either. A genuine port
/// defect is not hidden by this, because a defect shows up on samples whose reference is
/// stable, and the sweep counts and reports how many samples it discarded.
fn jos3_reference_is_ulp_unstable(
    base: &Bound<'_, PyAny>,
    perturbed: &Bound<'_, PyAny>,
    field: &str,
    step: usize,
) -> Result<bool, String> {
    let a = jos3_named_series(base, field)?;
    let b = jos3_named_series(perturbed, field)?;
    let last = step
        .min(a.len().saturating_sub(1))
        .min(b.len().saturating_sub(1));
    Ok((0..=last).any(|i| a[i] != b[i]))
}

/// Compare every field of a Rust [`Jos3Results`] against a Python `JOS3.results()`
/// object, over every step. `results()`, not `dict_results()` -- see
/// `tests/python_comparison.rs` for why (a genuine pythermalcomfort 4.4.0 bug makes
/// `dict_results()` return attribute-name strings instead of values for every
/// per-body-part field).
fn jos3_results_match(py: &Bound<'_, PyAny>, rust: &Jos3Results) -> Result<(), Jos3Mismatch> {
    let n = rust.simulation_time.len();

    let scalar_fields: [(&str, &Vec<f64>); 16] = [
        ("dt", &rust.dt),
        ("t_skin_mean", &rust.t_skin_mean),
        ("w_mean", &rust.w_mean),
        (
            "weight_loss_by_evap_and_res",
            &rust.weight_loss_by_evap_and_res,
        ),
        ("cardiac_output", &rust.cardiac_output),
        ("q_thermogenesis_total", &rust.q_thermogenesis_total),
        ("q_res", &rust.q_res),
        ("height", &rust.height),
        ("weight", &rust.weight),
        ("fat", &rust.fat),
        ("t_cb", &rust.t_cb),
        ("par", &rust.par),
        ("bf_ava_hand", &rust.bf_ava_hand),
        ("bf_ava_foot", &rust.bf_ava_foot),
        ("q_res_sensible", &rust.q_res_sensible),
        ("q_res_latent", &rust.q_res_latent),
    ];
    for (field, rust_series) in scalar_fields {
        if rust_series.len() != n {
            return Err(Jos3Mismatch::Shape(format!(
                "{field}: rust length {} != {n}",
                rust_series.len()
            )));
        }
        let py_series = jos3_f64_series(py, field)?;
        if py_series.len() != n {
            return Err(Jos3Mismatch::Shape(format!(
                "{field}: python length {} != {n}",
                py_series.len()
            )));
        }
        for (step, (&r, &p)) in rust_series.iter().zip(py_series.iter()).enumerate() {
            jos3_close(field, step, r, p)?;
        }
    }

    let body17_fields: [(&str, &Vec<[f64; 17]>); 31] = [
        ("t_skin", &rust.t_skin),
        ("t_core", &rust.t_core),
        ("w", &rust.w),
        ("q_skin2env", &rust.q_skin2env),
        ("bsa", &rust.bsa),
        ("t_core_set", &rust.t_core_set),
        ("t_skin_set", &rust.t_skin_set),
        ("t_artery", &rust.t_artery),
        ("t_vein", &rust.t_vein),
        ("to", &rust.to),
        ("r_t", &rust.r_t),
        ("r_et", &rust.r_et),
        ("tdb", &rust.tdb),
        ("tr", &rust.tr),
        ("rh", &rust.rh),
        ("v", &rust.v),
        ("clo", &rust.clo),
        ("e_skin", &rust.e_skin),
        ("e_max", &rust.e_max),
        ("e_sweat", &rust.e_sweat),
        ("bf_core", &rust.bf_core),
        ("bf_skin", &rust.bf_skin),
        ("q_bmr_core", &rust.q_bmr_core),
        ("q_bmr_skin", &rust.q_bmr_skin),
        ("q_work", &rust.q_work),
        ("q_shiv", &rust.q_shiv),
        ("q_nst", &rust.q_nst),
        ("q_thermogenesis_core", &rust.q_thermogenesis_core),
        ("q_thermogenesis_skin", &rust.q_thermogenesis_skin),
        ("q_skin2env_sensible", &rust.q_skin2env_sensible),
        ("q_skin2env_latent", &rust.q_skin2env_latent),
    ];
    for (field, rust_series) in body17_fields {
        if rust_series.len() != n {
            return Err(Jos3Mismatch::Shape(format!(
                "{field}: rust length {} != {n}",
                rust_series.len()
            )));
        }
        let py_series = jos3_body_part_series(py, field)?;
        for (part_idx, part_name) in JOS3_BODY_NAMES.iter().enumerate() {
            let py_part = &py_series[part_idx];
            if py_part.len() != n {
                return Err(Jos3Mismatch::Shape(format!(
                    "{field}.{part_name}: python length {} != {n}",
                    py_part.len()
                )));
            }
            for step in 0..n {
                jos3_close(
                    &format!("{field}.{part_name}"),
                    step,
                    rust_series[step][part_idx],
                    py_part[step],
                )?;
            }
        }
    }

    let body2_fields: [(&str, &Vec<[f64; 2]>); 8] = [
        ("t_muscle", &rust.t_muscle),
        ("t_fat", &rust.t_fat),
        ("bf_muscle", &rust.bf_muscle),
        ("bf_fat", &rust.bf_fat),
        ("q_bmr_muscle", &rust.q_bmr_muscle),
        ("q_bmr_fat", &rust.q_bmr_fat),
        ("q_thermogenesis_muscle", &rust.q_thermogenesis_muscle),
        ("q_thermogenesis_fat", &rust.q_thermogenesis_fat),
    ];
    for (field, rust_series) in body2_fields {
        if rust_series.len() != n {
            return Err(Jos3Mismatch::Shape(format!(
                "{field}: rust length {} != {n}",
                rust_series.len()
            )));
        }
        let (py_head, py_pelvis) = jos3_head_pelvis_series(py, field)?;
        for step in 0..n {
            jos3_close(
                &format!("{field}.head"),
                step,
                rust_series[step][0],
                py_head[step],
            )?;
            jos3_close(
                &format!("{field}.pelvis"),
                step,
                rust_series[step][1],
                py_pelvis[step],
            )?;
        }
    }

    if rust.t_superficial_vein.len() != n {
        return Err(Jos3Mismatch::Shape(format!(
            "t_superficial_vein: rust length {} != {n}",
            rust.t_superficial_vein.len()
        )));
    }
    let py_sfvein = jos3_superficial_vein_series(py)?;
    for (k, py_k) in py_sfvein.iter().enumerate() {
        for (step, &p) in py_k.iter().enumerate() {
            jos3_close(
                &format!("t_superficial_vein[{k}]"),
                step,
                rust.t_superficial_vein[step][k],
                p,
            )?;
        }
    }

    Ok(())
}

#[test]
fn sweep_jos3() {
    // Upstream rounds every JOS3 output field (`models/jos3.py`: 2 dp for most fields, 1
    // dp for `cardiac_output`, 3 dp for `r_t`/`r_et`, 5 dp for
    // `weight_loss_by_evap_and_res` -- see `Jos3Model::run_step`), so a comparison of
    // rounded values cannot see drift smaller than that rounding step. The Rust port
    // solves the same 85x85 system with an LU decomposition where Python computes an
    // explicit matrix inverse (`Jos3Model::run_step`'s own doc comment); that is more
    // numerically accurate, and the two are expected to diverge by a few ULP per step
    // in the *unrounded* internal body-temperature state, which is carried forward
    // unrounded from one step to the next on both sides. A run of a handful of steps
    // cannot accumulate that into anything visible after 2-5 dp rounding and would pass
    // whether or not the port were correct. `duration` below draws up to 500 steps (at
    // a drawn dtime of up to 120s, i.e. up to 16+ hours of simulated time) specifically
    // so that if the LU/inverse difference -- or any other genuine bug -- ever grows
    // large enough to cross a rounding boundary, this sweep is positioned to catch it.
    //
    // Two things follow from running that long, and both are handled below rather than
    // by loosening the comparison wholesale. First, a boundary straddle is expected, so
    // `jos3_tolerance` allows exactly one rounding step per field and no more. Second,
    // some samples reach the skin-wettedness saturation clip and become genuinely
    // chaotic, at which point pythermalcomfort's own output stops being a function of
    // its inputs at any resolution this sweep can see; `jos3_reference_is_ulp_unstable`
    // identifies those by re-running the reference with `tdb` moved one ULP, and they
    // are skipped and counted rather than compared against.
    let domain = Domain::new()
        .real("height", 1.3, 2.0)
        .real("weight", 35.0, 140.0)
        .real("age", 10.0, 90.0)
        .real("fat", 5.0, 40.0)
        .real("ci", 1.5, 4.0)
        .enumerated("bmr_equation", 3)
        .enumerated("bsa_equation", 4)
        .flag("sex")
        .real("tdb", 10.0, 35.0)
        .real("tr", 10.0, 35.0)
        .real("rh", 20.0, 80.0)
        .real("v", 0.0, 1.2)
        .real("clo", 0.2, 1.8)
        .real("par", 1.0, 2.5)
        .enumerated("posture", 5)
        .real("dtime", 30.0, 120.0)
        .enumerated("duration", 4);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        let ill_conditioned = std::cell::Cell::new(0usize);

        run_sweep("sweep_jos3", &domain, |s: &Sample| {
            let height = s.real("height");
            let weight = s.real("weight");
            let age = s.real("age").round() as i32;
            let fat = s.real("fat");
            let ci = s.real("ci");
            let (bmr, py_bmr) = match s.index("bmr_equation") {
                0 => (BmrEquation::HarrisBenedict, "harris-benedict"),
                1 => (
                    BmrEquation::HarrisBenedictOriginal,
                    "harris-benedict_origin",
                ),
                _ => (BmrEquation::Japanese, "japanese"),
            };
            let (bsa, py_bsa) = match s.index("bsa_equation") {
                0 => (BsaFormula::DuBois, "dubois"),
                1 => (BsaFormula::Takahira, "takahira"),
                2 => (BsaFormula::Fujimoto, "fujimoto"),
                _ => (BsaFormula::Kurazumi, "kurazumi"),
            };
            let (sex, py_sex) = if s.flag("sex") {
                (Sex::Female, "female")
            } else {
                (Sex::Male, "male")
            };
            let tdb = s.real("tdb");
            let tr = s.real("tr");
            let rh = s.real("rh");
            let v = s.real("v");
            let clo = s.real("clo");
            let par = s.real("par");
            let (posture, py_posture) = match s.index("posture") {
                0 => (Posture::Standing, "standing"),
                1 => (Posture::Sitting, "sitting"),
                2 => (Posture::Sedentary, "sedentary"),
                3 => (Posture::Lying, "lying"),
                _ => (Posture::Supine, "supine"),
            };
            let dtime = s.real("dtime");
            // At least "a few hundred steps" per the brief; see the comment above the
            // domain for why the run needs to be long, not just varied.
            let steps: u32 = match s.index("duration") {
                0 => 80,
                1 => 200,
                2 => 350,
                _ => 500,
            };

            let kwargs = [
                ("height", height.into_pyobject(py).unwrap().into_any()),
                ("weight", weight.into_pyobject(py).unwrap().into_any()),
                ("fat", fat.into_pyobject(py).unwrap().into_any()),
                ("age", age.into_pyobject(py).unwrap().into_any()),
                ("sex", py_sex.into_pyobject(py).unwrap().into_any()),
                ("ci", ci.into_pyobject(py).unwrap().into_any()),
                ("bmr_equation", py_bmr.into_pyobject(py).unwrap().into_any()),
                ("bsa_equation", py_bsa.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_results = match jos3_python_run(
                &models, &kwargs, tdb, tr, rh, v, clo, par, py_posture, steps, dtime,
            ) {
                Ok(results) => results,
                // A physically-invalid combination the thermoregulation loop cannot
                // solve (e.g. a negative convective/radiative coefficient) -- Python
                // raises a bare ValueError with no structured type to match on, mirrored
                // by `ThermoregulationError` on the Rust side. There is no reference
                // value to compare against, so the sample is skipped rather than
                // counted as a divergence, the same way `sweep_two_nodes_gagge_sleep`
                // skips upstream's OverflowError.
                Err(e) if e.is_instance_of::<PyValueError>(py) => return Ok(()),
                Err(e) => return Err(format!("python raised: {e}")),
            };

            let mut rust_model = Jos3Builder::new()
                .height(Length::from_meters(height))
                .weight(Mass::from_kilograms(weight))
                .age(age)
                .fat(BodyFat::new(fat).expect("domain stays inside [1, 90]"))
                .sex(sex)
                .cardiac_index(CardiacIndex::from_liters_per_minute_per_square_meter(ci))
                .bmr_equation(bmr)
                .bsa_equation(bsa)
                .build()
                .map_err(|e| format!("rust build failed where python succeeded: {e}"))?;

            let mut conditions = rust_model.conditions();
            conditions.tdb = PerBodyPart::Uniform(tdb);
            conditions.tr = PerBodyPart::Uniform(tr);
            conditions.rh = PerBodyPart::Uniform(rh);
            conditions.v = PerBodyPart::Uniform(v);
            conditions.clo = PerBodyPart::Uniform(clo);
            conditions.par = ActivityRatio::from_ratio(par);
            conditions.posture = posture;

            rust_model
                .advance(&conditions, steps, Duration::from_secs_f64(dtime))
                .map_err(|e| format!("rust advance failed where python succeeded: {e}"))?;

            match jos3_results_match(&py_results, rust_model.results()) {
                Ok(()) => Ok(()),
                Err(Jos3Mismatch::Shape(msg)) => Err(msg),
                Err(mismatch @ Jos3Mismatch::Value { .. }) => {
                    let Jos3Mismatch::Value { field, step, .. } = &mismatch else {
                        unreachable!("matched on the Value variant")
                    };
                    // The field disagreed by more than one rounding step. Before calling
                    // that a port defect, ask whether pythermalcomfort's own answer is
                    // even a function of the inputs here: re-run it with `tdb` moved by a
                    // single ULP and see whether the reference itself moves. See
                    // `jos3_reference_is_ulp_unstable`.
                    let probe = jos3_python_run(
                        &models,
                        &kwargs,
                        next_up(tdb),
                        tr,
                        rh,
                        v,
                        clo,
                        par,
                        py_posture,
                        steps,
                        dtime,
                    )
                    .map_err(|e| format!("python raised on the conditioning probe: {e}"))?;

                    if jos3_reference_is_ulp_unstable(&py_results, &probe, field, *step)? {
                        ill_conditioned.set(ill_conditioned.get() + 1);
                        Ok(())
                    } else {
                        Err(mismatch.describe())
                    }
                }
            }
        });

        // Not a silent cap: report the excluded share so it cannot drift upward unseen.
        eprintln!(
            "sweep_jos3: skipped {} sample(s) whose pythermalcomfort reference is not \
             stable under a 1-ULP change to tdb (skin-wettedness saturation chatter)",
            ill_conditioned.get()
        );
    });
}

/// Build and drive one Python `JOS3` run, returning its `results()`.
///
/// Factored out because the sweep runs it twice on a divergence: once for the reference
/// and once with `tdb` nudged, to measure the reference's own last-bit sensitivity.
#[allow(clippy::too_many_arguments)]
fn jos3_python_run<'py>(
    models: &Bound<'py, PyModule>,
    kwargs: &Bound<'py, PyDict>,
    tdb: f64,
    tr: f64,
    rh: f64,
    v: f64,
    clo: f64,
    par: f64,
    posture: &str,
    steps: u32,
    dtime: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let model = models.getattr("JOS3")?.call((), Some(kwargs))?;
    model.setattr("tdb", tdb)?;
    model.setattr("tr", tr)?;
    model.setattr("rh", rh)?;
    model.setattr("v", v)?;
    model.setattr("clo", clo)?;
    model.setattr("par", par)?;
    model.setattr("posture", posture)?;
    model.getattr("simulate")?.call1((steps, dtime))?;
    model.call_method0("results")
}

/// The next representable f64 above `value`.
///
/// Hand-rolled rather than `f64::next_up`, which stabilised in Rust 1.86 while this crate
/// pins `rust-version = "1.85"`. Only ever called on the sweep's `tdb` axis, which is
/// drawn from [10, 35], so the positive-finite bit-increment is the whole story.
fn next_up(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value > 0.0);
    f64::from_bits(value.to_bits() + 1)
}

#[test]
fn sweep_pmv_a() {
    // pythermalcomfort's pmv_a exposes no round_output and no model — its inner PMV
    // call always uses rounding and ISO 7730:2025. `PmvAOptions` mirrors that (see its
    // doc comment), so unlike a previous port there is no longer a `round_output` knob
    // here for the flag to prove inert; `round_output` is still drawn from
    // `pmv_domain()` but simply unused.
    let domain = pmv_domain().real("a_coefficient", 0.0, 1.0);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("a_pmv", 1e-9);

        run_sweep("sweep_pmv_a", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, wme, a_coefficient) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("a_coefficient"),
            );
            let limit_inputs = s.flag("limit_inputs");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("pmv_a")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo, a_coefficient), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_a(
                PmvAInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    a_coefficient,
                },
                PmvAOptions {
                    wme: MetabolicRate::from_met(wme),
                    limit_inputs,
                },
            );

            compare_field(&field, rust, py_float(&py_result, "a_pmv")?)
        });
    });
}

#[test]
fn sweep_pmv_e() {
    // See `sweep_pmv_a`: `PmvEOptions` likewise has no `round_output` or `model`.
    let domain = pmv_domain().real("e_coefficient", 0.0, 1.0);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("e_pmv", 1e-9);

        run_sweep("sweep_pmv_e", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, wme, e_coefficient) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("wme"),
                s.real("e_coefficient"),
            );
            let limit_inputs = s.flag("limit_inputs");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("pmv_e")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo, e_coefficient), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_e(
                PmvEInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    e_coefficient,
                },
                PmvEOptions {
                    wme: MetabolicRate::from_met(wme),
                    limit_inputs,
                },
            );

            compare_field(&field, rust, py_float(&py_result, "e_pmv")?)
        });
    });
}

#[test]
fn sweep_pmv_athb() {
    // `clo` is optional on both sides: Python takes `False` to mean "derive it", Rust
    // takes `None`. The flag axis exercises both branches.
    let domain = Domain::new()
        .real("tdb", 5.0, 45.0)
        .real("tr", 5.0, 45.0)
        .real("vr", 0.0, 2.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.6, 5.0)
        .real("clo", 0.0, 2.5)
        .real("t_running_mean", -10.0, 35.0)
        .flag("supply_clo");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("athb_pmv", 1e-9);

        run_sweep("sweep_pmv_athb", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, t_running_mean) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("t_running_mean"),
            );
            let supply_clo = s.flag("supply_clo");

            let py_clo = if supply_clo {
                clo.into_pyobject(py).unwrap().into_any()
            } else {
                PyBool::new(py, false).to_owned().into_any()
            };
            let kwargs = [("clo", py_clo)].into_py_dict(py).unwrap();

            let py_result = models
                .getattr("pmv_athb")
                .unwrap()
                .call((tdb, tr, vr, rh, met, t_running_mean), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_athb(
                PmvAthbInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    t_running_mean: Temperature::from_celsius(t_running_mean),
                },
                PmvAthbOptions {
                    clo: supply_clo.then(|| ClothingInsulation::from_clo(clo)),
                },
            );

            compare_field(&field, rust, py_float(&py_result, "athb_pmv")?)
        });
    });
}

/// Read an optional category string from a Python result, tolerating the 0-d numpy
/// object arrays the 4.x retyping produces.
fn py_category(obj: &Bound<'_, PyAny>, field: &str) -> Result<Option<String>, String> {
    let attr = obj
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python result: {e}"))?;
    if attr.is_none() {
        return Ok(None);
    }
    if let Ok(s) = attr.extract::<String>() {
        return Ok(Some(s));
    }
    // Out of applicability range Python sets the category to NaN, where the Rust port
    // uses `None`; an enum cannot hold a NaN, so these are the same "not applicable".
    if attr.extract::<f64>().is_ok_and(f64::is_nan) {
        return Ok(None);
    }
    attr.call_method0("item")
        .and_then(|i| i.extract::<String>())
        .map(Some)
        .map_err(|e| format!("{field}: could not read as a string: {e}"))
}

/// Compare a Rust category against Python's, both optional.
fn compare_category(
    name: &str,
    rust: Option<&'static str>,
    py: Option<String>,
) -> Result<(), String> {
    if rust.map(str::to_string) == py {
        Ok(())
    } else {
        Err(format!("{name}: Rust {rust:?}, Python {py:?}"))
    }
}

/// tdb/rh space shared by the humidity-driven indices.
fn tdb_rh_domain() -> Domain {
    Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0)
        .flag("round_output")
}

#[test]
fn sweep_heat_index_lu() {
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("hi", 1e-9);

        run_sweep("sweep_heat_index_lu", &domain, |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("heat_index_lu")
                .unwrap()
                .call((tdb, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = heat_index_lu(
                HeatIndexLuInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                HeatIndexLuOptions { round_output },
            );

            compare_field(&field, rust.hi, py_float(&py_result, "hi")?)?;
            compare_category(
                "stress_category",
                rust.stress_category.map(|c| c.as_str()),
                py_category(&py_result, "stress_category")?,
            )
        });
    });
}

#[test]
fn sweep_heat_index_rothfusz() {
    let domain = tdb_rh_domain().flag("limit_inputs");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("hi", 1e-9);

        run_sweep("sweep_heat_index_rothfusz", &domain, |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");
            let limit_inputs = s.flag("limit_inputs");

            let kwargs = [
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("heat_index_rothfusz")
                .unwrap()
                .call((tdb, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = heat_index_rothfusz(
                HeatIndexRothfuszInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                HeatIndexRothfuszOptions {
                    round_output,
                    limit_inputs,
                },
            );

            compare_field(&field, rust.hi, py_float(&py_result, "hi")?)?;
            compare_category(
                "stress_category",
                rust.stress_category.map(|c| c.as_str()),
                py_category(&py_result, "stress_category")?,
            )
        });
    });
}

#[test]
fn sweep_heat_index_schoen() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("hi", 1e-9);

        run_sweep("sweep_heat_index_schoen", &tdb_rh_domain(), |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("heat_index_schoen")
                .unwrap()
                .call((tdb, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = heat_index_schoen(
                HeatIndexSchoenInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                HeatIndexSchoenOptions { round_output },
            );

            compare_field(&field, rust.hi, py_float(&py_result, "hi")?)?;
            compare_category(
                "stress_category",
                rust.stress_category.map(|c| c.as_str()),
                py_category(&py_result, "stress_category")?,
            )
        });
    });
}

#[test]
fn sweep_humidex() {
    // `model` is an enumerated axis so both the Rana (upstream default) and Masterson
    // vapor-pressure formulations stay covered, rather than dropping one of them: they
    // are two branches of one Rust function (`HumidexOptions::model`), not the separate
    // `humidex`/`humidex_masterson` functions this used to sweep independently.
    let domain = tdb_rh_domain().enumerated("model", 2);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("humidex", 1e-9);

        run_sweep("sweep_humidex", &domain, |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");
            let (model, py_model) = match s.index("model") {
                0 => (HumidexModel::Rana, "rana"),
                _ => (HumidexModel::Masterson, "masterson"),
            };

            let kwargs = [
                ("model", py_model.into_pyobject(py).unwrap().into_any()),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("humidex")
                .unwrap()
                .call((tdb, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = humidex(
                HumidexInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                HumidexOptions {
                    model,
                    round_output,
                },
            );

            compare_field(&field, rust.humidex, py_float(&py_result, "humidex")?)?;
            compare_category(
                "discomfort",
                Some(rust.discomfort.as_str()),
                py_category(&py_result, "discomfort")?,
            )
        });
    });
}

#[test]
fn sweep_thi() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("thi", 1e-9);

        run_sweep("sweep_thi", &tdb_rh_domain(), |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("thi")
                .unwrap()
                .call((tdb, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = thi(
                ThiInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                ThiOptions { round_output },
            );

            compare_field(&field, rust, py_float(&py_result, "thi")?)
        });
    });
}

#[test]
fn sweep_discomfort_index() {
    // Python exposes no round_output here, so neither does the Rust port.
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("di", 1e-9);

        run_sweep("sweep_discomfort_index", &domain, |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));

            let py_result = models
                .getattr("discomfort_index")
                .unwrap()
                .call1((tdb, rh))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = discomfort_index(DiscomfortIndexInputs {
                tdb: Temperature::from_celsius(tdb),
                rh: Humidity::from_percent(rh),
            });

            compare_field(&field, rust.di, py_float(&py_result, "di")?)?;
            compare_category(
                "discomfort_condition",
                Some(rust.discomfort_condition.as_str()),
                py_category(&py_result, "discomfort_condition")?,
            )
        });
    });
}

#[test]
fn sweep_wci() {
    let domain = Domain::new()
        .real("tdb", -40.0, 20.0)
        .real("v", 0.0, 30.0)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("wci", 1e-9);

        run_sweep("sweep_wci", &domain, |s: &Sample| {
            let (tdb, v) = (s.real("tdb"), s.real("v"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("wci")
                .unwrap()
                .call((tdb, v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = wci(
                WciInputs {
                    tdb: Temperature::from_celsius(tdb),
                    v: Speed::from_meters_per_second(v),
                },
                WciOptions { round_output },
            );

            compare_field(&field, rust, py_float(&py_result, "wci")?)
        });
    });
}

#[test]
fn sweep_wind_chill_temperature() {
    let domain = Domain::new()
        .real("tdb", -40.0, 20.0)
        .real("v", 0.0, 30.0)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("wct", 1e-9);

        run_sweep("sweep_wind_chill_temperature", &domain, |s: &Sample| {
            let (tdb, v) = (s.real("tdb"), s.real("v"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("wind_chill_temperature")
                .unwrap()
                .call((tdb, v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            // Python documents this input as km/h, unlike `wci` next door which is m/s.
            let rust = wind_chill_temperature(
                WindChillTemperatureInputs {
                    tdb: Temperature::from_celsius(tdb),
                    v: Speed::from_kilometers_per_hour(v),
                },
                WindChillTemperatureOptions { round_output },
            );

            compare_field(&field, rust, py_float(&py_result, "wct")?)
        });
    });
}

#[test]
fn sweep_net() {
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0)
        .real("v", 0.0, 20.0)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("net", 1e-9);

        run_sweep("sweep_net", &domain, |s: &Sample| {
            let (tdb, rh, v) = (s.real("tdb"), s.real("rh"), s.real("v"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("net")
                .unwrap()
                .call((tdb, rh, v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = net(
                NetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                    v: Speed::from_meters_per_second(v),
                },
                NetOptions { round_output },
            );

            compare_field(&field, rust, py_float(&py_result, "net")?)
        });
    });
}

#[test]
fn sweep_at() {
    // `q` (net radiation) is optional on both sides; the flag exercises both branches.
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0)
        .real("v", 0.0, 20.0)
        .real("q", 0.0, 800.0)
        .flag("supply_q")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("at", 1e-9);

        run_sweep("sweep_at", &domain, |s: &Sample| {
            let (tdb, rh, v, q) = (s.real("tdb"), s.real("rh"), s.real("v"), s.real("q"));
            let supply_q = s.flag("supply_q");
            let round_output = s.flag("round_output");

            let py_q = if supply_q {
                q.into_pyobject(py).unwrap().into_any()
            } else {
                py.None().into_bound(py)
            };
            let kwargs = [
                ("q", py_q),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("at")
                .unwrap()
                .call((tdb, rh, v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = at(
                AtInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                    v: Speed::from_meters_per_second(v),
                },
                AtOptions {
                    q: supply_q.then_some(q),
                    round_output,
                },
            );

            compare_field(&field, rust, py_float(&py_result, "at")?)
        });
    });
}

#[test]
fn sweep_esi() {
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0)
        .real("sol_radiation_global", 0.0, 1200.0)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("esi", 1e-9);

        run_sweep("sweep_esi", &domain, |s: &Sample| {
            let (tdb, rh, sol) = (s.real("tdb"), s.real("rh"), s.real("sol_radiation_global"));
            let round_output = s.flag("round_output");

            let kwargs = [(
                "round_output",
                PyBool::new(py, round_output).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("esi")
                .unwrap()
                .call((tdb, rh, sol), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = esi(
                EsiInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                    sol_radiation_global: sol,
                },
                EsiOptions { round_output },
            );

            compare_field(&field, rust, py_float(&py_result, "esi")?)
        });
    });
}

#[test]
fn sweep_wbgt() {
    // `with_solar_load` requires tdb, so the flag drives both the option and whether the
    // dry-bulb argument is supplied at all.
    let domain = Domain::new()
        .real("twb", -10.0, 40.0)
        .real("tg", -10.0, 60.0)
        .real("tdb", -10.0, 55.0)
        .flag("with_solar_load")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("wbgt", 1e-9);

        run_sweep("sweep_wbgt", &domain, |s: &Sample| {
            let (twb, tg, tdb) = (s.real("twb"), s.real("tg"), s.real("tdb"));
            let with_solar_load = s.flag("with_solar_load");
            let round_output = s.flag("round_output");

            let py_tdb = if with_solar_load {
                tdb.into_pyobject(py).unwrap().into_any()
            } else {
                py.None().into_bound(py)
            };
            let kwargs = [
                ("tdb", py_tdb),
                (
                    "with_solar_load",
                    PyBool::new(py, with_solar_load).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("wbgt")
                .unwrap()
                .call((twb, tg), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = wbgt(
                WbgtInputs {
                    twb: Temperature::from_celsius(twb),
                    tg: Temperature::from_celsius(tg),
                },
                WbgtOptions {
                    tdb: with_solar_load.then(|| Temperature::from_celsius(tdb)),
                    with_solar_load,
                    round_output,
                },
            );

            compare_field(&field, rust, py_float(&py_result, "wbgt")?)
        });
    });
}

#[test]
fn sweep_pet_steady() {
    // PET is the one model with a documented no_std deviation, so the sweep runs the
    // full option space: age, sex, height, weight, atmospheric pressure, external work
    // and posture are all axes, and none of them is varied by a hand-written case.
    let domain = Domain::new()
        .real("tdb", -10.0, 45.0)
        .real("tr", -10.0, 60.0)
        .real("v", 0.0, 5.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.8, 4.0)
        // Floored at 0.02, not 0: Hoeppe's clothing-area polynomial
        // (173.51*clo - 2.36 - 100.76*clo^2 + 19.28*clo^3)/100 is *negative* below
        // clo = 0.0136, so the 3-node system there describes a body with negative
        // clothed area and the two solvers legitimately settle on different points of
        // it. That band is degenerate in pythermalcomfort too, not a Rust defect; it is
        // excluded deliberately rather than absorbed into a wider tolerance.
        .real("clo", 0.02, 2.0)
        .real("p_atm", 80_000.0, 105_000.0)
        .real("age", 18.0, 85.0)
        .real("weight", 45.0, 120.0)
        .real("height", 1.4, 2.1)
        // Both implementations use this as a work-efficiency fraction, `h = he*(1-wme)`,
        // so values above 1 make the metabolic source term negative and the two solvers
        // settle on different points of a system with no physical solution. Bounded to
        // [0, 1] to match every other model's wme axis.
        .real("wme", 0.0, 1.0)
        .enumerated("position", 2)
        .enumerated("sex", 2)
        // `PetOptions::forced_convection` exposes Python's third position,
        // "standing, forced convection" (hc = 8.6 * v**0.513), which was previously
        // unreachable from Rust. Only takes effect combined with Standing; see the
        // per-sample match below for how Sitting stays a no-op.
        .flag("forced_convection");

    Python::with_gil(|py| {
        // Force the version guard before the shim imports pythermalcomfort itself.
        import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        // scipy's fsolve does not always converge on PET's 3-node system; when it gives
        // up it still returns its last iterate, and pythermalcomfort passes that
        // straight through. Those samples have no trustworthy reference value, so the
        // shim reports whether a RuntimeWarning fired and the sweep skips them rather
        // than comparing against a number scipy itself disowns.
        let shim = PyModule::from_code(
            py,
            c_str!(
                r#"
import warnings
from pythermalcomfort.models import pet_steady

def call(args, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pet_steady(*args, **kwargs)
        converged = not any(
            issubclass(w.category, RuntimeWarning) for w in caught
        )
    return float(result.pet), converged
"#
            ),
            c_str!("pet_shim.py"),
            c_str!("pet_shim"),
        )
        .expect("failed to build the PET convergence shim");

        let skipped = std::cell::Cell::new(0usize);
        let unconverged = std::cell::Cell::new(0usize);

        run_sweep("sweep_pet_steady", &domain, |s: &Sample| {
            let (tdb, tr, v, rh, met, clo, p_atm, age, weight, height, wme) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("v"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("p_atm"),
                s.real("age"),
                s.real("weight"),
                s.real("height"),
                s.real("wme"),
            );
            let forced_convection = s.flag("forced_convection");
            // Python has no "sitting, forced convection" position; the Rust flag is a
            // documented no-op for Sitting, so only Standing selects the third string.
            let (posture, py_position) = match (s.index("position"), forced_convection) {
                (0, _) => (PetPosture::Sitting, "sitting"),
                (_, true) => (PetPosture::Standing, "standing, forced convection"),
                (_, false) => (PetPosture::Standing, "standing"),
            };
            let (sex, py_sex) = match s.index("sex") {
                0 => (Sex::Male, "male"),
                _ => (Sex::Female, "female"),
            };

            // Python takes atmospheric pressure in hPa here, not Pa.
            let kwargs = [
                (
                    "p_atm",
                    (p_atm / 100.0).into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "position",
                    py_position.into_pyobject(py).unwrap().into_any(),
                ),
                ("age", age.into_pyobject(py).unwrap().into_any()),
                ("sex", py_sex.into_pyobject(py).unwrap().into_any()),
                ("weight", weight.into_pyobject(py).unwrap().into_any()),
                ("height", height.into_pyobject(py).unwrap().into_any()),
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();

            let (py_pet, converged): (f64, bool) = shim
                .getattr("call")
                .unwrap()
                .call1(((tdb, tr, v, rh, met, clo), kwargs))
                .map_err(|e| format!("python raised: {e}"))?
                .extract()
                .map_err(|e| format!("shim returned an unexpected shape: {e}"))?;

            if !converged {
                skipped.set(skipped.get() + 1);
                return Ok(());
            }

            let rust = pet_steady(
                PetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                PetOptions {
                    age,
                    sex,
                    height: Length::from_meters(height),
                    weight: Mass::from_kilograms(weight),
                    p_atm: Pressure::from_pascals(p_atm),
                    wme: WorkEfficiency::new(wme)
                        .expect("the wme axis is bounded to the valid [0, 1] range"),
                    position: posture,
                    forced_convection,
                },
            );

            if rust.pet.is_nan() {
                unconverged.set(unconverged.get() + 1);
            }

            // `RustMayBeNan` is confined to this one documented case: PET's 3-node
            // Newton demands a 1e-5 residual, while scipy's fsolve stops on step size
            // and accepts points whose balance is still ~0.3 W/m2 out. Where no root
            // meets the stricter bar the Rust solver reports NaN rather than returning
            // a wrong number. Python-NaN against a Rust number is still a failure.
            //
            // Python always rounds to two decimals and Rust now does the same
            // unconditionally, so the two can straddle a rounding boundary and differ by
            // exactly one step: 0.0101 and no more. Rust used to expose a round_output
            // flag, which upstream has no equivalent for (pet_steady.py:474 rounds
            // inside its return), so the unrounded half of this sweep was comparing
            // against a value pythermalcomfort cannot produce. Removing the flag costs
            // the 0.0051 bound that half provided -- a fair price for comparing only
            // states upstream can actually reach.
            let tol = 0.0101;
            let field = FieldCmp::new(
                "pet",
                if std::env::var("PET_MEASURE").is_ok() {
                    1e9
                } else {
                    tol
                },
            )
            .nan(NanPolicy::RustMayBeNan);
            if std::env::var("PET_MEASURE").is_ok()
                && !rust.pet.is_nan()
                && (rust.pet - py_pet).abs() > 0.0051
            {
                unconverged.set(unconverged.get());
                eprintln!(
                    "DELTA {:.4} clo={:.3} tdb={:.1} rh={:.1} met={:.2} v={:.2}",
                    (rust.pet - py_pet).abs(),
                    s.real("clo"),
                    tdb,
                    rh,
                    met,
                    v
                );
            }
            compare_field(&field, rust.pet, py_pet)
        });

        // Not a silent cap: report the excluded share so it cannot drift upward unseen.
        eprintln!(
            "sweep_pet_steady: skipped {} sample(s) where scipy's fsolve did not \
             converge; {} sample(s) where the Rust 3-node solver could not reach its \
             own tolerance and returned NaN",
            skipped.get(),
            unconverged.get()
        );
    });
}

/// WBGT / metabolic-power space shared by the ISO and NIOSH work-capacity models.
fn work_capacity_met_domain() -> Domain {
    Domain::new()
        .real("wbgt", 5.0, 50.0)
        .real("met", 100.0, 600.0)
}

/// WBGT / work-intensity space shared by the Dunne and Hothaps models.
fn work_capacity_intensity_domain() -> Domain {
    Domain::new()
        .real("wbgt", 5.0, 50.0)
        .enumerated("work_intensity", 3)
}

fn work_intensity_at(index: usize) -> (WorkIntensity, &'static str) {
    match index {
        0 => (WorkIntensity::Heavy, "heavy"),
        1 => (WorkIntensity::Moderate, "moderate"),
        _ => (WorkIntensity::Light, "light"),
    }
}

#[test]
fn sweep_work_capacity_iso() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("capacity", 1e-9);

        run_sweep(
            "sweep_work_capacity_iso",
            &work_capacity_met_domain(),
            |s: &Sample| {
                let (wbgt_v, met) = (s.real("wbgt"), s.real("met"));
                let py_result = models
                    .getattr("work_capacity_iso")
                    .unwrap()
                    .call1((wbgt_v, met))
                    .map_err(|e| format!("python raised: {e}"))?;
                let rust =
                    work_capacity_iso(Temperature::from_celsius(wbgt_v), Power::from_watts(met));
                compare_field(&field, rust, py_float(&py_result, "capacity")?)
            },
        );
    });
}

#[test]
fn sweep_work_capacity_niosh() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("capacity", 1e-9);

        run_sweep(
            "sweep_work_capacity_niosh",
            &work_capacity_met_domain(),
            |s: &Sample| {
                let (wbgt_v, met) = (s.real("wbgt"), s.real("met"));
                let py_result = models
                    .getattr("work_capacity_niosh")
                    .unwrap()
                    .call1((wbgt_v, met))
                    .map_err(|e| format!("python raised: {e}"))?;
                let rust =
                    work_capacity_niosh(Temperature::from_celsius(wbgt_v), Power::from_watts(met));
                compare_field(&field, rust, py_float(&py_result, "capacity")?)
            },
        );
    });
}

#[test]
fn sweep_work_capacity_dunne() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("capacity", 1e-9);

        run_sweep(
            "sweep_work_capacity_dunne",
            &work_capacity_intensity_domain(),
            |s: &Sample| {
                let wbgt_v = s.real("wbgt");
                let (intensity, py_intensity) = work_intensity_at(s.index("work_intensity"));
                let kwargs = [(
                    "work_intensity",
                    py_intensity.into_pyobject(py).unwrap().into_any(),
                )]
                .into_py_dict(py)
                .unwrap();
                let py_result = models
                    .getattr("work_capacity_dunne")
                    .unwrap()
                    .call((wbgt_v,), Some(&kwargs))
                    .map_err(|e| format!("python raised: {e}"))?;
                let rust = work_capacity_dunne(
                    Temperature::from_celsius(wbgt_v),
                    WorkCapacityIntensityOptions {
                        work_intensity: intensity,
                    },
                );
                compare_field(&field, rust, py_float(&py_result, "capacity")?)
            },
        );
    });
}

#[test]
fn sweep_work_capacity_hothaps() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("capacity", 1e-9);

        run_sweep(
            "sweep_work_capacity_hothaps",
            &work_capacity_intensity_domain(),
            |s: &Sample| {
                let wbgt_v = s.real("wbgt");
                let (intensity, py_intensity) = work_intensity_at(s.index("work_intensity"));
                let kwargs = [(
                    "work_intensity",
                    py_intensity.into_pyobject(py).unwrap().into_any(),
                )]
                .into_py_dict(py)
                .unwrap();
                let py_result = models
                    .getattr("work_capacity_hothaps")
                    .unwrap()
                    .call((wbgt_v,), Some(&kwargs))
                    .map_err(|e| format!("python raised: {e}"))?;
                let rust = work_capacity_hothaps(
                    Temperature::from_celsius(wbgt_v),
                    WorkCapacityIntensityOptions {
                        work_intensity: intensity,
                    },
                );
                compare_field(&field, rust, py_float(&py_result, "capacity")?)
            },
        );
    });
}

/// Compare a Rust bool against Python's acceptability, which arrives as a numpy bool or
/// as a 0.0/1.0 float depending on the model.
fn compare_bool(name: &str, rust: bool, obj: &Bound<'_, PyAny>) -> Result<(), String> {
    let attr = obj
        .getattr(name)
        .map_err(|e| format!("{name}: missing on Python result: {e}"))?;
    let py: bool = if let Ok(b) = attr.extract::<bool>() {
        b
    } else {
        let value = attr
            .extract::<f64>()
            .or_else(|_| attr.call_method0("item").and_then(|i| i.extract::<f64>()))
            .map_err(|e| format!("{name}: could not read as a bool: {e}"))?;
        // Outside its applicability range Python sets acceptability to NaN. A Rust bool
        // cannot hold that, and the port spells "not applicable" as false alongside a
        // NaN result value, so NaN maps to false rather than to a truthy non-zero.
        if value.is_nan() { false } else { value != 0.0 }
    };
    if rust == py {
        Ok(())
    } else {
        Err(format!("{name}: Rust {rust}, Python {py}"))
    }
}

/// Input space shared by the two adaptive models.
fn adaptive_domain() -> Domain {
    Domain::new()
        .real("tdb", 5.0, 45.0)
        .real("tr", 5.0, 45.0)
        .real("t_running_mean", -5.0, 40.0)
        .real("v", 0.0, 2.0)
        .flag("limit_inputs")
        .flag("round_output")
        .enumerated("units", 2)
}

#[test]
fn sweep_adaptive_ashrae() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let tol_si = 1e-9;
        // A genuine SI->IP conversion of the result (adaptive_ashrae.py:120-130), so an
        // SI-scale tolerance becomes 9/5 as large once expressed in Fahrenheit.
        let tol_ip = tol_si * 9.0 / 5.0;
        let field_names = [
            "tmp_cmf",
            "tmp_cmf_80_low",
            "tmp_cmf_80_up",
            "tmp_cmf_90_low",
            "tmp_cmf_90_up",
        ];

        run_sweep("sweep_adaptive_ashrae", &adaptive_domain(), |s: &Sample| {
            let (tdb, tr, trm, v) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("t_running_mean"),
                s.real("v"),
            );
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");
            let (units, py_units) = match s.index("units") {
                0 => (Units::SI, "SI"),
                _ => (Units::IP, "IP"),
            };

            let kwargs = [
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
                ("units", py_units.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();

            // Python's `units="IP"` reinterprets its raw floats as Fahrenheit/fps and
            // converts them to SI before doing anything else (see `sweep_cooling_effect`,
            // which solved this the same way): feed it the IP equivalents of the same SI
            // environment Rust uses.
            let (py_tdb, py_tr, py_trm, py_v) = if units == Units::IP {
                (
                    tdb * 9.0 / 5.0 + 32.0,
                    tr * 9.0 / 5.0 + 32.0,
                    trm * 9.0 / 5.0 + 32.0,
                    v * 3.281,
                )
            } else {
                (tdb, tr, trm, v)
            };

            let py_result = models
                .getattr("adaptive_ashrae")
                .unwrap()
                .call((py_tdb, py_tr, py_trm, py_v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = adaptive_ashrae(
                AdaptiveInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    t_running_mean: Temperature::from_celsius(trm),
                    v: Speed::from_meters_per_second(v),
                },
                AdaptiveOptions {
                    limit_inputs,
                    round_output,
                    units,
                },
            );

            let values = [
                rust.tmp_cmf,
                rust.tmp_cmf_80_low,
                rust.tmp_cmf_80_up,
                rust.tmp_cmf_90_low,
                rust.tmp_cmf_90_up,
            ];
            let tol = match units {
                Units::SI => tol_si,
                Units::IP => tol_ip,
            };
            for (name, rust_value) in field_names.iter().zip(values) {
                let name = *name;
                let rust_value = match units {
                    Units::SI => rust_value.as_celsius(),
                    Units::IP => rust_value.as_fahrenheit(),
                };
                compare_field(
                    &FieldCmp::new(name, tol),
                    rust_value,
                    py_float(&py_result, name)?,
                )?;
            }
            compare_bool("acceptability_80", rust.acceptability_80, &py_result)?;
            compare_bool("acceptability_90", rust.acceptability_90, &py_result)
        });
    });
}

#[test]
fn sweep_adaptive_en() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let tol_si = 1e-9;
        // A genuine SI->IP conversion of the result (adaptive_en.py:140-150), so an
        // SI-scale tolerance becomes 9/5 as large once expressed in Fahrenheit.
        let tol_ip = tol_si * 9.0 / 5.0;
        let field_names = [
            "tmp_cmf",
            "tmp_cmf_cat_i_low",
            "tmp_cmf_cat_i_up",
            "tmp_cmf_cat_ii_low",
            "tmp_cmf_cat_ii_up",
            "tmp_cmf_cat_iii_low",
            "tmp_cmf_cat_iii_up",
        ];

        run_sweep("sweep_adaptive_en", &adaptive_domain(), |s: &Sample| {
            let (tdb, tr, trm, v) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("t_running_mean"),
                s.real("v"),
            );
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");
            let (units, py_units) = match s.index("units") {
                0 => (Units::SI, "SI"),
                _ => (Units::IP, "IP"),
            };

            let kwargs = [
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
                ("units", py_units.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();

            // See sweep_adaptive_ashrae: Python's `units="IP"` reinterprets its raw
            // floats as Fahrenheit/fps, so feed it the IP equivalents of the same SI
            // environment Rust uses.
            let (py_tdb, py_tr, py_trm, py_v) = if units == Units::IP {
                (
                    tdb * 9.0 / 5.0 + 32.0,
                    tr * 9.0 / 5.0 + 32.0,
                    trm * 9.0 / 5.0 + 32.0,
                    v * 3.281,
                )
            } else {
                (tdb, tr, trm, v)
            };

            let py_result = models
                .getattr("adaptive_en")
                .unwrap()
                .call((py_tdb, py_tr, py_trm, py_v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = adaptive_en(
                AdaptiveInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    t_running_mean: Temperature::from_celsius(trm),
                    v: Speed::from_meters_per_second(v),
                },
                AdaptiveOptions {
                    limit_inputs,
                    round_output,
                    units,
                },
            );

            let values = [
                rust.tmp_cmf,
                rust.tmp_cmf_cat_i_low,
                rust.tmp_cmf_cat_i_up,
                rust.tmp_cmf_cat_ii_low,
                rust.tmp_cmf_cat_ii_up,
                rust.tmp_cmf_cat_iii_low,
                rust.tmp_cmf_cat_iii_up,
            ];
            let tol = match units {
                Units::SI => tol_si,
                Units::IP => tol_ip,
            };
            for (name, rust_value) in field_names.iter().zip(values) {
                let name = *name;
                let rust_value = match units {
                    Units::SI => rust_value.as_celsius(),
                    Units::IP => rust_value.as_fahrenheit(),
                };
                compare_field(
                    &FieldCmp::new(name, tol),
                    rust_value,
                    py_float(&py_result, name)?,
                )?;
            }
            compare_bool("acceptability_cat_i", rust.acceptability_cat_i, &py_result)?;
            compare_bool(
                "acceptability_cat_ii",
                rust.acceptability_cat_ii,
                &py_result,
            )?;
            compare_bool(
                "acceptability_cat_iii",
                rust.acceptability_cat_iii,
                &py_result,
            )
        });
    });
}

#[test]
fn sweep_ankle_draft() {
    let domain = Domain::new()
        .real("tdb", 15.0, 35.0)
        .real("tr", 15.0, 35.0)
        // Python raises outright above 0.2 m/s ("only applicable for air speed lower
        // than 0.2 m/s"), so there is no reference value to compare against there.
        .real("vr", 0.0, 0.2)
        .real("rh", 0.0, 100.0)
        .real("met", 0.8, 2.0)
        .real("clo", 0.0, 1.5)
        .real("v_ankle", 0.0, 1.0)
        .flag("limit_inputs");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("ppd_ad", 1e-9);

        run_sweep("sweep_ankle_draft", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, v_ankle) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("v_ankle"),
            );
            let limit_inputs = s.flag("limit_inputs");

            let kwargs = [(
                "limit_inputs",
                PyBool::new(py, limit_inputs).to_owned().into_any(),
            )]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("ankle_draft")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo, v_ankle), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let (ppd_ad, acceptability) = ankle_draft(
                AnkleDraftInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    v_ankle: Speed::from_meters_per_second(v_ankle),
                },
                AnkleDraftOptions { limit_inputs },
            );

            compare_field(&field, ppd_ad, py_float(&py_result, "ppd_ad")?)?;
            compare_bool("acceptability", acceptability, &py_result)
        });
    });
}

#[test]
fn sweep_vertical_tmp_grad_ppd() {
    let domain = Domain::new()
        .real("tdb", 15.0, 35.0)
        .real("tr", 15.0, 35.0)
        .real("vr", 0.0, 1.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.8, 2.0)
        .real("clo", 0.0, 1.5)
        .real("vertical_tmp_grad", -2.0, 12.0)
        .flag("limit_inputs")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        run_sweep("sweep_vertical_tmp_grad_ppd", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, grad) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("vertical_tmp_grad"),
            );
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");

            let kwargs = [
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("vertical_tmp_grad_ppd")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo, grad), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let (ppd_vg, acceptability) = vertical_tmp_grad_ppd(
                VerticalTmpGradPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    vertical_tmp_grad: TemperatureDelta::from_celsius(grad),
                },
                VerticalTmpGradPpdOptions {
                    round_output,
                    limit_inputs,
                },
            );

            // round_output is now honoured (it used to be hardcoded to true here, so
            // the unrounded half of this sweep had to be compared at 0.051 -- wide
            // enough to hide any error smaller than the rounding it was compensating
            // for). Both halves are compared to full precision now.
            compare_field(
                &FieldCmp::new("ppd_vg", 1e-9),
                ppd_vg,
                py_float(&py_result, "ppd_vg")?,
            )?;
            compare_bool("acceptability", acceptability, &py_result)
        });
    });
}

#[test]
fn sweep_solar_gain() {
    // The angle axes deliberately run outside the table domain (altitude 0-90,
    // sharp 0-180) so the NaN guard is exercised on both sides rather than assumed.
    let domain = Domain::new()
        .real("sol_altitude", -20.0, 110.0)
        .real("sharp", -20.0, 200.0)
        .real("sol_radiation_dir", 0.0, 1200.0)
        .real("sol_transmittance", 0.0, 1.0)
        .real("f_svv", 0.0, 1.0)
        .real("f_bes", 0.0, 1.0)
        .real("asw", 0.0, 1.0)
        .real("floor_reflectance", 0.0, 1.0)
        .enumerated("posture", 3)
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        run_sweep("sweep_solar_gain", &domain, |s: &Sample| {
            let (alt, sharp, dir, trans, fsvv, fbes, asw, floor) = (
                s.real("sol_altitude"),
                s.real("sharp"),
                s.real("sol_radiation_dir"),
                s.real("sol_transmittance"),
                s.real("f_svv"),
                s.real("f_bes"),
                s.real("asw"),
                s.real("floor_reflectance"),
            );
            // Python accepts only these three for solar gain.
            let (posture, py_posture) = match s.index("posture") {
                0 => (Posture::Sitting, "sitting"),
                1 => (Posture::Standing, "standing"),
                _ => (Posture::Supine, "supine"),
            };
            let round_output = s.flag("round_output");

            let kwargs = [
                ("asw", asw.into_pyobject(py).unwrap().into_any()),
                ("posture", py_posture.into_pyobject(py).unwrap().into_any()),
                (
                    "floor_reflectance",
                    floor.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("solar_gain")
                .unwrap()
                .call((alt, sharp, dir, trans, fsvv, fbes), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = solar_gain(
                SolarGainInputs {
                    sol_altitude: Angle::from_degrees(alt),
                    sharp: Angle::from_degrees(sharp),
                    sol_radiation_dir: HeatFluxDensity::from_watts_per_square_meter(dir),
                    sol_transmittance: trans,
                    f_svv: fsvv,
                    f_bes: fbes,
                },
                SolarGainOptions {
                    asw,
                    posture,
                    floor_reflectance: floor,
                    round_output,
                },
            );

            // Both halves of the round_output sweep are compared to full precision now.
            // Previously Rust rounded unconditionally, so the unrounded half had to be
            // checked at 0.051 -- wide enough to hide any error smaller than the rounding
            // it was compensating for. Exposing it showed a residue of about 5e-11
            // relative, which is bilinear interpolation over the fp table amplifying
            // float associativity through the difference of two neighbouring entries,
            // not a formula gap: it stays at that scale across a 3000-sample deep run
            // instead of growing in any corner.
            for (name, rust_value) in [
                ("erf", rust.erf.as_watts_per_square_meter()),
                ("delta_mrt", rust.delta_mrt.as_celsius()),
            ] {
                compare_field(
                    &FieldCmp::new(name, 1e-9).rel(1e-9),
                    rust_value,
                    py_float(&py_result, name)?,
                )?;
            }
            Ok(())
        });
    });
}

/// Compare a duration-limited-exposure result. Python returns either a number of hours,
/// the string "more than 8", or NaN; the Rust port models those three as an enum.
fn compare_dle(
    field: &str,
    rust: DurationLimitedExposure,
    obj: &Bound<'_, PyAny>,
    tol: f64,
) -> Result<(), String> {
    let attr = obj
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python result: {e}"))?;

    if let Ok(text) = attr.extract::<String>() {
        return match (&rust, text.as_str()) {
            (DurationLimitedExposure::MoreThanEight, "more than 8") => Ok(()),
            (DurationLimitedExposure::NotApplicable, "nan") => Ok(()),
            _ => Err(format!("{field}: Rust {rust}, Python {text:?}")),
        };
    }

    let value: f64 = attr
        .extract()
        .or_else(|_| attr.call_method0("item").and_then(|i| i.extract()))
        .map_err(|e| format!("{field}: could not read: {e}"))?;

    match rust {
        DurationLimitedExposure::NotApplicable if value.is_nan() => Ok(()),
        DurationLimitedExposure::Hours(h) if (h - value).abs() <= tol => Ok(()),
        _ => Err(format!("{field}: Rust {rust}, Python {value}")),
    }
}

#[test]
fn sweep_ireq() {
    // `p` and `walk_sp` span both sides of the applicability mask, which no hand-written
    // case varies at all.
    let domain = Domain::new()
        .real("tdb", -50.0, 10.0)
        .real("tr", -50.0, 10.0)
        .real("vr", 0.0, 5.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.8, 4.0)
        .real("clo", 0.5, 4.0)
        .real("p", 5.0, 200.0)
        .real("walk_sp", 0.0, 1.5)
        .real("wme", 0.0, 1.0)
        .flag("limit_inputs")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        // 1e-6 absolute, not 1e-9: IREQ is solved iteratively and reaches into the
        // hundreds in severe cold, where the two implementations agree to about 7e-9
        // absolute - 2.7e-11 relative, i.e. float-representation noise rather than a
        // difference in the model.
        let fields = [
            FieldCmp::new("ireq_min", 1e-6),
            FieldCmp::new("ireq_neutral", 1e-6),
            FieldCmp::new("icl_min", 1e-6),
            FieldCmp::new("icl_neutral", 1e-6),
        ];

        run_sweep("sweep_ireq", &domain, |s: &Sample| {
            let (tdb, tr, vr, rh, met, clo, p, walk_sp, wme) = (
                s.real("tdb"),
                s.real("tr"),
                s.real("vr"),
                s.real("rh"),
                s.real("met"),
                s.real("clo"),
                s.real("p"),
                s.real("walk_sp"),
                s.real("wme"),
            );
            let limit_inputs = s.flag("limit_inputs");
            let round_output = s.flag("round_output");

            let kwargs = [
                ("wme", wme.into_pyobject(py).unwrap().into_any()),
                (
                    "limit_inputs",
                    PyBool::new(py, limit_inputs).to_owned().into_any(),
                ),
                (
                    "round_output",
                    PyBool::new(py, round_output).to_owned().into_any(),
                ),
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("ireq")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo, p, walk_sp), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = ireq(
                IreqInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    p: AirPermeability::from_l_per_m2_s(p),
                    walk_sp: Speed::from_meters_per_second(walk_sp),
                },
                IreqOptions {
                    wme: MetabolicRate::from_met(wme),
                    limit_inputs,
                    round_output,
                },
            );

            let values = [
                rust.ireq_min.as_clo(),
                rust.ireq_neutral.as_clo(),
                rust.icl_min.as_clo(),
                rust.icl_neutral.as_clo(),
            ];
            for (field, rust_value) in fields.iter().zip(values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
            }
            compare_dle("dle_min", rust.dle_min, &py_result, 1e-6)?;
            compare_dle("dle_neutral", rust.dle_neutral, &py_result, 1e-6)
        });
    });
}

/// The 34 sport presets, by the name both libraries use.
const SPORT_NAMES: [&str; 34] = [
    "ABSEILING",
    "ARCHERY",
    "AUSTRALIAN_FOOTBALL",
    "BASEBALL",
    "BASKETBALL",
    "BOWLS",
    "CANOEING",
    "CRICKET",
    "CROQUET",
    "CYCLING",
    "EQUESTRIAN",
    "FIELD_ATHLETICS",
    "FIELD_HOCKEY",
    "FISHING",
    "GOLF",
    "HORSEBACK",
    "KAYAKING",
    "RUNNING",
    "MTB",
    "NETBALL",
    "OZTAG",
    "PICKLEBALL",
    "CLIMBING",
    "ROWING",
    "RUGBY_LEAGUE",
    "RUGBY_UNION",
    "SAILING",
    "SHOOTING",
    "SOCCER",
    "SOFTBALL",
    "TENNIS",
    "TOUCH",
    "VOLLEYBALL",
    "WALKING",
];

fn sport_at(index: usize) -> SportsValues {
    match SPORT_NAMES[index] {
        "ABSEILING" => Sports::ABSEILING,
        "ARCHERY" => Sports::ARCHERY,
        "AUSTRALIAN_FOOTBALL" => Sports::AUSTRALIAN_FOOTBALL,
        "BASEBALL" => Sports::BASEBALL,
        "BASKETBALL" => Sports::BASKETBALL,
        "BOWLS" => Sports::BOWLS,
        "CANOEING" => Sports::CANOEING,
        "CRICKET" => Sports::CRICKET,
        "CROQUET" => Sports::CROQUET,
        "CYCLING" => Sports::CYCLING,
        "EQUESTRIAN" => Sports::EQUESTRIAN,
        "FIELD_ATHLETICS" => Sports::FIELD_ATHLETICS,
        "FIELD_HOCKEY" => Sports::FIELD_HOCKEY,
        "FISHING" => Sports::FISHING,
        "GOLF" => Sports::GOLF,
        "HORSEBACK" => Sports::HORSEBACK,
        "KAYAKING" => Sports::KAYAKING,
        "RUNNING" => Sports::RUNNING,
        "MTB" => Sports::MTB,
        "NETBALL" => Sports::NETBALL,
        "OZTAG" => Sports::OZTAG,
        "PICKLEBALL" => Sports::PICKLEBALL,
        "CLIMBING" => Sports::CLIMBING,
        "ROWING" => Sports::ROWING,
        "RUGBY_LEAGUE" => Sports::RUGBY_LEAGUE,
        "RUGBY_UNION" => Sports::RUGBY_UNION,
        "SAILING" => Sports::SAILING,
        "SHOOTING" => Sports::SHOOTING,
        "SOCCER" => Sports::SOCCER,
        "SOFTBALL" => Sports::SOFTBALL,
        "TENNIS" => Sports::TENNIS,
        "TOUCH" => Sports::TOUCH,
        "VOLLEYBALL" => Sports::VOLLEYBALL,
        "WALKING" => Sports::WALKING,
        other => unreachable!("unmapped sport {other}"),
    }
}

#[test]
fn sweep_sports_heat_stress_risk() {
    let domain = Domain::new()
        .real("tdb", 5.0, 50.0)
        .real("tr", 5.0, 60.0)
        .real("rh", 0.0, 100.0)
        .real("vr", 0.0, 5.0)
        .enumerated("sport", SPORT_NAMES.len());

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let sports_mod = import_reference(py, "pythermalcomfort.models.sports_heat_stress_risk")
            .expect("failed to import the sports preset namespace");
        let sports = sports_mod.getattr("Sports").unwrap();
        let fields = [
            FieldCmp::new("risk_level_interpolated", 1e-9),
            FieldCmp::new("t_medium", 1e-9),
            FieldCmp::new("t_high", 1e-9),
            FieldCmp::new("t_extreme", 1e-9),
        ];

        run_sweep("sweep_sports_heat_stress_risk", &domain, |s: &Sample| {
            let (tdb, tr, rh, vr) = (s.real("tdb"), s.real("tr"), s.real("rh"), s.real("vr"));
            let index = s.index("sport");
            let name = SPORT_NAMES[index];

            let py_sport = sports
                .getattr(name)
                .map_err(|e| format!("Python has no sport preset {name}: {e}"))?;

            let py_call = models
                .getattr("sports_heat_stress_risk")
                .unwrap()
                .call1((tdb, tr, rh, vr, py_sport));

            let rust = sports_heat_stress_risk(SportsHeatStressRiskInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                rh: Humidity::from_percent(rh),
                vr: Speed::from_meters_per_second(vr),
                sport: sport_at(index),
            });

            let py_result = match py_call {
                Ok(result) => result,
                // `_calc_risk_single_value` raises when `risk_level_interpolated` never
                // leaves its NaN seed (`sports_heat_stress_risk.py:372-373`), mirrored by
                // `SportsHeatStressRiskError::NanRiskLevel`. Not merely skipped the way
                // `sweep_two_nodes_gagge_sleep` skips upstream's OverflowError: since Rust
                // can now fail the same way, a Python raise is only in agreement if Rust
                // also rejected the sample, and it is a real divergence if Rust produced
                // an answer where Python declined to.
                Err(e) if e.is_instance_of::<PyValueError>(py) => {
                    return match rust {
                        Err(_) => Ok(()),
                        Ok(ok) => Err(format!(
                            "{name}: python raised ValueError but rust returned {ok:?}"
                        )),
                    };
                }
                Err(e) => return Err(format!("python raised: {e}")),
            };

            let rust =
                rust.map_err(|e| format!("{name}: rust rejected a sample python accepted: {e}"))?;

            let values = [
                rust.risk_level_interpolated,
                rust.t_medium.as_celsius(),
                rust.t_high.as_celsius(),
                rust.t_extreme.as_celsius(),
            ];
            for (field, rust_value) in fields.iter().zip(values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)
                    .map_err(|e| format!("{name}: {e}"))?;
            }
            compare_category(
                "recommendation",
                Some(rust.recommendation),
                py_category(&py_result, "recommendation")?,
            )
            .map_err(|e| format!("{name}: {e}"))
        });
    });
}

#[test]
fn sweep_ridge_regression_predict_t_re_t_sk() {
    // Python's applicability window is age 60-100; the sweep straddles it so the
    // limit_inputs mask is exercised in both states.
    let domain = Domain::new()
        .real("age", 50.0, 105.0)
        .real("height", 1.4, 2.1)
        .real("weight", 45.0, 120.0)
        .real("tdb", 15.0, 50.0)
        .real("rh", 0.0, 100.0)
        .real("t_re", 36.0, 39.0)
        .real("t_sk", 30.0, 38.0)
        .enumerated("duration", 4)
        .enumerated("sex", 2)
        .flag("supply_initials")
        .flag("limit_inputs")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        run_sweep(
            "sweep_ridge_regression_predict_t_re_t_sk",
            &domain,
            |s: &Sample| {
                let (age, height, weight, tdb, rh, t_re0, t_sk0) = (
                    s.real("age"),
                    s.real("height"),
                    s.real("weight"),
                    s.real("tdb"),
                    s.real("rh"),
                    s.real("t_re"),
                    s.real("t_sk"),
                );
                let duration = [15_usize, 30, 60, 120][s.index("duration")];
                let (sex, py_sex) = match s.index("sex") {
                    0 => (Sex::Male, "male"),
                    _ => (Sex::Female, "female"),
                };
                let supply_initials = s.flag("supply_initials");
                let limit_inputs = s.flag("limit_inputs");
                let round_output = s.flag("round_output");

                let (py_t_re, py_t_sk) = if supply_initials {
                    (
                        t_re0.into_pyobject(py).unwrap().into_any(),
                        t_sk0.into_pyobject(py).unwrap().into_any(),
                    )
                } else {
                    (py.None().into_bound(py), py.None().into_bound(py))
                };

                let kwargs = [
                    ("t_re", py_t_re),
                    ("t_sk", py_t_sk),
                    (
                        "limit_inputs",
                        PyBool::new(py, limit_inputs).to_owned().into_any(),
                    ),
                    (
                        "round_output",
                        PyBool::new(py, round_output).to_owned().into_any(),
                    ),
                ]
                .into_py_dict(py)
                .unwrap();

                let py_result = models
                    .getattr("ridge_regression_predict_t_re_t_sk")
                    .unwrap()
                    .call(
                        (py_sex, age, height, weight, tdb, rh, duration),
                        Some(&kwargs),
                    )
                    .map_err(|e| format!("python raised: {e}"))?;

                let rust = ridge_regression_predict_t_re_t_sk(
                    RidgeRegressionInputs {
                        sex,
                        age,
                        height: Length::from_meters(height),
                        weight: Mass::from_kilograms(weight),
                        tdb: Temperature::from_celsius(tdb),
                        rh: Humidity::from_percent(rh),
                        duration,
                    },
                    RidgeRegressionOptions {
                        t_re: supply_initials.then(|| Temperature::from_celsius(t_re0)),
                        t_sk: supply_initials.then(|| Temperature::from_celsius(t_sk0)),
                        limit_inputs,
                        round_output,
                    },
                );

                for (name, rust_series) in [("t_re", &rust.t_re), ("t_sk", &rust.t_sk)] {
                    let py_series = py_float_seq(&py_result, name)?;
                    if py_series.len() != rust_series.len() {
                        return Err(format!(
                            "{name}: Rust returned {} minutes, Python {}",
                            rust_series.len(),
                            py_series.len()
                        ));
                    }
                    let cmp = FieldCmp::new(name, 1e-9);
                    for (minute, (rust_value, py_value)) in
                        rust_series.iter().zip(py_series.iter()).enumerate()
                    {
                        compare_field(&cmp, rust_value.as_celsius(), *py_value)
                            .map_err(|e| format!("minute {minute}: {e}"))?;
                    }
                }
                Ok(())
            },
        );
    });
}

/// Call a `pythermalcomfort.utilities` function and read its scalar result.
fn py_util(
    utils: &Bound<'_, PyAny>,
    name: &str,
    args: impl for<'p> IntoPyObject<'p, Target = PyTuple>,
) -> Result<f64, String> {
    let value = utils
        .getattr(name)
        .map_err(|e| format!("{name}: missing from utilities: {e}"))?
        .call1(args)
        .map_err(|e| format!("{name} raised: {e}"))?;
    value
        .extract::<f64>()
        .or_else(|_| value.call_method0("item").and_then(|i| i.extract::<f64>()))
        .map_err(|e| format!("{name}: could not read as a number: {e}"))
}

#[test]
fn sweep_saturation_pressures() {
    // Three formulations that all return a saturation pressure but in three different
    // units upstream: p_sat in Pa, p_sat_torr in torr, antoine in kPa. The Rust port
    // normalises the first two to a `Pressure`, so the conversions belong in the test.
    let domain = Domain::new().real("tdb", -40.0, 90.0);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_saturation_pressures", &domain, |s: &Sample| {
            let tdb = s.real("tdb");
            let t = Temperature::from_celsius(tdb);

            compare_field(
                &FieldCmp::new("p_sat", 1e-6),
                p_sat(t).as_pascals(),
                py_util(&utils, "p_sat", (tdb,))?,
            )?;

            // Python returns torr; the port carries the equivalent `Pressure`.
            compare_field(
                &FieldCmp::new("p_sat_torr", 1e-9),
                p_sat_torr(t).as_pascals() / 133.322,
                py_util(&utils, "p_sat_torr", (tdb,))?,
            )?;

            // Python's `antoine` is kPa. The port exposes it both as a bare kPa float
            // and as `p_sat_antoine`, a `Pressure`; both are checked against it.
            let py_antoine = py_util(&utils, "antoine", (tdb,))?;
            compare_field(&FieldCmp::new("antoine", 1e-9), antoine(t), py_antoine)?;
            compare_field(
                &FieldCmp::new("p_sat_antoine", 1e-6),
                p_sat_antoine(t).as_pascals(),
                py_antoine * 1000.0,
            )
        });
    });
}

#[test]
fn sweep_v_relative_and_clo_dynamic() {
    // `ashrae_model` / `iso9920_model` each have only one legal value today (see
    // `Ashrae55Model`, `Iso9920Model`); they are drawn anyway so the axes exist for a
    // future edition, and matched exhaustively in the Rust call so a new variant would
    // be a compile error here rather than a silent gap.
    let domain = Domain::new()
        .real("v", 0.0, 4.0)
        .real("met", 0.6, 5.0)
        .real("clo", 0.0, 3.0)
        .real("i_a", 0.0, 1.5)
        .enumerated("ashrae_model", 1)
        .enumerated("iso9920_model", 1);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_v_relative_and_clo_dynamic", &domain, |s: &Sample| {
            let (v, met, clo, i_a) = (s.real("v"), s.real("met"), s.real("clo"), s.real("i_a"));
            let _ = s.index("ashrae_model");
            let _ = s.index("iso9920_model");
            let ashrae_model = Ashrae55Model::Ashrae552023;
            let iso9920_model = Iso9920Model::Iso99202007;

            compare_field(
                &FieldCmp::new("v_relative", 1e-9),
                v_relative(
                    Speed::from_meters_per_second(v),
                    MetabolicRate::from_met(met),
                )
                .as_meters_per_second(),
                py_util(&utils, "v_relative", (v, met))?,
            )?;

            compare_field(
                &FieldCmp::new("clo_dynamic_ashrae", 1e-9),
                clo_dynamic_ashrae(
                    CloDynamicAshraeInputs {
                        clo: ClothingInsulation::from_clo(clo),
                        met: MetabolicRate::from_met(met),
                    },
                    CloDynamicAshraeOptions {
                        model: ashrae_model,
                    },
                )
                .as_clo(),
                py_util(&utils, "clo_dynamic_ashrae", (clo, met))?,
            )?;

            compare_field(
                &FieldCmp::new("clo_dynamic_iso", 1e-9),
                clo_dynamic_iso(
                    CloDynamicIsoInputs {
                        clo: ClothingInsulation::from_clo(clo),
                        met: MetabolicRate::from_met(met),
                        v: Speed::from_meters_per_second(v),
                    },
                    CloDynamicIsoOptions {
                        i_a: ClothingInsulation::from_clo(i_a),
                        model: iso9920_model,
                    },
                ),
                py_util(&utils, "clo_dynamic_iso", (clo, met, v, i_a))?,
            )
        });
    });
}

#[test]
fn sweep_clo_insulation_helpers() {
    let domain = Domain::new()
        .real("vr", 0.0, 4.0)
        .real("v_walk", 0.0, 2.0)
        .real("i_cl", 0.0, 3.0)
        .real("i_a_static", 0.0, 1.5)
        .real("i_t", 0.0, 4.0)
        .real("tout", -30.0, 40.0);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        run_sweep("sweep_clo_insulation_helpers", &domain, |s: &Sample| {
            let (vr, v_walk, i_cl, i_a_static, i_t, tout) = (
                s.real("vr"),
                s.real("v_walk"),
                s.real("i_cl"),
                s.real("i_a_static"),
                s.real("i_t"),
                s.real("tout"),
            );

            compare_field(
                &FieldCmp::new("clo_area_factor", 1e-9),
                clo_area_factor(ClothingInsulation::from_clo(i_cl)),
                py_util(&utils, "clo_area_factor", (i_cl,))?,
            )?;

            compare_field(
                &FieldCmp::new("clo_insulation_air_layer", 1e-9),
                clo_insulation_air_layer(CloInsulationAirLayerInputs {
                    vr: Speed::from_meters_per_second(vr),
                    v_walk: Speed::from_meters_per_second(v_walk),
                    i_a_static: ClothingInsulation::from_clo(i_a_static),
                }),
                py_util(&utils, "clo_insulation_air_layer", (vr, v_walk, i_a_static))?,
            )?;

            compare_field(
                &FieldCmp::new("clo_correction_factor_environment", 1e-9),
                clo_correction_factor_environment(CloCorrectionFactorEnvironmentInputs {
                    vr: Speed::from_meters_per_second(vr),
                    v_walk: Speed::from_meters_per_second(v_walk),
                    i_cl: ClothingInsulation::from_clo(i_cl),
                }),
                py_util(
                    &utils,
                    "clo_correction_factor_environment",
                    (vr, v_walk, i_cl),
                )?,
            )?;

            compare_field(
                &FieldCmp::new("clo_total_insulation", 1e-9),
                clo_total_insulation(CloTotalInsulationInputs {
                    i_t: ClothingInsulation::from_clo(i_t),
                    vr: Speed::from_meters_per_second(vr),
                    v_walk: Speed::from_meters_per_second(v_walk),
                    i_a_static: ClothingInsulation::from_clo(i_a_static),
                    i_cl: ClothingInsulation::from_clo(i_cl),
                }),
                py_util(
                    &utils,
                    "clo_total_insulation",
                    (i_t, vr, v_walk, i_a_static, i_cl),
                )?,
            )?;

            let py_clo_tout = models
                .getattr("clo_tout")
                .unwrap()
                .call1((tout,))
                .map_err(|e| format!("clo_tout raised: {e}"))?;
            compare_field(
                &FieldCmp::new("clo_tout", 1e-9),
                clo_tout(Temperature::from_celsius(tout)),
                py_float(&py_clo_tout, "clo_tout")?,
            )
        });
    });
}

#[test]
fn sweep_body_surface_area_and_hr_to_rh() {
    let domain = Domain::new()
        .real("weight", 30.0, 150.0)
        .real("height", 1.2, 2.2)
        .real("hr", 0.0, 0.03)
        .real("tdb", -20.0, 55.0)
        .real("p_atm", 80_000.0, 105_000.0)
        .enumerated("formula", 4);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep(
            "sweep_body_surface_area_and_hr_to_rh",
            &domain,
            |s: &Sample| {
                let (weight, height, hr, tdb, p_atm) = (
                    s.real("weight"),
                    s.real("height"),
                    s.real("hr"),
                    s.real("tdb"),
                    s.real("p_atm"),
                );
                let (formula, py_formula) = match s.index("formula") {
                    0 => (BsaFormula::DuBois, "dubois"),
                    1 => (BsaFormula::Takahira, "takahira"),
                    2 => (BsaFormula::Fujimoto, "fujimoto"),
                    _ => (BsaFormula::Kurazumi, "kurazumi"),
                };

                let kwargs = [("formula", py_formula.into_pyobject(py).unwrap().into_any())]
                    .into_py_dict(py)
                    .unwrap();
                let py_bsa = utils
                    .getattr("body_surface_area")
                    .unwrap()
                    .call((weight, height), Some(&kwargs))
                    .map_err(|e| format!("body_surface_area raised: {e}"))?;
                let py_bsa: f64 = py_bsa
                    .extract()
                    .or_else(|_| py_bsa.call_method0("item").and_then(|i| i.extract()))
                    .map_err(|e| format!("body_surface_area: could not read: {e}"))?;

                compare_field(
                    &FieldCmp::new("body_surface_area", 1e-9),
                    body_surface_area(
                        BodySurfaceAreaInputs {
                            weight: Mass::from_kilograms(weight),
                            height: Length::from_meters(height),
                        },
                        BodySurfaceAreaOptions { formula },
                    )
                    .as_square_meters(),
                    py_bsa,
                )?;

                compare_field(
                    &FieldCmp::new("hr_to_rh", 1e-9),
                    hr_to_rh(
                        hr,
                        Temperature::from_celsius(tdb),
                        Pressure::from_pascals(p_atm),
                    ),
                    py_util(&utils, "hr_to_rh", (hr, tdb, p_atm))?,
                )
            },
        );
    });
}

#[test]
fn sweep_running_mean_and_ensemble() {
    // `alpha` is one of the parameters the hand-written cases never vary, and the
    // rounding bug found on 2026-08-09 lived exactly there.
    let domain = Domain::new()
        .real("t0", -20.0, 40.0)
        .real("t1", -20.0, 40.0)
        .real("t2", -20.0, 40.0)
        .real("t3", -20.0, 40.0)
        .real("t4", -20.0, 40.0)
        .real("t5", -20.0, 40.0)
        .real("t6", -20.0, 40.0)
        .real("alpha", 0.0, 1.0)
        .enumerated("units", 2)
        .real("g0", 0.0, 1.0)
        .real("g1", 0.0, 1.0)
        .real("g2", 0.0, 1.0);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_running_mean_and_ensemble", &domain, |s: &Sample| {
            let temps: [f64; 7] = [
                s.real("t0"),
                s.real("t1"),
                s.real("t2"),
                s.real("t3"),
                s.real("t4"),
                s.real("t5"),
                s.real("t6"),
            ];
            let alpha = s.real("alpha");
            let (units, py_units) = match s.index("units") {
                0 => (Units::SI, "SI"),
                _ => (Units::IP, "IP"),
            };

            // utilities.py:944-950's IP branch reinterprets its raw list as Fahrenheit
            // and converts to SI before averaging; feed it the IP-valued equivalents of
            // the same SI environment Rust uses (see `sweep_cooling_effect`/`sweep_utci`,
            // which solve this the same way for their own `units` axis).
            let py_temps: Vec<f64> = if units == Units::IP {
                temps.iter().map(|t| t * 9.0 / 5.0 + 32.0).collect()
            } else {
                temps.to_vec()
            };

            let py_rm = utils
                .getattr("running_mean_outdoor_temperature")
                .unwrap()
                .call1((py_temps, alpha, py_units))
                .map_err(|e| format!("running_mean_outdoor_temperature raised: {e}"))?;
            let py_rm: f64 = py_rm
                .extract()
                .or_else(|_| py_rm.call_method0("item").and_then(|i| i.extract()))
                .map_err(|e| format!("running_mean: could not read: {e}"))?;

            let rust_temps: Vec<Temperature> = temps
                .iter()
                .map(|t| Temperature::from_celsius(*t))
                .collect();
            let rust_rm = running_mean_outdoor_temperature(
                &rust_temps,
                RunningMeanOutdoorTemperatureOptions { alpha, units },
            );
            let rust_value = match units {
                Units::SI => rust_rm.as_celsius(),
                Units::IP => rust_rm.as_fahrenheit(),
            };
            compare_field(
                &FieldCmp::new("running_mean_outdoor_temperature", 1e-9),
                rust_value,
                py_rm,
            )?;

            let garments = [s.real("g0"), s.real("g1"), s.real("g2")];
            let rust_garments: Vec<ClothingInsulation> = garments
                .iter()
                .map(|c| ClothingInsulation::from_clo(*c))
                .collect();
            compare_field(
                &FieldCmp::new("clo_intrinsic_insulation_ensemble", 1e-9),
                clo_intrinsic_insulation_ensemble(&rust_garments),
                py_util(
                    &utils,
                    "clo_intrinsic_insulation_ensemble",
                    (garments.to_vec(),),
                )?,
            )
        });
    });
}

#[test]
fn sweep_f_svv_and_transpose_sharp_altitude() {
    let domain = Domain::new()
        .real("w", 0.1, 10.0)
        .real("h", 0.1, 10.0)
        .real("d", 0.1, 20.0)
        .real("sharp", -20.0, 200.0)
        .real("altitude", -20.0, 110.0);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep(
            "sweep_f_svv_and_transpose_sharp_altitude",
            &domain,
            |s: &Sample| {
                let (w, h, d, sharp, altitude) = (
                    s.real("w"),
                    s.real("h"),
                    s.real("d"),
                    s.real("sharp"),
                    s.real("altitude"),
                );

                compare_field(
                    &FieldCmp::new("f_svv", 1e-9),
                    f_svv(FSvvInputs {
                        width: Length::from_meters(w),
                        height: Length::from_meters(h),
                        distance: Length::from_meters(d),
                    }),
                    py_util(&utils, "f_svv", (w, h, d))?,
                )?;

                let py_pair = utils
                    .getattr("transpose_sharp_altitude")
                    .unwrap()
                    .call1((sharp, altitude))
                    .map_err(|e| format!("transpose_sharp_altitude raised: {e}"))?;
                let (py_sharp, py_alt): (f64, f64) = py_pair
                    .extract()
                    .map_err(|e| format!("transpose_sharp_altitude: could not read: {e}"))?;

                let (rust_sharp, rust_alt) = transpose_sharp_altitude(
                    Angle::from_degrees(sharp),
                    Angle::from_degrees(altitude),
                );
                compare_field(
                    &FieldCmp::new("sharp", 1e-9),
                    rust_sharp.as_degrees(),
                    py_sharp,
                )?;
                compare_field(
                    &FieldCmp::new("altitude", 1e-9),
                    rust_alt.as_degrees(),
                    py_alt,
                )
            },
        );
    });
}

#[test]
fn sweep_clo_lookup_tables() {
    // Not randomised: every entry of both tables is checked, because a transcription
    // slip in one row is exactly what a sampled sweep would miss.
    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        for (py_table, label) in [
            ("clo_individual_garments", "garment"),
            ("clo_typical_ensembles", "ensemble"),
        ] {
            let table = utils
                .getattr(py_table)
                .unwrap_or_else(|e| panic!("{py_table} missing from utilities: {e}"));
            let items: Vec<(String, f64)> = table
                .call_method0("items")
                .and_then(|i| i.call_method0("__iter__"))
                .and_then(|i| py.get_type::<pyo3::types::PyList>().call1((i,)))
                .and_then(|l| l.extract())
                .unwrap_or_else(|e| panic!("{py_table} is not a name->value mapping: {e}"));

            assert!(!items.is_empty(), "{py_table} is empty");

            for (name, py_value) in items {
                let rust_value = if label == "garment" {
                    clo_individual_garment(&name)
                } else {
                    clo_typical_ensemble(&name)
                };
                let rust_value = rust_value
                    .unwrap_or_else(|| panic!("{label} {name:?} is missing from the Rust table"));
                assert!(
                    (rust_value - py_value).abs() < 1e-9,
                    "{label} {name:?}: Rust {rust_value}, Python {py_value}"
                );
            }
        }
    });
}

#[test]
fn sweep_psychrometrics() {
    // These six live in src/psychrometrics.rs rather than src/utilities.rs, which is how
    // they were missed when the utility sweeps were first written: they had hand-written
    // parity cases, so the coverage checker stayed green while the space went unswept.
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0)
        .real("p_atm", 80_000.0, 105_000.0)
        .real("hr", 0.0, 0.03)
        .real("tr", -20.0, 60.0)
        .real("tg", -20.0, 70.0)
        .real("v", 0.0, 5.0)
        .real("d", 0.05, 0.3)
        .real("emissivity", 0.5, 1.0)
        .flag("use_iso")
        .flag("use_ashrae");

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_psychrometrics", &domain, |s: &Sample| {
            let (tdb, rh, p_atm, hr, tr, tg, v, d, emissivity) = (
                s.real("tdb"),
                s.real("rh"),
                s.real("p_atm"),
                s.real("hr"),
                s.real("tr"),
                s.real("tg"),
                s.real("v"),
                s.real("d"),
                s.real("emissivity"),
            );
            let t = Temperature::from_celsius(tdb);

            compare_field(
                &FieldCmp::new("enthalpy_air", 1e-6),
                enthalpy_air(t, hr),
                py_util(&utils, "enthalpy_air", (tdb, hr))?,
            )?;

            compare_field(
                &FieldCmp::new("wet_bulb_temperature", 1e-9),
                wet_bulb_temperature(t, Humidity::from_percent(rh)).as_celsius(),
                py_util(&utils, "wet_bulb_tmp", (tdb, rh))?,
            )?;

            compare_field(
                &FieldCmp::new("dew_point_temperature", 1e-9),
                dew_point_temperature(t, Humidity::from_percent(rh)).as_celsius(),
                py_util(&utils, "dew_point_tmp", (tdb, rh))?,
            )?;

            // Both standards on both helpers: the Rust port spells the choice as a bool,
            // Python as a string, and neither pairing was previously exercised.
            let use_iso = s.flag("use_iso");
            let py_standard = if use_iso { "ISO" } else { "Mixed Convection" };
            compare_field(
                &FieldCmp::new("mean_radiant_temperature", 1e-9),
                mean_radiant_temperature(
                    MeanRadiantTemperatureInputs {
                        tg: Temperature::from_celsius(tg),
                        tdb: t,
                        v: Speed::from_meters_per_second(v),
                    },
                    MeanRadiantTemperatureOptions {
                        d: Length::from_meters(d),
                        emissivity,
                        use_iso,
                    },
                )
                .as_celsius(),
                py_util(
                    &utils,
                    "mean_radiant_tmp",
                    (tg, tdb, v, d, emissivity, py_standard),
                )?,
            )?;

            let use_ashrae = s.flag("use_ashrae");
            let py_op_standard = if use_ashrae { "ASHRAE" } else { "ISO" };
            compare_field(
                &FieldCmp::new("operative_temperature", 1e-9),
                operative_temperature(
                    OperativeTemperatureInputs {
                        tdb: t,
                        tr: Temperature::from_celsius(tr),
                        v: Speed::from_meters_per_second(v),
                    },
                    OperativeTemperatureOptions { use_ashrae },
                )
                .as_celsius(),
                py_util(&utils, "operative_tmp", (tdb, tr, v, py_op_standard))?,
            )?;

            // psy_ta_rh returns six fields; Python names two of them differently from
            // the Rust struct, so they are paired explicitly rather than by name.
            let py_psy = utils
                .getattr("psy_ta_rh")
                .unwrap()
                .call1((tdb, rh, p_atm))
                .map_err(|e| format!("psy_ta_rh raised: {e}"))?;
            let rust_psy = psy_ta_rh(
                PsyTaRhInputs {
                    tdb: t,
                    rh: Humidity::from_percent(rh),
                },
                PsyTaRhOptions {
                    p_atm: Pressure::from_pascals(p_atm),
                },
            );

            for (py_name, rust_value, tol) in [
                ("p_sat", rust_psy.p_sat.as_pascals(), 1e-6),
                ("p_vap", rust_psy.p_vap.as_pascals(), 1e-6),
                ("hr", rust_psy.hr, 1e-9),
                ("wet_bulb_tmp", rust_psy.t_wb.as_celsius(), 1e-9),
                ("dew_point_tmp", rust_psy.t_dp.as_celsius(), 1e-9),
                ("h", rust_psy.h, 1e-6),
            ] {
                compare_field(
                    &FieldCmp::new(py_name, tol),
                    rust_value,
                    py_float(&py_psy, py_name)?,
                )?;
            }
            Ok(())
        });
    });
}
