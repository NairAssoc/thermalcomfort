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

use pyo3::exceptions::PyOverflowError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAnyMethods, PyBool, PyModule, PyTuple};
use support::compare::{FieldCmp, NanPolicy, compare_field};
use support::domain::{Domain, Sample};
use support::sweep::{import_reference, run_sweep};
use thermalcomfort::models::pmv::PmvPpdOptions;
use thermalcomfort::models::two_nodes_gagge::{GaggeTwoNodesJiOptions, two_nodes_gagge_ji};
use thermalcomfort::models::{
    AdaptiveOptions, CoolingEffectOptions, DurationLimitedExposure, GaggeTwoNodesOptions,
    GaggeTwoNodesSleepOptions, IreqOptions, Iso7933Model, PetOptions, PetPosture, PhsOptions,
    PhsPosture, SleepInputs, two_nodes_gagge_sleep,
    RidgeRegressionOptions, SetOptions, Sports, SportsValues, UtciOptions, WbgtOptions,
    WorkIntensity, adaptive_ashrae, adaptive_en, ankle_draft, at, cooling_effect, discomfort_index,
    esi, heat_index_lu, heat_index_rothfusz, heat_index_schoen, humidex, humidex_masterson, ireq,
    net, pet_steady, phs, pmv_a, pmv_athb, pmv_e, pmv_ppd_ashrae, pmv_ppd_iso,
    ridge_regression_predict_t_re_t_sk, set_tmp, solar_gain, sports_heat_stress_risk, thi,
    two_nodes_gagge, use_fans_heatwaves, utci, vertical_tmp_grad_ppd, wbgt, wci,
    wind_chill_temperature, work_capacity_dunne, work_capacity_hothaps, work_capacity_iso,
    work_capacity_niosh,
};
use thermalcomfort::models::{f_svv, transpose_sharp_altitude};
use thermalcomfort::psychrometrics::{
    dew_point_temperature, enthalpy_air, mean_radiant_temperature, operative_temperature,
    psy_ta_rh, wet_bulb_temperature,
};
use thermalcomfort::utilities::{
    BsaFormula, Posture, antoine, body_surface_area, clo_area_factor,
    clo_correction_factor_environment, clo_dynamic_ashrae, clo_dynamic_iso, clo_individual_garment,
    clo_insulation_air_layer, clo_intrinsic_insulation_ensemble, clo_total_insulation, clo_tout,
    clo_typical_ensemble, hr_to_rh, p_sat, p_sat_antoine, p_sat_torr,
    running_mean_outdoor_temperature, v_relative,
};
use thermalcomfort::{
    AirPermeability, Area, ClothingInsulation, Humidity, Length, Mass, MetabolicRate, Power,
    Pressure, Sex, Speed, Temperature, TemperatureDelta, WorkEfficiency,
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
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = [FieldCmp::new("pmv", 0.01), FieldCmp::new("ppd", 0.11)];

        run_sweep("sweep_pmv_ppd_iso", &pmv_domain(), |s: &Sample| {
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
                .getattr("pmv_ppd_iso")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_ppd_iso(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                PmvPpdOptions {
                    wme: MetabolicRate::from_met(wme),
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
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = [FieldCmp::new("pmv", 0.01), FieldCmp::new("ppd", 0.11)];

        run_sweep("sweep_pmv_ppd_ashrae", &pmv_domain(), |s: &Sample| {
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
                .getattr("pmv_ppd_ashrae")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = pmv_ppd_ashrae(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                PmvPpdOptions {
                    wme: MetabolicRate::from_met(wme),
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                GaggeTwoNodesOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    posture,
                    max_skin_blood_flow: msbf,
                    max_sweating: msw,
                    round_output,
                    ..Default::default()
                },
            );

            let rust_values = [
                rust.e_skin,
                rust.e_rsw,
                rust.e_max,
                rust.q_sensible,
                rust.q_skin,
                rust.q_res,
                rust.t_core,
                rust.t_skin,
                rust.m_bl,
                rust.m_rsw,
                rust.w,
                rust.w_max,
                rust.set,
                rust.et,
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                SetOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    posture,
                    limit_inputs,
                    round_output,
                    calculate_ce: false,
                },
            );

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
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("utci", 0.06);

        run_sweep("sweep_utci", &domain, |s: &Sample| {
            let (tdb, tr, v, rh) = (s.real("tdb"), s.real("tr"), s.real("v"), s.real("rh"));
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
                .getattr("utci")
                .unwrap()
                .call((tdb, tr, v, rh), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = utci(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                UtciOptions {
                    limit_inputs,
                    round_output,
                },
            );

            compare_field(&field, rust.utci, py_float(&py_result, "utci")?)
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
        .real("wme", 0.0, 1.0);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("ce", 0.011);

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

            let kwargs = [("wme", wme.into_pyobject(py).unwrap().into_any())]
                .into_py_dict(py)
                .unwrap();
            let py_result = models
                .getattr("cooling_effect")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = cooling_effect(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                CoolingEffectOptions {
                    wme: MetabolicRate::from_met(wme),
                    ..Default::default()
                },
            );

            compare_field(&field, rust, py_float(&py_result, "ce")?)
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                MetabolicRate::from_met(wme),
                Area::from_square_meters(bsa),
                Pressure::from_pascals(p_atm),
                posture,
                msbf,
                msw,
                s.flag("limit_inputs"),
                s.flag("round_output"),
            );

            let values = [
                rust.e_skin,
                rust.e_rsw,
                rust.e_max,
                rust.q_sensible,
                rust.q_skin,
                rust.q_res,
                rust.t_core,
                rust.t_skin,
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                posture,
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
                rust.t_re,
                rust.t_sk,
                rust.t_cr,
                rust.t_cr_eq,
                rust.t_sk_t_cr_wg,
                rust.d_lim_loss_50,
                rust.d_lim_loss_95,
                rust.d_lim_t_re,
                rust.sweat_loss_g,
                rust.sweat_rate_watt,
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
        .enumerated("position", 3)
        .flag("acclimatized")
        .flag("round_output");

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let utilities = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_two_nodes_gagge_ji", &domain, |s: &Sample| {
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
            let round_output = s.flag("round_output");

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
            ]
            .into_py_dict(py)
            .unwrap();

            let py_result = models
                .getattr("two_nodes_gagge_ji")
                .unwrap()
                .call((tdb, tr, v, met, clo, vapor_pressure), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = two_nodes_gagge_ji(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                GaggeTwoNodesJiOptions {
                    wme: MetabolicRate::from_met(wme),
                    body_surface_area: Area::from_square_meters(bsa),
                    p_atm: Pressure::from_pascals(p_atm),
                    posture,
                    acclimatized,
                    round_output,
                },
            );

            // Python does not round, so when the Rust side does, the reference is the
            // Python trajectory put through the same rounding. The unrounded half of
            // the sweep is what proves the underlying values agree.
            let (tol, round) = if round_output {
                (0.0051, true)
            } else {
                (1e-9, false)
            };

            for (name, rust_series) in [("t_core", &rust.t_core), ("t_skin", &rust.t_skin)] {
                let py_series = py_float_seq(&py_result, name)?;
                if py_series.len() != rust_series.len() {
                    return Err(format!(
                        "{name}: Rust returned {} minutes, Python {}",
                        rust_series.len(),
                        py_series.len()
                    ));
                }
                let cmp = FieldCmp::new(name, tol);
                for (minute, (rust_value, py_value)) in
                    rust_series.iter().zip(py_series.iter()).enumerate()
                {
                    let expected = if round {
                        (py_value * 100.0).round() / 100.0
                    } else {
                        *py_value
                    };
                    compare_field(&cmp, *rust_value, expected)
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
            let thickness = ramp(
                s.real("thickness"),
                s.real("thickness_drift"),
                0.0,
                30.0,
            );

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
                ("t_core", rust.t_core.iter().map(|t| t.as_celsius()).collect()),
                ("t_skin", rust.t_skin.iter().map(|t| t.as_celsius()).collect()),
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

#[test]
fn sweep_pmv_a() {
    // pythermalcomfort's pmv_a exposes no round_output; the flag is swept anyway to
    // prove the Rust option cannot change the answer.
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                a_coefficient,
                PmvPpdOptions {
                    wme: MetabolicRate::from_met(wme),
                    limit_inputs,
                    round_output: s.flag("round_output"),
                },
            );

            compare_field(&field, rust, py_float(&py_result, "a_pmv")?)
        });
    });
}

#[test]
fn sweep_pmv_e() {
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                e_coefficient,
                PmvPpdOptions {
                    wme: MetabolicRate::from_met(wme),
                    limit_inputs,
                    round_output: s.flag("round_output"),
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                supply_clo.then(|| ClothingInsulation::from_clo(clo)),
                Temperature::from_celsius(t_running_mean),
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
    // The Rust port has no round_output knob; it always rounds, so Python is called the
    // same way.
    let domain = Domain::new()
        .real("tdb", -20.0, 55.0)
        .real("rh", 0.0, 100.0);

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("hi", 1e-9);

        run_sweep("sweep_heat_index_lu", &domain, |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));

            let py_result = models
                .getattr("heat_index_lu")
                .unwrap()
                .call1((tdb, rh))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = heat_index_lu(Temperature::from_celsius(tdb), Humidity::from_percent(rh));

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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                round_output,
                limit_inputs,
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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                round_output,
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
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("humidex", 1e-9);

        run_sweep("sweep_humidex", &tdb_rh_domain(), |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");

            let kwargs = [
                ("model", "rana".into_pyobject(py).unwrap().into_any()),
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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                round_output,
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
fn sweep_humidex_masterson() {
    // Python spells this `humidex(model="masterson")`; the Rust port splits it into its
    // own function, so it is a rename rather than the missing counterpart the coverage
    // checker's EXEMPT list used to claim.
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let field = FieldCmp::new("humidex", 1e-9);

        run_sweep("sweep_humidex_masterson", &tdb_rh_domain(), |s: &Sample| {
            let (tdb, rh) = (s.real("tdb"), s.real("rh"));
            let round_output = s.flag("round_output");

            let kwargs = [
                ("model", "masterson".into_pyobject(py).unwrap().into_any()),
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

            let rust = humidex_masterson(
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                round_output,
            );

            compare_field(&field, rust, py_float(&py_result, "humidex")?)
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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                round_output,
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

            let rust = discomfort_index(Temperature::from_celsius(tdb), Humidity::from_percent(rh));

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
                Temperature::from_celsius(tdb),
                Speed::from_meters_per_second(v),
                round_output,
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
                Temperature::from_celsius(tdb),
                Speed::from_kilometers_per_hour(v),
                round_output,
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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                Speed::from_meters_per_second(v),
                round_output,
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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                Speed::from_meters_per_second(v),
                supply_q.then_some(q),
                round_output,
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
                Temperature::from_celsius(tdb),
                Humidity::from_percent(rh),
                sol,
                round_output,
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
                Temperature::from_celsius(twb),
                Temperature::from_celsius(tg),
                with_solar_load.then(|| Temperature::from_celsius(tdb)),
                WbgtOptions {
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
        .flag("round_output");

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
            let (posture, py_position) = match s.index("position") {
                0 => (PetPosture::Sitting, "sitting"),
                _ => (PetPosture::Standing, "standing"),
            };
            let (sex, py_sex) = match s.index("sex") {
                0 => (Sex::Male, "male"),
                _ => (Sex::Female, "female"),
            };
            let round_output = s.flag("round_output");

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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(v),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                PetOptions {
                    age,
                    sex,
                    height: Length::from_meters(height),
                    weight: Mass::from_kilograms(weight),
                    p_atm: Pressure::from_pascals(p_atm),
                    wme: WorkEfficiency::new(wme)
                        .expect("the wme axis is bounded to the valid [0, 1] range"),
                    posture,
                    round_output,
                },
            );

            // Python has no round_output and always rounds to two decimals, so the
            // rounded half of the sweep must match exactly and the unrounded half only
            // to within that rounding step.
            if rust.pet.is_nan() {
                unconverged.set(unconverged.get() + 1);
            }

            // `RustMayBeNan` is confined to this one documented case: PET's 3-node
            // Newton demands a 1e-5 residual, while scipy's fsolve stops on step size
            // and accepts points whose balance is still ~0.3 W/m2 out. Where no root
            // meets the stricter bar the Rust solver reports NaN rather than returning
            // a wrong number. Python-NaN against a Rust number is still a failure.
            //
            // Python always rounds to two decimals and Rust rounds independently, so
            // when both round the results can straddle a boundary and differ by exactly
            // one step. That is why the rounded half allows 0.0101 and no more; the
            // unrounded half holds the underlying values to 0.0051, which is what
            // actually proves they agree.
            let tol = if round_output { 0.0101 } else { 0.0051 };
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
                let rust = work_capacity_dunne(Temperature::from_celsius(wbgt_v), intensity);
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
                let rust = work_capacity_hothaps(Temperature::from_celsius(wbgt_v), intensity);
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
}

#[test]
fn sweep_adaptive_ashrae() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");
        let fields = [
            FieldCmp::new("tmp_cmf", 1e-9),
            FieldCmp::new("tmp_cmf_80_low", 1e-9),
            FieldCmp::new("tmp_cmf_80_up", 1e-9),
            FieldCmp::new("tmp_cmf_90_low", 1e-9),
            FieldCmp::new("tmp_cmf_90_up", 1e-9),
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
                .getattr("adaptive_ashrae")
                .unwrap()
                .call((tdb, tr, trm, v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = adaptive_ashrae(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Temperature::from_celsius(trm),
                Speed::from_meters_per_second(v),
                AdaptiveOptions {
                    limit_inputs,
                    round_output,
                },
            );

            let values = [
                rust.tmp_cmf,
                rust.tmp_cmf_80_low,
                rust.tmp_cmf_80_up,
                rust.tmp_cmf_90_low,
                rust.tmp_cmf_90_up,
            ];
            for (field, rust_value) in fields.iter().zip(values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
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
        let fields = [
            FieldCmp::new("tmp_cmf", 1e-9),
            FieldCmp::new("tmp_cmf_cat_i_low", 1e-9),
            FieldCmp::new("tmp_cmf_cat_i_up", 1e-9),
            FieldCmp::new("tmp_cmf_cat_ii_low", 1e-9),
            FieldCmp::new("tmp_cmf_cat_ii_up", 1e-9),
            FieldCmp::new("tmp_cmf_cat_iii_low", 1e-9),
            FieldCmp::new("tmp_cmf_cat_iii_up", 1e-9),
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
                .getattr("adaptive_en")
                .unwrap()
                .call((tdb, tr, trm, v), Some(&kwargs))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = adaptive_en(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Temperature::from_celsius(trm),
                Speed::from_meters_per_second(v),
                AdaptiveOptions {
                    limit_inputs,
                    round_output,
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
            for (field, rust_value) in fields.iter().zip(values) {
                compare_field(field, rust_value, py_float(&py_result, field.name)?)?;
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                Speed::from_meters_per_second(v_ankle),
                limit_inputs,
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                TemperatureDelta::from_celsius(grad),
                limit_inputs,
            );

            // The Rust port has no round_output knob here, so the reference is Python's
            // rounded value; the unrounded half only has to agree within that step.
            let tol = if round_output { 1e-9 } else { 0.051 };
            compare_field(
                &FieldCmp::new("ppd_vg", tol),
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

            let rust = solar_gain(alt, sharp, dir, trans, fsvv, fbes, asw, posture, floor);

            // Rust always rounds; compare against Python's rounded value.
            let tol = if round_output { 1e-9 } else { 0.051 };
            for (name, rust_value) in [("erf", rust.erf), ("delta_mrt", rust.delta_mrt)] {
                compare_field(
                    &FieldCmp::new(name, tol),
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
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Speed::from_meters_per_second(vr),
                Humidity::from_percent(rh),
                MetabolicRate::from_met(met),
                ClothingInsulation::from_clo(clo),
                AirPermeability::from_l_per_m2_s(p),
                Speed::from_meters_per_second(walk_sp),
                IreqOptions {
                    wme: MetabolicRate::from_met(wme),
                    limit_inputs,
                    round_output,
                },
            );

            let values = [
                rust.ireq_min,
                rust.ireq_neutral,
                rust.icl_min,
                rust.icl_neutral,
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

            let py_result = models
                .getattr("sports_heat_stress_risk")
                .unwrap()
                .call1((tdb, tr, rh, vr, py_sport))
                .map_err(|e| format!("python raised: {e}"))?;

            let rust = sports_heat_stress_risk(
                Temperature::from_celsius(tdb),
                Temperature::from_celsius(tr),
                Humidity::from_percent(rh),
                Speed::from_meters_per_second(vr),
                sport_at(index),
            );

            let values = [
                rust.risk_level_interpolated,
                rust.t_medium,
                rust.t_high,
                rust.t_extreme,
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
        .real("t_re_initial", 36.0, 39.0)
        .real("t_sk_initial", 30.0, 38.0)
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
                    s.real("t_re_initial"),
                    s.real("t_sk_initial"),
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
                    sex,
                    age,
                    Length::from_meters(height),
                    Mass::from_kilograms(weight),
                    Temperature::from_celsius(tdb),
                    Humidity::from_percent(rh),
                    duration,
                    RidgeRegressionOptions {
                        t_re_initial: supply_initials.then(|| Temperature::from_celsius(t_re0)),
                        t_sk_initial: supply_initials.then(|| Temperature::from_celsius(t_sk0)),
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
                        compare_field(&cmp, *rust_value, *py_value)
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
    let domain = Domain::new()
        .real("v", 0.0, 4.0)
        .real("met", 0.6, 5.0)
        .real("clo", 0.0, 3.0)
        .real("i_a", 0.0, 1.5);

    Python::with_gil(|py| {
        let utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("failed to import pythermalcomfort.utilities");

        run_sweep("sweep_v_relative_and_clo_dynamic", &domain, |s: &Sample| {
            let (v, met, clo, i_a) = (s.real("v"), s.real("met"), s.real("clo"), s.real("i_a"));

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
                    ClothingInsulation::from_clo(clo),
                    MetabolicRate::from_met(met),
                )
                .as_clo(),
                py_util(&utils, "clo_dynamic_ashrae", (clo, met))?,
            )?;

            compare_field(
                &FieldCmp::new("clo_dynamic_iso", 1e-9),
                clo_dynamic_iso(
                    ClothingInsulation::from_clo(clo),
                    MetabolicRate::from_met(met),
                    Speed::from_meters_per_second(v),
                    ClothingInsulation::from_clo(i_a),
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
                clo_insulation_air_layer(
                    Speed::from_meters_per_second(vr),
                    Speed::from_meters_per_second(v_walk),
                    ClothingInsulation::from_clo(i_a_static),
                ),
                py_util(&utils, "clo_insulation_air_layer", (vr, v_walk, i_a_static))?,
            )?;

            compare_field(
                &FieldCmp::new("clo_correction_factor_environment", 1e-9),
                clo_correction_factor_environment(
                    Speed::from_meters_per_second(vr),
                    Speed::from_meters_per_second(v_walk),
                    ClothingInsulation::from_clo(i_cl),
                ),
                py_util(
                    &utils,
                    "clo_correction_factor_environment",
                    (vr, v_walk, i_cl),
                )?,
            )?;

            compare_field(
                &FieldCmp::new("clo_total_insulation", 1e-9),
                clo_total_insulation(
                    ClothingInsulation::from_clo(i_t),
                    Speed::from_meters_per_second(vr),
                    Speed::from_meters_per_second(v_walk),
                    ClothingInsulation::from_clo(i_a_static),
                    ClothingInsulation::from_clo(i_cl),
                ),
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
                        Mass::from_kilograms(weight),
                        Length::from_meters(height),
                        formula,
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

            let py_rm = utils
                .getattr("running_mean_outdoor_temperature")
                .unwrap()
                .call1((temps.to_vec(), alpha))
                .map_err(|e| format!("running_mean_outdoor_temperature raised: {e}"))?;
            let py_rm: f64 = py_rm
                .extract()
                .or_else(|_| py_rm.call_method0("item").and_then(|i| i.extract()))
                .map_err(|e| format!("running_mean: could not read: {e}"))?;

            let rust_temps: Vec<Temperature> = temps
                .iter()
                .map(|t| Temperature::from_celsius(*t))
                .collect();
            compare_field(
                &FieldCmp::new("running_mean_outdoor_temperature", 1e-9),
                running_mean_outdoor_temperature(&rust_temps, alpha).as_celsius(),
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
                    f_svv(
                        Length::from_meters(w),
                        Length::from_meters(h),
                        Length::from_meters(d),
                    ),
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

                let (rust_sharp, rust_alt) = transpose_sharp_altitude(sharp, altitude);
                compare_field(&FieldCmp::new("sharp", 1e-9), rust_sharp, py_sharp)?;
                compare_field(&FieldCmp::new("altitude", 1e-9), rust_alt, py_alt)
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
                    Temperature::from_celsius(tg),
                    t,
                    Speed::from_meters_per_second(v),
                    Length::from_meters(d),
                    emissivity,
                    use_iso,
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
                    t,
                    Temperature::from_celsius(tr),
                    Speed::from_meters_per_second(v),
                    use_ashrae,
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
            let rust_psy = psy_ta_rh(t, Humidity::from_percent(rh), Pressure::from_pascals(p_atm));

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
