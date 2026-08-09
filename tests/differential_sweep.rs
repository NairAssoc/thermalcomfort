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

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAnyMethods, PyBool};
use support::compare::{FieldCmp, compare_field};
use support::domain::{Domain, Sample};
use support::sweep::{import_reference, run_sweep};
use thermalcomfort::models::pmv::PmvPpdOptions;
use thermalcomfort::models::{
    GaggeTwoNodesOptions, SetOptions, UtciOptions, pmv_ppd_ashrae, pmv_ppd_iso, set_tmp,
    two_nodes_gagge, utci,
};
use thermalcomfort::utilities::Posture;
use thermalcomfort::{
    Area, ClothingInsulation, Humidity, MetabolicRate, Pressure, Speed, Temperature,
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
