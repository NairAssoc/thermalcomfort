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

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAnyMethods, PyBool, PyModule};
use support::compare::{FieldCmp, NanPolicy, compare_field};
use support::domain::{Domain, Sample};
use support::sweep::{import_reference, run_sweep};
use thermalcomfort::models::pmv::PmvPpdOptions;
use thermalcomfort::models::two_nodes_gagge::{GaggeTwoNodesJiOptions, two_nodes_gagge_ji};
use thermalcomfort::models::{
    CoolingEffectOptions, GaggeTwoNodesOptions, Iso7933Model, PetOptions, PetPosture, PhsOptions,
    PhsPosture, SetOptions, UtciOptions, WbgtOptions, at, cooling_effect, discomfort_index, esi,
    heat_index_lu, heat_index_rothfusz, heat_index_schoen, humidex, humidex_masterson, net,
    pet_steady, phs, pmv_a, pmv_athb, pmv_e, pmv_ppd_ashrae, pmv_ppd_iso, set_tmp, thi,
    two_nodes_gagge, use_fans_heatwaves, utci, wbgt, wci, wind_chill_temperature,
};
use thermalcomfort::utilities::Posture;
use thermalcomfort::{
    Area, ClothingInsulation, Humidity, Length, Mass, MetabolicRate, Pressure, Sex, Speed,
    Temperature,
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
        .real("wme", 0.0, 2.0)
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
                    wme: MetabolicRate::from_met(wme),
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
            let tol = if round_output { 1e-9 } else { 0.0051 };
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
