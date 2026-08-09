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
use thermalcomfort::models::two_nodes_gagge::{GaggeTwoNodesJiOptions, two_nodes_gagge_ji};
use thermalcomfort::models::{
    CoolingEffectOptions, GaggeTwoNodesOptions, Iso7933Model, PhsOptions, PhsPosture, SetOptions,
    UtciOptions, cooling_effect, phs, pmv_ppd_ashrae, pmv_ppd_iso, set_tmp, two_nodes_gagge,
    use_fans_heatwaves, utci,
};
use thermalcomfort::utilities::Posture;
use thermalcomfort::{
    Area, ClothingInsulation, Humidity, Length, Mass, MetabolicRate, Pressure, Speed, Temperature,
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
