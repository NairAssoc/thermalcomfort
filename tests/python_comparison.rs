//! Comprehensive tests comparing Rust implementation with Python pythermalcomfort
//!
//! These tests ensure the Rust port produces identical results to the original
//! Python package across a wide range of inputs and edge cases.

use approx::assert_abs_diff_eq;
use core::time::Duration;
use measurements::{Angle, Humidity, Length, Power, Pressure, Speed, Temperature};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAnyMethods, PyBool};
use std::sync::atomic::{AtomicBool, Ordering};
use thermalcomfort::models::adaptive::{AdaptiveInputs, AdaptiveOptions};
use thermalcomfort::models::jos3::{Jos3Builder, Jos3Posture, Jos3Results, PerBodyPart};
use thermalcomfort::models::pmv::{
    PmvAInputs, PmvAOptions, PmvAthbInputs, PmvAthbOptions, PmvEInputs, PmvEOptions,
    PmvPpdAshraeOptions, PmvPpdInputs, PmvPpdIsoOptions,
};
use thermalcomfort::models::specialty::{
    AnkleDraftInputs, AnkleDraftOptions, FSvvInputs, VerticalTmpGradPpdInputs,
    VerticalTmpGradPpdOptions, f_svv,
};
use thermalcomfort::models::{
    AtInputs, CoolingEffectInputs, DiscomfortIndexInputs, DurationLimitedExposure, EsiInputs,
    GaggeTwoNodesInputs, GaggeTwoNodesJiInputs, HeatIndexLuInputs, HeatIndexLuOptions,
    HeatIndexRothfuszInputs, HeatIndexSchoenInputs, HumidexInputs, HumidexModel, HumidexOptions,
    IreqInputs, IreqOptions, Iso7933Model, NetInputs, PetInputs, PetOptions, PhsInputs, PhsOptions,
    PhsPosture, RidgeRegressionInputs, SetInputs, SleepInputs, SolarGainInputs, SolarGainOptions,
    SportsHeatStressRiskInputs, ThiInputs, UseFansHeatwavesInputs, UseFansHeatwavesOptions,
    UtciInputs, UtciOptions, WbgtInputs, WbgtOptions, WciInputs, WindChillTemperatureInputs,
    WorkCapacityIntensityOptions, WorkIntensity, adaptive_ashrae, adaptive_en, ankle_draft, at,
    cooling_effect, discomfort_index, esi, heat_index_lu, heat_index_rothfusz, heat_index_schoen,
    humidex, ireq, net, pet_steady, phs, pmv_a, pmv_athb, pmv_e, pmv_ppd_ashrae, pmv_ppd_iso,
    ridge_regression_predict_t_re_t_sk, set_tmp, solar_gain, thi, transpose_sharp_altitude,
    two_nodes_gagge, two_nodes_gagge_ji, two_nodes_gagge_sleep, use_fans_heatwaves, utci,
    vertical_tmp_grad_ppd, wbgt, wci, wind_chill_temperature, work_capacity_dunne,
    work_capacity_hothaps, work_capacity_iso, work_capacity_niosh,
};
use thermalcomfort::psychrometrics::{
    MeanRadiantTemperatureInputs, MeanRadiantTemperatureOptions, OperativeTemperatureInputs,
    OperativeTemperatureOptions, PsyTaRhInputs, PsyTaRhOptions, dew_point_temperature,
    enthalpy_air, mean_radiant_temperature, operative_temperature, psy_ta_rh, wet_bulb_temperature,
};
use thermalcomfort::utilities::{
    BodySurfaceAreaInputs, BodySurfaceAreaOptions, BsaFormula, CLO_INDIVIDUAL_GARMENTS,
    CLO_TYPICAL_ENSEMBLES, CloCorrectionFactorEnvironmentInputs, CloDynamicAshraeInputs,
    CloDynamicAshraeOptions, CloDynamicIsoInputs, CloDynamicIsoOptions,
    CloInsulationAirLayerInputs, CloTotalInsulationInputs, Posture,
    RunningMeanOutdoorTemperatureOptions, Units, antoine, body_surface_area, clo_area_factor,
    clo_correction_factor_environment, clo_dynamic_ashrae, clo_individual_garment,
    clo_insulation_air_layer, clo_intrinsic_insulation_ensemble, clo_total_insulation, clo_tout,
    clo_typical_ensemble, hr_to_rh, p_sat, p_sat_antoine, p_sat_torr,
    running_mean_outdoor_temperature, v_relative,
};
use thermalcomfort::{
    ActivityRatio, AirPermeability, BmrEquation, BodyFat, CardiacIndex, ClothingInsulation,
    HeatFluxDensity, Mass, MetabolicRate, Sex, TemperatureDelta,
};

/// Guard against validating the port against the wrong pythermalcomfort.
///
/// Every other test in this file compares against whatever `pythermalcomfort` happens to
/// be importable. If that is a stale version, those comparisons quietly stop meaning
/// anything — the suite can go green while the port matches a version nobody is targeting,
/// or fail in ways that look like Rust bugs. This has bitten this repo twice: CI was
/// pinned to 3.8.0 for four releases, and the local install sat at 2.7.0.
///
/// The crate version IS the pythermalcomfort version being ported, by convention (see the
/// version-bump workflow), so `CARGO_PKG_VERSION` is the single source of truth.
#[test]
fn test_pythermalcomfort_version_matches_crate() {
    Python::with_gil(assert_reference_version);
}

/// Import a pythermalcomfort module, asserting once per process that the reference is
/// the version this crate ports.
///
/// Drop-in for `PyModule::import` - it returns the same `PyResult`, so call sites keep
/// their existing `.expect(...)`/`.unwrap()`. Routing every import through here is what
/// makes the check unskippable: as a standalone `#[test]` it was bypassed by any
/// `cargo test <name>` filter, and absent entirely for tests outside this target.
fn import_reference<'py>(py: Python<'py>, module: &str) -> PyResult<Bound<'py, PyModule>> {
    // Deliberately an atomic flag rather than a `Once`. Tests run on parallel threads,
    // and Python's import machinery can release the GIL mid-import; a blocking
    // `Once::call_once` around it deadlocks, because a second thread acquires the
    // released GIL and then waits on the `Once` the first thread needs the GIL to
    // finish. A relaxed flag can let a few threads race and verify redundantly, which
    // is harmless - the check is a cheap attribute read and the assertion is identical.
    static CHECKED: AtomicBool = AtomicBool::new(false);
    if !CHECKED.swap(true, Ordering::Relaxed) {
        assert_reference_version(py);
    }
    PyModule::import(py, module)
}

/// Assert the importable pythermalcomfort is the version this crate ports.
fn assert_reference_version(py: Python<'_>) {
    // A Rust pre-release suffix (4.4.0-rc.1) marks a revision of the *port*, not of
    // upstream, and PEP 440 spells pre-releases differently anyway. Compare the release
    // triple only.
    let full = env!("CARGO_PKG_VERSION");
    let expected = full.split('-').next().unwrap_or(full);

    {
        let ptc = PyModule::import(py, "pythermalcomfort").unwrap_or_else(|e| {
            panic!(
                "could not import pythermalcomfort, so no parity test in this suite is \
                 actually verifying anything.\n\
                 Expected version {expected}. Set one up with:\n  \
                 python3 -m venv /tmp/ptc_venv && \
                 /tmp/ptc_venv/bin/pip install pythermalcomfort=={expected}\n  \
                 PYTHONPATH=/tmp/ptc_venv/lib/python3.*/site-packages cargo test\n\
                 Underlying error: {e}"
            )
        });

        let actual: String = ptc
            .getattr("__version__")
            .expect("pythermalcomfort has no __version__")
            .extract()
            .expect("pythermalcomfort.__version__ is not a string");

        assert_eq!(
            actual, expected,
            "\n\npythermalcomfort version mismatch: the parity tests in this suite are \
             comparing against {actual}, but this crate ports {expected}.\n\
             Every parity result below is therefore meaningless.\n\
             Fix with:\n  \
             python3 -m venv /tmp/ptc_venv && \
             /tmp/ptc_venv/bin/pip install pythermalcomfort=={expected}\n  \
             PYTHONPATH=/tmp/ptc_venv/lib/python3.*/site-packages cargo test\n\
             If you are intentionally bumping the port, update Cargo.toml and README \
             together with the models.\n"
        );
    }
}

/// Extract a category/label field from a pythermalcomfort result.
///
/// Since the 4.x `NumericInput` retyping these fields come back as 0-dimensional numpy
/// object arrays rather than plain `str`, so a direct `extract::<String>()` fails.
/// Returns `None` when the field is `nan`, which is what Python yields for a category
/// whose underlying value fell outside the model's applicability limits.
fn extract_category(obj: &Bound<'_, PyAny>) -> Option<String> {
    // Plain Python str
    if let Ok(s) = obj.extract::<String>() {
        return Some(s);
    }
    // 0-d numpy array wrapping either a str or a nan
    if let Ok(item) = obj.call_method0("item") {
        if let Ok(s) = item.extract::<String>() {
            return Some(s);
        }
        if let Ok(f) = item.extract::<f64>() {
            if f.is_nan() {
                return None;
            }
        }
    }
    // Bare float nan
    if let Ok(f) = obj.extract::<f64>() {
        if f.is_nan() {
            return None;
        }
    }
    panic!(
        "could not interpret category field: {obj:?}.\n\
         If this is a number, the Rust side likely produced a non-numeric variant where \
         Python produced a value (or vice versa) - compare the two directly rather than \
         loosening this helper, which would swallow real mismatches."
    );
}

/// Read an optional boolean pythermalcomfort field, in every shape it comes back as.
///
/// `pmv_ppd_ashrae`'s `compliance` is a plain `bool` for a scalar call with the default
/// `limit_inputs`, a 0-d numpy object array wrapping `np.True_` when `limit_inputs` is
/// off, and a float NaN when the inputs left the applicability range (where the Rust port
/// uses `None`). A bare `extract::<bool>().ok()` silently yields `None` for the numpy
/// shape, which turns a real comparison into a vacuous one -- exactly the failure mode
/// this suite keeps finding, so the extraction panics on anything it does not recognise
/// rather than degrading to `None`.
fn extract_optional_bool(obj: &Bound<'_, PyAny>) -> Option<bool> {
    if let Ok(b) = obj.extract::<bool>() {
        return Some(b);
    }
    if let Ok(item) = obj.call_method0("item") {
        if let Ok(b) = item.extract::<bool>() {
            return Some(b);
        }
        if item.extract::<f64>().is_ok_and(f64::is_nan) {
            return None;
        }
    }
    if obj.extract::<f64>().is_ok_and(f64::is_nan) {
        return None;
    }
    panic!(
        "could not interpret optional-bool field: {obj:?}.\n\
         Do not loosen this to return None on an unknown shape - that is how a compliance \
         comparison becomes vacuous."
    );
}

/// Describe a pythermalcomfort field for comparison without panicking on unexpected
/// types.
///
/// [`extract_category`] aborts on anything that is neither a string nor a NaN, so a
/// genuine disagreement over a mixed-type field surfaced as an opaque "could not
/// interpret category field: 1.2" with no field name, inputs, or Rust value.
fn describe_field(obj: &Bound<'_, PyAny>) -> String {
    if let Ok(s) = obj.extract::<String>() {
        return format!("{s:?}");
    }
    if let Ok(item) = obj.call_method0("item") {
        if let Ok(s) = item.extract::<String>() {
            return format!("{s:?}");
        }
        if let Ok(f) = item.extract::<f64>() {
            return f.to_string();
        }
    }
    if let Ok(f) = obj.extract::<f64>() {
        return f.to_string();
    }
    format!("{obj:?}")
}

#[test]
fn test_pmv_ppd_iso_standard_conditions() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep spans cold → warm so the `tsv` band assignment is exercised
        // across SlightlyCool / Neutral / SlightlyWarm. The 18 C row now returns NaN
        // under 4.4.0's pmv applicability clamp; Cold / Cool / Warm are covered by
        // test_pmv_ppd_iso_extreme_conditions, which disables limit_inputs.
        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            (20.0, 20.0, 0.1, 50.0, 1.0, 1.0),
            (28.0, 28.0, 0.3, 60.0, 1.5, 0.3),
            (22.0, 24.0, 0.15, 40.0, 1.1, 0.7),
            (26.0, 26.0, 0.2, 55.0, 1.3, 0.6),
            (18.0, 18.0, 0.1, 50.0, 1.0, 0.7),
            (29.0, 29.0, 0.1, 50.0, 1.4, 0.4),
            // Inherited from a pyo3 test that used to live in src/models/pmv.rs, where
            // it ran with no reference-version guard at all.
            (25.0, 25.0, 0.22, 50.0, 1.4, 0.5),
        ];

        for (tdb, tr, vr, rh, met, clo) in test_cases {
            println!(
                "\nTesting: tdb={}, tr={}, vr={}, rh={}, met={}, clo={}",
                tdb, tr, vr, rh, met, clo
            );

            // Call Python function
            let py_result = pythermal
                .getattr("pmv_ppd_iso")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
            let py_ppd: f64 = py_result.getattr("ppd").unwrap().extract().unwrap();
            let py_tsv = extract_category(&py_result.getattr("tsv").unwrap());

            // Call Rust function with measurement types
            let rust_result = pmv_ppd_iso(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            println!("  Python - PMV: {:.2}, PPD: {:.1}", py_pmv, py_ppd);
            println!(
                "  Rust   - PMV: {:.2}, PPD: {:.1}",
                rust_result.pmv, rust_result.ppd
            );

            // Inputs outside the ISO 7730 applicability limits yield NaN on both sides;
            // NaN never compares equal, so check that case explicitly.
            if py_pmv.is_nan() {
                assert!(
                    rust_result.pmv.is_nan() && rust_result.ppd.is_nan(),
                    "Python returned NaN but Rust did not at tdb={tdb} tr={tr} vr={vr} rh={rh} met={met} clo={clo}",
                );
            } else {
                assert_abs_diff_eq!(rust_result.pmv, py_pmv, epsilon = 0.02);
                assert_abs_diff_eq!(rust_result.ppd, py_ppd, epsilon = 0.2);
            }
            assert_eq!(
                rust_result.tsv.map(|t| t.as_str().to_string()),
                py_tsv,
                "tsv mismatch at tdb={tdb} tr={tr} vr={vr} rh={rh} met={met} clo={clo}",
            );
            // ISO does not populate the ASHRAE compliance check.
            assert_eq!(rust_result.compliance, None);
        }
    });
}

#[test]
fn test_pmv_ppd_iso_extreme_conditions() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test edge cases with limit_inputs=False
        let test_cases = vec![
            // Very hot conditions
            (35.0, 35.0, 0.5, 60.0, 1.0, 0.3),
            // Very cold conditions
            (15.0, 15.0, 0.1, 50.0, 1.5, 1.5),
            // High air speed
            (25.0, 25.0, 0.8, 50.0, 1.2, 0.5),
            // High metabolic rate
            (22.0, 22.0, 0.2, 50.0, 3.0, 0.5),
            // Cold / Cool / Warm. Without these the tsv mapping was only ever verified
            // against Python for Neutral, SlightlyCool, SlightlyWarm and Hot; the other
            // three bands rested on hand-transcribed unit tests.
            (14.0, 14.0, 0.2, 50.0, 1.2, 0.5),
            (19.0, 19.0, 0.1, 50.0, 1.0, 0.6),
            (32.0, 32.0, 0.1, 50.0, 1.6, 0.6),
        ];

        let options = PmvPpdIsoOptions {
            limit_inputs: false,
            ..Default::default()
        };

        for (tdb, tr, vr, rh, met, clo) in test_cases {
            println!(
                "\nTesting extreme: tdb={}, tr={}, vr={}, rh={}, met={}, clo={}",
                tdb, tr, vr, rh, met, clo
            );

            // Call Python with limit_inputs=False
            let kwargs = [("limit_inputs", false)].into_py_dict(py).unwrap();
            let py_result = pythermal
                .getattr("pmv_ppd_iso")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
            let py_ppd: f64 = py_result.getattr("ppd").unwrap().extract().unwrap();

            // Call Rust function with measurement types
            let rust_result = pmv_ppd_iso(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                options,
            );

            println!("  Python - PMV: {:.2}, PPD: {:.1}", py_pmv, py_ppd);
            println!(
                "  Rust   - PMV: {:.2}, PPD: {:.1}",
                rust_result.pmv, rust_result.ppd
            );

            let py_tsv = extract_category(&py_result.getattr("tsv").unwrap());

            assert_abs_diff_eq!(rust_result.pmv, py_pmv, epsilon = 0.02);
            assert_abs_diff_eq!(rust_result.ppd, py_ppd, epsilon = 0.2);
            assert_eq!(
                rust_result.tsv.map(|t| t.as_str().to_string()),
                py_tsv,
                "tsv mismatch at tdb={tdb} tr={tr} vr={vr} rh={rh} met={met} clo={clo}",
            );
            assert_eq!(rust_result.compliance, None);
        }
    });
}

#[test]
fn test_pmv_ppd_ashrae() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Inputs are chosen to span compliant (-0.5 < PMV < 0.5) and non-compliant
        // outcomes, and to include vr > 0.1 cases so the ASHRAE Appendix H3
        // cooling-effect correction is exercised against the Python reference.
        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            (23.0, 23.0, 0.1, 45.0, 1.1, 0.7),
            (27.0, 27.0, 0.1, 55.0, 1.4, 0.4),
            (20.0, 20.0, 0.1, 50.0, 1.0, 1.0),
            (30.0, 30.0, 0.1, 60.0, 1.4, 0.3),
            (18.0, 18.0, 0.1, 40.0, 1.1, 0.6),
            // Cooling-effect path (vr > 0.1):
            (28.0, 28.0, 0.4, 50.0, 1.2, 0.5),
            (26.0, 26.0, 0.8, 55.0, 1.4, 0.4),
            (30.0, 30.0, 1.2, 50.0, 1.2, 0.5),
        ];

        for (tdb, tr, vr, rh, met, clo) in test_cases {
            println!(
                "\nTesting ASHRAE: tdb={}, tr={}, vr={}, rh={}, met={}, clo={}",
                tdb, tr, vr, rh, met, clo
            );

            // Call Python function
            let py_result = pythermal
                .getattr("pmv_ppd_ashrae")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
            let py_ppd: f64 = py_result.getattr("ppd").unwrap().extract().unwrap();
            let py_compliance = extract_optional_bool(&py_result.getattr("compliance").unwrap());

            // Call Rust function with measurement types
            let rust_result = pmv_ppd_ashrae(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            println!("  Python - PMV: {:.2}, PPD: {:.1}", py_pmv, py_ppd);
            println!(
                "  Rust   - PMV: {:.2}, PPD: {:.1}",
                rust_result.pmv, rust_result.ppd
            );

            assert_abs_diff_eq!(rust_result.pmv, py_pmv, epsilon = 0.02);
            assert_abs_diff_eq!(rust_result.ppd, py_ppd, epsilon = 0.2);
            assert_eq!(
                rust_result.compliance, py_compliance,
                "PMV ASHRAE compliance mismatch at tdb={} tr={} vr={} rh={} met={} clo={}",
                tdb, tr, vr, rh, met, clo,
            );
        }
    });
}

#[test]
fn test_compare_heat_index_schoen() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Spans the no risk / caution / extreme caution / danger bands
        let test_cases = vec![
            (20.0, 40.0),
            (25.0, 50.0),
            (29.0, 50.0),
            (32.0, 45.0),
            (35.0, 60.0),
            (38.0, 70.0),
            (40.0, 80.0),
        ];

        for (tdb, rh) in test_cases {
            let py_result = pythermal
                .getattr("heat_index_schoen")
                .unwrap()
                .call1((tdb, rh))
                .unwrap();

            let py_hi: f64 = py_result.getattr("hi").unwrap().extract().unwrap();
            let py_category = extract_category(&py_result.getattr("stress_category").unwrap());

            let rust_result = heat_index_schoen(
                HeatIndexSchoenInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.hi, py_hi, epsilon = 0.05);
            assert_eq!(
                rust_result.stress_category.map(|c| c.as_str().to_string()),
                py_category,
                "stress_category mismatch at tdb={tdb} rh={rh}",
            );
        }
    });
}

#[test]
fn test_compare_ireq() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb, tr, vr, rh, met, clo, p, walk_sp). Covers unlimited exposure, limited
        // exposure, and the tdb > 10 °C applicability cutoff.
        let test_cases = vec![
            (-15.0, -15.0, 2.0, 55.0, 175.0 / 58.15, 2.8, 50.0, 1.1),
            (-5.0, -5.0, 0.5, 80.0, 2.0, 1.5, 50.0, 0.5),
            (-30.0, -30.0, 5.0, 40.0, 3.0, 4.0, 100.0, 1.2),
            (0.0, 0.0, 1.0, 60.0, 1.5, 2.0, 20.0, 0.3),
            (10.0, 10.0, 0.4, 50.0, 2.5, 1.0, 50.0, 0.6),
            (-20.0, -25.0, 3.0, 70.0, 2.2, 3.5, 75.0, 0.9),
            // Outside applicability (tdb > 10) -> NaN on both sides
            (20.0, 20.0, 2.0, 55.0, 2.0, 2.8, 50.0, 1.1),
        ];

        for (tdb, tr, vr, rh, met, clo, p, walk_sp) in test_cases {
            let py_result = pythermal
                .getattr("ireq")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo, p, walk_sp))
                .unwrap();

            let rust_result = ireq(
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
                IreqOptions::default(),
            );

            for (field, rust_value) in [
                ("ireq_min", rust_result.ireq_min.as_clo()),
                ("ireq_neutral", rust_result.ireq_neutral.as_clo()),
                ("icl_min", rust_result.icl_min.as_clo()),
                ("icl_neutral", rust_result.icl_neutral.as_clo()),
            ] {
                let py_value: f64 = py_result.getattr(field).unwrap().extract().unwrap();
                if py_value.is_nan() {
                    assert!(
                        rust_value.is_nan(),
                        "{field}: Python NaN but Rust {rust_value} at tdb={tdb} vr={vr}",
                    );
                } else {
                    assert_abs_diff_eq!(rust_value, py_value, epsilon = 0.05);
                }
            }

            // dle is a mixed type upstream: a float, the string "more than 8", or nan
            for (field, rust_dle) in [
                ("dle_min", rust_result.dle_min),
                ("dle_neutral", rust_result.dle_neutral),
            ] {
                let py_dle = py_result.getattr(field).unwrap();
                let py_desc = describe_field(&py_dle);
                let context = format!(
                    "{field} at tdb={tdb} vr={vr} met={met} clo={clo}: \
                     Rust {rust_dle}, Python {py_desc}"
                );
                match rust_dle {
                    DurationLimitedExposure::MoreThanEight => {
                        assert_eq!(py_desc, "\"more than 8\"", "{context}");
                    }
                    DurationLimitedExposure::NotApplicable => {
                        assert_eq!(py_desc, "NaN", "{context}");
                    }
                    DurationLimitedExposure::Hours(h) => {
                        let py_hours: f64 =
                            py_dle.extract().unwrap_or_else(|_| panic!("{context}"));
                        assert_abs_diff_eq!(h, py_hours, epsilon = 0.05);
                    }
                }
            }
        }
    });
}

#[test]
fn test_compare_transpose_sharp_altitude() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for (sharp, altitude) in [
            (0.0, 0.0),
            (30.0, 45.0),
            (45.0, 0.0),
            (90.0, 20.0),
            (120.0, 60.0),
            (150.0, 30.0),
            (180.0, 10.0),
            (60.0, 75.0),
        ] {
            let py_pair = pythermal_utils
                .getattr("transpose_sharp_altitude")
                .unwrap()
                .call1((sharp, altitude))
                .unwrap();
            let (py_sharp, py_altitude): (f64, f64) = py_pair.extract().unwrap();

            let (rust_sharp, rust_altitude) =
                transpose_sharp_altitude(Angle::from_degrees(sharp), Angle::from_degrees(altitude));

            assert_abs_diff_eq!(rust_sharp.as_degrees(), py_sharp, epsilon = 1e-3);
            assert_abs_diff_eq!(rust_altitude.as_degrees(), py_altitude, epsilon = 1e-3);
        }
    });
}

#[test]
fn test_compare_saturation_pressures() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for tdb in [-20.0, -5.0, 0.0, 10.0, 25.0, 35.0, 50.0] {
            let call = |name: &str| -> f64 {
                pythermal_utils
                    .getattr(name)
                    .unwrap()
                    .call1((tdb,))
                    .unwrap()
                    .extract()
                    .unwrap()
            };

            // p_sat is in Pa on both sides
            assert_abs_diff_eq!(
                p_sat(Temperature::from_celsius(tdb)).as_pascals(),
                call("p_sat"),
                epsilon = 0.1
            );
            // Python returns torr here; the Rust port normalises to Pa
            assert_abs_diff_eq!(
                p_sat_torr(Temperature::from_celsius(tdb)).as_pascals(),
                call("p_sat_torr") * 133.322,
                epsilon = 0.1
            );
            // Python's antoine returns kPa; the Rust port normalises to Pa
            assert_abs_diff_eq!(
                p_sat_antoine(Temperature::from_celsius(tdb)).as_pascals(),
                call("antoine") * 1000.0,
                epsilon = 0.1
            );
        }
    });
}

#[test]
fn test_compare_enthalpy_air() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for (tdb, hr) in [
            (0.0, 0.001),
            (15.0, 0.005),
            (25.0, 0.01),
            (30.0, 0.02),
            (40.0, 0.03),
        ] {
            let py_h: f64 = pythermal_utils
                .getattr("enthalpy_air")
                .unwrap()
                .call1((tdb, hr))
                .unwrap()
                .extract()
                .unwrap();

            let rust_h = enthalpy_air(Temperature::from_celsius(tdb), hr);
            assert_abs_diff_eq!(rust_h, py_h, epsilon = 1.0);
        }
    });
}

#[test]
fn test_compare_f_svv() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for (w, h, d) in [
            (3.0, 2.0, 1.0),
            (1.0, 1.0, 1.0),
            (5.0, 3.0, 2.0),
            (2.0, 4.0, 0.5),
            (10.0, 10.0, 5.0),
        ] {
            let py_f: f64 = pythermal_utils
                .getattr("f_svv")
                .unwrap()
                .call1((w, h, d))
                .unwrap()
                .extract()
                .unwrap();

            let rust_f = f_svv(FSvvInputs {
                width: Length::from_meters(w),
                height: Length::from_meters(h),
                distance: Length::from_meters(d),
            });
            assert_abs_diff_eq!(rust_f, py_f, epsilon = 1e-6);
        }
    });
}

#[test]
fn test_compare_clo_area_factor() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for clo in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0] {
            let py_f: f64 = pythermal_utils
                .getattr("clo_area_factor")
                .unwrap()
                .call1((clo,))
                .unwrap()
                .extract()
                .unwrap();

            let rust_f = clo_area_factor(ClothingInsulation::from_clo(clo));
            assert_abs_diff_eq!(rust_f, py_f, epsilon = 1e-9);
        }
    });
}

#[test]
fn test_compare_operative_temperature() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for (tdb, tr, v) in [
            (25.0, 27.0, 0.3),
            (20.0, 20.0, 0.1),
            (30.0, 35.0, 0.8),
            (18.0, 22.0, 0.05),
            (28.0, 26.0, 1.5),
        ] {
            // Both editions: Rust's `use_ashrae` flag selects between them
            for (use_ashrae, standard) in [(false, "ISO"), (true, "ASHRAE")] {
                let kwargs = [("standard", standard)].into_py_dict(py).unwrap();
                let py_to: f64 = pythermal_utils
                    .getattr("operative_tmp")
                    .unwrap()
                    .call((tdb, tr, v), Some(&kwargs))
                    .unwrap()
                    .extract()
                    .unwrap();

                let rust_to = operative_temperature(
                    OperativeTemperatureInputs {
                        tdb: Temperature::from_celsius(tdb),
                        tr: Temperature::from_celsius(tr),
                        v: Speed::from_meters_per_second(v),
                    },
                    OperativeTemperatureOptions { use_ashrae },
                );
                assert_abs_diff_eq!(rust_to.as_celsius(), py_to, epsilon = 1e-6);
            }
        }
    });
}

#[test]
fn test_compare_body_surface_area() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for (weight, height) in [(70.0, 1.8), (50.0, 1.6), (95.0, 1.9), (60.0, 1.7)] {
            for (formula, name) in [
                (BsaFormula::DuBois, "dubois"),
                (BsaFormula::Takahira, "takahira"),
                (BsaFormula::Fujimoto, "fujimoto"),
                (BsaFormula::Kurazumi, "kurazumi"),
            ] {
                let py_bsa: f64 = pythermal_utils
                    .getattr("body_surface_area")
                    .unwrap()
                    .call1((weight, height, name))
                    .unwrap()
                    .extract()
                    .unwrap();

                let rust_bsa = body_surface_area(
                    BodySurfaceAreaInputs {
                        weight: Mass::from_kilograms(weight),
                        height: Length::from_meters(height),
                    },
                    BodySurfaceAreaOptions { formula },
                );
                assert_abs_diff_eq!(rust_bsa.as_square_meters(), py_bsa, epsilon = 1e-6);
            }
        }
    });
}

#[test]
fn test_compare_clo_insulation_helpers() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // (vr, v_walk, i_a_static, i_cl, i_t)
        for (vr, v_walk, i_a, i_cl, i_t) in [
            (0.3, 0.2, 0.7, 1.0, 1.2),
            (0.1, 0.0, 0.7, 0.5, 0.9),
            (0.8, 0.5, 0.7, 1.5, 2.0),
            (1.5, 0.7, 0.6, 0.3, 0.8),
            (0.2, 0.1, 0.7, 0.0, 0.7),
        ] {
            let py_ccfe: f64 = pythermal_utils
                .getattr("clo_correction_factor_environment")
                .unwrap()
                .call1((vr, v_walk, i_cl))
                .unwrap()
                .extract()
                .unwrap();
            assert_abs_diff_eq!(
                clo_correction_factor_environment(CloCorrectionFactorEnvironmentInputs {
                    vr: Speed::from_meters_per_second(vr),
                    v_walk: Speed::from_meters_per_second(v_walk),
                    i_cl: ClothingInsulation::from_clo(i_cl),
                }),
                py_ccfe,
                epsilon = 1e-6
            );

            let py_cial: f64 = pythermal_utils
                .getattr("clo_insulation_air_layer")
                .unwrap()
                .call1((vr, v_walk, i_a))
                .unwrap()
                .extract()
                .unwrap();
            assert_abs_diff_eq!(
                clo_insulation_air_layer(CloInsulationAirLayerInputs {
                    vr: Speed::from_meters_per_second(vr),
                    v_walk: Speed::from_meters_per_second(v_walk),
                    i_a_static: ClothingInsulation::from_clo(i_a),
                }),
                py_cial,
                epsilon = 1e-6
            );

            let py_cti: f64 = pythermal_utils
                .getattr("clo_total_insulation")
                .unwrap()
                .call1((i_t, vr, v_walk, i_a, i_cl))
                .unwrap()
                .extract()
                .unwrap();
            assert_abs_diff_eq!(
                clo_total_insulation(CloTotalInsulationInputs {
                    i_t: ClothingInsulation::from_clo(i_t),
                    vr: Speed::from_meters_per_second(vr),
                    v_walk: Speed::from_meters_per_second(v_walk),
                    i_a_static: ClothingInsulation::from_clo(i_a),
                    i_cl: ClothingInsulation::from_clo(i_cl),
                }),
                py_cti,
                epsilon = 1e-6
            );
        }
    });
}

#[test]
fn test_compare_clo_dynamic_ashrae() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Spans the met <= 1.2 branch (no correction) and the active branch
        for (clo, met) in [
            (0.5, 1.0),
            (0.5, 1.2),
            (0.5, 1.4),
            (1.0, 2.0),
            (1.5, 3.0),
            (0.3, 4.0),
        ] {
            let py_clo: f64 = pythermal_utils
                .getattr("clo_dynamic_ashrae")
                .unwrap()
                .call1((clo, met))
                .unwrap()
                .extract()
                .unwrap();

            let rust_clo = clo_dynamic_ashrae(
                CloDynamicAshraeInputs {
                    clo: ClothingInsulation::from_clo(clo),
                    met: MetabolicRate::from_met(met),
                },
                CloDynamicAshraeOptions::default(),
            );
            assert_abs_diff_eq!(rust_clo.as_clo(), py_clo, epsilon = 1e-6);
        }
    });
}

#[test]
fn test_compare_mean_radiant_temperature() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        for (tg, tdb, v, d, emissivity) in [
            (30.0, 25.0, 0.3, 0.15, 0.95),
            (40.0, 30.0, 0.1, 0.15, 0.95),
            (22.0, 24.0, 0.8, 0.1, 0.9),
            (50.0, 35.0, 1.5, 0.2, 0.95),
        ] {
            // Rust's `use_iso` flag selects between the two standards
            for (use_iso, standard) in [(false, "Mixed Convection"), (true, "ISO")] {
                let kwargs = [("standard", standard)].into_py_dict(py).unwrap();
                let py_mrt: f64 = pythermal_utils
                    .getattr("mean_radiant_tmp")
                    .unwrap()
                    .call((tg, tdb, v, d, emissivity), Some(&kwargs))
                    .unwrap()
                    .extract()
                    .unwrap();

                let rust_mrt = mean_radiant_temperature(
                    MeanRadiantTemperatureInputs {
                        tg: Temperature::from_celsius(tg),
                        tdb: Temperature::from_celsius(tdb),
                        v: Speed::from_meters_per_second(v),
                    },
                    MeanRadiantTemperatureOptions {
                        d: Length::from_meters(d),
                        emissivity,
                        use_iso,
                    },
                );
                // d=0.2 is outside the [0.04, 0.15] applicability range, so both
                // sides yield NaN under Mixed Convection; NaN never compares equal.
                if py_mrt.is_nan() {
                    assert!(
                        rust_mrt.as_celsius().is_nan(),
                        "{standard}: Python NaN but Rust {} at tg={tg} d={d}",
                        rust_mrt.as_celsius()
                    );
                } else {
                    assert_abs_diff_eq!(rust_mrt.as_celsius(), py_mrt, epsilon = 1e-4);
                }
            }
        }
    });
}

#[test]
fn test_compare_running_mean_outdoor_temperature() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        let series: [&[f64]; 4] = [
            &[20.0, 19.0, 18.0, 21.0, 22.0, 23.0, 20.0],
            &[10.0, 12.0, 14.0, 11.0, 9.0, 8.0, 13.0],
            &[30.0, 31.0, 29.0],
            &[25.0],
        ];

        for temps in series {
            for alpha in [0.8, 0.6, 0.9] {
                for (units, py_units) in [(Units::SI, "SI"), (Units::IP, "IP")] {
                    // Python's units="IP" reinterprets its raw list as Fahrenheit and
                    // converts to SI before averaging (utilities.py:944-950); feed it the
                    // IP-valued equivalents of the same SI environment Rust uses, as
                    // `sweep_running_mean_and_ensemble` does.
                    let py_temps: Vec<f64> = if units == Units::IP {
                        temps.iter().map(|t| t * 9.0 / 5.0 + 32.0).collect()
                    } else {
                        temps.to_vec()
                    };

                    let py_rmot: f64 = pythermal_utils
                        .getattr("running_mean_outdoor_temperature")
                        .unwrap()
                        .call1((py_temps, alpha, py_units))
                        .unwrap()
                        .extract()
                        .unwrap();

                    let rust_input: Vec<Temperature> = temps
                        .iter()
                        .map(|t| Temperature::from_celsius(*t))
                        .collect();
                    let rust_rmot = running_mean_outdoor_temperature(
                        &rust_input,
                        RunningMeanOutdoorTemperatureOptions { alpha, units },
                    );

                    let rust_value = match units {
                        Units::SI => rust_rmot.as_celsius(),
                        Units::IP => rust_rmot.as_fahrenheit(),
                    };
                    assert_abs_diff_eq!(rust_value, py_rmot, epsilon = 1e-6);
                }
            }
        }
    });
}

#[test]
fn test_compare_use_fans_heatwaves() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Heatwave conditions where fan use is the question: hot, spanning humidities
        // that flip the heat-strain flags.
        for (tdb, tr, v, rh, met, clo) in [
            (40.0, 40.0, 0.6, 40.0, 1.2, 0.5),
            (40.0, 40.0, 0.2, 40.0, 1.2, 0.5),
            (45.0, 45.0, 0.8, 20.0, 1.1, 0.3),
            (38.0, 38.0, 4.0, 60.0, 1.2, 0.5),
            (42.0, 42.0, 0.6, 70.0, 1.3, 0.6),
            (35.0, 35.0, 0.2, 30.0, 1.0, 0.4),
        ] {
            let py_result = pythermal
                .getattr("use_fans_heatwaves")
                .unwrap()
                .call1((tdb, tr, v, rh, met, clo))
                .unwrap();

            let rust_result = use_fans_heatwaves(
                UseFansHeatwavesInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            let numeric_fields: [(&str, f64); 12] = [
                ("e_skin", rust_result.e_skin.as_watts_per_square_meter()),
                ("e_rsw", rust_result.e_rsw.as_watts_per_square_meter()),
                ("e_max", rust_result.e_max.as_watts_per_square_meter()),
                (
                    "q_sensible",
                    rust_result.q_sensible.as_watts_per_square_meter(),
                ),
                ("q_skin", rust_result.q_skin.as_watts_per_square_meter()),
                ("q_res", rust_result.q_res.as_watts_per_square_meter()),
                ("t_core", rust_result.t_core.as_celsius()),
                ("t_skin", rust_result.t_skin.as_celsius()),
                ("m_bl", rust_result.m_bl),
                ("m_rsw", rust_result.m_rsw),
                ("w", rust_result.w),
                ("w_max", rust_result.w_max),
            ];
            for (field, rust_value) in numeric_fields {
                let py_value: f64 = py_result.getattr(field).unwrap().extract().unwrap();
                if py_value.is_nan() {
                    assert!(
                        rust_value.is_nan(),
                        "{field}: Python NaN but Rust {rust_value} at tdb={tdb} v={v} rh={rh}",
                    );
                } else {
                    assert_abs_diff_eq!(rust_value, py_value, epsilon = 0.1);
                }
            }

            // The heat-strain flags are the model's actual verdict on fan use
            for (field, rust_flag) in [
                ("heat_strain", rust_result.heat_strain),
                ("heat_strain_blood_flow", rust_result.heat_strain_blood_flow),
                ("heat_strain_w", rust_result.heat_strain_w),
                ("heat_strain_sweating", rust_result.heat_strain_sweating),
            ] {
                // Python reports these as float64 0.0/1.0 rather than bool
                // Python reports these as float64 0.0/1.0, or NaN when the inputs fall
                // outside the applicability limits, which maps to None on the Rust side.
                let py_raw: f64 = py_result.getattr(field).unwrap().extract().unwrap();
                let py_flag = if py_raw.is_nan() {
                    None
                } else {
                    Some(py_raw != 0.0)
                };
                assert_eq!(
                    rust_flag, py_flag,
                    "{field} mismatch at tdb={tdb} v={v} rh={rh}: Rust {rust_flag:?}, Python {py_flag:?}",
                );
            }
        }
    });
}

#[test]
fn test_compare_hr_to_rh() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        let test_cases = vec![
            (0.001, 5.0),
            (0.005, 20.0),
            (0.01, 25.0),
            (0.015, 28.0),
            (0.02, 30.0),
            (0.025, 35.0),
        ];

        for (hr, tdb) in test_cases {
            let py_rh: f64 = pythermal_utils
                .getattr("hr_to_rh")
                .unwrap()
                .call1((hr, tdb))
                .unwrap()
                .extract()
                .unwrap();

            let rust_rh = hr_to_rh(
                hr,
                Temperature::from_celsius(tdb),
                Pressure::from_pascals(101325.0),
            );

            assert_abs_diff_eq!(rust_rh, py_rh, epsilon = 1e-4);
        }
    });
}

#[test]
fn test_compare_clo_dynamic_iso() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // met spans the range where the ISO 9920 walking-speed formula clips at both
        // ends: 0 below ~1.19 met, and 0.7 m/s above ~3.3 met.
        let test_cases = vec![
            (0.5, 1.0, 0.1),
            (0.5, 1.2, 0.1),
            (0.7, 1.5, 0.2),
            (1.0, 2.0, 0.3),
            (1.5, 3.0, 0.5),
            (1.0, 4.0, 0.2),
        ];

        for (clo, met, v) in test_cases {
            let py_clo_dyn: f64 = pythermal_utils
                .getattr("clo_dynamic_iso")
                .unwrap()
                .call1((clo, met, v))
                .unwrap()
                .extract()
                .unwrap();

            let rust_clo_dyn = thermalcomfort::utilities::clo_dynamic_iso(
                CloDynamicIsoInputs {
                    clo: ClothingInsulation::from_clo(clo),
                    met: MetabolicRate::from_met(met),
                    v: Speed::from_meters_per_second(v),
                },
                CloDynamicIsoOptions::default(),
            );

            assert_abs_diff_eq!(rust_clo_dyn, py_clo_dyn, epsilon = 1e-5);
        }
    });
}

#[test]
fn test_v_relative() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        let test_cases = vec![
            (0.1, 1.0),  // met <= 1.0, should return v
            (0.1, 1.2),  // low activity
            (0.1, 1.4),  // medium activity
            (0.15, 2.0), // high activity
            (0.2, 3.0),  // very high activity
            (0.5, 1.5),  // higher base velocity
        ];

        for (v, met) in test_cases {
            println!("\nTesting v_relative: v={}, met={}", v, met);

            // Call Python function
            let py_vr: f64 = pythermal_utils
                .getattr("v_relative")
                .unwrap()
                .call1((v, met))
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_vr = v_relative(
                Speed::from_meters_per_second(v),
                MetabolicRate::from_met(met),
            );

            println!("  Python: {:.3}", py_vr);
            println!("  Rust:   {:.3}", rust_vr.as_meters_per_second());

            assert_abs_diff_eq!(rust_vr.as_meters_per_second(), py_vr, epsilon = 0.001);
        }
    });
}

#[test]
fn test_wet_bulb_temperature() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        let test_cases = vec![
            (25.0, 50.0), // standard conditions
            (20.0, 60.0), // cool and humid
            (30.0, 40.0), // hot and dry
            (15.0, 80.0), // cold and humid
            (28.0, 30.0), // hot and dry
            (10.0, 90.0), // cold and very humid
        ];

        for (tdb, rh) in test_cases {
            println!("\nTesting wet_bulb_temperature: tdb={}, rh={}", tdb, rh);

            // Call Python function
            let py_twb: f64 = pythermal_utils
                .getattr("wet_bulb_tmp")
                .unwrap()
                .call1((tdb, rh))
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_twb =
                wet_bulb_temperature(Temperature::from_celsius(tdb), Humidity::from_percent(rh));

            println!("  Python: {:.2}°C", py_twb);
            println!("  Rust:   {:.2}°C", rust_twb.as_celsius());

            assert_abs_diff_eq!(rust_twb.as_celsius(), py_twb, epsilon = 0.1);
        }
    });
}

#[test]
fn test_dew_point_temperature() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        let test_cases = vec![
            (25.0, 50.0),
            (20.0, 60.0),
            (30.0, 40.0),
            (15.0, 80.0),
            (28.0, 30.0),
        ];

        for (tdb, rh) in test_cases {
            println!("\nTesting dew_point_temperature: tdb={}, rh={}", tdb, rh);

            // Call Python function
            let py_tdp: f64 = pythermal_utils
                .getattr("dew_point_tmp")
                .unwrap()
                .call1((tdb, rh))
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_tdp =
                dew_point_temperature(Temperature::from_celsius(tdb), Humidity::from_percent(rh));

            println!("  Python: {:.2}°C", py_tdp);
            println!("  Rust:   {:.2}°C", rust_tdp.as_celsius());

            assert_abs_diff_eq!(rust_tdp.as_celsius(), py_tdp, epsilon = 0.1);
        }
    });
}

#[test]
fn test_psychrometrics() {
    Python::with_gil(|py| {
        let pythermal_utils = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        let test_cases = vec![
            (25.0, 50.0, 101325.0),
            (20.0, 60.0, 101325.0),
            (30.0, 40.0, 101325.0),
            (22.0, 55.0, 101325.0),
        ];

        for (tdb, rh, p_atm) in test_cases {
            println!(
                "\nTesting psychrometrics: tdb={}, rh={}, p_atm={}",
                tdb, rh, p_atm
            );

            // Call Python function
            let py_result = pythermal_utils
                .getattr("psy_ta_rh")
                .unwrap()
                .call1((tdb, rh, p_atm))
                .unwrap();

            let py_p_sat: f64 = py_result.getattr("p_sat").unwrap().extract().unwrap();
            let py_p_vap: f64 = py_result.getattr("p_vap").unwrap().extract().unwrap();
            let py_hr: f64 = py_result.getattr("hr").unwrap().extract().unwrap();
            let py_twb: f64 = py_result
                .getattr("wet_bulb_tmp")
                .unwrap()
                .extract()
                .unwrap();
            let py_tdp: f64 = py_result
                .getattr("dew_point_tmp")
                .unwrap()
                .extract()
                .unwrap();
            let py_h: f64 = py_result.getattr("h").unwrap().extract().unwrap();

            // Call Rust function
            let rust_result = psy_ta_rh(
                PsyTaRhInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                PsyTaRhOptions {
                    p_atm: Pressure::from_pascals(p_atm),
                },
            );

            println!(
                "  p_sat: Python={:.1}, Rust={:.1}",
                py_p_sat,
                rust_result.p_sat.as_pascals()
            );
            println!(
                "  p_vap: Python={:.1}, Rust={:.1}",
                py_p_vap,
                rust_result.p_vap.as_pascals()
            );
            println!("  hr: Python={:.5}, Rust={:.5}", py_hr, rust_result.hr);
            println!(
                "  t_wb: Python={:.2}, Rust={:.2}",
                py_twb,
                rust_result.t_wb.as_celsius()
            );
            println!(
                "  t_dp: Python={:.2}, Rust={:.2}",
                py_tdp,
                rust_result.t_dp.as_celsius()
            );
            println!("  h: Python={:.1}, Rust={:.1}", py_h, rust_result.h);

            assert_abs_diff_eq!(rust_result.p_sat.as_pascals(), py_p_sat, epsilon = 1.0);
            assert_abs_diff_eq!(rust_result.p_vap.as_pascals(), py_p_vap, epsilon = 1.0);
            assert_abs_diff_eq!(rust_result.hr, py_hr, epsilon = 0.0001);
            assert_abs_diff_eq!(rust_result.t_wb.as_celsius(), py_twb, epsilon = 0.1);
            assert_abs_diff_eq!(rust_result.t_dp.as_celsius(), py_tdp, epsilon = 0.1);
            assert_abs_diff_eq!(rust_result.h, py_h, epsilon = 10.0);
        }
    });
}

#[test]
fn test_pmv_ppd_iso_outside_limits() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test cases that are outside ISO limits (should return NaN with limit_inputs=true)
        let test_cases = vec![
            (5.0, 25.0, 0.1, 50.0, 1.2, 0.5, "tdb too low"),
            (35.0, 25.0, 0.1, 50.0, 1.2, 0.5, "tdb too high"),
            (25.0, 5.0, 0.1, 50.0, 1.2, 0.5, "tr too low"),
            (25.0, 45.0, 0.1, 50.0, 1.2, 0.5, "tr too high"),
            (25.0, 25.0, 2.0, 50.0, 1.2, 0.5, "vr too high"),
            (25.0, 25.0, 0.1, 50.0, 0.5, 0.5, "met too low"),
            (25.0, 25.0, 0.1, 50.0, 5.0, 0.5, "met too high"),
            (25.0, 25.0, 0.1, 50.0, 1.2, 2.5, "clo too high"),
        ];

        for (tdb, tr, vr, rh, met, clo, description) in test_cases {
            println!(
                "\nTesting outside limits ({}): tdb={}, tr={}, vr={}, rh={}, met={}, clo={}",
                description, tdb, tr, vr, rh, met, clo
            );

            // Call Python function with limit_inputs=True (default)
            let py_result = pythermal
                .getattr("pmv_ppd_iso")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();

            // Call Rust function with measurement types
            let rust_result = pmv_ppd_iso(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            println!("  Python PMV is NaN: {}", py_pmv.is_nan());
            println!("  Rust PMV is NaN: {}", rust_result.pmv.is_nan());

            // Both should return NaN
            assert_eq!(
                py_pmv.is_nan(),
                rust_result.pmv.is_nan(),
                "Mismatch in NaN behavior for {}",
                description
            );
        }
    });
}

/// `tsv` on the band edges, where ISO and ASHRAE are supposed to disagree.
///
/// ISO 7730 maps PMV to a sensation band with `right=False` and ASHRAE 55 with
/// `right=True`, so a PMV sitting *exactly* on an edge lands one band apart between the
/// two models. Conflating the two band sets was a real bug here (`b88bbc0`).
///
/// **The differential sweep cannot cover this**, which is why it needs a fixture test.
/// Adding `tsv` to `sweep_pmv_ppd_iso`/`sweep_pmv_ppd_ashrae` catches a mislabelled band
/// (verified by mislabelling one), but *not* swapping ISO onto ASHRAE's right-closed
/// bands: the closure direction only changes an answer when PMV equals an edge exactly,
/// and the sweep's random reals never land there. Giving ISO `right=True` leaves both
/// sweeps green -- verified by doing it.
///
/// The `tdb` values below were found by scanning for inputs whose rounded PMV is exactly
/// an edge. They are fixtures, not magic: each case asserts Python still returns that
/// exact PMV, so if an upstream formula change moves them this fails loudly rather than
/// quietly testing an interior point.
#[test]
fn test_pmv_tsv_band_edges_iso_versus_ashrae() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb, the exact PMV it produces, ISO's band, ASHRAE's band)
        let cases = [
            (16.551, -2.5, "Cool", "Cold"),
            (19.772, -1.5, "Slightly Cool", "Cool"),
            (23.013, -0.5, "Neutral", "Slightly Cool"),
            (26.369, 0.5, "Slightly Warm", "Neutral"),
            (29.641, 1.5, "Warm", "Slightly Warm"),
            (32.827, 2.5, "Hot", "Warm"),
        ];

        for (tdb, edge, iso_band, ashrae_band) in cases {
            let (tr, vr, rh, met, clo) = (tdb, 0.1, 50.0, 1.2, 0.5);
            let kwargs = [("limit_inputs", PyBool::new(py, false).to_owned().into_any())]
                .into_py_dict(py)
                .unwrap();

            for (py_fn, expected_band) in
                [("pmv_ppd_iso", iso_band), ("pmv_ppd_ashrae", ashrae_band)]
            {
                let label = format!("{py_fn} at pmv={edge}");
                let py_result = pythermal
                    .getattr(py_fn)
                    .unwrap()
                    .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                    .unwrap_or_else(|e| panic!("{label}: raised: {e}"));

                let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
                let py_tsv: String = py_result.getattr("tsv").unwrap().extract().unwrap();

                // The fixture guard: if this trips, the inputs no longer sit on the edge
                // and the rest of the case proves nothing.
                assert_eq!(
                    py_pmv, edge,
                    "{label}: tdb={tdb} no longer lands on the band edge (got {py_pmv}); \
                     rescan for a tdb whose PMV is exactly {edge}"
                );
                assert_eq!(
                    py_tsv, expected_band,
                    "{label}: upstream's band for an exact edge changed"
                );

                let inputs = PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                };
                let rust_tsv = if py_fn == "pmv_ppd_iso" {
                    pmv_ppd_iso(
                        inputs,
                        PmvPpdIsoOptions {
                            limit_inputs: false,
                            ..Default::default()
                        },
                    )
                    .tsv
                } else {
                    pmv_ppd_ashrae(
                        inputs,
                        PmvPpdAshraeOptions {
                            limit_inputs: false,
                            ..Default::default()
                        },
                    )
                    .tsv
                };

                assert_eq!(
                    rust_tsv.map(|c| c.as_str()),
                    Some(expected_band),
                    "{label}: Rust band differs from Python's"
                );
            }
        }
    });
}

/// `compliance` either side of ASHRAE 55's -0.5 < PMV < 0.5 comfort band.
///
/// Like `tsv`, this is a discrete output driven by a float, and like `tsv` the
/// differential sweep cannot pin its edges: `sweep_pmv_ppd_ashrae` catches an *inverted*
/// compliance flag but not a band widened by 0.01, nor evaluating the criterion on the
/// rounded PMV instead of the unrounded one -- both verified by injecting them. Each
/// needs a PMV inside a window a few thousandths wide, which uniform random sampling
/// does not reliably visit.
///
/// The `round_output = true` half is what pins the rounded-vs-unrounded distinction:
/// upstream evaluates the criterion on the unrounded PMV, so an input whose PMV is
/// 0.4961 is compliant even though the reported PMV rounds to 0.5, which the criterion
/// would reject. Two of the cases below sit in exactly that gap.
///
/// As in [`test_pmv_tsv_band_edges_iso_versus_ashrae`], the `tdb` values are fixtures and
/// each case asserts Python's PMV still falls in the intended window.
///
/// What this deliberately does **not** cover: swapping the strict `<`/`>` for `<=`/`>=`.
/// That changes an answer only when the *unrounded* PMV is exactly ±0.5, and no input
/// produces that -- the criterion runs before rounding, so the exact-edge case `tsv` has
/// (where `round_to(pmv, 2)` lands on 0.5 constantly) does not arise here. The
/// substitution is untestable rather than untested; do not add a case chasing it.
#[test]
fn test_pmv_compliance_band_edges() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb, the PMV window it must land in, expected compliance)
        let cases = [
            // Rounds to 0.5 but is below it: compliant only on the unrounded value.
            (26.3725, (0.4950, 0.4999), true),
            // Just outside the band, and inside a 0.01-widened one.
            (26.3860, (0.5001, 0.5099), false),
            // Rounds to -0.5 but is above it: the negative-side mirror.
            (23.0295, (-0.4999, -0.4950), true),
            (23.0155, (-0.5099, -0.5001), false),
        ];

        for (tdb, (lo, hi), expected) in cases {
            let (tr, vr, rh, met, clo) = (tdb, 0.1, 50.0, 1.2, 0.5);

            for round_output in [false, true] {
                let label = format!("tdb={tdb} round_output={round_output}");
                let kwargs = [
                    ("limit_inputs", PyBool::new(py, false).to_owned().into_any()),
                    (
                        "round_output",
                        PyBool::new(py, round_output).to_owned().into_any(),
                    ),
                ]
                .into_py_dict(py)
                .unwrap();

                let py_result = pythermal
                    .getattr("pmv_ppd_ashrae")
                    .unwrap()
                    .call((tdb, tr, vr, rh, met, clo), Some(&kwargs))
                    .unwrap_or_else(|e| panic!("{label}: raised: {e}"));

                let py_compliance =
                    extract_optional_bool(&py_result.getattr("compliance").unwrap());

                let rust_result = pmv_ppd_ashrae(
                    PmvPpdInputs {
                        tdb: Temperature::from_celsius(tdb),
                        tr: Temperature::from_celsius(tr),
                        vr: Speed::from_meters_per_second(vr),
                        rh: Humidity::from_percent(rh),
                        met: MetabolicRate::from_met(met),
                        clo: ClothingInsulation::from_clo(clo),
                    },
                    PmvPpdAshraeOptions {
                        limit_inputs: false,
                        round_output,
                        ..Default::default()
                    },
                );

                // The fixture guard: read the unrounded PMV so the window check means
                // the same thing on both passes. If this trips, `tdb` has drifted out of
                // the window and the case no longer probes the edge it was chosen for.
                let unrounded = pmv_ppd_ashrae(
                    PmvPpdInputs {
                        tdb: Temperature::from_celsius(tdb),
                        tr: Temperature::from_celsius(tr),
                        vr: Speed::from_meters_per_second(vr),
                        rh: Humidity::from_percent(rh),
                        met: MetabolicRate::from_met(met),
                        clo: ClothingInsulation::from_clo(clo),
                    },
                    PmvPpdAshraeOptions {
                        limit_inputs: false,
                        round_output: false,
                        ..Default::default()
                    },
                )
                .pmv;
                assert!(
                    (lo..=hi).contains(&unrounded),
                    "{label}: PMV {unrounded} left the ({lo}, {hi}) window this case \
                     exists to probe; rescan for a tdb inside it"
                );

                assert_eq!(
                    py_compliance,
                    Some(expected),
                    "{label}: upstream's compliance for this window changed"
                );
                assert_eq!(
                    rust_result.compliance, py_compliance,
                    "{label}: Rust compliance differs from Python's"
                );
            }
        }
    });
}

#[test]
fn test_pmv_sequential_scenarios() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test multiple scenarios in sequence
        let scenarios = vec![
            // Neutral comfort
            (22.0, 22.0, 0.1, 50.0, 1.2, 1.0),
            // Slightly warm
            (26.0, 26.0, 0.1, 50.0, 1.2, 0.5),
            // Slightly cool
            (20.0, 20.0, 0.1, 50.0, 1.2, 1.0),
        ];

        for (tdb, tr, vr, rh, met, clo) in scenarios {
            let py_result = pythermal
                .getattr("pmv_ppd_iso")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
            let rust_result = pmv_ppd_iso(
                PmvPpdInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.pmv, py_pmv, epsilon = 0.02);
        }
    });
}

#[test]
fn test_compare_two_nodes_gagge() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Broadened sweep: cool/comfortable/hot conditions across the activity
        // (met) and clothing (clo) ranges, with low and elevated air speeds and
        // varied humidity. Loose ε reflects the iterative nature of the model.
        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            (20.0, 20.0, 0.1, 50.0, 1.0, 1.0),
            (28.0, 28.0, 0.3, 60.0, 1.5, 0.3),
            (22.0, 24.0, 0.15, 40.0, 1.1, 0.7),
            (32.0, 32.0, 0.5, 70.0, 1.6, 0.3),
            (18.0, 18.0, 0.1, 30.0, 1.0, 1.2),
            (26.0, 26.0, 0.8, 50.0, 2.0, 0.4),
        ];

        for (tdb, tr, v, rh, met, clo) in test_cases {
            println!(
                "\nTesting two_nodes_gagge: tdb={}, tr={}, v={}, rh={}, met={}, clo={}",
                tdb, tr, v, rh, met, clo
            );

            // Call Python function
            let py_result = pythermal
                .getattr("two_nodes_gagge")
                .unwrap()
                .call1((tdb, tr, v, rh, met, clo))
                .unwrap();

            let get =
                |name: &str| -> f64 { py_result.getattr(name).unwrap().extract::<f64>().unwrap() };
            let py_set = get("set");
            let py_e_skin = get("e_skin");
            let py_e_rsw = get("e_rsw");
            let py_e_max = get("e_max");
            let py_q_sensible = get("q_sensible");
            let py_q_skin = get("q_skin");
            let py_q_res = get("q_res");
            let py_t_core = get("t_core");
            let py_t_skin = get("t_skin");
            let py_m_bl = get("m_bl");
            let py_m_rsw = get("m_rsw");
            let py_w = get("w");
            let py_w_max = get("w_max");
            let py_et = get("et");
            let py_pmv_gagge = get("pmv_gagge");
            let py_pmv_set = get("pmv_set");
            let py_disc = get("disc");
            let py_t_sens = get("t_sens");

            // Call Rust function with measurement types
            let rust_result = two_nodes_gagge(
                GaggeTwoNodesInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            // Two-node model is iterative — tolerances scale with field magnitude.
            assert_abs_diff_eq!(rust_result.set.as_celsius(), py_set, epsilon = 0.15);
            assert_abs_diff_eq!(
                rust_result.e_skin.as_watts_per_square_meter(),
                py_e_skin,
                epsilon = 1.0
            );
            assert_abs_diff_eq!(
                rust_result.e_rsw.as_watts_per_square_meter(),
                py_e_rsw,
                epsilon = 1.0
            );
            assert_abs_diff_eq!(
                rust_result.e_max.as_watts_per_square_meter(),
                py_e_max,
                epsilon = 1.5
            );
            assert_abs_diff_eq!(
                rust_result.q_sensible.as_watts_per_square_meter(),
                py_q_sensible,
                epsilon = 1.0
            );
            assert_abs_diff_eq!(
                rust_result.q_skin.as_watts_per_square_meter(),
                py_q_skin,
                epsilon = 1.0
            );
            assert_abs_diff_eq!(
                rust_result.q_res.as_watts_per_square_meter(),
                py_q_res,
                epsilon = 0.5
            );
            assert_abs_diff_eq!(rust_result.t_core.as_celsius(), py_t_core, epsilon = 0.1);
            assert_abs_diff_eq!(rust_result.t_skin.as_celsius(), py_t_skin, epsilon = 0.3);
            assert_abs_diff_eq!(rust_result.m_bl, py_m_bl, epsilon = 2.0);
            assert_abs_diff_eq!(rust_result.m_rsw, py_m_rsw, epsilon = 5.0);
            assert_abs_diff_eq!(rust_result.w, py_w, epsilon = 0.03);
            assert_abs_diff_eq!(rust_result.w_max, py_w_max, epsilon = 0.02);
            assert_abs_diff_eq!(rust_result.et.as_celsius(), py_et, epsilon = 0.3);
            assert_abs_diff_eq!(rust_result.pmv_gagge, py_pmv_gagge, epsilon = 0.05);
            assert_abs_diff_eq!(rust_result.pmv_set, py_pmv_set, epsilon = 0.05);
            assert_abs_diff_eq!(rust_result.disc, py_disc, epsilon = 0.2);
            assert_abs_diff_eq!(rust_result.t_sens, py_t_sens, epsilon = 0.2);
        }
    });
}

#[test]
fn test_compare_utci() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep covers every UTCI stress band so the categorical mapping is
        // exercised end-to-end.
        // Bands (°C): <-40, [-40,-27), [-27,-13), [-13,0), [0,9), [9,26),
        //             [26,32), [32,38), [38,46), >=46
        let test_cases = vec![
            (-45.0, -45.0, 3.0, 70.0),
            (-30.0, -30.0, 4.0, 60.0),
            (-15.0, -15.0, 3.0, 70.0),
            (-5.0, -5.0, 3.0, 80.0),
            (5.0, 5.0, 2.0, 70.0),
            (20.0, 20.0, 2.0, 50.0),
            (25.0, 25.0, 1.0, 50.0),
            (30.0, 30.0, 0.5, 60.0),
            (35.0, 35.0, 1.5, 40.0),
            (42.0, 42.0, 1.0, 50.0),
        ];

        for (tdb, tr, v, rh) in test_cases {
            // Call Python function
            let py_result = pythermal
                .getattr("utci")
                .unwrap()
                .call1((tdb, tr, v, rh))
                .unwrap();

            let py_utci: f64 = py_result.getattr("utci").unwrap().extract().unwrap();
            let py_stress = extract_category(&py_result.getattr("stress_category").unwrap());

            // Call Rust function with measurement types
            let rust_result = utci(
                UtciInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );

            // Compare results (UTCI polynomial should match very closely)
            assert_abs_diff_eq!(rust_result.utci, py_utci, epsilon = 0.1);
            assert_eq!(
                rust_result.stress_category.map(|c| c.as_str().to_string()),
                py_stress,
                "UTCI stress_category mismatch at tdb={tdb} tr={tr} v={v} rh={rh}",
            );
        }
    });
}

/// Every banded index compared against Python at its exact band edges.
///
/// The crate already has exact-edge tests for `StressCategory`, `HeatIndexStress`,
/// `HumidexDiscomfort` and `DiscomfortCondition`, and they are not useless: flipping
/// UTCI's bands from right-inclusive to right-open does fail
/// `models::utci::tests::test_stress_categories`. But they are `#[cfg(test)]` unit tests
/// that call the banding function with a literal and compare against a **transcribed
/// constant**, so they can only catch a later *regression* in a rule -- never a rule that
/// was wrong when it was transcribed, because the expectation was written from the same
/// reading of the standard as the code. That is exactly what `b88bbc0` was: ISO's
/// right-open bands conflated with ASHRAE's right-closed ones, self-consistently.
/// Comparing against pythermalcomfort is the only thing that can catch that class.
///
/// The differential sweep does not cover it either: with UTCI banded right-open instead
/// of right-inclusive, `sweep_utci` still passes (verified), because a randomised real
/// never lands exactly on an edge.
///
/// So these drive the public function end to end and compare the category against
/// Python's, at inputs whose index value lands exactly on each edge. The inputs were
/// found by scanning; each case asserts the index value is still the edge, so an upstream
/// formula change fails loudly rather than silently moving to an interior point.
///
/// Every band here is right-inclusive: the value on the edge belongs to the *lower* band.
#[test]
fn test_banded_indices_at_exact_edges() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb, the index value it produces, the category on that edge)
        let utci_cases = [
            (-39.320, -40.0, "extreme cold stress"),
            (-25.654, -27.0, "very strong cold stress"),
            (-11.208, -13.0, "strong cold stress"),
            (0.694, 0.0, "moderate cold stress"),
            (9.284, 9.0, "slight cold stress"),
            (26.188, 26.0, "no thermal stress"),
            (31.332, 32.0, "moderate heat stress"),
            (36.054, 38.0, "strong heat stress"),
            (41.312, 46.0, "very strong heat stress"),
        ];
        for (tdb, edge, expected) in utci_cases {
            let label = format!("utci edge {edge}");
            let kwargs = [("limit_inputs", PyBool::new(py, false).to_owned().into_any())]
                .into_py_dict(py)
                .unwrap();
            let py_result = models
                .getattr("utci")
                .unwrap()
                .call((tdb, tdb, 1.0, 50.0), Some(&kwargs))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_value: f64 = py_result.getattr("utci").unwrap().extract().unwrap();
            let py_category = extract_category(&py_result.getattr("stress_category").unwrap());

            let rust = utci(
                UtciInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tdb),
                    v: Speed::from_meters_per_second(1.0),
                    rh: Humidity::from_percent(50.0),
                },
                UtciOptions {
                    limit_inputs: false,
                    ..Default::default()
                },
            );

            assert_eq!(
                py_value, edge,
                "{label}: tdb={tdb} no longer lands on the edge (got {py_value}); rescan"
            );
            assert_eq!(
                py_category.as_deref(),
                Some(expected),
                "{label}: upstream's category on the edge changed"
            );
            assert_eq!(
                rust.stress_category.map(|c| c.as_str()),
                py_category.as_deref(),
                "{label}: Rust category differs from Python's"
            );
        }

        // Humidex bands the *rounded* value, so these edges are exactly reachable.
        let humidex_cases = [
            (26.1205, 30.0, "Little or no discomfort"),
            (29.2455, 35.0, "Noticeable discomfort"),
            (32.1895, 40.0, "Evident discomfort"),
            (34.9620, 45.0, "Intense discomfort; avoid exertion"),
            (39.5570, 54.0, "Dangerous discomfort"),
        ];
        for (tdb, edge, expected) in humidex_cases {
            let label = format!("humidex edge {edge}");
            let py_result = models
                .getattr("humidex")
                .unwrap()
                .call1((tdb, 50.0))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_value: f64 = py_result.getattr("humidex").unwrap().extract().unwrap();
            let py_category = extract_category(&py_result.getattr("discomfort").unwrap());

            let rust = humidex(
                HumidexInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(50.0),
                },
                HumidexOptions::default(),
            );

            assert_eq!(
                py_value, edge,
                "{label}: tdb={tdb} no longer lands on the edge (got {py_value}); rescan"
            );
            assert_eq!(
                py_category.as_deref(),
                Some(expected),
                "{label}: upstream changed"
            );
            assert_eq!(
                Some(rust.discomfort.as_str()),
                py_category.as_deref(),
                "{label}: Rust category differs from Python's"
            );
        }

        let hi_cases = [
            (26.4790, 27.0, "no risk"),
            (30.5885, 32.0, "caution"),
            (35.1155, 41.0, "extreme caution"),
            (39.7480, 54.0, "danger"),
        ];
        for (tdb, edge, expected) in hi_cases {
            let label = format!("heat_index_rothfusz edge {edge}");
            let kwargs = [("limit_inputs", PyBool::new(py, false).to_owned().into_any())]
                .into_py_dict(py)
                .unwrap();
            let py_result = models
                .getattr("heat_index_rothfusz")
                .unwrap()
                .call((tdb, 50.0), Some(&kwargs))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_value: f64 = py_result.getattr("hi").unwrap().extract().unwrap();
            let py_category = extract_category(&py_result.getattr("stress_category").unwrap());

            let rust = heat_index_rothfusz(
                HeatIndexRothfuszInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(50.0),
                },
                thermalcomfort::models::HeatIndexRothfuszOptions {
                    limit_inputs: false,
                    ..Default::default()
                },
            );

            assert_eq!(
                py_value, edge,
                "{label}: tdb={tdb} no longer lands on the edge (got {py_value}); rescan"
            );
            assert_eq!(
                py_category.as_deref(),
                Some(expected),
                "{label}: upstream changed"
            );
            assert_eq!(
                rust.stress_category.map(|c| c.as_str()),
                py_category.as_deref(),
                "{label}: Rust category differs from Python's"
            );
        }
    });
}

/// Weaker sibling of [`assert_is_really_the_edge`] for a bound that cannot be probed on
/// the edge itself: require only that Python's verdict differs a millidegree either side,
/// which still proves the edge lies between the two probes.
fn assert_brackets_the_edge(verdicts: [Option<bool>; 3], is_lower: bool, label: &str) {
    let [below, _, above] = verdicts;
    let (inside, outside) = if is_lower {
        (above, below)
    } else {
        (below, above)
    };
    assert_eq!(
        inside,
        Some(true),
        "{label}: Python rejects a millidegree inside the bound, so the bound is wrong"
    );
    assert_eq!(
        outside,
        Some(false),
        "{label}: Python still accepts a millidegree outside the bound, so this case does \
         not bracket the edge and proves nothing"
    );
}

/// Prove a reported band bound really is the edge: Python's own verdict must flip across
/// it. `is_lower` picks which side is expected to be outside the band.
fn assert_is_really_the_edge(verdicts: [Option<bool>; 3], is_lower: bool, label: &str) {
    let [below, at, above] = verdicts;
    assert_eq!(
        at,
        Some(true),
        "{label}: Python does not accept the bound itself, so it is not the edge"
    );
    let outside = if is_lower { below } else { above };
    assert_eq!(
        outside,
        Some(false),
        "{label}: Python still accepts a millidegree outside the bound, so the reported \
         (rounded) bound is not the true (unrounded) edge and this case proves nothing"
    );
}

/// `two_nodes_gagge`'s `calculate_ce=True` path, which upstream reaches through a wrapper
/// that silently drops most of its arguments.
///
/// `_gagge_two_nodes_optimized_return_set` (`two_nodes_gagge.py:580-604`) takes only
/// `(tdb, tr, v, met, clo, vapor_pressure, wme, body_surface_area, p_atm, position)`. Four
/// things follow, none of them in the docstring:
///
/// 1. `max_skin_blood_flow`, `max_sweating` and `w_max` never reach the kernel and fall
///    back to its defaults of 90, 500 and "compute from air speed" (`:215-223`). Whatever
///    the caller passed is discarded.
/// 2. `position` is passed as the literal `1`, which the kernel compares against the
///    *string* `Postures.sitting.value`, so the branch is always "standing".
/// 3. The branch `return SET(set=result)`s at `:142`, before the `if round_output:` block
///    at `:201`, so this path is never rounded.
/// 4. Only `set` exists on the result; it is a `SET`, not a `GaggeTwoNodes`.
///
/// This port got (1) and (3) wrong until 2026-08-26 and nothing noticed, because
/// `calculate_ce` had never been swept -- it sat at its `false` default, so an entire
/// upstream entry point was unexercised. Both bugs are invisible until a cap binds: below
/// saturation the two agree to ~1e-9. The case below has met=3.05, where `m_bl` reaches 90
/// upstream while this port was holding it at the caller's 82.37.
#[test]
fn test_two_nodes_gagge_calculate_ce_drops_caps_and_skips_rounding() {
    use measurements::{Area, Pressure};
    use thermalcomfort::models::GaggeTwoNodesOptions;
    use thermalcomfort::utilities::Posture;

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let (tdb, tr, v, rh, met, clo): (f64, f64, f64, f64, f64, f64) =
            (11.31472, 28.428285, 0.157185, 63.277979, 3.048367, 1.977141);
        // Deliberately *not* the kernel defaults, so passing them through would show.
        let (msbf, msw): (f64, f64) = (82.3708, 346.446716);

        for round_output in [false, true] {
            let label = format!("calculate_ce round_output={round_output}");
            let kwargs = [
                ("wme", 0.221401_f64.into_pyobject(py).unwrap().into_any()),
                (
                    "body_surface_area",
                    1.512569_f64.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "p_atm",
                    80765.553751_f64.into_pyobject(py).unwrap().into_any(),
                ),
                ("position", "sitting".into_pyobject(py).unwrap().into_any()),
                (
                    "max_skin_blood_flow",
                    msbf.into_pyobject(py).unwrap().into_any(),
                ),
                ("max_sweating", msw.into_pyobject(py).unwrap().into_any()),
                ("calculate_ce", PyBool::new(py, true).to_owned().into_any()),
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
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_set: f64 = py_result.getattr("set").unwrap().extract().unwrap();

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
                    wme: MetabolicRate::from_met(0.221401),
                    body_surface_area: Area::from_square_meters(1.512569),
                    p_atm: Pressure::from_pascals(80765.553751),
                    position: Posture::Sitting,
                    max_skin_blood_flow: msbf,
                    max_sweating: msw,
                    round_output,
                    w_max: Some(0.4),
                    calculate_ce: true,
                },
            );

            // The fixture guard: if a cap stops binding, the case stops testing (1).
            assert!(
                py_set > 36.0,
                "{label}: upstream returned {py_set}; these inputs no longer saturate, so \
                 the dropped-caps behaviour is not being exercised"
            );
            assert_abs_diff_eq!(rust.set.as_celsius(), py_set, epsilon = 1e-9);
        }
    });
}

/// `sports_heat_stress_risk`'s floor-with-nudge and its 4.9 ceiling.
///
/// `risk_level_interpolated` is `min(floor((risk + 1e-9) * 10) / 10, 4.9)`, and the
/// recommendation is derived from that floored value, so both the nudge and the clamp are
/// discrete-output logic driven by a float.
///
/// **The `1e-9` nudge was undetectable by the entire suite.** Removing it leaves the lib
/// tests, `sweep_sports_heat_stress_risk` and every parity test green -- verified by
/// removing it. It exists to stop a risk that should be exactly a tenth from flooring a
/// tenth *down* when the interpolation lands a few ULPs short, and it changes the answer
/// only for a risk within 1e-9 below a tenth. `d(risk)/d(tdb)` is about 0.22 here, so that
/// is a ~4.5e-9 window in `tdb` -- narrow, but four orders of magnitude wider than the
/// `Temperature` round-trip error, so unlike `ireq`'s ceiling this one *is* reachable
/// through the public API.
///
/// `tdb = 25.29999999` sits in that window: upstream and this port both report 1.2, and
/// without the nudge this port reports 1.1.
///
/// The 4.9 clamp is caught by `test_extreme_band_caps_at_4_9` in the model's own unit
/// tests but by nothing cross-library, so it is pinned against Python here too.
#[test]
fn test_sports_risk_level_floor_nudge_and_ceiling() {
    use thermalcomfort::models::sports_heat_stress_risk::{Sports, sports_heat_stress_risk};

    Python::with_gil(|py| {
        let sports_mod = import_reference(py, "pythermalcomfort.models.sports_heat_stress_risk")
            .expect("Failed to import sports_heat_stress_risk");
        let py_func = sports_mod.getattr("sports_heat_stress_risk").unwrap();
        let py_sports = sports_mod.getattr("Sports").unwrap();

        // (tdb, expected floored risk, what the case is for)
        let cases = [
            (25.29999999_f64, 1.2, "inside the 1e-9 floor nudge window"),
            (
                70.0_f64,
                4.9,
                "far above t_extreme, clamped by the 4.9 ceiling",
            ),
        ];

        for (tdb, expected, what) in cases {
            let label = format!("sports_heat_stress_risk [{what}]");
            let (tr, rh, vr) = (30.0, 50.0, 1.0);

            let kwargs = [("sport", py_sports.getattr("RUNNING").unwrap())]
                .into_py_dict(py)
                .unwrap();
            let py_result = py_func
                .call((tdb, tr, rh, vr), Some(&kwargs))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_risk: f64 = py_result
                .getattr("risk_level_interpolated")
                .unwrap()
                .extract()
                .unwrap();

            // The fixture guard: if upstream's value moved, the input has slid out of the
            // window it was chosen for and the case no longer probes anything.
            assert_eq!(
                py_risk, expected,
                "{label}: upstream now reports {py_risk}, not {expected}; re-bisect for an \
                 input inside the window"
            );

            let rust = sports_heat_stress_risk(SportsHeatStressRiskInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                rh: Humidity::from_percent(rh),
                vr: Speed::from_meters_per_second(vr),
                sport: Sports::RUNNING,
            })
            .unwrap_or_else(|e| panic!("{label}: Rust returned {e}"));

            assert_eq!(
                rust.risk_level_interpolated, py_risk,
                "{label}: Rust {}, Python {py_risk}",
                rust.risk_level_interpolated
            );

            let py_recommendation = extract_category(&py_result.getattr("recommendation").unwrap());
            assert_eq!(
                Some(rust.recommendation),
                py_recommendation.as_deref(),
                "{label}: recommendation differs"
            );
        }
    });
}

/// `use_fans_heatwaves`' heat-strain flags at and just below the saturation caps.
///
/// These are not band edges: `m_bl`, `w` and `m_rsw` are *clamped* to their caps inside
/// the two-node model, so a saturated value is bit-identical to the cap and upstream tests
/// it with `==`. The failure mode is therefore the opposite of the usual one -- not
/// "the edge was never reached" but "a tolerance was used where equality was meant",
/// which reports strain for a value merely *near* the cap.
///
/// That was a real bug here: a 1e-3 window called `heat_strain_w` true where Python
/// reports false. `use_fans_heatwaves.rs` cites `tdb=38.1, tr=43.7, v=2.16` for it, but
/// **those inputs no longer demonstrate it** -- at 4.4.2 they put `w` 0.032 away from
/// `w_max`, far outside any plausible window, so a test built on them would prove
/// nothing. The pair below was re-derived by scanning:
///
/// - **saturated**: `w == w_max` exactly, both `0.6710727624360568`; strain is true.
/// - **near-miss**: `w` and `w_max` differ by 4.7e-05; strain is false. This is the case
///   that discriminates -- any tolerance down to 1e-4 flips it.
#[test]
fn test_use_fans_heatwaves_strain_flags_at_the_caps() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb, tr, v, rh, saturated?, description)
        let cases = [
            (34.0, 39.0, 0.2, 70.0, true, "w == w_max exactly"),
            (35.5, 40.5, 4.0, 70.0, false, "w within 4.7e-05 of w_max"),
        ];

        for (tdb, tr, v, rh, expect_strain_w, what) in cases {
            let (met, clo) = (1.2, 0.5);
            let label = format!("use_fans_heatwaves [{what}]");

            let kwargs = [
                ("round_output", PyBool::new(py, false).to_owned().into_any()),
                ("limit_inputs", PyBool::new(py, false).to_owned().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();
            let py_result = pythermal
                .getattr("use_fans_heatwaves")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo), Some(&kwargs))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));

            let py_w: f64 = py_result.getattr("w").unwrap().extract().unwrap();
            let py_w_max: f64 = py_result.getattr("w_max").unwrap().extract().unwrap();
            let py_strain_w: f64 = py_result
                .getattr("heat_strain_w")
                .unwrap()
                .extract()
                .unwrap();

            // The fixture guard: each case only probes what it claims to if `w` is still
            // in the right relationship to the cap.
            if expect_strain_w {
                assert_eq!(
                    py_w, py_w_max,
                    "{label}: w is no longer clamped to the cap; rescan for a saturating input"
                );
            } else {
                let gap = (py_w_max - py_w).abs();
                assert!(
                    gap > 0.0 && gap < 1e-4,
                    "{label}: w is {gap} from the cap, not inside the near-miss window \
                     this case exists to probe; rescan"
                );
            }
            assert_eq!(
                py_strain_w != 0.0,
                expect_strain_w,
                "{label}: upstream's verdict changed"
            );

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
                    round_output: false,
                    limit_inputs: false,
                    ..Default::default()
                },
            );

            assert_eq!(
                rust.heat_strain_w,
                Some(expect_strain_w),
                "{label}: Rust {:?}, Python {expect_strain_w}",
                rust.heat_strain_w
            );
        }
    });
}

/// `ireq`'s `dle` either side of the 8-hour reporting ceiling.
///
/// `DurationLimitedExposure` is `Hours(_)` up to the ceiling and `MoreThanEight` past it,
/// so it is a discrete output driven by a float and carries the same risk as every band
/// edge in this file. Two details make it worth its own test rather than trusting the
/// sweep:
///
/// - The comparison is on the **unrounded** `dle` (upstream: `(dle > 8.0) | (dle < 0)` in
///   `ireq.py:_format_dle`), while `Hours` reports the rounded one. The `Hours` case below
///   is `7.957...`, reported as `8.0`.
///
/// Measured, by injecting each fault: moving the ceiling from 8.0 to 8.1 is **caught**
/// here and **missed** by `sweep_ireq`. Classifying the rounded `dle` instead of the
/// unrounded one is **not** caught by either -- and that is a property of the model, not a
/// hole to plug. The two orders differ only for a `dle` in `(8.0, 8.05]`, and the
/// discontinuity at the ceiling means no input produces one: the value jumps from ~7.96
/// straight past the window. Do not add cases chasing it.
/// - `dle = -40 / storage` is discontinuous where `storage` crosses zero, so the value
///   jumps across the ceiling rather than creeping over it.
///
/// The bracket is 1e-4 wide in `tdb`, not tighter, and that is forced rather than chosen.
/// Bisection puts the crossing between two *adjacent* f64s, but
/// `Temperature::from_celsius(-0.3483882704890564).as_celsius()` is
/// `-0.3483882704890675` -- off by ~1e-13, which is millions of ULPs at this magnitude,
/// so the Rust API cannot deliver an input on the correct side of the crossing at all.
/// A tighter bracket would test the newtype's representation instead of the ceiling
/// comparison. See `2-temperature-newtype-is-lossy-at-the-boundary.md`; this is the same
/// divergence found via `adaptive_en`, in a place where it bites harder.
#[test]
fn test_ireq_dle_at_the_eight_hour_ceiling() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Straddling the ceiling, everything else held fixed. The true crossing is at
        // tdb = -0.34838827048905638; these bracket it by 1e-4.
        let below = -0.3484_f64;
        let above = -0.3483_f64;

        for (tdb, expect_more_than_eight) in [(below, false), (above, true)] {
            let label = format!("ireq dle_neutral at tdb={tdb:?}");
            let (tr, vr, rh, met, clo, p, walk_sp) = (tdb, 1.0, 50.0, 2.0, 2.0, 101.325, 0.0);

            let kwargs = [("limit_inputs", PyBool::new(py, false).to_owned().into_any())]
                .into_py_dict(py)
                .unwrap();
            let py_result = pythermal
                .getattr("ireq")
                .unwrap()
                .call((tdb, tr, vr, rh, met, clo, p, walk_sp), Some(&kwargs))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_desc = describe_field(&py_result.getattr("dle_neutral").unwrap());

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
                    limit_inputs: false,
                    ..Default::default()
                },
            );

            // The fixture guard: the pair only brackets the ceiling if Python actually
            // reports different sides for the two probes.
            if expect_more_than_eight {
                assert_eq!(
                    py_desc, "\"more than 8\"",
                    "{label}: upstream no longer reports the over-ceiling side here; \
                     re-bisect for a pair straddling dle = 8.0"
                );
                assert_eq!(
                    rust.dle_neutral,
                    DurationLimitedExposure::MoreThanEight,
                    "{label}: Rust {} where Python says {py_desc}",
                    rust.dle_neutral
                );
            } else {
                let py_hours: f64 = py_result
                    .getattr("dle_neutral")
                    .unwrap()
                    .extract()
                    .unwrap_or_else(|_| {
                        panic!(
                            "{label}: upstream reports {py_desc} rather than a number of \
                             hours; re-bisect for a pair straddling dle = 8.0"
                        )
                    });
                assert_eq!(
                    py_hours, 8.0,
                    "{label}: upstream no longer reports 8.0 h here; re-bisect"
                );
                assert_eq!(
                    rust.dle_neutral,
                    DurationLimitedExposure::Hours(8.0),
                    "{label}: Rust {} where Python says {py_desc}",
                    rust.dle_neutral
                );
            }
        }
    });
}

/// The adaptive acceptability flags exactly on their band bounds.
///
/// `acceptability_80`/`_90` and `acceptability_cat_i`/`_ii`/`_iii` are `low <= to <= up`
/// against bounds the model reports itself, so unlike the fixed-literal indices the edge
/// *moves with the input*. That makes them unreachable by the scan used in
/// [`test_banded_indices_at_exact_edges`], and the differential sweep never lands on them
/// either -- randomised reals do not hit a computed bound.
///
/// The technique: ask Python for the bounds, then feed each one back as `tdb = tr = bound`
/// so the operative temperature *is* the bound exactly, and require Rust to agree with
/// Python at the bound and a millidegree either side. The test asserts agreement rather
/// than a hardcoded verdict, so it stays correct if upstream changes the inclusivity --
/// it is a parity test, not a transcription of the rule.
///
/// **The catch this guards against:** acceptability is evaluated against the *unrounded*
/// bounds, while the reported `tmp_cmf_*` fields are rounded. Feeding a reported bound
/// back only probes the true edge where the two coincide. `t_running_mean = 20` is chosen
/// because `0.31 * 20 + 17.8` is exactly 24.0, so every bound is exact -- and
/// `assert_is_really_the_edge` below proves it rather than trusting it, by requiring
/// Python's own verdict to flip across the bound. Without that guard this test would
/// quietly degrade into an interior-point check if the arithmetic ever stopped landing
/// exactly, which is the failure mode this whole worklist item exists to prevent.
#[test]
fn test_adaptive_acceptability_at_band_bounds() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let t_running_mean = 20.0;
        let v = 0.1;

        // ASHRAE: (bound field, acceptability field)
        let ashrae_bounds = [
            ("tmp_cmf_80_low", "acceptability_80"),
            ("tmp_cmf_80_up", "acceptability_80"),
            ("tmp_cmf_90_low", "acceptability_90"),
            ("tmp_cmf_90_up", "acceptability_90"),
        ];

        let seed = models
            .getattr("adaptive_ashrae")
            .unwrap()
            .call1((25.0, 25.0, t_running_mean, v))
            .expect("adaptive_ashrae raised");

        for (bound_field, acc_field) in ashrae_bounds {
            let bound: f64 = seed.getattr(bound_field).unwrap().extract().unwrap();
            let mut verdicts = [None; 3];

            for (slot, offset) in [-0.001, 0.0, 0.001].into_iter().enumerate() {
                let to = bound + offset;
                let label = format!("adaptive_ashrae {bound_field}={bound} offset={offset}");

                let py_result = models
                    .getattr("adaptive_ashrae")
                    .unwrap()
                    .call1((to, to, t_running_mean, v))
                    .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
                let py_acc = extract_optional_bool(&py_result.getattr(acc_field).unwrap());

                let rust = adaptive_ashrae(
                    AdaptiveInputs {
                        tdb: Temperature::from_celsius(to),
                        tr: Temperature::from_celsius(to),
                        t_running_mean: Temperature::from_celsius(t_running_mean),
                        v: Speed::from_meters_per_second(v),
                    },
                    Default::default(),
                );
                let rust_acc = if acc_field == "acceptability_80" {
                    rust.acceptability_80
                } else {
                    rust.acceptability_90
                };

                assert_eq!(
                    Some(rust_acc),
                    py_acc,
                    "{label}: Rust {rust_acc}, Python {py_acc:?}"
                );
                verdicts[slot] = py_acc;
            }
            assert_is_really_the_edge(
                verdicts,
                bound_field.ends_with("_low"),
                &format!("adaptive_ashrae {bound_field}"),
            );
        }

        // EN: three nested category bands.
        //
        // Two things differ from the ASHRAE half. First, `t_running_mean = 10.0`: EN
        // converts each bound to the output unit and only then rounds it
        // (`adaptive_en.py:140-162`), so at t_rm = 20 the reported bounds are *not* the
        // unrounded ones acceptability is evaluated against, and every case would have
        // degenerated into an interior-point check. `assert_is_really_the_edge` caught
        // that.
        //
        // Second, EN is probed a millidegree either side of each bound and NOT on the
        // bound itself, because its bounds are not binary-representable. ASHRAE's are
        // (t_cmf is rounded before the +/-3.5 and +/-2.5 offsets, giving values like
        // 20.5 and 27.5, which survive `Temperature`'s kelvin round-trip exactly), so
        // that half can sit on the edge. EN's cat_i_up at t_rm = 10 is 24.1, and
        // `Temperature::from_celsius(24.1).as_celsius()` is 24.100000000000023 -- above
        // the bound. Probing there would measure the newtype's representation, not the
        // banding rule this test is for. See the worklist item for that divergence,
        // which is real and separate.
        let t_running_mean = 10.0;
        let en_bounds = [
            ("tmp_cmf_cat_i_low", "acceptability_cat_i"),
            ("tmp_cmf_cat_i_up", "acceptability_cat_i"),
            ("tmp_cmf_cat_ii_low", "acceptability_cat_ii"),
            ("tmp_cmf_cat_ii_up", "acceptability_cat_ii"),
            ("tmp_cmf_cat_iii_low", "acceptability_cat_iii"),
            ("tmp_cmf_cat_iii_up", "acceptability_cat_iii"),
        ];

        let seed = models
            .getattr("adaptive_en")
            .unwrap()
            .call1((25.0, 25.0, t_running_mean, v))
            .expect("adaptive_en raised");

        for (bound_field, acc_field) in en_bounds {
            let bound: f64 = seed.getattr(bound_field).unwrap().extract().unwrap();
            let mut verdicts = [None; 3];

            for (slot, offset) in [-0.001, f64::NAN, 0.001].into_iter().enumerate() {
                if offset.is_nan() {
                    // The on-the-bound probe, skipped for EN: see the comment above.
                    continue;
                }
                let to = bound + offset;
                let label = format!("adaptive_en {bound_field}={bound} offset={offset}");

                let py_result = models
                    .getattr("adaptive_en")
                    .unwrap()
                    .call1((to, to, t_running_mean, v))
                    .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
                let py_acc = extract_optional_bool(&py_result.getattr(acc_field).unwrap());

                let rust = adaptive_en(
                    AdaptiveInputs {
                        tdb: Temperature::from_celsius(to),
                        tr: Temperature::from_celsius(to),
                        t_running_mean: Temperature::from_celsius(t_running_mean),
                        v: Speed::from_meters_per_second(v),
                    },
                    Default::default(),
                );
                let rust_acc = match acc_field {
                    "acceptability_cat_i" => rust.acceptability_cat_i,
                    "acceptability_cat_ii" => rust.acceptability_cat_ii,
                    _ => rust.acceptability_cat_iii,
                };

                assert_eq!(
                    Some(rust_acc),
                    py_acc,
                    "{label}: Rust {rust_acc}, Python {py_acc:?}"
                );
                verdicts[slot] = py_acc;
            }
            assert_brackets_the_edge(
                verdicts,
                bound_field.ends_with("_low"),
                &format!("adaptive_en {bound_field}"),
            );
        }
    });
}

/// `discomfort_index` bands the *unrounded* DI and rounds only the reported value --
/// the opposite order from `utci`, `humidex` and `heat_index_rothfusz`, which round
/// first and band the rounded value.
///
/// Both inputs below report `di = 21.0` and yet fall in **different** bands, because
/// their unrounded DI straddles 21.0. A port that banded the rounded value would give
/// both the same category, and this test fails on it.
///
/// Unlike the inclusivity flips in [`test_banded_indices_at_exact_edges`], `sweep_di`
/// *does* catch this one -- round-before-band changes the category for every sample
/// within 0.05 of an edge, which is a wide enough target for randomised reals to hit.
/// The unit tests in `thermal_indices.rs` do not (verified: they pass with the order
/// swapped). This test is kept anyway because the sweep catches it by sampling luck
/// while these two inputs pin it deterministically, and because it is the only place the
/// band-before-round order -- which is the opposite of the other three indices -- is
/// written down as intentional rather than incidental.
#[test]
fn test_discomfort_index_bands_before_rounding() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb, expected category) -- both report di = 21.0
        let cases = [
            (23.41036, "No discomfort"),
            (23.46552, "Less than 50% feels discomfort"),
        ];

        for (tdb, expected) in cases {
            let label = format!("discomfort_index tdb={tdb}");
            let py_result = models
                .getattr("discomfort_index")
                .unwrap()
                .call1((tdb, 50.0))
                .unwrap_or_else(|e| panic!("{label}: raised: {e}"));
            let py_di: f64 = py_result.getattr("di").unwrap().extract().unwrap();
            let py_category = extract_category(&py_result.getattr("discomfort_condition").unwrap());

            let rust = discomfort_index(DiscomfortIndexInputs {
                tdb: Temperature::from_celsius(tdb),
                rh: Humidity::from_percent(50.0),
            });

            assert_eq!(
                py_di, 21.0,
                "{label}: no longer reports di=21.0 (got {py_di}); the two cases must \
                 share a rounded DI for this test to mean anything"
            );
            assert_eq!(
                py_category.as_deref(),
                Some(expected),
                "{label}: upstream changed"
            );
            assert_eq!(
                Some(rust.discomfort_condition.as_str()),
                py_category.as_deref(),
                "{label}: Rust category differs from Python's"
            );
        }
    });
}

/// UTCI's `units` (SI/IP) rounds in the OUTPUT unit, not by converting an
/// already-rounded SI value -- cover the IP path against pythermalcomfort directly.
///
/// Under `units="IP"`, pythermalcomfort's raw floats are °F (tdb/tr) and fps (v), not
/// °C/m-per-s, so the test cases below are chosen in those units: `v` in particular
/// must convert to at least 0.5 m/s (the model's applicability floor) or both sides
/// return NaN and the comparison is vacuous.
#[test]
fn test_compare_utci_ip_units() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb_f, tr_f, v_fps, rh)
        let test_cases = vec![
            (77.0, 77.0, 3.0, 50.0),
            (86.0, 86.0, 10.0, 60.0),
            (41.0, 41.0, 6.0, 70.0),
        ];

        for (tdb_f, tr_f, v_fps, rh) in test_cases {
            let kwargs = [("units", "IP")].into_py_dict(py).unwrap();
            let py_result = pythermal
                .getattr("utci")
                .unwrap()
                .call((tdb_f, tr_f, v_fps, rh), Some(&kwargs))
                .unwrap();

            let py_utci: f64 = py_result.getattr("utci").unwrap().extract().unwrap();

            let rust_result = utci(
                UtciInputs {
                    tdb: Temperature::from_fahrenheit(tdb_f),
                    tr: Temperature::from_fahrenheit(tr_f),
                    v: Speed::from_meters_per_second(v_fps * 0.3048),
                    rh: Humidity::from_percent(rh),
                },
                UtciOptions {
                    units: thermalcomfort::utilities::Units::IP,
                    ..Default::default()
                },
            );

            assert_abs_diff_eq!(rust_result.utci, py_utci, epsilon = 0.2);
        }
    });
}

#[test]
fn test_compare_pmv_a() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 0.0),
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 0.5),
            (28.0, 28.0, 0.2, 60.0, 1.4, 0.4, 0.3),
        ];

        for (tdb, tr, vr, rh, met, clo, a_coeff) in test_cases {
            let py_result = pythermal
                .getattr("pmv_a")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo, a_coeff))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("a_pmv").unwrap().extract().unwrap();

            let rust_result = pmv_a(
                PmvAInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    a_coefficient: a_coeff,
                },
                PmvAOptions::default(),
            );

            assert_abs_diff_eq!(rust_result, py_pmv, epsilon = 0.02);
        }
    });
}

#[test]
fn test_compare_pmv_e() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 1.0),
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 0.7),
            (28.0, 28.0, 0.2, 60.0, 1.4, 0.4, 0.9),
        ];

        for (tdb, tr, vr, rh, met, clo, e_coeff) in test_cases {
            let py_result = pythermal
                .getattr("pmv_e")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo, e_coeff))
                .unwrap();

            let py_pmv: f64 = py_result.getattr("e_pmv").unwrap().extract().unwrap();

            let rust_result = pmv_e(
                PmvEInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    e_coefficient: e_coeff,
                },
                PmvEOptions::default(),
            );

            assert_abs_diff_eq!(rust_result, py_pmv, epsilon = 0.02);
        }
    });
}

#[test]
fn test_compare_pmv_athb() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 20.0),
            (28.0, 28.0, 0.2, 60.0, 1.4, 0.6, 25.0),
        ];

        for (tdb, tr, vr, rh, met, clo, t_rm) in test_cases {
            let py_result = pythermal
                .getattr("pmv_athb")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, t_rm, clo))
                .unwrap(); // Note: Python signature is (tdb, tr, vr, rh, met, t_running_mean, clo)

            let py_pmv: f64 = py_result.getattr("athb_pmv").unwrap().extract().unwrap();

            let rust_result = pmv_athb(
                PmvAthbInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    t_running_mean: Temperature::from_celsius(t_rm),
                },
                PmvAthbOptions {
                    clo: Some(ClothingInsulation::from_clo(clo)),
                },
            );

            assert_abs_diff_eq!(rust_result, py_pmv, epsilon = 0.05);
        }
    });
}

#[test]
fn test_compare_set_tmp() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5),
            (28.0, 28.0, 0.3, 60.0, 1.5, 0.3),
            (22.0, 24.0, 0.15, 40.0, 1.1, 0.7),
        ];

        for (tdb, tr, v, rh, met, clo) in test_cases {
            let py_result = pythermal
                .getattr("set_tmp")
                .unwrap()
                .call1((tdb, tr, v, rh, met, clo))
                .unwrap();

            let py_set: f64 = py_result.getattr("set").unwrap().extract().unwrap();

            let rust_result = set_tmp(
                SetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            )
            .as_celsius();

            assert_abs_diff_eq!(rust_result, py_set, epsilon = 1.5);
        }
    });
}

#[test]
fn test_compare_cooling_effect() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (28.0, 28.0, 0.8, 50.0, 1.2, 0.5),
            (30.0, 30.0, 1.0, 60.0, 1.3, 0.4),
        ];

        for (tdb, tr, vr, rh, met, clo) in test_cases {
            let py_result = pythermal
                .getattr("cooling_effect")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo))
                .unwrap();

            let py_ce: f64 = py_result.getattr("ce").unwrap().extract().unwrap();

            let rust_result = cooling_effect(
                CoolingEffectInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.as_celsius(), py_ce, epsilon = 0.15);
        }
    });
}

#[test]
fn test_compare_adaptive_ashrae() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep covers:
        //   - t_running_mean across the model's [10, 33.5] range
        //   - air speed below and above the 0.6/0.9/1.2 cooling-effect thresholds
        //   - operative temperatures both inside and outside the 80%/90% bands so
        //     `acceptability_*` flips both ways
        let test_cases = vec![
            (25.0, 25.0, 0.1, 20.0),
            (28.0, 28.0, 0.3, 25.0),
            (22.0, 22.0, 0.2, 18.0),
            (24.0, 24.0, 0.7, 22.0), // ce tier 1 (0.6 <= v < 0.9)
            (26.0, 26.0, 1.0, 28.0), // ce tier 2 (0.9 <= v < 1.2)
            (27.0, 27.0, 1.5, 30.0), // ce tier 3 (v >= 1.2)
            (32.0, 32.0, 0.1, 32.0), // top end (outside 90% band)
            (15.0, 15.0, 0.1, 12.0), // low end
        ];

        for (tdb, tr, v, t_running_mean) in test_cases {
            let py_result = pythermal
                .getattr("adaptive_ashrae")
                .unwrap()
                .call1((tdb, tr, t_running_mean, v))
                .unwrap();

            let py_tmp_cmf: f64 = py_result.getattr("tmp_cmf").unwrap().extract().unwrap();
            let py_80_low: f64 = py_result
                .getattr("tmp_cmf_80_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_80_up: f64 = py_result
                .getattr("tmp_cmf_80_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_90_low: f64 = py_result
                .getattr("tmp_cmf_90_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_90_up: f64 = py_result
                .getattr("tmp_cmf_90_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_acc_80: bool = py_result
                .getattr("acceptability_80")
                .unwrap()
                .extract()
                .unwrap();
            let py_acc_90: bool = py_result
                .getattr("acceptability_90")
                .unwrap()
                .extract()
                .unwrap();

            let rust_result = adaptive_ashrae(
                AdaptiveInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    t_running_mean: Temperature::from_celsius(t_running_mean),
                    v: Speed::from_meters_per_second(v),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.tmp_cmf.as_celsius(), py_tmp_cmf, epsilon = 0.1);
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_80_low.as_celsius(),
                py_80_low,
                epsilon = 0.1
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_80_up.as_celsius(),
                py_80_up,
                epsilon = 0.1
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_90_low.as_celsius(),
                py_90_low,
                epsilon = 0.1
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_90_up.as_celsius(),
                py_90_up,
                epsilon = 0.1
            );
            assert_eq!(
                rust_result.acceptability_80, py_acc_80,
                "acceptability_80 mismatch at tdb={tdb} tr={tr} v={v} trm={t_running_mean}",
            );
            assert_eq!(
                rust_result.acceptability_90, py_acc_90,
                "acceptability_90 mismatch at tdb={tdb} tr={tr} v={v} trm={t_running_mean}",
            );
        }
    });
}

#[test]
fn test_compare_adaptive_en() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep covers operating points inside Category I, II, III and outside
        // all categories so acceptability bools flip across the cases.
        let test_cases = vec![
            (25.0, 25.0, 0.1, 20.0),
            (28.0, 28.0, 0.3, 25.0),
            (22.0, 22.0, 0.1, 18.0),
            (24.0, 24.0, 0.1, 22.0),
            (28.0, 28.0, 0.7, 26.0),
            (29.0, 29.0, 1.0, 28.0),
        ];

        for (tdb, tr, v, t_running_mean) in test_cases {
            let py_result = pythermal
                .getattr("adaptive_en")
                .unwrap()
                .call1((tdb, tr, t_running_mean, v))
                .unwrap();

            let py_tmp_cmf: f64 = py_result.getattr("tmp_cmf").unwrap().extract().unwrap();
            let py_cat_i_low: f64 = py_result
                .getattr("tmp_cmf_cat_i_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_i_up: f64 = py_result
                .getattr("tmp_cmf_cat_i_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_ii_low: f64 = py_result
                .getattr("tmp_cmf_cat_ii_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_ii_up: f64 = py_result
                .getattr("tmp_cmf_cat_ii_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_iii_low: f64 = py_result
                .getattr("tmp_cmf_cat_iii_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_iii_up: f64 = py_result
                .getattr("tmp_cmf_cat_iii_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_acc_i: bool = py_result
                .getattr("acceptability_cat_i")
                .unwrap()
                .extract()
                .unwrap();
            let py_acc_ii: bool = py_result
                .getattr("acceptability_cat_ii")
                .unwrap()
                .extract()
                .unwrap();
            let py_acc_iii: bool = py_result
                .getattr("acceptability_cat_iii")
                .unwrap()
                .extract()
                .unwrap();

            let rust_result = adaptive_en(
                AdaptiveInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    t_running_mean: Temperature::from_celsius(t_running_mean),
                    v: Speed::from_meters_per_second(v),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.tmp_cmf.as_celsius(), py_tmp_cmf, epsilon = 0.15);
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_i_low.as_celsius(),
                py_cat_i_low,
                epsilon = 0.15
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_i_up.as_celsius(),
                py_cat_i_up,
                epsilon = 0.15
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_ii_low.as_celsius(),
                py_cat_ii_low,
                epsilon = 0.15
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_ii_up.as_celsius(),
                py_cat_ii_up,
                epsilon = 0.15
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_iii_low.as_celsius(),
                py_cat_iii_low,
                epsilon = 0.15
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_iii_up.as_celsius(),
                py_cat_iii_up,
                epsilon = 0.15
            );
            assert_eq!(
                rust_result.acceptability_cat_i, py_acc_i,
                "acceptability_cat_i mismatch at tdb={tdb} tr={tr} v={v} trm={t_running_mean}",
            );
            assert_eq!(
                rust_result.acceptability_cat_ii, py_acc_ii,
                "acceptability_cat_ii mismatch at tdb={tdb} tr={tr} v={v} trm={t_running_mean}",
            );
            assert_eq!(
                rust_result.acceptability_cat_iii, py_acc_iii,
                "acceptability_cat_iii mismatch at tdb={tdb} tr={tr} v={v} trm={t_running_mean}",
            );
        }
    });
}

#[test]
fn test_compare_adaptive_round_output_false() {
    // Mirrors the headline 3.9.8 change for the adaptive models: when
    // round_output=False the unrounded t_cmf and derived bounds must agree with
    // pythermalcomfort. Inputs are chosen so the unrounded value differs from
    // the rounded one (e.g. trm=27 → ashrae t_cmf=26.17, not 26.2).
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 22.0), // ashrae 24.62, en 26.06
            (25.0, 25.0, 0.1, 27.0), // ashrae 26.17, en 27.71
            (25.0, 25.0, 0.1, 23.5), // ashrae 25.085, en 26.555
        ];

        let opts = AdaptiveOptions {
            round_output: false,
            ..Default::default()
        };

        for (tdb, tr, v, trm) in test_cases {
            // ASHRAE
            let kwargs = [("round_output", false)].into_py_dict(py).unwrap();
            let py_result = pythermal
                .getattr("adaptive_ashrae")
                .unwrap()
                .call((tdb, tr, trm, v), Some(&kwargs))
                .unwrap();
            let py_tmp_cmf: f64 = py_result.getattr("tmp_cmf").unwrap().extract().unwrap();
            let py_80_low: f64 = py_result
                .getattr("tmp_cmf_80_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_80_up: f64 = py_result
                .getattr("tmp_cmf_80_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_90_low: f64 = py_result
                .getattr("tmp_cmf_90_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_90_up: f64 = py_result
                .getattr("tmp_cmf_90_up")
                .unwrap()
                .extract()
                .unwrap();

            let rust_result = adaptive_ashrae(
                AdaptiveInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    t_running_mean: Temperature::from_celsius(trm),
                    v: Speed::from_meters_per_second(v),
                },
                opts,
            );

            // Tight ε: with rounding disabled we should match to floating-point precision.
            assert_abs_diff_eq!(rust_result.tmp_cmf.as_celsius(), py_tmp_cmf, epsilon = 1e-9);
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_80_low.as_celsius(),
                py_80_low,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_80_up.as_celsius(),
                py_80_up,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_90_low.as_celsius(),
                py_90_low,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_90_up.as_celsius(),
                py_90_up,
                epsilon = 1e-9
            );

            // EN
            let py_result = pythermal
                .getattr("adaptive_en")
                .unwrap()
                .call((tdb, tr, trm, v), Some(&kwargs))
                .unwrap();
            let py_tmp_cmf: f64 = py_result.getattr("tmp_cmf").unwrap().extract().unwrap();
            let py_cat_i_low: f64 = py_result
                .getattr("tmp_cmf_cat_i_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_i_up: f64 = py_result
                .getattr("tmp_cmf_cat_i_up")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_ii_low: f64 = py_result
                .getattr("tmp_cmf_cat_ii_low")
                .unwrap()
                .extract()
                .unwrap();
            let py_cat_ii_up: f64 = py_result
                .getattr("tmp_cmf_cat_ii_up")
                .unwrap()
                .extract()
                .unwrap();

            let rust_result = adaptive_en(
                AdaptiveInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    t_running_mean: Temperature::from_celsius(trm),
                    v: Speed::from_meters_per_second(v),
                },
                opts,
            );

            assert_abs_diff_eq!(rust_result.tmp_cmf.as_celsius(), py_tmp_cmf, epsilon = 1e-9);
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_i_low.as_celsius(),
                py_cat_i_low,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_i_up.as_celsius(),
                py_cat_i_up,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_ii_low.as_celsius(),
                py_cat_ii_low,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf_cat_ii_up.as_celsius(),
                py_cat_ii_up,
                epsilon = 1e-9
            );

            // Sanity: confirm the unrounded value is NOT just the rounded value
            // (otherwise this test wouldn't be exercising the false branch).
            let rounded_tmp_cmf = libm::round(rust_result.tmp_cmf.as_celsius() * 10.0) / 10.0;
            assert!(
                (rust_result.tmp_cmf.as_celsius() - rounded_tmp_cmf).abs() > 0.0
                    || (rust_result.tmp_cmf.as_celsius() * 10.0).fract().abs() < 1e-9,
                "round_output=false should preserve sub-0.1 precision (trm={trm})",
            );
        }
    });
}

/// `adaptive_ashrae`/`adaptive_en` gained `units` (SI/IP) alongside `utci`. Upstream
/// rounds `t_cmf` in SI, THEN converts to °F, so the IP output carries the extra
/// decimals from that conversion -- cover the IP path directly against
/// pythermalcomfort rather than trusting a derived Celsius comparison.
#[test]
fn test_compare_adaptive_ip_units() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // (tdb_f, tr_f, trm_f, v_fps)
        let test_cases = vec![(77.0, 77.0, 68.0, 0.3), (82.0, 82.0, 77.0, 1.0)];

        let ip_opts = AdaptiveOptions {
            units: thermalcomfort::utilities::Units::IP,
            ..Default::default()
        };

        for (tdb_f, tr_f, trm_f, v_fps) in test_cases {
            let kwargs = [("units", "IP")].into_py_dict(py).unwrap();

            let tdb_c = Temperature::from_fahrenheit(tdb_f);
            let tr_c = Temperature::from_fahrenheit(tr_f);
            let trm_c = Temperature::from_fahrenheit(trm_f);
            let v = Speed::from_meters_per_second(v_fps * 0.3048);

            // ASHRAE
            let py_result = pythermal
                .getattr("adaptive_ashrae")
                .unwrap()
                .call((tdb_f, tr_f, trm_f, v_fps), Some(&kwargs))
                .unwrap();
            let py_tmp_cmf: f64 = py_result.getattr("tmp_cmf").unwrap().extract().unwrap();

            let rust_result = adaptive_ashrae(
                AdaptiveInputs {
                    tdb: tdb_c,
                    tr: tr_c,
                    t_running_mean: trm_c,
                    v,
                },
                ip_opts,
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf.as_fahrenheit(),
                py_tmp_cmf,
                epsilon = 0.2
            );

            // EN
            let py_result = pythermal
                .getattr("adaptive_en")
                .unwrap()
                .call((tdb_f, tr_f, trm_f, v_fps), Some(&kwargs))
                .unwrap();
            let py_tmp_cmf: f64 = py_result.getattr("tmp_cmf").unwrap().extract().unwrap();

            let rust_result = adaptive_en(
                AdaptiveInputs {
                    tdb: tdb_c,
                    tr: tr_c,
                    t_running_mean: trm_c,
                    v,
                },
                ip_opts,
            );
            assert_abs_diff_eq!(
                rust_result.tmp_cmf.as_fahrenheit(),
                py_tmp_cmf,
                epsilon = 0.2
            );
        }
    });
}

#[test]
fn test_compare_wbgt() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(30.0, 25.0, 35.0), (28.0, 24.0, 32.0)];

        for (twb, tg, tdb) in test_cases {
            let py_result = pythermal
                .getattr("wbgt")
                .unwrap()
                .call1((twb, tg, tdb))
                .unwrap();

            let py_wbgt: f64 = py_result.getattr("wbgt").unwrap().extract().unwrap();

            let rust_result = wbgt(
                WbgtInputs {
                    twb: Temperature::from_celsius(twb),
                    tg: Temperature::from_celsius(tg),
                },
                WbgtOptions {
                    tdb: Some(Temperature::from_celsius(tdb)),
                    ..Default::default()
                },
            );

            assert_abs_diff_eq!(rust_result, py_wbgt, epsilon = 0.1);
        }
    });
}

#[test]
fn test_compare_heat_index_rothfusz() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep covers every stress band: no_risk, caution, extreme caution,
        // danger, extreme danger.
        let test_cases = vec![
            (28.0, 70.0),
            (30.0, 50.0),
            (33.0, 60.0),
            (35.0, 60.0),
            (38.0, 70.0),
            (40.0, 80.0),
            (45.0, 90.0),
        ];

        for (tdb, rh) in test_cases {
            let py_result = pythermal
                .getattr("heat_index_rothfusz")
                .unwrap()
                .call1((tdb, rh))
                .unwrap();

            let py_hi: f64 = py_result.getattr("hi").unwrap().extract().unwrap();
            let py_category: Option<String> =
                py_result.getattr("stress_category").unwrap().extract().ok();

            let rust_result = heat_index_rothfusz(
                HeatIndexRothfuszInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.hi, py_hi, epsilon = 0.5);
            assert_eq!(
                rust_result.stress_category.map(|c| c.as_str()),
                py_category.as_deref(),
                "heat_index_rothfusz stress_category mismatch at tdb={} rh={}",
                tdb,
                rh,
            );
        }
    });
}

#[test]
fn test_compare_heat_index_lu() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 50.0), (30.0, 60.0), (28.0, 55.0)];

        for (tdb, rh) in test_cases {
            let py_result = pythermal
                .getattr("heat_index_lu")
                .unwrap()
                .call1((tdb, rh))
                .unwrap();

            let py_hi: f64 = py_result.getattr("hi").unwrap().extract().unwrap();

            let rust_result = heat_index_lu(
                HeatIndexLuInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );

            // Lu model uses iterative solver, allow larger tolerance
            assert_abs_diff_eq!(rust_result.hi, py_hi, epsilon = 1.0);
            // pythermalcomfort leaves stress_category unset for the Lu model.
            assert!(rust_result.stress_category.is_none());
        }
    });
}

/// `round_output` was previously hardcoded to `true`; cover both settings against
/// pythermalcomfort's `round_output` keyword.
#[test]
fn test_compare_heat_index_lu_round_output() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 50.0), (30.0, 60.0)];

        for (tdb, rh) in test_cases {
            for round_output in [true, false] {
                let kwargs = [("round_output", round_output)].into_py_dict(py).unwrap();
                let py_result = pythermal
                    .getattr("heat_index_lu")
                    .unwrap()
                    .call((tdb, rh), Some(&kwargs))
                    .unwrap();

                let py_hi: f64 = py_result.getattr("hi").unwrap().extract().unwrap();

                let rust_result = heat_index_lu(
                    HeatIndexLuInputs {
                        tdb: Temperature::from_celsius(tdb),
                        rh: Humidity::from_percent(rh),
                    },
                    HeatIndexLuOptions { round_output },
                );

                assert_abs_diff_eq!(rust_result.hi, py_hi, epsilon = 1.0);
            }
        }
    });
}

#[test]
fn test_compare_humidex() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep covers all six discomfort bands so the categorical mapping is exercised.
        let test_cases = vec![
            (20.0, 30.0),
            (25.0, 50.0),
            (30.0, 60.0),
            (32.0, 70.0),
            (35.0, 80.0),
            (38.0, 85.0),
            (42.0, 90.0),
        ];

        for (tdb, rh) in test_cases {
            let py_result = pythermal
                .getattr("humidex")
                .unwrap()
                .call1((tdb, rh))
                .unwrap();

            let py_humidex: f64 = py_result.getattr("humidex").unwrap().extract().unwrap();
            let py_discomfort = extract_category(&py_result.getattr("discomfort").unwrap())
                .expect("humidex always yields a discomfort category");

            let rust_result = humidex(
                HumidexInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result.humidex, py_humidex, epsilon = 0.1);
            assert_eq!(
                rust_result.discomfort.as_str(),
                py_discomfort,
                "humidex discomfort mismatch at tdb={} rh={}",
                tdb,
                rh,
            );
        }
    });
}

/// `humidex_masterson` no longer exists as a separate function upstream; the Masterson
/// model is now selected via `humidex(..., model="masterson")`, and `HumidexModel::Masterson`
/// here. Keep it covered alongside the default Rana model above.
#[test]
fn test_compare_humidex_masterson() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (20.0, 30.0),
            (25.0, 50.0),
            (30.0, 60.0),
            (35.0, 80.0),
            (42.0, 90.0),
        ];

        for (tdb, rh) in test_cases {
            let kwargs = [("model", "masterson")].into_py_dict(py).unwrap();
            let py_result = pythermal
                .getattr("humidex")
                .unwrap()
                .call((tdb, rh), Some(&kwargs))
                .unwrap();

            let py_humidex: f64 = py_result.getattr("humidex").unwrap().extract().unwrap();
            let py_discomfort = extract_category(&py_result.getattr("discomfort").unwrap())
                .expect("humidex always yields a discomfort category");

            let rust_result = humidex(
                HumidexInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                HumidexOptions {
                    model: HumidexModel::Masterson,
                    round_output: true,
                },
            );

            assert_abs_diff_eq!(rust_result.humidex, py_humidex, epsilon = 0.2);
            assert_eq!(
                rust_result.discomfort.as_str(),
                py_discomfort,
                "humidex (masterson) discomfort mismatch at tdb={} rh={}",
                tdb,
                rh,
            );
        }
    });
}

#[test]
fn test_compare_thi() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 50.0), (28.0, 60.0)];

        for (tdb, rh) in test_cases {
            let py_result = pythermal.getattr("thi").unwrap().call1((tdb, rh)).unwrap();

            let py_thi: f64 = py_result.getattr("thi").unwrap().extract().unwrap();

            let rust_result = thi(
                ThiInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result, py_thi, epsilon = 0.1);
        }
    });
}

#[test]
fn test_compare_discomfort_index() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep spans every band so the categorical mapping is exercised.
        let test_cases = vec![
            (18.0, 40.0),
            (22.0, 50.0),
            (25.0, 50.0),
            (26.0, 70.0),
            (28.0, 60.0),
            (30.0, 70.0),
            (33.0, 80.0),
            (38.0, 80.0),
        ];

        for (tdb, rh) in test_cases {
            let py_result = pythermal
                .getattr("discomfort_index")
                .unwrap()
                .call1((tdb, rh))
                .unwrap();

            let py_di: f64 = py_result.getattr("di").unwrap().extract().unwrap();
            let py_condition: String = py_result
                .getattr("discomfort_condition")
                .unwrap()
                .extract()
                .unwrap();

            let rust_result = discomfort_index(DiscomfortIndexInputs {
                tdb: Temperature::from_celsius(tdb),
                rh: Humidity::from_percent(rh),
            });

            assert_abs_diff_eq!(rust_result.di, py_di, epsilon = 0.1);
            assert_eq!(
                rust_result.discomfort_condition.as_str(),
                py_condition,
                "DI discomfort_condition mismatch at tdb={} rh={}",
                tdb,
                rh,
            );
        }
    });
}

#[test]
fn test_compare_at() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 25.0, 1.0, 50.0), (30.0, 30.0, 0.5, 60.0)];

        for (tdb, _tr, v, rh) in test_cases {
            let py_result = pythermal
                .getattr("at")
                .unwrap()
                .call1((tdb, rh, v))
                .unwrap(); // Python signature is at(tdb, rh, v)

            let py_at: f64 = py_result.getattr("at").unwrap().extract().unwrap();

            let rust_result = at(
                AtInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                    v: Speed::from_meters_per_second(v),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result, py_at, epsilon = 0.2);
        }
    });
}

#[test]
fn test_compare_net() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 25.0, 1.0, 50.0), (30.0, 30.0, 0.5, 60.0)];

        for (tdb, _tr, v, rh) in test_cases {
            let py_result = pythermal
                .getattr("net")
                .unwrap()
                .call1((tdb, rh, v))
                .unwrap();

            let py_net: f64 = py_result.getattr("net").unwrap().extract().unwrap();

            let rust_result = net(
                NetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                    v: Speed::from_meters_per_second(v),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result, py_net, epsilon = 0.2);
        }
    });
}

#[test]
fn test_compare_esi() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 50.0), (28.0, 60.0)];

        for (tdb, rh) in test_cases {
            let py_result = pythermal
                .getattr("esi")
                .unwrap()
                .call1((tdb, rh, 0.0))
                .unwrap();

            let py_esi: f64 = py_result.getattr("esi").unwrap().extract().unwrap();

            let rust_result = esi(
                EsiInputs {
                    tdb: Temperature::from_celsius(tdb),
                    rh: Humidity::from_percent(rh),
                    sol_radiation_global: 0.0,
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result, py_esi, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_wci() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(5.0, 5.0), (-10.0, 10.0), (0.0, 8.0)];

        for (tdb, v) in test_cases {
            let py_result = pythermal.getattr("wci").unwrap().call1((tdb, v)).unwrap();

            let py_wci: f64 = py_result.getattr("wci").unwrap().extract().unwrap();

            let rust_result = wci(
                WciInputs {
                    tdb: Temperature::from_celsius(tdb),
                    v: Speed::from_meters_per_second(v),
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result, py_wci, epsilon = 10.0);
        }
    });
}

#[test]
fn test_compare_wind_chill_temperature() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(5.0, 10.0), (-10.0, 15.0), (0.0, 20.0)];

        for (tdb, v) in test_cases {
            let py_result = pythermal
                .getattr("wind_chill_temperature")
                .unwrap()
                .call1((tdb, v))
                .unwrap();

            let py_wct: f64 = py_result.getattr("wct").unwrap().extract().unwrap();

            let rust_result = wind_chill_temperature(
                WindChillTemperatureInputs {
                    tdb: Temperature::from_celsius(tdb),
                    v: Speed::from_kilometers_per_hour(v), // Python expects km/h
                },
                Default::default(),
            );

            assert_abs_diff_eq!(rust_result, py_wct, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_work_capacity_iso() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 300.0), (30.0, 350.0), (35.0, 400.0)];

        for (wbgt, met) in test_cases {
            let py_result = pythermal
                .getattr("work_capacity_iso")
                .unwrap()
                .call1((wbgt, met))
                .unwrap();

            let py_capacity: f64 = py_result.getattr("capacity").unwrap().extract().unwrap();

            let rust_result =
                work_capacity_iso(Temperature::from_celsius(wbgt), Power::from_watts(met));

            assert_abs_diff_eq!(rust_result, py_capacity, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_work_capacity_niosh() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, 300.0), (30.0, 350.0)];

        for (wbgt, met) in test_cases {
            let py_result = pythermal
                .getattr("work_capacity_niosh")
                .unwrap()
                .call1((wbgt, met))
                .unwrap();

            let py_capacity: f64 = py_result.getattr("capacity").unwrap().extract().unwrap();

            let rust_result =
                work_capacity_niosh(Temperature::from_celsius(wbgt), Power::from_watts(met));

            assert_abs_diff_eq!(rust_result, py_capacity, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_work_capacity_dunne() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, "Heavy"), (30.0, "Moderate"), (28.0, "Light")];

        for (wbgt, intensity_str) in test_cases {
            let py_result = pythermal
                .getattr("work_capacity_dunne")
                .unwrap()
                .call1((wbgt, intensity_str))
                .unwrap();

            let py_capacity: f64 = py_result.getattr("capacity").unwrap().extract().unwrap();

            let intensity = match intensity_str {
                "Heavy" => WorkIntensity::Heavy,
                "Moderate" => WorkIntensity::Moderate,
                "Light" => WorkIntensity::Light,
                _ => WorkIntensity::Heavy,
            };

            let rust_result = work_capacity_dunne(
                Temperature::from_celsius(wbgt),
                WorkCapacityIntensityOptions {
                    work_intensity: intensity,
                },
            );

            assert_abs_diff_eq!(rust_result, py_capacity, epsilon = 1.0);
        }
    });
}

#[test]
fn test_compare_work_capacity_hothaps() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![(25.0, "Heavy"), (30.0, "Moderate")];

        for (wbgt, intensity_str) in test_cases {
            let py_result = pythermal
                .getattr("work_capacity_hothaps")
                .unwrap()
                .call1((wbgt, intensity_str))
                .unwrap();

            let py_capacity: f64 = py_result.getattr("capacity").unwrap().extract().unwrap();

            let intensity = match intensity_str {
                "Heavy" => WorkIntensity::Heavy,
                "Moderate" => WorkIntensity::Moderate,
                "Light" => WorkIntensity::Light,
                _ => WorkIntensity::Heavy,
            };

            let rust_result = work_capacity_hothaps(
                Temperature::from_celsius(wbgt),
                WorkCapacityIntensityOptions {
                    work_intensity: intensity,
                },
            );

            assert_abs_diff_eq!(rust_result, py_capacity, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_ankle_draft() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 0.15), // v_ankle must be < 0.2 m/s
            (23.0, 23.0, 0.1, 45.0, 1.1, 0.7, 0.18),
        ];

        for (tdb, tr, vr, rh, met, clo, v_ankle) in test_cases {
            let py_result = pythermal
                .getattr("ankle_draft")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo, v_ankle))
                .unwrap();

            let py_ppd: f64 = py_result.getattr("ppd_ad").unwrap().extract().unwrap();

            let (rust_ppd, _) = ankle_draft(
                AnkleDraftInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    vr: Speed::from_meters_per_second(vr),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    v_ankle: Speed::from_meters_per_second(v_ankle),
                },
                AnkleDraftOptions { limit_inputs: true },
            );

            assert_abs_diff_eq!(rust_ppd, py_ppd, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_vertical_tmp_grad_ppd() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5, 3.0),
            (23.0, 23.0, 0.1, 45.0, 1.1, 0.7, 2.0),
        ];

        for (tdb, tr, vr, rh, met, clo, grad) in test_cases {
            let py_result = pythermal
                .getattr("vertical_tmp_grad_ppd")
                .unwrap()
                .call1((tdb, tr, vr, rh, met, clo, grad))
                .unwrap();

            let py_ppd: f64 = py_result.getattr("ppd_vg").unwrap().extract().unwrap();

            let (rust_ppd, _) = vertical_tmp_grad_ppd(
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
                    round_output: true,
                    limit_inputs: true,
                },
            );

            assert_abs_diff_eq!(rust_ppd, py_ppd, epsilon = 0.5);
        }
    });
}

#[test]
fn test_compare_solar_gain() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Sweep solar altitudes from horizon to overhead, varied SHARP, beam
        // radiation, transmittance, view fractions, posture, and floor reflectance
        // so both `erf` and `delta_mrt` are exercised over a broad range.
        let test_cases = vec![
            (0.0, 120.0, 800.0, 0.5, 0.5, 0.5, 0.7, "sitting", 0.6),
            (45.0, 90.0, 600.0, 0.7, 0.6, 0.7, 0.7, "standing", 0.6),
            (15.0, 0.0, 400.0, 0.5, 0.3, 0.4, 0.6, "sitting", 0.4),
            (60.0, 180.0, 900.0, 0.8, 0.7, 0.8, 0.5, "standing", 0.7),
            (30.0, 45.0, 500.0, 0.6, 0.5, 0.6, 0.8, "sitting", 0.5),
        ];

        for (alt, sharp, sol_rad, sol_trans, f_svv_val, f_bes, asw, posture_str, floor_refl) in
            test_cases
        {
            let py_result = pythermal
                .getattr("solar_gain")
                .unwrap()
                .call1((
                    alt,
                    sharp,
                    sol_rad,
                    sol_trans,
                    f_svv_val,
                    f_bes,
                    asw,
                    posture_str,
                    floor_refl,
                ))
                .unwrap();

            let py_erf: f64 = py_result.getattr("erf").unwrap().extract().unwrap();
            let py_delta_mrt: f64 = py_result.getattr("delta_mrt").unwrap().extract().unwrap();

            let posture = match posture_str {
                "sitting" => Posture::Sitting,
                "standing" => Posture::Standing,
                _ => Posture::Standing,
            };

            let rust_result = solar_gain(
                SolarGainInputs {
                    sol_altitude: Angle::from_degrees(alt),
                    sharp: Angle::from_degrees(sharp),
                    sol_radiation_dir: HeatFluxDensity::from_watts_per_square_meter(sol_rad),
                    sol_transmittance: sol_trans,
                    f_svv: f_svv_val,
                    f_bes,
                },
                SolarGainOptions {
                    asw,
                    posture,
                    floor_reflectance: floor_refl,
                    ..Default::default()
                },
            );

            // Both sides round to one decimal by default, so these agree exactly. The
            // previous 1.0 and 0.5 tolerances were inherited from an era when this
            // comparison was approximate.
            assert_abs_diff_eq!(
                rust_result.erf.as_watts_per_square_meter(),
                py_erf,
                epsilon = 1e-9
            );
            assert_abs_diff_eq!(
                rust_result.delta_mrt.as_celsius(),
                py_delta_mrt,
                epsilon = 1e-9
            );
        }
    });
}

#[test]
fn test_compare_clo_tout() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![27.0, 25.0, 10.0, -10.0, 30.0];

        for tout in test_cases {
            let py_result = pythermal
                .getattr("clo_tout")
                .unwrap()
                .call1((tout,))
                .unwrap();

            let py_clo: f64 = py_result.getattr("clo_tout").unwrap().extract().unwrap();

            let rust_result = clo_tout(Temperature::from_celsius(tout));

            assert_abs_diff_eq!(rust_result, py_clo, epsilon = 0.01);
        }
    });
}

// ============================================================================
// README Example Tests
// ============================================================================
// These tests verify that all examples shown in README.md work correctly
// and produce results matching the Python implementation.

#[test]
fn test_readme_example_basic_pmv_ppd() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Example from README: Basic PMV/PPD Calculation
        let tdb = 25.0; // dry bulb temperature [°C]
        let tr = 25.0; // mean radiant temperature [°C]
        let rh = 50.0; // relative humidity [%]
        let v = 0.1; // air speed [m/s]
        let met = 1.4; // metabolic rate [met]
        let clo = 0.5; // clothing insulation [clo]

        // Calculate relative air speed (accounts for body movement)
        let vr = v_relative(
            Speed::from_meters_per_second(v),
            MetabolicRate::from_met(met),
        );

        // Python calculation
        let py_result = pythermal
            .getattr("pmv_ppd_iso")
            .unwrap()
            .call1((tdb, tr, vr.as_meters_per_second(), rh, met, clo))
            .unwrap();
        let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();
        let py_ppd: f64 = py_result.getattr("ppd").unwrap().extract().unwrap();

        // Rust calculation with measurement types
        let result = pmv_ppd_iso(
            PmvPpdInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                vr,
                rh: Humidity::from_percent(rh),
                met: MetabolicRate::from_met(met),
                clo: ClothingInsulation::from_clo(clo),
            },
            Default::default(),
        );

        // Verify results match README comments: PMV ~0.17, PPD ~5.6%
        assert_abs_diff_eq!(result.pmv, py_pmv, epsilon = 0.01);
        assert_abs_diff_eq!(result.ppd, py_ppd, epsilon = 0.1);

        // Check that results are close to documented values
        assert!((result.pmv - 0.17).abs() < 0.1, "PMV should be ~0.17");
        assert!((result.ppd - 5.6).abs() < 1.0, "PPD should be ~5.6%");
    });
}

#[test]
fn test_readme_example_psychrometric() {
    Python::with_gil(|py| {
        let pyutil = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Example from README: Psychrometric Calculations
        let tdb = 25.0; // dry bulb temperature [°C]
        let rh = 50.0; // relative humidity [%]
        let p_atm = 101325.0; // atmospheric pressure [Pa]

        // Python calculation
        let py_result = pyutil
            .getattr("psy_ta_rh")
            .unwrap()
            .call1((tdb, rh, p_atm))
            .unwrap();
        let py_t_wb: f64 = py_result
            .getattr("wet_bulb_tmp")
            .unwrap()
            .extract()
            .unwrap();
        let py_t_dp: f64 = py_result
            .getattr("dew_point_tmp")
            .unwrap()
            .extract()
            .unwrap();

        // Rust calculation
        let psychro = psy_ta_rh(
            PsyTaRhInputs {
                tdb: Temperature::from_celsius(tdb),
                rh: Humidity::from_percent(rh),
            },
            PsyTaRhOptions {
                p_atm: Pressure::from_pascals(p_atm),
            },
        );

        // Verify results match
        assert_abs_diff_eq!(psychro.t_wb.as_celsius(), py_t_wb, epsilon = 0.1);
        assert_abs_diff_eq!(psychro.t_dp.as_celsius(), py_t_dp, epsilon = 0.1);

        // Check that results are close to documented values
        assert!(
            (psychro.t_wb.as_celsius() - 17.7).abs() < 0.5,
            "Wet bulb temp should be ~17.7°C"
        );
        assert!(
            (psychro.t_dp.as_celsius() - 13.9).abs() < 0.5,
            "Dew point should be ~13.9°C"
        );
    });
}

#[test]
fn test_readme_example_custom_pmv_options() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Example from README: Custom PMV/PPD Options
        let options = PmvPpdIsoOptions {
            wme: MetabolicRate::from_met(0.0), // external work [met]
            model: Default::default(),
            limit_inputs: false, // don't limit to standard ranges
            round_output: true,  // round output values
        };

        // Python calculation with same options
        let kwargs = [("limit_inputs", false)].into_py_dict(py).unwrap();
        let py_result = pythermal
            .getattr("pmv_ppd_iso")
            .unwrap()
            .call((30.0, 30.0, 0.1, 50.0, 1.2, 0.5), Some(&kwargs))
            .unwrap();
        let py_pmv: f64 = py_result.getattr("pmv").unwrap().extract().unwrap();

        // Rust calculation with measurement types
        let result = pmv_ppd_iso(
            PmvPpdInputs {
                tdb: Temperature::from_celsius(30.0),
                tr: Temperature::from_celsius(30.0),
                vr: Speed::from_meters_per_second(0.1),
                rh: Humidity::from_percent(50.0),
                met: MetabolicRate::from_met(1.2),
                clo: ClothingInsulation::from_clo(0.5),
            },
            options,
        );

        assert_abs_diff_eq!(result.pmv, py_pmv, epsilon = 0.02);
    });
}

#[test]
fn test_readme_example_set() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Example from README: Standard Effective Temperature (SET)
        let tdb = 25.0; // dry bulb temperature [°C]
        let tr = 25.0; // mean radiant temperature [°C]
        let v = 0.3; // air speed [m/s]
        let rh = 50.0; // relative humidity [%]
        let met = 1.2; // metabolic rate [met]
        let clo = 0.5; // clothing insulation [clo]

        // Python calculation
        let py_result = pythermal
            .getattr("set_tmp")
            .unwrap()
            .call1((tdb, tr, v, rh, met, clo))
            .unwrap();
        let py_set: f64 = py_result.getattr("set").unwrap().extract().unwrap();

        // Rust calculation with measurement types
        let set = set_tmp(
            SetInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                v: Speed::from_meters_per_second(v),
                rh: Humidity::from_percent(rh),
                met: MetabolicRate::from_met(met),
                clo: ClothingInsulation::from_clo(clo),
            },
            Default::default(),
        )
        .as_celsius();

        // SET has some numerical differences due to iterative solvers
        assert_abs_diff_eq!(set, py_set, epsilon = 1.0);
    });
}

#[test]
fn test_readme_example_cooling_effect() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Example from README: Cooling Effect
        let tdb = 28.0; // dry bulb temperature [°C]
        let tr = 28.0; // mean radiant temperature [°C]
        let vr = 0.8; // relative air speed [m/s]
        let rh = 50.0; // relative humidity [%]
        let met = 1.2; // metabolic rate [met]
        let clo = 0.5; // clothing insulation [clo]

        // Python calculation
        let py_result = pythermal
            .getattr("cooling_effect")
            .unwrap()
            .call1((tdb, tr, vr, rh, met, clo))
            .unwrap();
        let py_ce: f64 = py_result.getattr("ce").unwrap().extract().unwrap();

        // Rust calculation with measurement types
        let ce = cooling_effect(
            CoolingEffectInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                vr: Speed::from_meters_per_second(vr),
                rh: Humidity::from_percent(rh),
                met: MetabolicRate::from_met(met),
                clo: ClothingInsulation::from_clo(clo),
            },
            Default::default(),
        );

        assert_abs_diff_eq!(ce.as_celsius(), py_ce, epsilon = 0.15);
    });
}

#[test]
fn test_readme_example_utci() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Example from README: UTCI (Universal Thermal Climate Index)
        let tdb = 25.0; // dry bulb temperature [°C]
        let tr = 27.0; // mean radiant temperature [°C]
        let v = 1.0; // wind speed at 10m [m/s]
        let rh = 50.0; // relative humidity [%]

        // Python calculation
        let py_result = pythermal
            .getattr("utci")
            .unwrap()
            .call1((tdb, tr, v, rh))
            .unwrap();
        let py_utci: f64 = py_result.getattr("utci").unwrap().extract().unwrap();
        let py_stress = extract_category(&py_result.getattr("stress_category").unwrap());

        // Rust calculation with measurement types
        let result = utci(
            UtciInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                v: Speed::from_meters_per_second(v),
                rh: Humidity::from_percent(rh),
            },
            Default::default(),
        );

        assert_abs_diff_eq!(result.utci, py_utci, epsilon = 0.15);

        // Verify stress category matches
        assert_eq!(
            result.stress_category.map(|c| c.as_str().to_string()),
            py_stress
        );

        // Check that results are close to documented values: UTCI: 25.2°C
        assert!((result.utci - 25.2).abs() < 0.5, "UTCI should be ~25.2°C");
        assert_eq!(
            result.stress_category.map(|c| c.as_str()),
            Some("no thermal stress")
        );
    });
}

#[test]
fn test_clothing_typical_ensembles() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Get Python's typical ensembles dictionary
        let ensembles_obj = pythermal.getattr("clo_typical_ensembles").unwrap();
        let py_ensembles = ensembles_obj.downcast::<pyo3::types::PyDict>().unwrap();

        // Test all ensembles match
        for (name, clo) in CLO_TYPICAL_ENSEMBLES.iter() {
            let py_value: f64 = py_ensembles
                .get_item(name)
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let rust_value = clo_typical_ensemble(name).unwrap();

            println!(
                "Ensemble: '{}' - Python: {}, Rust: {}",
                name, py_value, rust_value
            );

            assert_abs_diff_eq!(rust_value, py_value, epsilon = 0.01);
            assert_abs_diff_eq!(rust_value, *clo, epsilon = 0.001);
        }

        // Verify count matches
        assert_eq!(
            CLO_TYPICAL_ENSEMBLES.len(),
            py_ensembles.len(),
            "Number of typical ensembles should match Python"
        );
    });
}

#[test]
fn test_clothing_individual_garments() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Get Python's individual garments dictionary
        let garments_obj = pythermal.getattr("clo_individual_garments").unwrap();
        let py_garments = garments_obj.downcast::<pyo3::types::PyDict>().unwrap();

        // Test all garments match
        for (name, clo) in CLO_INDIVIDUAL_GARMENTS.iter() {
            let py_value: f64 = py_garments
                .get_item(name)
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let rust_value = clo_individual_garment(name).unwrap();

            assert_abs_diff_eq!(rust_value, py_value, epsilon = 0.01);
            assert_abs_diff_eq!(rust_value, *clo, epsilon = 0.001);
        }

        // Verify count matches
        assert_eq!(
            CLO_INDIVIDUAL_GARMENTS.len(),
            py_garments.len(),
            "Number of individual garments should match Python"
        );
    });
}

#[test]
fn test_clo_intrinsic_insulation_ensemble_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Test cases with different garment combinations
        let test_cases = vec![
            vec![0.25, 0.24, 0.04], // shirt, pants, underwear
            vec![0.5],              // single garment
            vec![0.1, 0.15, 0.2],   // light ensemble
            vec![0.36, 0.44, 0.06], // heavier ensemble
        ];

        for garments in test_cases {
            // Call Python function
            let py_result: f64 = pythermal
                .getattr("clo_intrinsic_insulation_ensemble")
                .unwrap()
                .call1((garments.clone(),))
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_garments: Vec<ClothingInsulation> = garments
                .iter()
                .map(|c| ClothingInsulation::from_clo(*c))
                .collect();
            let rust_result = clo_intrinsic_insulation_ensemble(&rust_garments);

            println!(
                "Garments: {:?} - Python: {:.3}, Rust: {:.3}",
                garments, py_result, rust_result
            );

            assert_abs_diff_eq!(rust_result, py_result, epsilon = 0.01);
        }
    });
}

/// One sleep-model case: a label plus the six per-minute schedules
/// (`tdb`, `tr`, `v`, `rh`, `clo`, `thickness_quilt`), all the same length.
type SleepCase = (
    &'static str,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
);

#[test]
fn test_two_nodes_gagge_sleep_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Whole-night schedules, one value per minute. The third case varies every
        // driving variable over the night — the shape the previous single-value Rust
        // signature could not express at all, and the reason the old test compared one
        // steady-state point with a 2 °C tolerance.
        let cases: Vec<SleepCase> = vec![
            (
                "steady, thin quilt",
                vec![25.0; 30],
                vec![25.0; 30],
                vec![0.1; 30],
                vec![50.0; 30],
                vec![0.5; 30],
                vec![1.0; 30],
            ),
            (
                "steady, thick quilt",
                vec![22.0; 30],
                vec![22.0; 30],
                vec![0.1; 30],
                vec![40.0; 30],
                vec![1.0; 30],
                vec![9.0; 30],
            ),
            (
                "varying overnight",
                (0..45).map(|i| 20.0 + 0.2 * i as f64).collect(),
                (0..45).map(|i| 21.0 + 0.15 * i as f64).collect(),
                (0..45).map(|i| 0.1 + 0.004 * i as f64).collect(),
                (0..45).map(|i| 40.0 + 0.5 * i as f64).collect(),
                (0..45).map(|i| 0.3 + 0.01 * i as f64).collect(),
                (0..45).map(|i| 1.0 + 0.15 * i as f64).collect(),
            ),
        ];

        for (name, tdb, tr, v, rh, clo, thickness) in cases {
            println!("\nTesting sleep schedule: {name} ({} minutes)", tdb.len());

            let py_result = pythermal
                .getattr("two_nodes_gagge_sleep")
                .unwrap()
                .call1((
                    tdb.clone(),
                    tr.clone(),
                    v.clone(),
                    rh.clone(),
                    clo.clone(),
                    thickness.clone(),
                ))
                .unwrap();

            let py_field = |name: &str| -> Vec<f64> {
                py_result
                    .getattr(name)
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract()
                    .unwrap()
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

            let rust_result = two_nodes_gagge_sleep(
                SleepInputs {
                    tdb: &rust_tdb,
                    tr: &rust_tr,
                    v: &rust_v,
                    rh: &rust_rh,
                    clo: &rust_clo,
                    thickness_quilt: &rust_quilt,
                },
                Default::default(),
            )
            .expect("all six schedules are the same length");

            // Every field upstream returns, over the whole trajectory — not just the
            // final minute, and not just the three fields the old test looked at.
            let comparisons: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
                (
                    "set",
                    rust_result.set.iter().map(|t| t.as_celsius()).collect(),
                    py_field("set"),
                ),
                (
                    "t_core",
                    rust_result.t_core.iter().map(|t| t.as_celsius()).collect(),
                    py_field("t_core"),
                ),
                (
                    "t_skin",
                    rust_result.t_skin.iter().map(|t| t.as_celsius()).collect(),
                    py_field("t_skin"),
                ),
                ("wet", rust_result.wet.clone(), py_field("wet")),
                ("t_sens", rust_result.t_sens.clone(), py_field("t_sens")),
                ("disc", rust_result.disc.clone(), py_field("disc")),
                (
                    "e_skin",
                    rust_result
                        .e_skin
                        .iter()
                        .map(|q| q.as_watts_per_square_meter())
                        .collect(),
                    py_field("e_skin"),
                ),
                (
                    "met_shivering",
                    rust_result
                        .met_shivering
                        .iter()
                        .map(|q| q.as_watts_per_square_meter())
                        .collect(),
                    py_field("met_shivering"),
                ),
                ("alfa", rust_result.alfa.clone(), py_field("alfa")),
                (
                    "skin_blood_flow",
                    rust_result.skin_blood_flow.clone(),
                    py_field("skin_blood_flow"),
                ),
            ];

            for (field, rust_vals, py_vals) in comparisons {
                assert_eq!(
                    rust_vals.len(),
                    py_vals.len(),
                    "{name}/{field}: trajectory length"
                );
                for (i, (r, p)) in rust_vals.iter().zip(&py_vals).enumerate() {
                    assert_abs_diff_eq!(r, p, epsilon = 1e-9);
                    let _ = i;
                }
            }
            println!("  all 10 fields match over {} minutes", tdb.len());
        }
    });
}

#[test]
fn test_clo_tout_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test across full temperature range
        let test_temps = vec![
            -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 26.0, 27.0, 30.0,
        ];

        for tout in test_temps {
            // Call Python function
            let py_result = pythermal
                .getattr("clo_tout")
                .unwrap()
                .call1((tout,))
                .unwrap();

            let py_clo: f64 = py_result.getattr("clo_tout").unwrap().extract().unwrap();

            // Call Rust function
            let rust_clo = clo_tout(Temperature::from_celsius(tout));

            println!(
                "Tout: {:.1}°C - Python: {:.2} clo, Rust: {:.2} clo",
                tout, py_clo, rust_clo
            );

            // Should match exactly as this is a simple lookup formula
            assert_abs_diff_eq!(rust_clo, py_clo, epsilon = 0.01);
        }
    });
}

#[test]
fn test_antoine_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Test across temperature range
        let test_temps = vec![0.0, 10.0, 20.0, 25.0, 30.0, 40.0];

        for t in test_temps {
            // Call Python function
            let py_result: f64 = pythermal
                .getattr("antoine")
                .unwrap()
                .call1((t,))
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_result = antoine(Temperature::from_celsius(t));

            println!(
                "T: {:.1}°C - Python: {:.6} kPa, Rust: {:.6} kPa",
                t, py_result, rust_result
            );

            // Should match exactly as this is the same formula
            assert_abs_diff_eq!(rust_result, py_result, epsilon = 0.000001);
        }
    });
}

#[test]
fn test_ridge_regression_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test case: Male, 60 years old, hot environment
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("sex", "male").unwrap();
        kwargs.set_item("age", 60).unwrap();
        kwargs.set_item("height", 1.8).unwrap();
        kwargs.set_item("weight", 75).unwrap();
        kwargs.set_item("tdb", 35).unwrap();
        kwargs.set_item("rh", 60).unwrap();
        kwargs.set_item("duration", 60).unwrap();

        let py_result = pythermal
            .getattr("ridge_regression_predict_t_re_t_sk")
            .unwrap()
            .call((), Some(&kwargs))
            .unwrap();

        let py_t_re: Vec<f64> = py_result.getattr("t_re").unwrap().extract().unwrap();
        let py_t_sk: Vec<f64> = py_result.getattr("t_sk").unwrap().extract().unwrap();

        // Call Rust function
        let rust_result = ridge_regression_predict_t_re_t_sk(
            RidgeRegressionInputs {
                sex: Sex::Male,
                age: 60.0,
                height: Length::from_meters(1.8),
                weight: Mass::from_kilograms(75.0),
                tdb: Temperature::from_celsius(35.0),
                rh: Humidity::from_percent(60.0),
                duration: 60,
            },
            Default::default(),
        );

        println!("Duration: {} minutes", rust_result.t_re.len());
        println!(
            "Final temps - Python: t_re={:.2}, t_sk={:.2}",
            py_t_re.last().unwrap(),
            py_t_sk.last().unwrap()
        );
        println!(
            "Final temps - Rust: t_re={:.2}, t_sk={:.2}",
            rust_result.t_re.last().unwrap().as_celsius(),
            rust_result.t_sk.last().unwrap().as_celsius()
        );

        // Check lengths match
        assert_eq!(rust_result.t_re.len(), 60);
        assert_eq!(rust_result.t_sk.len(), 60);
        assert_eq!(rust_result.t_re.len(), py_t_re.len());
        assert_eq!(rust_result.t_sk.len(), py_t_sk.len());

        // Compare a few time points
        // Initial (minute 0)
        assert_abs_diff_eq!(rust_result.t_re[0].as_celsius(), py_t_re[0], epsilon = 0.01);
        assert_abs_diff_eq!(rust_result.t_sk[0].as_celsius(), py_t_sk[0], epsilon = 0.01);

        // Middle (minute 30)
        assert_abs_diff_eq!(
            rust_result.t_re[30].as_celsius(),
            py_t_re[30],
            epsilon = 0.01
        );
        assert_abs_diff_eq!(
            rust_result.t_sk[30].as_celsius(),
            py_t_sk[30],
            epsilon = 0.01
        );

        // Final (minute 59)
        assert_abs_diff_eq!(
            rust_result.t_re.last().unwrap().as_celsius(),
            *py_t_re.last().unwrap(),
            epsilon = 0.01
        );
        assert_abs_diff_eq!(
            rust_result.t_sk.last().unwrap().as_celsius(),
            *py_t_sk.last().unwrap(),
            epsilon = 0.01
        );
    });
}

#[test]
fn test_two_nodes_gagge_ji_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");
        let pyutil = import_reference(py, "pythermalcomfort.utilities")
            .expect("Failed to import pythermalcomfort.utilities");

        // Test cases for elderly (JI model)
        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.2, 0.5), // Typical conditions
            (28.0, 28.0, 0.2, 60.0, 1.0, 0.5), // Warmer
            (22.0, 22.0, 0.1, 40.0, 1.1, 1.0), // Cooler
        ];

        for (tdb, tr, v, rh, met, clo) in test_cases {
            println!(
                "\nTesting JI: tdb={}, tr={}, v={}, rh={}, met={}, clo={}",
                tdb, tr, v, rh, met, clo
            );

            // Calculate vapor pressure from RH
            let p_sat: f64 = pyutil
                .getattr("p_sat_torr")
                .unwrap()
                .call1((tdb,))
                .unwrap()
                .extract()
                .unwrap();
            let vapor_pressure = rh * p_sat / 100.0;

            // Call Python function (uses vapor pressure, not RH)
            let py_result = pythermal
                .getattr("two_nodes_gagge_ji")
                .unwrap()
                .call1((tdb, tr, v, met, clo, vapor_pressure))
                .unwrap();

            // Python JI model returns time series (120 values)
            let py_t_core_array = py_result.getattr("t_core").unwrap();
            let py_t_skin_array = py_result.getattr("t_skin").unwrap();

            // Get final value from numpy array
            let py_t_core_final: f64 = py_t_core_array
                .call_method1("__getitem__", (-1,))
                .unwrap()
                .extract()
                .unwrap();
            let py_t_skin_final: f64 = py_t_skin_array
                .call_method1("__getitem__", (-1,))
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_result = two_nodes_gagge_ji(
                GaggeTwoNodesJiInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    vapor_pressure: Pressure::from_torrs(vapor_pressure),
                },
                Default::default(),
            );

            println!(
                "  Python - T_core (final): {:.2}, T_skin (final): {:.2}",
                py_t_core_final, py_t_skin_final
            );
            let rust_t_core_final = rust_result.t_core.last().unwrap().as_celsius();
            let rust_t_skin_final = rust_result.t_skin.last().unwrap().as_celsius();
            println!(
                "  Rust   - T_core (final): {:.2}, T_skin (final): {:.2}",
                rust_t_core_final, rust_t_skin_final
            );

            // Check length
            assert_eq!(rust_result.t_core.len(), 120);
            assert_eq!(rust_result.t_skin.len(), 120);

            // Compare final values
            // Ji model has acceptable accuracy within 0.5°C for skin temperature
            assert_abs_diff_eq!(rust_t_core_final, py_t_core_final, epsilon = 0.1);
            assert_abs_diff_eq!(
                rust_t_skin_final,
                py_t_skin_final,
                epsilon = 0.5 // Larger tolerance for skin temp due to numerical differences
            );
        }
    });
}

#[test]
fn test_pet_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test cases: (tdb, tr, v, rh, met, clo)
        let test_cases = vec![
            (25.0, 25.0, 0.1, 50.0, 1.0, 0.5),
            (35.0, 35.0, 1.0, 60.0, 1.2, 0.5),
            (5.0, 5.0, 2.0, 50.0, 1.5, 1.0),
        ];

        for (tdb, tr, v, rh, met, clo) in test_cases {
            println!(
                "\nTesting PET: tdb={}, tr={}, v={}, rh={}, met={}, clo={}",
                tdb, tr, v, rh, met, clo
            );

            // Call Python function
            let py_result = pythermal
                .getattr("pet_steady")
                .unwrap()
                .call1((tdb, tr, v, rh, met, clo))
                .unwrap();

            let py_pet: f64 = py_result.getattr("pet").unwrap().extract().unwrap();

            // Call Rust function
            let rust_result = pet_steady(
                PetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                Default::default(),
            );

            println!("  Python - PET: {:.2}°C", py_pet);
            println!("  Rust   - PET: {:.2}°C", rust_result.pet);

            // Compare results (PET can have larger differences due to numerical solving)
            let diff = (rust_result.pet - py_pet).abs();
            println!("  Difference: {:.2}°C", diff);

            // PET accuracy with full-matrix Newton solver:
            // - Normal conditions (20-35°C): <0.1°C (excellent)
            // - Cold + high wind (5°C, 2m/s): ~2.5°C (acceptable)
            // Full-matrix Jacobian improved cold case from 9.2°C to 2.5°C error
            let tolerance = if tdb < 10.0 && v > 1.5 {
                3.0 // Cold + high wind case
            } else {
                0.5 // Normal conditions
            };
            assert_abs_diff_eq!(rust_result.pet, py_pet, epsilon = tolerance);
        }
    });
}

/// `position="standing, forced convection"` (`pet_steady.py`) uses a different
/// convective coefficient (`hc = 8.6 * v**0.513`) than plain "standing"; this was
/// unreachable before `PetOptions::forced_convection` existed.
#[test]
fn test_pet_forced_convection_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let test_cases = vec![
            (25.0, 25.0, 1.0, 50.0, 1.0, 0.5),
            (30.0, 30.0, 2.0, 60.0, 1.2, 0.4),
        ];

        for (tdb, tr, v, rh, met, clo) in test_cases {
            let kwargs = [("position", "standing, forced convection")]
                .into_py_dict(py)
                .unwrap();
            let py_result = pythermal
                .getattr("pet_steady")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo), Some(&kwargs))
                .unwrap();

            let py_pet: f64 = py_result.getattr("pet").unwrap().extract().unwrap();

            let rust_result = pet_steady(
                PetInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                },
                PetOptions {
                    position: thermalcomfort::models::pet::Posture::Standing,
                    forced_convection: true,
                    ..Default::default()
                },
            );

            assert_abs_diff_eq!(rust_result.pet, py_pet, epsilon = 0.5);
        }
    });
}

#[test]
fn test_phs_iso2023_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test cases: (tdb, tr, v, rh, met, clo, posture)
        let test_cases = vec![
            // Standard hot condition
            (40.0, 40.0, 0.3, 33.85, 2.5, 0.5, "standing"),
            // Higher humidity
            (38.0, 38.0, 0.5, 50.0, 2.0, 0.5, "standing"),
            // Lower activity
            (35.0, 35.0, 0.3, 40.0, 1.8, 0.6, "sitting"),
            // Higher activity
            (42.0, 42.0, 0.4, 30.0, 3.0, 0.4, "standing"),
        ];

        for (tdb, tr, v, rh, met, clo, posture) in test_cases {
            println!(
                "\nTesting PHS: tdb={}, tr={}, v={}, rh={}, met={}, clo={}, posture={}",
                tdb, tr, v, rh, met, clo, posture
            );

            // Call Python function
            let kwargs = [("duration", 480)].into_py_dict(py).unwrap();
            let py_result = pythermal
                .getattr("phs")
                .unwrap()
                .call((tdb, tr, v, rh, met, clo, posture), Some(&kwargs))
                .unwrap();

            let py_t_re: f64 = py_result.getattr("t_re").unwrap().extract().unwrap();
            let py_t_sk: f64 = py_result.getattr("t_sk").unwrap().extract().unwrap();
            let py_t_cr: f64 = py_result.getattr("t_cr").unwrap().extract().unwrap();
            let py_d_lim_loss_50: f64 = py_result
                .getattr("d_lim_loss_50")
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust function
            let rust_posture = match posture {
                "standing" => PhsPosture::Standing,
                "sitting" => PhsPosture::Sitting,
                "crouching" => PhsPosture::Crouching,
                _ => panic!("Unknown posture"),
            };

            let rust_result = phs(
                PhsInputs {
                    tdb: Temperature::from_celsius(tdb),
                    tr: Temperature::from_celsius(tr),
                    v: Speed::from_meters_per_second(v),
                    rh: Humidity::from_percent(rh),
                    met: MetabolicRate::from_met(met),
                    clo: ClothingInsulation::from_clo(clo),
                    posture: rust_posture,
                },
                PhsOptions::default(),
            );

            println!(
                "  Python - t_re: {:.1}°C, t_sk: {:.1}°C, t_cr: {:.1}°C, d_lim_50: {:.0} min",
                py_t_re, py_t_sk, py_t_cr, py_d_lim_loss_50
            );
            println!(
                "  Rust   - t_re: {:.1}°C, t_sk: {:.1}°C, t_cr: {:.1}°C, d_lim_50: {:.0} min",
                rust_result.t_re.as_celsius(),
                rust_result.t_sk.as_celsius(),
                rust_result.t_cr.as_celsius(),
                rust_result.d_lim_loss_50
            );

            // Compare results
            assert_abs_diff_eq!(rust_result.t_re.as_celsius(), py_t_re, epsilon = 0.2);
            assert_abs_diff_eq!(rust_result.t_sk.as_celsius(), py_t_sk, epsilon = 0.2);
            assert_abs_diff_eq!(rust_result.t_cr.as_celsius(), py_t_cr, epsilon = 0.2);
            // Exposure time limits can differ slightly due to rounding
            assert_abs_diff_eq!(rust_result.d_lim_loss_50, py_d_lim_loss_50, epsilon = 2.0);
        }
    });
}

#[test]
fn test_phs_iso2004_comparison() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test case for ISO 2004 model
        let (tdb, tr, v, rh, met, clo) = (35.0, 35.0, 0.5, 50.0, 2.0, 0.5);

        println!(
            "\nTesting PHS ISO 2004: tdb={}, tr={}, v={}, rh={}, met={}, clo={}",
            tdb, tr, v, rh, met, clo
        );

        // Call Python function with model="7933-2004"
        let py_dict = pyo3::types::PyDict::new(py);
        py_dict.set_item("duration", 480).unwrap();
        py_dict.set_item("model", "7933-2004").unwrap();
        let py_result = pythermal
            .getattr("phs")
            .unwrap()
            .call((tdb, tr, v, rh, met, clo, "standing"), Some(&py_dict))
            .unwrap();

        let py_t_re: f64 = py_result.getattr("t_re").unwrap().extract().unwrap();
        let py_t_sk: f64 = py_result.getattr("t_sk").unwrap().extract().unwrap();
        let py_t_cr: f64 = py_result.getattr("t_cr").unwrap().extract().unwrap();

        // Call Rust function with ISO 2004 model
        let options = PhsOptions {
            model: Iso7933Model::Iso2004,
            ..Default::default()
        };

        let rust_result = phs(
            PhsInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                v: Speed::from_meters_per_second(v),
                rh: Humidity::from_percent(rh),
                met: MetabolicRate::from_met(met),
                clo: ClothingInsulation::from_clo(clo),
                posture: PhsPosture::Standing,
            },
            options,
        );

        println!(
            "  Python - t_re: {:.1}°C, t_sk: {:.1}°C, t_cr: {:.1}°C",
            py_t_re, py_t_sk, py_t_cr
        );
        println!(
            "  Rust   - t_re: {:.1}°C, t_sk: {:.1}°C, t_cr: {:.1}°C",
            rust_result.t_re.as_celsius(),
            rust_result.t_sk.as_celsius(),
            rust_result.t_cr.as_celsius()
        );

        // Compare results
        assert_abs_diff_eq!(rust_result.t_re.as_celsius(), py_t_re, epsilon = 0.2);
        assert_abs_diff_eq!(rust_result.t_sk.as_celsius(), py_t_sk, epsilon = 0.2);
        assert_abs_diff_eq!(rust_result.t_cr.as_celsius(), py_t_cr, epsilon = 0.2);
    });
}

#[test]
fn test_phs_short_duration() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        // Test shorter duration (60 minutes)
        let (tdb, tr, v, rh, met, clo) = (40.0, 40.0, 0.3, 50.0, 2.5, 0.5);

        println!("\nTesting PHS short duration (60 min)");

        // Call Python function
        let kwargs = [("duration", 60)].into_py_dict(py).unwrap();
        let py_result = pythermal
            .getattr("phs")
            .unwrap()
            .call((tdb, tr, v, rh, met, clo, "standing"), Some(&kwargs))
            .unwrap();

        let py_t_re: f64 = py_result.getattr("t_re").unwrap().extract().unwrap();
        let py_sweat_loss_g: f64 = py_result
            .getattr("sweat_loss_g")
            .unwrap()
            .extract()
            .unwrap();

        // Call Rust function
        let options = PhsOptions {
            duration: 60,
            ..Default::default()
        };

        let rust_result = phs(
            PhsInputs {
                tdb: Temperature::from_celsius(tdb),
                tr: Temperature::from_celsius(tr),
                v: Speed::from_meters_per_second(v),
                rh: Humidity::from_percent(rh),
                met: MetabolicRate::from_met(met),
                clo: ClothingInsulation::from_clo(clo),
                posture: PhsPosture::Standing,
            },
            options,
        );

        println!(
            "  Python - t_re: {:.1}°C, sweat_loss: {:.0} g",
            py_t_re, py_sweat_loss_g
        );
        println!(
            "  Rust   - t_re: {:.1}°C, sweat_loss: {:.0} g",
            rust_result.t_re.as_celsius(),
            rust_result.sweat_loss_g.as_grams()
        );

        // Compare results
        // Note: Short duration simulations can have slightly larger temperature differences
        // due to numerical precision in the time-stepping process
        assert_abs_diff_eq!(rust_result.t_re.as_celsius(), py_t_re, epsilon = 0.5);
        assert_abs_diff_eq!(
            rust_result.sweat_loss_g.as_grams(),
            py_sweat_loss_g,
            epsilon = 50.0
        );
    });
}

/// The ISO 7933:2023 Annex E minute-1 skin-temperature special case (added upstream in
/// 4.4.1): `t_sk` is forced to its equilibrium value on the first minute rather than
/// exponentially lagging from `t_sk0`. The 2004 edition has no such case.
///
/// This needs its own test because **no other PHS test can detect it.** The special case
/// perturbs `t_sk` only on minute 1, and the lag constant is `exp(-1/3)`, so the
/// difference decays by ~0.717 per minute: it is ~0.57 °C at minute 1 but ~6e-10 °C by
/// minute 60. `test_phs_short_duration` (60 min) and the two 480-minute comparisons all
/// pass unchanged with the special case deleted -- verified by deleting it. Only a
/// simulation of a few minutes is short enough to see it.
///
/// `round_output` is off and the tolerance is 1e-9, so this also pins that the special
/// case fires on minute 1 exactly, not minute 0 or 2.
#[test]
fn test_phs_minute_one_skin_temperature_special_case() {
    Python::with_gil(|py| {
        let pythermal = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let (tdb, tr, v, rh, met, clo) = (40.0, 40.0, 0.3, 33.85, 2.5, 0.5);

        for (model, py_model) in [
            (Iso7933Model::Iso2023, "7933-2023"),
            (Iso7933Model::Iso2004, "7933-2004"),
        ] {
            for duration in [1_i32, 2, 3] {
                let label = format!("{py_model} duration={duration}");

                let kwargs = [
                    ("duration", duration.into_pyobject(py).unwrap().into_any()),
                    ("model", py_model.into_pyobject(py).unwrap().into_any()),
                    (
                        "round_output",
                        false.into_pyobject(py).unwrap().to_owned().into_any(),
                    ),
                ]
                .into_py_dict(py)
                .unwrap();
                let py_result = pythermal
                    .getattr("phs")
                    .unwrap()
                    .call((tdb, tr, v, rh, met, clo, "standing"), Some(&kwargs))
                    .unwrap_or_else(|e| panic!("{label}: phs raised: {e}"));

                let py_t_sk: f64 = py_result.getattr("t_sk").unwrap().extract().unwrap();
                let py_t_re: f64 = py_result.getattr("t_re").unwrap().extract().unwrap();
                let py_sweat_loss_g: f64 = py_result
                    .getattr("sweat_loss_g")
                    .unwrap()
                    .extract()
                    .unwrap();

                let rust_result = phs(
                    PhsInputs {
                        tdb: Temperature::from_celsius(tdb),
                        tr: Temperature::from_celsius(tr),
                        v: Speed::from_meters_per_second(v),
                        rh: Humidity::from_percent(rh),
                        met: MetabolicRate::from_met(met),
                        clo: ClothingInsulation::from_clo(clo),
                        posture: PhsPosture::Standing,
                    },
                    PhsOptions {
                        duration,
                        model,
                        round_output: false,
                        ..Default::default()
                    },
                );

                // Plain `assert!` rather than `assert_abs_diff_eq!` so the failure names
                // the model and duration that diverged.
                assert!(
                    (rust_result.t_sk.as_celsius() - py_t_sk).abs() < 1e-9,
                    "{label}: t_sk {} != {py_t_sk}",
                    rust_result.t_sk.as_celsius()
                );
                assert!(
                    (rust_result.t_re.as_celsius() - py_t_re).abs() < 1e-9,
                    "{label}: t_re {} != {py_t_re}",
                    rust_result.t_re.as_celsius()
                );
                assert!(
                    (rust_result.sweat_loss_g.as_grams() - py_sweat_loss_g).abs() < 1e-9,
                    "{label}: sweat_loss_g {} != {py_sweat_loss_g}",
                    rust_result.sweat_loss_g.as_grams()
                );
            }
        }
    });
}

/// Test sports_heat_stress_risk against Python pythermalcomfort
#[test]
fn test_sports_heat_stress_risk_comparison() {
    use thermalcomfort::models::sports_heat_stress_risk::{Sports, sports_heat_stress_risk};

    Python::with_gil(|py| {
        let sports_mod = import_reference(py, "pythermalcomfort.models.sports_heat_stress_risk")
            .expect("Failed to import sports_heat_stress_risk module");
        let py_sports_class = sports_mod
            .getattr("Sports")
            .expect("Failed to get Sports class");
        let py_func = sports_mod
            .getattr("sports_heat_stress_risk")
            .expect("Failed to get sports_heat_stress_risk function");

        // Test cases: (tdb, tr, rh, vr, sport_name, rust_sport)
        let test_cases: Vec<(
            f64,
            f64,
            f64,
            f64,
            &str,
            thermalcomfort::models::SportsValues,
        )> = vec![
            (35.0, 35.0, 40.0, 0.1, "RUNNING", Sports::RUNNING),
            (30.0, 30.0, 50.0, 0.5, "SOCCER", Sports::SOCCER),
            (20.0, 20.0, 50.0, 0.5, "WALKING", Sports::WALKING),
            (45.0, 45.0, 30.0, 0.5, "CYCLING", Sports::CYCLING),
            (33.0, 70.0, 60.0, 0.1, "TENNIS", Sports::TENNIS),
        ];

        for (tdb, tr, rh, vr, sport_name, rust_sport) in &test_cases {
            println!(
                "\nTest: {} at tdb={}, tr={}, rh={}, vr={}",
                sport_name, tdb, tr, rh, vr
            );

            // Call Python
            let py_sport = py_sports_class.getattr(*sport_name).unwrap();
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("tdb", tdb).unwrap();
            kwargs.set_item("tr", tr).unwrap();
            kwargs.set_item("rh", rh).unwrap();
            kwargs.set_item("vr", vr).unwrap();
            kwargs.set_item("sport", py_sport).unwrap();
            let py_result = py_func.call((), Some(&kwargs)).unwrap();

            // Python returns numpy arrays for scalar inputs; use .item() to extract
            let py_risk: f64 = py_result
                .getattr("risk_level_interpolated")
                .unwrap()
                .call_method0("item")
                .unwrap()
                .extract()
                .unwrap();
            let py_t_medium: f64 = py_result
                .getattr("t_medium")
                .unwrap()
                .call_method0("item")
                .unwrap()
                .extract()
                .unwrap();
            let py_t_high: f64 = py_result
                .getattr("t_high")
                .unwrap()
                .call_method0("item")
                .unwrap()
                .extract()
                .unwrap();
            let py_t_extreme: f64 = py_result
                .getattr("t_extreme")
                .unwrap()
                .call_method0("item")
                .unwrap()
                .extract()
                .unwrap();
            let py_recommendation: String = py_result
                .getattr("recommendation")
                .unwrap()
                .call_method0("item")
                .unwrap()
                .extract()
                .unwrap();

            // Call Rust
            let rust_result = sports_heat_stress_risk(SportsHeatStressRiskInputs {
                tdb: Temperature::from_celsius(*tdb),
                tr: Temperature::from_celsius(*tr),
                rh: Humidity::from_percent(*rh),
                vr: Speed::from_meters_per_second(*vr),
                sport: *rust_sport,
            })
            .expect("none of the fixed test cases have a NaN tdb");

            println!(
                "  Python - risk: {}, t_med: {}, t_high: {}, t_ext: {}",
                py_risk, py_t_medium, py_t_high, py_t_extreme
            );
            println!(
                "  Rust   - risk: {}, t_med: {}, t_high: {}, t_ext: {}",
                rust_result.risk_level_interpolated,
                rust_result.t_medium.as_celsius(),
                rust_result.t_high.as_celsius(),
                rust_result.t_extreme.as_celsius()
            );

            // Compare results
            assert_abs_diff_eq!(rust_result.risk_level_interpolated, py_risk, epsilon = 0.1);
            assert_abs_diff_eq!(
                rust_result.t_medium.as_celsius(),
                py_t_medium,
                epsilon = 0.5
            );
            assert_abs_diff_eq!(rust_result.t_high.as_celsius(), py_t_high, epsilon = 0.5);
            assert_abs_diff_eq!(
                rust_result.t_extreme.as_celsius(),
                py_t_extreme,
                epsilon = 0.5
            );
            assert_eq!(rust_result.recommendation, py_recommendation.as_str());
        }
    });
}

/// A NaN `tdb` leaves every risk-band comparison false on both sides: Python raises
/// `ValueError("Risk level could not be determined due to NaN thresholds.")`
/// (`sports_heat_stress_risk.py:372-373`) and Rust must return
/// `Err(SportsHeatStressRiskError::NanRiskLevel)` at the same point, rather than the two
/// disagreeing about whether the inputs are even determinate.
#[test]
fn test_sports_heat_stress_risk_nan_tdb_matches_python_raise() {
    use thermalcomfort::models::sports_heat_stress_risk::{
        Sports, SportsHeatStressRiskError, sports_heat_stress_risk,
    };

    Python::with_gil(|py| {
        let sports_mod = import_reference(py, "pythermalcomfort.models.sports_heat_stress_risk")
            .expect("Failed to import sports_heat_stress_risk module");
        let py_sports_class = sports_mod
            .getattr("Sports")
            .expect("Failed to get Sports class");
        let py_func = sports_mod
            .getattr("sports_heat_stress_risk")
            .expect("Failed to get sports_heat_stress_risk function");
        let py_sport = py_sports_class.getattr("RUNNING").unwrap();

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("tdb", f64::NAN).unwrap();
        kwargs.set_item("tr", 35.0).unwrap();
        kwargs.set_item("rh", 40.0).unwrap();
        kwargs.set_item("vr", 0.1).unwrap();
        kwargs.set_item("sport", py_sport).unwrap();
        let py_call = py_func.call((), Some(&kwargs));

        match py_call {
            Err(e) if e.is_instance_of::<PyValueError>(py) => {}
            Err(e) => panic!("python raised an unexpected error: {e}"),
            Ok(result) => panic!("python did not raise for a NaN tdb: {result:?}"),
        }

        let rust_result = sports_heat_stress_risk(SportsHeatStressRiskInputs {
            tdb: Temperature::from_celsius(f64::NAN),
            tr: Temperature::from_celsius(35.0),
            rh: Humidity::from_percent(40.0),
            vr: Speed::from_meters_per_second(0.1),
            sport: Sports::RUNNING,
        });
        assert_eq!(rust_result, Err(SportsHeatStressRiskError::NanRiskLevel));
    });
}

// ---------------------------------------------------------------------------
// JOS3
// ---------------------------------------------------------------------------
//
// JOS3 is a stateful, 17-body-segment, 85-node thermoregulation model. Python drives it
// through mutable properties (`model.tdb = ...`) plus `simulate(times, dtime)`; the Rust
// port uses a builder ([`Jos3Builder`]) and an explicit [`Jos3Model::advance`] call that
// takes the whole environment snapshot ([`Jos3Conditions`]) for that call. `results()`
// accumulates one row per step (plus the constructor's own reference-steady-state row),
// mirroring Python's `JOS3.results()`/`JOS3.dict_results()`.
//
// Compared against `results()`, NOT `dict_results()`, deliberately: `dict_results()` is
// broken in pythermalcomfort 4.4.0 for every per-body-part field. Its implementation
// (`models/jos3.py`, `dict_results`) does `values = value.__dict__` and then
// `zip(keys, values)` -- iterating a *dict* yields its keys, not its values -- so e.g.
// `dict_results()['t_skin_head']` is `['head', 'head', 'head', ...]` at every timestep,
// never the temperature. Confirmed directly against 4.4.0. `results()` shares the exact
// same underlying `_history` and is not affected by this bug (`results().t_skin.head`
// gives the real trajectory). Do NOT "fix" this back to `dict_results()`: the point of
// this suite is to match upstream's *values*, not to reproduce its defects.

/// The 17 JOS3 body segment names, canonical order. Matches
/// [`thermalcomfort::models::jos3::Jos3Model::body_names`] and pythermalcomfort's
/// `JOS3BodyParts.get_attribute_names()`.
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

/// A plain numeric field on Python's `JOS3Output`/`results()`, read across every step.
fn jos3_f64_series(py_output: &Bound<'_, PyAny>, field: &str) -> Vec<f64> {
    py_output
        .getattr(field)
        .unwrap_or_else(|e| panic!("{field}: missing on Python JOS3Output: {e}"))
        .call_method0("tolist")
        .unwrap_or_else(|e| panic!("{field}: tolist() raised: {e}"))
        .extract()
        .unwrap_or_else(|e| panic!("{field}: could not extract as Vec<f64>: {e}"))
}

/// `sex` is the one string-valued field.
fn jos3_sex_series(py_output: &Bound<'_, PyAny>) -> Vec<Sex> {
    let raw: Vec<String> = py_output
        .getattr("sex")
        .unwrap()
        .call_method0("tolist")
        .unwrap()
        .extract()
        .unwrap();
    raw.iter()
        .map(|s| match s.as_str() {
            "male" => Sex::Male,
            "female" => Sex::Female,
            other => panic!("sex: unexpected value {other:?}"),
        })
        .collect()
}

/// `simulation_time` is `datetime.timedelta`, not a plain number, so it needs its own
/// accessor rather than [`jos3_f64_series`].
fn jos3_simulation_time_series(py_output: &Bound<'_, PyAny>) -> Vec<f64> {
    let obj = py_output.getattr("simulation_time").unwrap();
    let n = obj.len().unwrap();
    (0..n)
        .map(|i| {
            obj.get_item(i)
                .unwrap()
                .call_method0("total_seconds")
                .unwrap()
                .extract()
                .unwrap()
        })
        .collect()
}

/// One series per body segment, in [`JOS3_BODY_NAMES`] order, for a 17-wide field
/// (`t_skin`, `tdb`, `bf_core`, ...).
fn jos3_body_part_series(py_output: &Bound<'_, PyAny>, field: &str) -> [Vec<f64>; 17] {
    let body = py_output
        .getattr(field)
        .unwrap_or_else(|e| panic!("{field}: missing on Python JOS3Output: {e}"));
    std::array::from_fn(|i| {
        body.getattr(JOS3_BODY_NAMES[i])
            .unwrap_or_else(|e| panic!("{field}.{}: {e}", JOS3_BODY_NAMES[i]))
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap()
    })
}

/// The head/pelvis pair for a muscle- or fat-layer field (`t_muscle`, `bf_fat`, ...) --
/// upstream only ever populates those two of `JOS3BodyParts`'s 17 attributes for these.
fn jos3_head_pelvis_series(py_output: &Bound<'_, PyAny>, field: &str) -> (Vec<f64>, Vec<f64>) {
    let body = py_output
        .getattr(field)
        .unwrap_or_else(|e| panic!("{field}: missing on Python JOS3Output: {e}"));
    let get = |name: &str| -> Vec<f64> {
        body.getattr(name)
            .unwrap_or_else(|e| panic!("{field}.{name}: {e}"))
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap()
    };
    (get("head"), get("pelvis"))
}

/// `t_superficial_vein`'s 12 raw values, positionally.
///
/// `pass_values_to_jos3_body_parts` (`jos3_functions/construction.py`) zips the *full*
/// 17-name list against the 12 raw superficial-vein values with `strict=False`, because
/// `t_superficial_vein`'s field is built with the default `body_parts=None` rather than
/// the explicit `["head", "pelvis"]` the muscle/fat fields use. That leaves the values
/// sitting under the wrong body-part names (raw index 0 -- physically `left_shoulder`'s
/// reading -- ends up under the `head` attribute, and so on for the first 12 canonical
/// names) and the last 5 canonical attributes (`left_leg` through `right_foot`) unset.
/// This mislabeling is a pre-existing upstream quirk, not something this port reproduces
/// or needs to: since the zip is positional, reading the first 12 canonical names in
/// order recovers the 12 raw values in the same order
/// [`thermalcomfort::models::jos3::Jos3Model::t_superficial_vein`]/
/// `Jos3Results::t_superficial_vein` use. Verified directly against 4.4.0.
fn jos3_superficial_vein_series(py_output: &Bound<'_, PyAny>) -> [Vec<f64>; 12] {
    let body = py_output
        .getattr("t_superficial_vein")
        .unwrap_or_else(|e| panic!("t_superficial_vein: missing on Python JOS3Output: {e}"));
    std::array::from_fn(|i| {
        body.getattr(JOS3_BODY_NAMES[i])
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap()
    })
}

/// One value comparison, tolerant of both sides agreeing on NaN (not currently expected
/// for any case swept here, but cheap to allow rather than panic obscurely on).
fn assert_jos3_close(label: &str, field: &str, step: usize, rust: f64, py: f64) {
    let ok = (rust.is_nan() && py.is_nan()) || (rust - py).abs() <= 1e-9;
    assert!(
        ok,
        "{label}: {field}[{step}] mismatch: rust={rust}, python={py}, diff={:.3e}",
        (rust - py).abs()
    );
}

/// Compare every field of a Rust [`Jos3Results`] against `py_model.results()` (see the
/// module-level comment above for why `results()` and not `dict_results()`), over every
/// step in the trajectory.
fn assert_jos3_matches_python(py_model: &Bound<'_, PyAny>, rust: &Jos3Results, label: &str) {
    let py = py_model
        .call_method0("results")
        .unwrap_or_else(|e| panic!("{label}: JOS3.results() raised: {e}"));

    let n = rust.simulation_time.len();
    assert_eq!(
        jos3_f64_series(&py, "t_skin_mean").len(),
        n,
        "{label}: trajectory length"
    );

    // -- simulation_time (timedelta) --
    let py_sim_time = jos3_simulation_time_series(&py);
    for (step, (&r, &p)) in rust
        .simulation_time
        .iter()
        .zip(py_sim_time.iter())
        .enumerate()
    {
        assert_jos3_close(label, "simulation_time", step, r, p);
    }

    // -- age (int) --
    let py_age: Vec<i32> = py
        .getattr("age")
        .unwrap()
        .call_method0("tolist")
        .unwrap()
        .extract()
        .unwrap();
    assert_eq!(rust.age, py_age, "{label}: age");

    // -- sex (string) --
    assert_eq!(rust.sex, jos3_sex_series(&py), "{label}: sex");

    // -- plain scalar fields --
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
        assert_eq!(rust_series.len(), n, "{label}: {field} rust length");
        let py_series = jos3_f64_series(&py, field);
        assert_eq!(py_series.len(), n, "{label}: {field} python length");
        for (step, (&r, &p)) in rust_series.iter().zip(py_series.iter()).enumerate() {
            assert_jos3_close(label, field, step, r, p);
        }
    }

    // -- 17-wide per-body-part fields --
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
        assert_eq!(rust_series.len(), n, "{label}: {field} rust length");
        let py_series = jos3_body_part_series(&py, field);
        for (part_idx, part_name) in JOS3_BODY_NAMES.iter().enumerate() {
            let py_part = &py_series[part_idx];
            assert_eq!(py_part.len(), n, "{label}: {field}.{part_name} length");
            for step in 0..n {
                assert_jos3_close(
                    label,
                    &format!("{field}.{part_name}"),
                    step,
                    rust_series[step][part_idx],
                    py_part[step],
                );
            }
        }
    }

    // -- 2-wide (head, pelvis) fields --
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
        assert_eq!(rust_series.len(), n, "{label}: {field} rust length");
        let (py_head, py_pelvis) = jos3_head_pelvis_series(&py, field);
        assert_eq!(py_head.len(), n, "{label}: {field}.head length");
        assert_eq!(py_pelvis.len(), n, "{label}: {field}.pelvis length");
        for step in 0..n {
            assert_jos3_close(
                label,
                &format!("{field}.head"),
                step,
                rust_series[step][0],
                py_head[step],
            );
            assert_jos3_close(
                label,
                &format!("{field}.pelvis"),
                step,
                rust_series[step][1],
                py_pelvis[step],
            );
        }
    }

    // -- t_superficial_vein (12-wide, positional -- see jos3_superficial_vein_series) --
    assert_eq!(
        rust.t_superficial_vein.len(),
        n,
        "{label}: t_superficial_vein rust length"
    );
    let py_sfvein = jos3_superficial_vein_series(&py);
    for (k, py_k) in py_sfvein.iter().enumerate() {
        assert_eq!(py_k.len(), n, "{label}: t_superficial_vein[{k}] length");
        for (step, &p) in py_k.iter().enumerate() {
            assert_jos3_close(
                label,
                &format!("t_superficial_vein[{k}]"),
                step,
                rust.t_superficial_vein[step][k],
                p,
            );
        }
    }
}

/// Constructor variation: all 3 `bmr_equation` values, all 4 `bsa_equation` values, both
/// sexes, and 4 distinct height/weight/age/fat/ci combinations, cycled together so every
/// (bmr, bsa, sex) triple in the 3x4x2 = 24-way cross-product is checked against a
/// different physique rather than all against the same one. Each combination also takes
/// 3 steps at its own reference environment (`Jos3Model::conditions()`, i.e. whatever
/// `reset_setpoint` left active) so `run_step` and the 85x85 matrix solve are exercised
/// for every combination, not just the constructor's own steady-state search.
#[test]
fn test_jos3_constructor_variation_comparison() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        struct BodySpec {
            height_m: f64,
            weight_kg: f64,
            age: i32,
            fat_percent: f64,
            ci: f64,
        }
        let specs = [
            BodySpec {
                height_m: 1.72,
                weight_kg: 74.43,
                age: 20,
                fat_percent: 15.0,
                ci: 2.59,
            }, // pythermalcomfort's own defaults
            BodySpec {
                height_m: 1.55,
                weight_kg: 55.0,
                age: 65,
                fat_percent: 28.0,
                ci: 3.2,
            },
            BodySpec {
                height_m: 1.90,
                weight_kg: 95.0,
                age: 35,
                fat_percent: 12.0,
                ci: 2.1,
            },
            BodySpec {
                height_m: 1.60,
                weight_kg: 48.0,
                age: 8,
                fat_percent: 20.0,
                ci: 2.8,
            },
        ];

        let bmr_equations = [
            (BmrEquation::HarrisBenedict, "harris-benedict"),
            (
                BmrEquation::HarrisBenedictOriginal,
                "harris-benedict_origin",
            ),
            (BmrEquation::Japanese, "japanese"),
        ];
        let bsa_equations = [
            (BsaFormula::DuBois, "dubois"),
            (BsaFormula::Takahira, "takahira"),
            (BsaFormula::Fujimoto, "fujimoto"),
            (BsaFormula::Kurazumi, "kurazumi"),
        ];
        let sexes = [(Sex::Male, "male"), (Sex::Female, "female")];

        let mut combo = 0usize;
        for &(bmr, bmr_str) in &bmr_equations {
            for &(bsa, bsa_str) in &bsa_equations {
                for &(sex, sex_str) in &sexes {
                    let spec = &specs[combo % specs.len()];
                    let label = format!("constructor[{bmr_str}/{bsa_str}/{sex_str}/#{combo}]");
                    combo += 1;

                    let kwargs = [
                        (
                            "height",
                            spec.height_m.into_pyobject(py).unwrap().into_any(),
                        ),
                        (
                            "weight",
                            spec.weight_kg.into_pyobject(py).unwrap().into_any(),
                        ),
                        (
                            "fat",
                            spec.fat_percent.into_pyobject(py).unwrap().into_any(),
                        ),
                        ("age", spec.age.into_pyobject(py).unwrap().into_any()),
                        ("sex", sex_str.into_pyobject(py).unwrap().into_any()),
                        ("ci", spec.ci.into_pyobject(py).unwrap().into_any()),
                        (
                            "bmr_equation",
                            bmr_str.into_pyobject(py).unwrap().into_any(),
                        ),
                        (
                            "bsa_equation",
                            bsa_str.into_pyobject(py).unwrap().into_any(),
                        ),
                    ]
                    .into_py_dict(py)
                    .unwrap();

                    let py_model = models
                        .getattr("JOS3")
                        .unwrap()
                        .call((), Some(&kwargs))
                        .unwrap_or_else(|e| panic!("{label}: JOS3() raised: {e}"));

                    let mut rust_model = Jos3Builder::new()
                        .height(Length::from_meters(spec.height_m))
                        .weight(Mass::from_kilograms(spec.weight_kg))
                        .age(spec.age)
                        .fat(BodyFat::new(spec.fat_percent).expect("in range"))
                        .sex(sex)
                        .cardiac_index(CardiacIndex::from_liters_per_minute_per_square_meter(
                            spec.ci,
                        ))
                        .bmr_equation(bmr)
                        .bsa_equation(bsa)
                        .build()
                        .unwrap_or_else(|e| panic!("{label}: Jos3Builder::build failed: {e}"));

                    py_model
                        .getattr("simulate")
                        .unwrap()
                        .call1((3, 60))
                        .unwrap_or_else(|e| panic!("{label}: JOS3.simulate raised: {e}"));

                    let conditions = rust_model.conditions();
                    rust_model
                        .advance(&conditions, 3, Duration::from_secs(60))
                        .unwrap_or_else(|e| panic!("{label}: advance failed: {e}"));

                    assert_jos3_matches_python(&py_model, rust_model.results(), &label);
                }
            }
        }
    });
}

/// Posture variation: every posture string upstream's setter actually recognizes
/// (`models/jos3.py:1471-1491`) -- `standing`; `sitting`/`sedentary`; `lying`/`supine`.
/// `reclining`/`crouching` are not among them, and upstream's setter silently leaves the
/// previous posture in place for an unrecognized string rather than raising; `Jos3Posture`
/// has no variant for either, so there is nothing to sweep.
///
/// All five strings are driven against three Rust variants on purpose. `Jos3Posture`
/// collapses `sedentary` into `Sitting` and `supine` into `Lying` on the claim that
/// upstream treats them as aliases; driving Python with the alias string and Rust with
/// the collapsed variant is what keeps that claim under test.
#[test]
fn test_jos3_posture_variants_comparison() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        let postures = [
            (Jos3Posture::Standing, "standing"),
            (Jos3Posture::Sitting, "sitting"),
            (Jos3Posture::Sitting, "sedentary"),
            (Jos3Posture::Lying, "lying"),
            (Jos3Posture::Lying, "supine"),
        ];

        for (posture, posture_str) in postures {
            let label = format!("posture[{posture_str}]");

            let py_model = models
                .getattr("JOS3")
                .unwrap()
                .call0()
                .unwrap_or_else(|e| panic!("{label}: JOS3() raised: {e}"));
            py_model.setattr("posture", posture_str).unwrap();
            py_model.setattr("tdb", 24.0).unwrap();
            py_model.setattr("tr", 24.0).unwrap();
            py_model.setattr("rh", 55.0).unwrap();
            py_model.setattr("v", 0.2).unwrap();
            py_model.setattr("clo", 0.6).unwrap();
            py_model
                .getattr("simulate")
                .unwrap()
                .call1((5, 60))
                .unwrap_or_else(|e| panic!("{label}: JOS3.simulate raised: {e}"));

            let mut rust_model = Jos3Builder::new()
                .build()
                .unwrap_or_else(|e| panic!("{label}: build failed: {e}"));
            let mut conditions = rust_model.conditions();
            conditions.tdb = PerBodyPart::Uniform(24.0);
            conditions.tr = PerBodyPart::Uniform(24.0);
            conditions.rh = PerBodyPart::Uniform(55.0);
            conditions.v = PerBodyPart::Uniform(0.2);
            conditions.clo = PerBodyPart::Uniform(0.6);
            conditions.posture = posture;
            rust_model
                .advance(&conditions, 5, Duration::from_secs(60))
                .unwrap_or_else(|e| panic!("{label}: advance failed: {e}"));

            assert_jos3_matches_python(&py_model, rust_model.results(), &label);
        }
    });
}

/// Multi-phase run: two subjects, each advanced through three phases that change the
/// environment between calls -- exactly the scenario the builder+advance API exists for
/// (see the `jos3` module docs). Phase 1 is uniform conditions at the default dtime;
/// phase 2 switches to per-body-part-by-segment conditions, a posture change, a `par`
/// change, and a non-default (fractional) dtime; phase 3 switches to per-body-part-by-name
/// conditions, another posture change, and a `to` override (which takes over both `tdb`
/// and `tr`, mirroring Python's `to` setter). Every phase builds its conditions from
/// `Jos3Model::conditions()` (the model's current environment) and overrides only what
/// changes, exactly mirroring how Python's per-property setters leave everything else
/// untouched between `simulate` calls.
#[test]
fn test_jos3_multiphase_run_comparison() {
    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort.models");

        struct Subject {
            label: &'static str,
            height_m: f64,
            weight_kg: f64,
            age: i32,
            fat_percent: f64,
            sex: Sex,
            py_sex: &'static str,
        }
        let subjects = [
            Subject {
                label: "female/25/1.65",
                height_m: 1.65,
                weight_kg: 58.0,
                age: 25,
                fat_percent: 22.0,
                sex: Sex::Female,
                py_sex: "female",
            },
            Subject {
                label: "male/50/1.80",
                height_m: 1.80,
                weight_kg: 82.0,
                age: 50,
                fat_percent: 18.0,
                sex: Sex::Male,
                py_sex: "male",
            },
        ];

        for subject in subjects {
            let kwargs = [
                (
                    "height",
                    subject.height_m.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "weight",
                    subject.weight_kg.into_pyobject(py).unwrap().into_any(),
                ),
                (
                    "fat",
                    subject.fat_percent.into_pyobject(py).unwrap().into_any(),
                ),
                ("age", subject.age.into_pyobject(py).unwrap().into_any()),
                ("sex", subject.py_sex.into_pyobject(py).unwrap().into_any()),
            ]
            .into_py_dict(py)
            .unwrap();
            let py_model = models
                .getattr("JOS3")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap_or_else(|e| panic!("{}: JOS3() raised: {e}", subject.label));

            let mut rust_model = Jos3Builder::new()
                .height(Length::from_meters(subject.height_m))
                .weight(Mass::from_kilograms(subject.weight_kg))
                .age(subject.age)
                .fat(BodyFat::new(subject.fat_percent).expect("in range"))
                .sex(subject.sex)
                .build()
                .unwrap_or_else(|e| panic!("{}: build failed: {e}", subject.label));

            // -- Phase 1: uniform conditions, standing, default dtime. --
            py_model.setattr("tdb", 26.0).unwrap();
            py_model.setattr("tr", 26.0).unwrap();
            py_model.setattr("rh", 50.0).unwrap();
            py_model.setattr("v", 0.1).unwrap();
            py_model.setattr("clo", 0.5).unwrap();
            py_model.setattr("par", 1.2).unwrap();
            py_model
                .getattr("simulate")
                .unwrap()
                .call1((5, 60))
                .unwrap_or_else(|e| panic!("{}: phase 1 simulate raised: {e}", subject.label));

            let mut conditions = rust_model.conditions();
            conditions.tdb = PerBodyPart::Uniform(26.0);
            conditions.tr = PerBodyPart::Uniform(26.0);
            conditions.rh = PerBodyPart::Uniform(50.0);
            conditions.v = PerBodyPart::Uniform(0.1);
            conditions.clo = PerBodyPart::Uniform(0.5);
            conditions.par = ActivityRatio::from_ratio(1.2);
            rust_model
                .advance(&conditions, 5, Duration::from_secs(60))
                .unwrap_or_else(|e| panic!("{}: phase 1 advance failed: {e}", subject.label));

            // -- Phase 2: per-body-part-by-segment conditions, sitting, a `par` change,
            // and a non-default, fractional dtime (37.5s). --
            let tdb2: [f64; 17] = std::array::from_fn(|i| 20.0 + 0.4 * i as f64);
            let tr2: [f64; 17] = std::array::from_fn(|i| 21.0 + 0.3 * i as f64);
            let rh2: [f64; 17] = std::array::from_fn(|i| 35.0 + 1.5 * i as f64);
            let v2: [f64; 17] = std::array::from_fn(|i| 0.05 + 0.02 * i as f64);
            let clo2: [f64; 17] = std::array::from_fn(|i| 0.2 + 0.03 * i as f64);

            py_model.setattr("tdb", tdb2.to_vec()).unwrap();
            py_model.setattr("tr", tr2.to_vec()).unwrap();
            py_model.setattr("rh", rh2.to_vec()).unwrap();
            py_model.setattr("v", v2.to_vec()).unwrap();
            py_model.setattr("clo", clo2.to_vec()).unwrap();
            py_model.setattr("par", 1.5).unwrap();
            py_model.setattr("posture", "sitting").unwrap();
            py_model
                .getattr("simulate")
                .unwrap()
                .call1((4, 37.5))
                .unwrap_or_else(|e| panic!("{}: phase 2 simulate raised: {e}", subject.label));

            let mut conditions = rust_model.conditions();
            conditions.tdb = PerBodyPart::BySegment(tdb2);
            conditions.tr = PerBodyPart::BySegment(tr2);
            conditions.rh = PerBodyPart::BySegment(rh2);
            conditions.v = PerBodyPart::BySegment(v2);
            conditions.clo = PerBodyPart::BySegment(clo2);
            conditions.par = ActivityRatio::from_ratio(1.5);
            conditions.posture = Jos3Posture::Sitting;
            rust_model
                .advance(&conditions, 4, Duration::from_secs_f64(37.5))
                .unwrap_or_else(|e| panic!("{}: phase 2 advance failed: {e}", subject.label));

            // -- Phase 3: per-body-part-by-name conditions via a `to` override (which
            // takes over both `tdb` and `tr` -- Python's `to` setter), lying. --
            let by_name: Vec<(&str, f64)> = JOS3_BODY_NAMES
                .iter()
                .enumerate()
                .map(|(i, name)| (*name, 30.0 - 0.2 * i as f64))
                .collect();
            let rh3: [f64; 17] = std::array::from_fn(|i| 60.0 - 0.5 * i as f64);

            let py_to_dict = by_name.clone().into_py_dict(py).unwrap();
            py_model.setattr("to", py_to_dict).unwrap();
            py_model.setattr("rh", rh3.to_vec()).unwrap();
            py_model.setattr("v", 0.15).unwrap();
            py_model.setattr("clo", 0.8).unwrap();
            py_model.setattr("posture", "lying").unwrap();
            py_model
                .getattr("simulate")
                .unwrap()
                .call1((3, 90))
                .unwrap_or_else(|e| panic!("{}: phase 3 simulate raised: {e}", subject.label));

            let mut conditions = rust_model.conditions();
            conditions.rh = PerBodyPart::BySegment(rh3);
            conditions.v = PerBodyPart::Uniform(0.15);
            conditions.clo = PerBodyPart::Uniform(0.8);
            conditions.posture = Jos3Posture::Lying;
            conditions.to = Some(PerBodyPart::ByName(&by_name));
            rust_model
                .advance(&conditions, 3, Duration::from_secs(90))
                .unwrap_or_else(|e| panic!("{}: phase 3 advance failed: {e}", subject.label));

            assert_jos3_matches_python(&py_model, rust_model.results(), subject.label);
        }
    });
}
