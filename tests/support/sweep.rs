//! Sweep driver: draw, evaluate, and shrink on failure.

use crate::support::domain::{Axis, Domain, Sample, draw};
use crate::support::rng::Rng;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::atomic::{AtomicBool, Ordering};

/// Samples per model. Override with `SWEEP_N=5000` for a deep run.
pub fn sweep_n() -> usize {
    std::env::var("SWEEP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500)
}

/// Report how many samples a sweep excluded, and fail if the share exceeds `ceiling`.
///
/// A sweep that skips samples has to say so, or it looks like it covered ground it
/// dropped. Printing alone does not achieve that: `cargo test` captures the output of a
/// *passing* test and discards it, so an `eprintln!` here is read only by someone who
/// already passed `--nocapture` and went looking. The assertion is what makes the claim
/// enforceable -- if a change starts pushing samples into the excluded bucket, the sweep
/// fails instead of quietly narrowing.
///
/// `ceiling` is a fraction of the samples drawn, so it means the same thing at
/// `SWEEP_N=500` and `SWEEP_N=3000`.
pub fn report_skipped(label: &str, skipped: usize, ceiling: f64, why: &str) {
    let n = sweep_n();
    let share = skipped as f64 / n as f64;
    eprintln!("{label}: skipped {skipped} of {n} sample(s) -- {why}");
    assert!(
        share <= ceiling,
        "\n{label} excluded {skipped} of {n} samples ({:.2}%), over the {:.2}% this sweep \
         is allowed to drop.\nThe exclusion is for samples where the pythermalcomfort \
         reference is not a function of its inputs at the compared resolution ({why}).\n\
         A jump here means either the port started diverging on ordinary inputs, or the \
         gate has become too permissive -- investigate rather than raising this number.\n",
        share * 100.0,
        ceiling * 100.0,
    );
}

/// Base seed. Override with `SWEEP_SEED=…` to reproduce a specific run.
pub fn seed() -> u64 {
    std::env::var("SWEEP_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x00C0_FFEE)
}

/// Import a pythermalcomfort module, asserting once per process that the reference is
/// the version this crate ports.
///
/// Drop-in for `PyModule::import`. Routing every import through here is what makes the
/// version check unskippable by test-name filtering.
pub fn import_reference<'py>(py: Python<'py>, module: &str) -> PyResult<Bound<'py, PyModule>> {
    // An atomic flag, not a `Once`: Python's import machinery can release the GIL
    // mid-import, and a blocking `Once` around it deadlocks once a second test thread
    // takes the released GIL and waits on the `Once` the first thread needs to finish.
    static CHECKED: AtomicBool = AtomicBool::new(false);
    if !CHECKED.swap(true, Ordering::Relaxed) {
        assert_reference_version(py);
    }
    PyModule::import(py, module)
}

/// Assert the importable pythermalcomfort is the version this crate ports.
pub fn assert_reference_version(py: Python<'_>) {
    let full = env!("CARGO_PKG_VERSION");
    // A pre-release suffix marks a revision of the *port*, not of upstream, and PEP 440
    // spells pre-releases differently anyway. Compare the release triple only.
    let expected = full.split('-').next().unwrap_or(full);

    let ptc = PyModule::import(py, "pythermalcomfort").unwrap_or_else(|e| {
        panic!(
            "could not import pythermalcomfort, so no parity test is verifying \
             anything.\nExpected version {expected}. Set one up with:\n  \
             make setup-parity\nUnderlying error: {e}"
        )
    });

    let actual: String = ptc
        .getattr("__version__")
        .expect("pythermalcomfort has no __version__")
        .extract()
        .expect("pythermalcomfort.__version__ is not a string");

    assert_eq!(
        actual, expected,
        "\n\npythermalcomfort version mismatch: comparing against {actual}, but this \
         crate ports {expected}. Every parity result is therefore meaningless.\n\
         Fix with: make setup-parity\n"
    );
}

/// Draw `sweep_n()` samples, evaluate each, and on the first failure shrink before
/// panicking. A raw multi-axis random vector is a poor bug report; a shrunk one is not.
pub fn run_sweep<F>(label: &str, domain: &Domain, eval: F)
where
    F: Fn(&Sample) -> Result<(), String>,
{
    let mut rng = Rng::new(seed());
    for i in 0..sweep_n() {
        let sample = draw(domain, &mut rng);
        if let Err(err) = eval(&sample) {
            let (shrunk, shrunk_err) = shrink(domain, &sample, &err, &eval);
            panic!(
                "\n{label}: divergence on sample {i} of {}\n\
                 reproduce: SWEEP_SEED={} SWEEP_N={} cargo test --features std \
                 --test differential_sweep {label}\n\
                 original: {}\n  {}\n\
                 shrunk:   {}\n  {}\n",
                sweep_n(),
                seed(),
                sweep_n(),
                sample.describe(),
                err,
                shrunk.describe(),
                shrunk_err,
            );
        }
    }
}

/// Move each real axis toward the midpoint of its declared range while the failure
/// persists. Halving rather than binary-searching keeps this cheap; the aim is a
/// readable bug report, not a provably minimal one.
fn shrink<F>(domain: &Domain, failing: &Sample, err: &str, eval: &F) -> (Sample, String)
where
    F: Fn(&Sample) -> Result<(), String>,
{
    let mut best = failing.clone();
    let mut best_err = err.to_string();

    for axis in &domain.axes {
        let Axis::Real { name, lo, hi } = axis else {
            continue;
        };
        let mid = (lo + hi) / 2.0;
        for _ in 0..12 {
            let current = best.real(name);
            let candidate_value = (current + mid) / 2.0;
            if (candidate_value - current).abs() < 1e-9 {
                break;
            }
            let candidate = best.with_real(name, candidate_value);
            match eval(&candidate) {
                Err(e) => {
                    best = candidate;
                    best_err = e;
                }
                Ok(()) => break,
            }
        }
    }

    (best, best_err)
}
