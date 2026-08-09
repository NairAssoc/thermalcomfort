# Randomised Differential Parity Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-picked parity spot-checks with a seeded randomised differential sweep that compares every Rust model against pythermalcomfort across the full input *and* optional-parameter space, so divergences are found by the machine rather than by luck.

**Architecture:** A new integration test target (`tests/differential_sweep.rs`) drives every model through N pseudo-random input vectors drawn from per-model domains. Each vector is evaluated in Rust and in Python via pyo3 in the same process, and all output fields are compared field-by-field with per-field tolerances. A deterministic seed makes failures reproducible; a shrinking pass narrows a failing vector to a minimal case for the bug report. Optional parameters (`wme`, `p_atm`, `posture`, `max_skin_blood_flow`, …) are part of the sampled space, not pinned at defaults.

**Tech Stack:** Rust 2024 (`rust-version = 1.85`), pyo3 0.23 (`auto-initialize`), approx 0.5, pythermalcomfort 4.4.0 in `.parity-venv`. No new runtime dependencies — the sweep uses a hand-rolled PRNG so the crate gains no dev-dependency on `rand`/`proptest`.

## Global Constraints

- The crate is `#![no_std]` by default; `std` is an opt-in feature. **Nothing in `src/` may gain a `std` dependency.** All sweep code lives in `tests/`, which is always `std`.
- Both configurations must pass: `cargo test --release` AND `cargo test --release --features std`. Use `make test`, which runs both.
- The reference Python is `.parity-venv` at the version in `Cargo.toml`. Never install into the user's `~/.local` site-packages. Set up with `make setup-parity`.
- Every Python-touching test **must** obtain modules via `import_reference(py, "…")` (defined in `tests/python_comparison.rs`) or an equivalent guarded helper, never `PyModule::import` directly. The reference-version guard rides on that call.
- `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings` must stay clean. `make lint` runs both plus `scripts/check_parity_coverage.py`.
- `scripts/check_parity_coverage.py` `KNOWN_GAPS` is **empty** and must stay empty. Adding an entry is a regression.
- Math in `src/` uses `libm::*` (no `f64::sin` etc.), because `no_std` lacks them. Test code may use `f64` methods freely.
- Commit after each task. Message style: imperative summary line, then a body explaining *why*, wrapped at ~76 columns.

## Background: why this exists

Coverage today is "every function is called at least once", which the checker enforces. It is **not** "every function is verified". Measured on 2026-08-09:

- 39/39 Python models and 22/25 utilities are called by a parity test.
- **103 optional parameters across the 39 models; 49 are never varied from their default.** Concentrated in `wme` (13 models), `p_atm` (6), `position`/`posture` (6), `calculate_ce` (2), `max_skin_blood_flow`/`max_sweating` (2 each), `work_intensity` (2).
- Median parity test uses **5 hand-picked input cases**; several use one.

Three genuine bugs (`transpose_sharp_altitude`, `use_fans_heatwaves` double-rounding, `running_mean_outdoor_temperature` missing rounding) were found on 2026-08-09 by hand-picked cases alone. The residue is what this sweep is for.

## File Structure

| File | Responsibility |
|---|---|
| `tests/support/mod.rs` (new) | Shared test scaffolding: guarded Python import, seeded PRNG, tolerance types, field-comparison helper, shrink driver. No model knowledge. |
| `tests/support/rng.rs` (new) | Deterministic SplitMix64 PRNG + uniform/choice sampling. Pure, unit-testable, no pyo3. |
| `tests/support/domain.rs` (new) | `Domain` describing one model's samplable inputs and optional parameters, plus `Sample` (one drawn input vector). No model knowledge. |
| `tests/support/compare.rs` (new) | `FieldCmp` (name, tolerance, NaN policy) and the routine that compares a Rust field set against a Python result object. |
| `tests/differential_sweep.rs` (new) | One `#[test]` per model. Each declares its `Domain`, its Rust adapter closure, its Python call, and its `FieldCmp` list. This is the only file with model knowledge. |
| `tests/python_comparison.rs` (modify) | Unchanged in behaviour; `import_reference`/`assert_reference_version` move to `tests/support/mod.rs` and are re-used, so the guard has exactly one implementation. |
| `scripts/check_parity_coverage.py` (modify) | Also scan `tests/differential_sweep.rs` for call-shaped evidence, so sweep-only coverage counts. |
| `Makefile` (modify) | `SWEEP_N` knob and a `make sweep` target for long runs. |
| `README.md` (modify) | Document the sweep, the seed, and how to reproduce a failure. |

## Ordering and rationale

Tasks 1–4 build the harness bottom-up with no model knowledge, so each is independently testable without Python. Task 5 proves the harness end-to-end on the simplest model. Tasks 6–10 add models in increasing complexity. Task 11 wires CI. Task 12 is the newtype API audit, which is independent of the sweep and can be done in any order after Task 1.

---

### Task 1: Deterministic PRNG

**Files:**
- Create: `tests/support/rng.rs`
- Create: `tests/support/mod.rs`
- Create: `tests/harness_selftest.rs`

**Interfaces:**
- Produces: `pub struct Rng { … }` with `Rng::new(seed: u64) -> Rng`, `rng.next_u64() -> u64`, `rng.uniform(lo: f64, hi: f64) -> f64`, `rng.choice<'a, T>(&mut self, items: &'a [T]) -> &'a T`, `rng.bool() -> bool`.

Why hand-rolled: adding `rand` as a dev-dependency pulls a tree into a crate that deliberately has almost none, and the sweep needs *reproducibility across machines and versions*, which a pinned algorithm gives and `rand`'s unspecified defaults do not.

- [ ] **Step 1: Write the failing test**

Create `tests/harness_selftest.rs`:

```rust
mod support;

use support::rng::Rng;

#[test]
fn rng_is_deterministic_for_a_seed() {
    let a: Vec<u64> = (0..5).map(|_| Rng::new(42).next_u64()).collect();
    // Same seed, fresh instance each time -> identical first value
    assert!(a.windows(2).all(|w| w[0] == w[1]));

    let mut r1 = Rng::new(7);
    let mut r2 = Rng::new(7);
    let s1: Vec<u64> = (0..100).map(|_| r1.next_u64()).collect();
    let s2: Vec<u64> = (0..100).map(|_| r2.next_u64()).collect();
    assert_eq!(s1, s2, "same seed must produce the same stream");

    let mut r3 = Rng::new(8);
    let s3: Vec<u64> = (0..100).map(|_| r3.next_u64()).collect();
    assert_ne!(s1, s3, "different seeds must diverge");
}

#[test]
fn uniform_stays_in_range_and_covers_it() {
    let mut r = Rng::new(1);
    let mut lo_seen = false;
    let mut hi_seen = false;
    for _ in 0..10_000 {
        let x = r.uniform(-5.0, 5.0);
        assert!((-5.0..=5.0).contains(&x), "out of range: {x}");
        if x < -4.0 {
            lo_seen = true;
        }
        if x > 4.0 {
            hi_seen = true;
        }
    }
    assert!(lo_seen && hi_seen, "uniform should cover the whole interval");
}

#[test]
fn choice_returns_a_member_and_varies() {
    let items = [1_u32, 2, 3, 4];
    let mut r = Rng::new(3);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..500 {
        let c = *r.choice(&items);
        assert!(items.contains(&c));
        seen.insert(c);
    }
    assert_eq!(seen.len(), 4, "all variants should eventually be chosen");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features std --test harness_selftest`
Expected: FAIL — `file not found for module 'support'` (the module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `tests/support/mod.rs`:

```rust
//! Shared scaffolding for the differential parity sweep.
//!
//! Lives under `tests/` because it is `std`-only; the crate itself is `no_std`.
#![allow(dead_code)]

pub mod rng;
```

Create `tests/support/rng.rs`:

```rust
//! Deterministic PRNG for the differential sweep.
//!
//! SplitMix64: tiny, well-distributed, and — critically — a *fixed, specified*
//! algorithm. A failing sweep must be reproducible from its seed alone, on any
//! machine and any future toolchain, which rules out `rand`'s unspecified defaults.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1). 53 bits of mantissa, matching f64 precision.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Uniform in [lo, hi].
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }

    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        assert!(!items.is_empty(), "choice() needs a non-empty slice");
        let i = (self.next_u64() % items.len() as u64) as usize;
        &items[i]
    }

    pub fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features std --test harness_selftest`
Expected: PASS — `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/support/mod.rs tests/support/rng.rs tests/harness_selftest.rs
git commit -m "Add deterministic PRNG for the differential parity sweep"
```

---

### Task 2: Field comparison with per-field tolerance and NaN policy

**Files:**
- Create: `tests/support/compare.rs`
- Modify: `tests/support/mod.rs`
- Modify: `tests/harness_selftest.rs`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `pub enum NanPolicy { MustMatch, RustMayBeNan }`
  - `pub struct FieldCmp { pub name: &'static str, pub tol: f64, pub nan: NanPolicy }` with `FieldCmp::new(name, tol)` and `.nan(policy)`.
  - `pub fn compare_field(field: &FieldCmp, rust: f64, py: f64) -> Result<(), String>`

`NanPolicy::MustMatch` is the default and the strict one: NaN on one side and a number on the other is a failure. `RustMayBeNan` exists only for the documented `no_std` PET deviation and must not be used to paper over anything else.

- [ ] **Step 1: Write the failing test**

Append to `tests/harness_selftest.rs`:

```rust
use support::compare::{compare_field, FieldCmp, NanPolicy};

#[test]
fn compare_field_accepts_within_tolerance() {
    let f = FieldCmp::new("t_core", 0.05);
    assert!(compare_field(&f, 37.10, 37.14).is_ok());
    assert!(compare_field(&f, 37.10, 37.20).is_err());
}

#[test]
fn compare_field_treats_matching_nan_as_equal() {
    let f = FieldCmp::new("pmv", 0.01);
    assert!(compare_field(&f, f64::NAN, f64::NAN).is_ok());
}

#[test]
fn compare_field_rejects_one_sided_nan() {
    let f = FieldCmp::new("pmv", 0.01);
    let err = compare_field(&f, f64::NAN, 0.5).unwrap_err();
    assert!(err.contains("pmv"), "message must name the field: {err}");
    assert!(compare_field(&f, 0.5, f64::NAN).is_err());
}

#[test]
fn rust_may_be_nan_policy_allows_only_that_direction() {
    let f = FieldCmp::new("pet", 0.1).nan(NanPolicy::RustMayBeNan);
    assert!(compare_field(&f, f64::NAN, 25.0).is_ok());
    // Python NaN with a Rust number is still a failure
    assert!(compare_field(&f, 25.0, f64::NAN).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features std --test harness_selftest`
Expected: FAIL — `unresolved import 'support::compare'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/support/compare.rs`:

```rust
//! Field-by-field comparison of a Rust result against a Python one.

/// How a NaN on one side should be treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanPolicy {
    /// NaN on one side and a number on the other is a failure. The default.
    MustMatch,
    /// Rust may return NaN where Python returns a number. Reserved for the
    /// documented `no_std` PET solver deviation — do not use it to silence
    /// anything else, because it hides exactly the class of bug we are hunting.
    RustMayBeNan,
}

#[derive(Debug, Clone)]
pub struct FieldCmp {
    pub name: &'static str,
    pub tol: f64,
    pub nan: NanPolicy,
}

impl FieldCmp {
    pub fn new(name: &'static str, tol: f64) -> Self {
        Self {
            name,
            tol,
            nan: NanPolicy::MustMatch,
        }
    }

    pub fn nan(mut self, policy: NanPolicy) -> Self {
        self.nan = policy;
        self
    }
}

/// Compare one field. Returns a human-readable error naming the field and both
/// values, so a sweep failure is actionable without re-running under a debugger.
pub fn compare_field(field: &FieldCmp, rust: f64, py: f64) -> Result<(), String> {
    match (rust.is_nan(), py.is_nan()) {
        (true, true) => Ok(()),
        (true, false) => {
            if field.nan == NanPolicy::RustMayBeNan {
                Ok(())
            } else {
                Err(format!("{}: Rust NaN, Python {py}", field.name))
            }
        }
        (false, true) => Err(format!("{}: Rust {rust}, Python NaN", field.name)),
        (false, false) => {
            let delta = (rust - py).abs();
            if delta <= field.tol {
                Ok(())
            } else {
                Err(format!(
                    "{}: Rust {rust}, Python {py} (delta {delta:.6}, tol {})",
                    field.name, field.tol
                ))
            }
        }
    }
}
```

Modify `tests/support/mod.rs` — add the module:

```rust
pub mod compare;
pub mod rng;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features std --test harness_selftest`
Expected: PASS — `test result: ok. 7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/support/compare.rs tests/support/mod.rs tests/harness_selftest.rs
git commit -m "Add per-field tolerance and NaN policy for the sweep"
```

---

### Task 3: Sampling domains

**Files:**
- Create: `tests/support/domain.rs`
- Modify: `tests/support/mod.rs`
- Modify: `tests/harness_selftest.rs`

**Interfaces:**
- Consumes: `support::rng::Rng` from Task 1.
- Produces:
  - `pub enum Axis { Real { name: &'static str, lo: f64, hi: f64 }, Enum { name: &'static str, n: usize }, Flag { name: &'static str } }`
  - `pub struct Domain { pub axes: Vec<Axis> }` with `Domain::new()`, `.real(name, lo, hi)`, `.enumerated(name, n)`, `.flag(name)`
  - `pub struct Sample { … }` with `sample.real(name) -> f64`, `sample.index(name) -> usize`, `sample.flag(name) -> bool`, `sample.describe() -> String`
  - `pub fn draw(domain: &Domain, rng: &mut Rng) -> Sample`

`Sample::describe()` renders the vector as a reproducible one-liner, which is what a failure message prints.

- [ ] **Step 1: Write the failing test**

Append to `tests/harness_selftest.rs`:

```rust
use support::domain::{draw, Domain};

#[test]
fn draw_respects_axis_bounds_and_is_reproducible() {
    let domain = Domain::new()
        .real("tdb", 10.0, 40.0)
        .real("rh", 0.0, 100.0)
        .enumerated("posture", 3)
        .flag("limit_inputs");

    let mut r1 = Rng::new(99);
    let mut r2 = Rng::new(99);
    for _ in 0..200 {
        let s1 = draw(&domain, &mut r1);
        let s2 = draw(&domain, &mut r2);

        assert!((10.0..=40.0).contains(&s1.real("tdb")));
        assert!((0.0..=100.0).contains(&s1.real("rh")));
        assert!(s1.index("posture") < 3);

        assert_eq!(s1.real("tdb"), s2.real("tdb"), "same seed -> same sample");
        assert_eq!(s1.index("posture"), s2.index("posture"));
        assert_eq!(s1.flag("limit_inputs"), s2.flag("limit_inputs"));
    }
}

#[test]
fn describe_names_every_axis() {
    let domain = Domain::new().real("tdb", 20.0, 21.0).flag("round_output");
    let mut r = Rng::new(5);
    let s = draw(&domain, &mut r);
    let text = s.describe();
    assert!(text.contains("tdb="), "missing tdb: {text}");
    assert!(text.contains("round_output="), "missing flag: {text}");
}

#[test]
#[should_panic(expected = "unknown axis")]
fn unknown_axis_panics_rather_than_returning_zero() {
    let domain = Domain::new().real("tdb", 20.0, 21.0);
    let mut r = Rng::new(5);
    draw(&domain, &mut r).real("nope");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features std --test harness_selftest`
Expected: FAIL — `unresolved import 'support::domain'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/support/domain.rs`:

```rust
//! Description of one model's samplable input space.
//!
//! A `Domain` is deliberately dumb — it knows names and ranges, not models. The
//! mapping from a drawn `Sample` to a Rust call and a Python call lives with the
//! model, in `tests/differential_sweep.rs`.

use crate::support::rng::Rng;

#[derive(Debug, Clone)]
pub enum Axis {
    Real {
        name: &'static str,
        lo: f64,
        hi: f64,
    },
    Enum {
        name: &'static str,
        n: usize,
    },
    Flag {
        name: &'static str,
    },
}

#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub axes: Vec<Axis>,
}

impl Domain {
    pub fn new() -> Self {
        Self { axes: Vec::new() }
    }

    pub fn real(mut self, name: &'static str, lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "axis {name}: lo {lo} > hi {hi}");
        self.axes.push(Axis::Real { name, lo, hi });
        self
    }

    pub fn enumerated(mut self, name: &'static str, n: usize) -> Self {
        assert!(n > 0, "axis {name}: needs at least one variant");
        self.axes.push(Axis::Enum { name, n });
        self
    }

    pub fn flag(mut self, name: &'static str) -> Self {
        self.axes.push(Axis::Flag { name });
        self
    }
}

#[derive(Debug, Clone)]
enum Value {
    Real(f64),
    Index(usize),
    Flag(bool),
}

#[derive(Debug, Clone)]
pub struct Sample {
    values: Vec<(&'static str, Value)>,
}

impl Sample {
    fn get(&self, name: &str) -> &Value {
        self.values
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("unknown axis {name:?}"))
    }

    pub fn real(&self, name: &str) -> f64 {
        match self.get(name) {
            Value::Real(x) => *x,
            other => panic!("axis {name:?} is not real: {other:?}"),
        }
    }

    pub fn index(&self, name: &str) -> usize {
        match self.get(name) {
            Value::Index(i) => *i,
            other => panic!("axis {name:?} is not enumerated: {other:?}"),
        }
    }

    pub fn flag(&self, name: &str) -> bool {
        match self.get(name) {
            Value::Flag(b) => *b,
            other => panic!("axis {name:?} is not a flag: {other:?}"),
        }
    }

    /// One-line rendering used in failure messages, so a failing vector can be
    /// pasted straight into a reproduction.
    pub fn describe(&self) -> String {
        self.values
            .iter()
            .map(|(n, v)| match v {
                Value::Real(x) => format!("{n}={x:.6}"),
                Value::Index(i) => format!("{n}=#{i}"),
                Value::Flag(b) => format!("{n}={b}"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Replace one real axis, used by the shrink pass in Task 4.
    pub fn with_real(&self, name: &str, value: f64) -> Sample {
        let mut out = self.clone();
        for (n, v) in out.values.iter_mut() {
            if *n == name {
                *v = Value::Real(value);
            }
        }
        out
    }

    pub fn real_axes(&self) -> Vec<(&'static str, f64)> {
        self.values
            .iter()
            .filter_map(|(n, v)| match v {
                Value::Real(x) => Some((*n, *x)),
                _ => None,
            })
            .collect()
    }
}

pub fn draw(domain: &Domain, rng: &mut Rng) -> Sample {
    let values = domain
        .axes
        .iter()
        .map(|axis| match axis {
            Axis::Real { name, lo, hi } => (*name, Value::Real(rng.uniform(*lo, *hi))),
            Axis::Enum { name, n } => {
                (*name, Value::Index((rng.next_u64() % *n as u64) as usize))
            }
            Axis::Flag { name } => (*name, Value::Flag(rng.bool())),
        })
        .collect();
    Sample { values }
}
```

Modify `tests/support/mod.rs`:

```rust
pub mod compare;
pub mod domain;
pub mod rng;
```

Note: `domain.rs` refers to `crate::support::rng::Rng`. In an integration test the crate root is the test file, so `mod support;` at the top of each test target makes `crate::support::…` resolve. Keep that `mod support;` line in every test target that uses it.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --features std --test harness_selftest`
Expected: PASS — `test result: ok. 10 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/support/domain.rs tests/support/mod.rs tests/harness_selftest.rs
git commit -m "Add sampling domains for the differential sweep"
```

---

### Task 4: Sweep driver with shrinking, and the shared version guard

**Files:**
- Create: `tests/support/sweep.rs`
- Modify: `tests/support/mod.rs`
- Modify: `tests/python_comparison.rs` (move the guard out, import it back)
- Modify: `tests/harness_selftest.rs`

**Interfaces:**
- Consumes: `Domain`, `Sample`, `draw`, `Rng`, `FieldCmp`, `compare_field`.
- Produces:
  - `pub fn import_reference<'py>(py: Python<'py>, module: &str) -> PyResult<Bound<'py, PyModule>>`
  - `pub fn assert_reference_version(py: Python<'_>)`
  - `pub fn sweep_n() -> usize` — reads `SWEEP_N` env var, default 500.
  - `pub fn seed() -> u64` — reads `SWEEP_SEED` env var, default 0xC0FFEE.
  - `pub fn run_sweep<F>(label: &str, domain: &Domain, eval: F) where F: Fn(&Sample) -> Result<(), String>`

`run_sweep` draws `sweep_n()` samples, calls `eval`, and on the first error shrinks: for each real axis in turn, repeatedly move that axis halfway toward the midpoint of its range while the failure persists, then report the minimal failing sample. Shrinking matters because a raw 12-axis random vector is nearly useless as a bug report.

- [ ] **Step 1: Write the failing test**

Append to `tests/harness_selftest.rs`:

```rust
use support::sweep::{run_sweep, sweep_n};

#[test]
fn sweep_runs_the_configured_number_of_samples() {
    let domain = Domain::new().real("x", 0.0, 1.0);
    let count = std::cell::Cell::new(0usize);
    run_sweep("counting", &domain, |_s| {
        count.set(count.get() + 1);
        Ok(())
    });
    assert_eq!(count.get(), sweep_n());
}

#[test]
#[should_panic(expected = "shrunk")]
fn sweep_shrinks_a_failing_sample_and_panics() {
    let domain = Domain::new().real("x", -100.0, 100.0);
    // Fails for any positive x; shrinking should drive |x| down toward 0
    run_sweep("shrinking", &domain, |s| {
        if s.real("x") > 0.0 {
            Err(format!("x was positive: {}", s.real("x")))
        } else {
            Ok(())
        }
    });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --features std --test harness_selftest`
Expected: FAIL — `unresolved import 'support::sweep'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/support/sweep.rs`:

```rust
//! Sweep driver: draw, evaluate, and shrink on failure.

use crate::support::domain::{draw, Domain, Sample};
use crate::support::rng::Rng;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::atomic::{AtomicBool, Ordering};

/// Number of samples per model. Override with `SWEEP_N=5000` for a deep run.
pub fn sweep_n() -> usize {
    std::env::var("SWEEP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500)
}

/// Base seed. Override with `SWEEP_SEED=…` to reproduce a specific run.
pub fn seed() -> u64 {
    std::env::var("SWEEP_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x00C0_FFEE)
}

/// Import a pythermalcomfort module, asserting once per process that the
/// reference is the version this crate ports.
///
/// Drop-in for `PyModule::import`. Routing every import through here is what
/// makes the version check unskippable by test-name filtering.
pub fn import_reference<'py>(py: Python<'py>, module: &str) -> PyResult<Bound<'py, PyModule>> {
    // An atomic flag, not a `Once`: Python's import machinery can release the
    // GIL mid-import, and a blocking `Once` around it deadlocks when a second
    // test thread takes the released GIL and waits on the `Once` the first
    // thread needs the GIL to finish.
    static CHECKED: AtomicBool = AtomicBool::new(false);
    if !CHECKED.swap(true, Ordering::Relaxed) {
        assert_reference_version(py);
    }
    PyModule::import(py, module)
}

/// Assert the importable pythermalcomfort is the version this crate ports.
pub fn assert_reference_version(py: Python<'_>) {
    let full = env!("CARGO_PKG_VERSION");
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

/// Draw `sweep_n()` samples, evaluate each, and on the first failure shrink the
/// sample before panicking.
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
                 seed: SWEEP_SEED={} SWEEP_N={}\n\
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

/// Move each real axis toward the midpoint of its declared range while the
/// failure persists. Halving rather than binary-searching keeps this cheap; the
/// aim is a readable bug report, not a provably minimal one.
fn shrink<F>(domain: &Domain, failing: &Sample, err: &str, eval: &F) -> (Sample, String)
where
    F: Fn(&Sample) -> Result<(), String>,
{
    use crate::support::domain::Axis;

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
```

Modify `tests/support/mod.rs`:

```rust
pub mod compare;
pub mod domain;
pub mod rng;
pub mod sweep;
```

Modify `tests/python_comparison.rs`: delete the local `import_reference` and `assert_reference_version` definitions and the `use std::sync::atomic::{AtomicBool, Ordering};` line, then add near the other imports:

```rust
mod support;

use support::sweep::{assert_reference_version, import_reference};
```

Keep `test_pythermalcomfort_version_matches_crate` in `python_comparison.rs`, rewritten as:

```rust
#[test]
fn test_pythermalcomfort_version_matches_crate() {
    Python::with_gil(assert_reference_version);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --features std --test harness_selftest`
Expected: PASS — 12 passed.

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test python_comparison`
Expected: PASS — 74 passed, proving the guard still works after moving.

- [ ] **Step 5: Commit**

```bash
git add tests/support/ tests/python_comparison.rs tests/harness_selftest.rs
git commit -m "Add sweep driver with shrinking; share the version guard"
```

---

### Task 5: First model end-to-end — `pmv_ppd_iso`

**Files:**
- Create: `tests/differential_sweep.rs`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: the per-model pattern every later task copies.

`pmv_ppd_iso` is first because it has a small input set, an options struct exercising `wme`/`limit_inputs`/`round_output`, and a known applicability clamp that makes NaN handling real rather than theoretical.

- [ ] **Step 1: Write the failing test**

Create `tests/differential_sweep.rs`:

```rust
//! Randomised differential sweep against pythermalcomfort.
//!
//! Every model is driven through `SWEEP_N` pseudo-random input vectors covering
//! its optional parameters as well as its physical inputs, and every output
//! field is compared against Python. This exists because the hand-picked parity
//! tests in `python_comparison.rs` leave 49 of 103 optional parameters pinned at
//! their defaults, and three real bugs were found in that residue on 2026-08-09.
//!
//! Reproduce a failure with the seed printed in the panic:
//!   SWEEP_SEED=12345 SWEEP_N=500 cargo test --features std --test differential_sweep

mod support;

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use support::compare::{compare_field, FieldCmp};
use support::domain::{Domain, Sample};
use support::sweep::{import_reference, run_sweep};
use thermalcomfort::models::pmv::PmvPpdOptions;
use thermalcomfort::models::pmv_ppd_iso;
use thermalcomfort::{ClothingInsulation, Humidity, MetabolicRate, Speed, Temperature};

#[test]
fn sweep_pmv_ppd_iso() {
    let domain = Domain::new()
        .real("tdb", 5.0, 45.0)
        .real("tr", 5.0, 45.0)
        .real("vr", 0.0, 2.0)
        .real("rh", 0.0, 100.0)
        .real("met", 0.6, 5.0)
        .real("clo", 0.0, 2.5)
        .real("wme", 0.0, 1.0)
        .flag("limit_inputs")
        .flag("round_output");

    let fields = [
        FieldCmp::new("pmv", 0.01),
        FieldCmp::new("ppd", 0.1),
    ];

    Python::with_gil(|py| {
        let models = import_reference(py, "pythermalcomfort.models")
            .expect("failed to import pythermalcomfort.models");

        run_sweep("pmv_ppd_iso", &domain, |s: &Sample| {
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
                ("limit_inputs", limit_inputs.into_pyobject(py).unwrap().to_owned().into_any()),
                ("round_output", round_output.into_pyobject(py).unwrap().to_owned().into_any()),
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
                let py_value: f64 = py_result
                    .getattr(field.name)
                    .unwrap()
                    .extract()
                    .map_err(|e| format!("{}: could not extract: {e}", field.name))?;
                compare_field(field, rust_value, py_value)?;
            }
            Ok(())
        });
    });
}
```

- [ ] **Step 2: Run test to verify it fails or reveals a divergence**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep -- --nocapture`

Expected: either a compile error to fix, or a **divergence panic naming a field and a shrunk sample**. A divergence here is a *result*, not a setup problem — investigate it as a real bug before touching tolerances.

**If it panics with a divergence:** stop and diagnose. Compare the shrunk case in isolation against Python. Only widen a tolerance if you can show the difference is float-representation noise (≤ ~1e-9 relative); anything larger is a bug in `src/`, and the fix belongs in `src/`, not the tolerance. Record the finding in the commit message.

- [ ] **Step 3: Resolve every divergence**

For each divergence: reproduce with the printed `SWEEP_SEED`, read both implementations side by side, fix `src/` if Rust is wrong, and add a targeted case to `tests/python_comparison.rs` so the specific bug is pinned by a fast test as well as by the sweep.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep`
Expected: PASS — `test result: ok. 1 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/differential_sweep.rs src/
git commit -m "Add differential sweep for pmv_ppd_iso"
```

---

### Task 6: Remaining PMV-family models

**Files:**
- Modify: `tests/differential_sweep.rs`

**Interfaces:**
- Consumes: the Task 5 pattern.
- Produces: `sweep_pmv_ppd_ashrae`, `sweep_pmv_a`, `sweep_pmv_e`, `sweep_pmv_athb`.

Cover in this task: `pmv_ppd_ashrae` (same axes as Task 5 plus `airspeed_control` if the Rust API exposes it — it does not today, so omit and note it), `pmv_a` (extra real axis `a_coefficient`, range 0.0–1.0), `pmv_e` (extra real axis `e_coefficient`, range 0.0–1.0), `pmv_athb` (no options struct; axes tdb/tr/vr/rh/met/t_running_mean).

- [ ] **Step 1: Write the four sweeps**

Follow the Task 5 body exactly, substituting the model name, axes and fields. For each, the field list is `pmv` (tol 0.01) and `ppd` (tol 0.1); `pmv_athb` returns only `pmv`.

Check each Rust signature before writing, with:

```bash
grep -n "pub fn pmv_ppd_ashrae\|pub fn pmv_a\|pub fn pmv_e\|pub fn pmv_athb" -A12 src/models/pmv.rs
```

and each Python signature with:

```bash
.parity-venv/bin/python -c "
import inspect, pythermalcomfort.models as m
for n in ['pmv_ppd_ashrae','pmv_a','pmv_e','pmv_athb']:
    print(n, inspect.signature(getattr(m,n)))"
```

- [ ] **Step 2: Run to see divergences**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep -- --nocapture`
Expected: divergences are results — diagnose each per Task 5 Step 3.

- [ ] **Step 3: Fix every divergence in `src/`**

- [ ] **Step 4: Run to verify all pass**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/differential_sweep.rs src/
git commit -m "Add differential sweeps for the PMV family"
```

---

### Task 7: Two-node and SET family

**Files:**
- Modify: `tests/differential_sweep.rs`

**Interfaces:**
- Produces: `sweep_two_nodes_gagge`, `sweep_two_nodes_gagge_ji`, `sweep_two_nodes_gagge_sleep`, `sweep_set_tmp`, `sweep_use_fans_heatwaves`, `sweep_cooling_effect`.

This is the highest-value task: these models carry the most untested optional parameters (`wme`, `body_surface_area`, `p_atm`, `posture`, `max_skin_blood_flow`, `max_sweating`, `calculate_ce`), and this family already produced two bugs on 2026-08-09.

Axes for `two_nodes_gagge`: tdb 10–45, tr 10–45, v 0.0–4.0, rh 0–100, met 0.8–4.0, clo 0.0–2.0, wme 0.0–1.0, body_surface_area 1.5–2.2, p_atm 80000–105000, posture enumerated 2 (`Standing`, `Sitting`), max_skin_blood_flow 40–110, max_sweating 200–700, flags round_output and calculate_ce.

Compare all numeric fields the Rust result exposes: `set`, `e_skin`, `e_rsw`, `e_max`, `q_sensible`, `q_skin`, `q_res`, `t_core`, `t_skin`, `m_bl`, `m_rsw`, `w`, `w_max`, plus the `pmv_gagge`/`pmv_set`/`disc`/`t_sens` fields if present. Tolerance 0.01 for temperatures and 0.05 for heat flows when `round_output` is false; when it is true, use 0.051 because Python and Rust may land on opposite sides of a rounding boundary — but **only** after confirming the unrounded values agree, otherwise you are hiding a real difference behind rounding.

- [ ] **Step 1: Enumerate the Rust and Python field sets before writing**

```bash
grep -n "pub struct GaggeTwoNodesResult" -A40 src/models/two_nodes_gagge.rs | grep "pub "
.parity-venv/bin/python -c "
import warnings; warnings.filterwarnings('ignore')
from pythermalcomfort.models import two_nodes_gagge as g
print(list(g(25,25,0.1,50,1.2,0.5).__dataclass_fields__))"
```

- [ ] **Step 2: Write the six sweeps following the Task 5 pattern**

- [ ] **Step 3: Run and diagnose divergences**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep -- --nocapture`

- [ ] **Step 4: Fix every divergence in `src/`, then verify**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep`
Expected: PASS — 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/differential_sweep.rs src/
git commit -m "Add differential sweeps for the two-node and SET family"
```

---

### Task 8: Heat-stress and index models

**Files:**
- Modify: `tests/differential_sweep.rs`

**Interfaces:**
- Produces: `sweep_utci`, `sweep_phs`, `sweep_pet_steady`, `sweep_wbgt`, `sweep_heat_index_lu`, `sweep_heat_index_rothfusz`, `sweep_heat_index_schoen`, `sweep_humidex`, `sweep_at`, `sweep_wci`, `sweep_wind_chill_temperature`, `sweep_net`, `sweep_thi`, `sweep_discomfort_index`, `sweep_esi`.

`phs` carries `posture` (3 variants), `wme`, `i_mst`, `a_p`, `drink`, `weight`, `height`, `walk_sp`, `theta`, `acclimatized`, and the `Iso7933Model` enum — the richest option space in the crate. `pet_steady` is the one model with a documented `no_std` deviation: give its `pet` field `NanPolicy::RustMayBeNan` **only** if a divergence is confirmed to be the known extreme cold+wind case, and note it in the commit.

- [ ] **Step 1: Write the sweeps, checking each signature pair first**

- [ ] **Step 2: Run and diagnose divergences**

- [ ] **Step 3: Fix divergences in `src/`, then verify**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep`
Expected: PASS — 26 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/differential_sweep.rs src/
git commit -m "Add differential sweeps for heat-stress and index models"
```

---

### Task 9: Adaptive, specialty, work-capacity and remaining models

**Files:**
- Modify: `tests/differential_sweep.rs`

**Interfaces:**
- Produces: `sweep_adaptive_ashrae`, `sweep_adaptive_en`, `sweep_ankle_draft`, `sweep_vertical_tmp_grad_ppd`, `sweep_solar_gain`, `sweep_sports_heat_stress_risk`, `sweep_ireq`, `sweep_clo_tout`, `sweep_ridge_regression_predict_t_re_t_sk`, `sweep_work_capacity_dunne`, `sweep_work_capacity_hothaps`, `sweep_work_capacity_iso`, `sweep_work_capacity_niosh`.

`solar_gain` needs its angle axes clamped to the table domain (sol_altitude 0–90, sharp 0–180) plus deliberate excursions outside it to exercise the NaN guard: sample sol_altitude −20–110 and sharp −20–200, and expect NaN on both sides outside the valid box. `sports_heat_stress_risk` needs the `Sports` preset as an enumerated axis over all presets. `ireq` needs `p` 5–200 and `walk_sp` 0–1.5 so the applicability mask is exercised in both states.

- [ ] **Step 1: Write the sweeps**

- [ ] **Step 2: Run and diagnose divergences**

- [ ] **Step 3: Fix divergences in `src/`, then verify**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep`
Expected: PASS — 39 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/differential_sweep.rs src/
git commit -m "Add differential sweeps for adaptive, specialty and work-capacity models"
```

---

### Task 10: Utility function sweeps

**Files:**
- Modify: `tests/differential_sweep.rs`

**Interfaces:**
- Produces: `sweep_utilities` — one test covering the 22 utilities with Python counterparts.

Cover: `v_relative`, `clo_dynamic_iso`, `clo_dynamic_ashrae`, `clo_area_factor`, `clo_correction_factor_environment`, `clo_insulation_air_layer`, `clo_total_insulation`, `clo_tout`, `body_surface_area` (all four formulae as an enumerated axis), `p_sat`, `p_sat_torr`, `p_sat_antoine`, `enthalpy_air`, `hr_to_rh`, `dew_point_temperature`, `wet_bulb_temperature`, `psy_ta_rh` (all returned fields), `operative_temperature` (both standards), `mean_radiant_temperature` (both standards), `running_mean_outdoor_temperature`, `f_svv`, `transpose_sharp_altitude`.

Remember the unit conversions: Python's `antoine` returns **kPa** and `p_sat_torr` returns **torr**, while the Rust port normalises both to Pa. Multiply by 1000.0 and 133.322 respectively.

- [ ] **Step 1: Write the sweep**

- [ ] **Step 2: Run and diagnose divergences**

- [ ] **Step 3: Fix divergences in `src/`, then verify**

Run: `PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages) cargo test --features std --test differential_sweep`
Expected: PASS — 40 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/differential_sweep.rs src/
git commit -m "Add differential sweep for utility functions"
```

---

### Task 11: Wire the sweep into the build, CI and docs

**Files:**
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/check_parity_coverage.py`
- Modify: `README.md`

The sweep must run in CI at a modest `SWEEP_N` so it gates merges without making them slow, plus a nightly deep run. It must also count toward parity coverage, otherwise the checker would still demand a hand-written test for a function the sweep already covers.

- [ ] **Step 1: Add the Makefile target**

In `Makefile`, after the `test` target:

```makefile
# Deep randomised differential sweep. CI runs a short one inside `make test`;
# this is the long-form version for a release check or a bug hunt.
SWEEP_N ?= 20000
sweep:
	@echo "Running differential sweep with SWEEP_N=$(SWEEP_N)..."
	@$(PARITY_ENV) SWEEP_N=$(SWEEP_N) cargo test --release --features std --test differential_sweep -- --nocapture
```

Add `sweep` to the `.PHONY` line at the top of the file.

- [ ] **Step 2: Teach the coverage checker about the sweep**

In `scripts/check_parity_coverage.py`, replace the single-file constant:

```python
PARITY_TESTS = REPO / "tests" / "python_comparison.rs"
```

with:

```python
# Both targets count as parity coverage: hand-written cases pin specific known
# bugs, the sweep covers the space. A function verified by either is verified.
PARITY_TEST_FILES = [
    REPO / "tests" / "python_comparison.rs",
    REPO / "tests" / "differential_sweep.rs",
]
```

Then in `main()`, replace the existence check and the read with:

```python
    missing_files = [p for p in PARITY_TEST_FILES if not p.exists()]
    if missing_files:
        for p in missing_files:
            print(f"error: {p} not found", file=sys.stderr)
        return 1

    tests = "\n".join(
        executable_test_source(p.read_text()) for p in PARITY_TEST_FILES
    )
```

and update the two failure messages that name `PARITY_TESTS.relative_to(REPO)` to list both files.

- [ ] **Step 3: Add the CI job**

In `.github/workflows/ci.yml`, add a job alongside `python-comparison`:

```yaml
  differential-sweep:
    name: Differential Sweep
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install the pythermalcomfort version this crate ports
        shell: bash
        run: |
          PTC_VERSION="$(make parity-version)"
          echo "Installing pythermalcomfort==${PTC_VERSION}"
          pip install "pythermalcomfort==${PTC_VERSION}"

      # A short sweep on every push; the nightly workflow runs a deep one.
      - name: Run differential sweep
        shell: bash
        run: SWEEP_N=2000 cargo test --release --features std --test differential_sweep
```

- [ ] **Step 4: Document it in the README**

In `README.md`, under `## Testing`, after the two-guards list, add:

```markdown
### Differential sweep

Beyond the hand-written parity cases, `tests/differential_sweep.rs` drives every
model through pseudo-random input vectors — including the optional parameters
(`wme`, `p_atm`, posture, blood-flow and sweating caps) that fixed cases leave at
their defaults — and compares every output field against Python.

```bash
make sweep                      # deep run, SWEEP_N=20000
SWEEP_N=500 cargo test --features std --test differential_sweep
```

Failures print the seed and a shrunk input vector. Reproduce with:

```bash
SWEEP_SEED=<seed> SWEEP_N=<n> cargo test --features std --test differential_sweep -- --nocapture
```

Widening a tolerance to make a sweep pass is almost always wrong: the sweep
exists to find the differences that fixed cases miss.
```

- [ ] **Step 5: Verify everything**

Run: `make verify`
Expected: lint clean, coverage clean, both configurations pass.

Run: `SWEEP_N=2000 make sweep`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Makefile .github/workflows/ci.yml scripts/check_parity_coverage.py README.md
git commit -m "Wire the differential sweep into make, CI and docs"
```

---

### Task 12: Newtype audit of the public API

**Files:**
- Modify: `src/models/ireq.rs`
- Modify: `src/utilities.rs`
- Modify: `src/lib.rs`
- Modify: `tests/python_comparison.rs`
- Modify: `README.md`

This task is independent of the sweep and may be done at any point after Task 1.

**Findings from the 2026-08-09 audit.** The crate's actual convention is **newtypes on inputs, bare `f64` on outputs** — every result struct (`PetResult.pet`, `UtciResult.utci`, `PhsResult.t_re`, `AdaptiveEnResult.tmp_cmf`) uses `f64`. `IreqResult`'s `f64` fields therefore follow convention and are **not** a defect. On the input side, 18 public functions take bare `f64`. Classify before changing anything:

*Legitimately dimensionless — leave alone:* `mean_radiant_temperature.emissivity`, `running_mean_outdoor_temperature.alpha`, `pmv_a.a_coefficient`, `pmv_e.e_coefficient`, `solar_gain.{sol_transmittance,f_svv,f_bes,asw,floor_reflectance}`, `enthalpy_air.hr` and `hr_to_rh.hr` (humidity ratio is kg/kg), `valid_range`, `round_to`, `celsius_to_temp`, `ms_to_speed` (adapters), `vertical_tmp_grad_ppd.vertical_temp_gradient` (a temperature *difference*; `measurements::Temperature` is absolute, so wrapping it would be wrong).

*Genuine misses:*
1. `clo_intrinsic_insulation_ensemble(clo_garments: &[f64])` — these are clo values and `ClothingInsulation` exists. **Pre-existing.**
2. `ireq(p: f64)` — air permeability [l/(m²·s)]. Dimensional, no existing newtype. **Introduced 2026-08-09.**
3. `transpose_sharp_altitude(sharp: f64, altitude: f64)` and `solar_gain(sol_altitude: f64, sharp: f64, …)` — angles in degrees; `measurements::Angle` exists but the crate does not re-export it. **Pre-existing.**
4. `esi(sol_radiation_global: f64)` and `solar_gain(sol_radiation_dir: f64)` — irradiance [W/m²]; `measurements` has no irradiance type. **Pre-existing.**
5. `use_fans_heatwaves(max_skin_blood_flow: f64, max_sweating: f64)` — dimensional rates, no `measurements` type. **Pre-existing.**

Scope this task to items 1 and 2 only. Items 3–5 need either a new re-export (`Angle`) or new newtypes, which is a wider API break better decided separately — record them in the README rather than changing them here.

- [ ] **Step 1: Write the failing test for `clo_intrinsic_insulation_ensemble`**

In `tests/python_comparison.rs`, change the existing call site to pass newtypes. Find it with:

```bash
grep -n "clo_intrinsic_insulation_ensemble" tests/python_comparison.rs
```

Change the Rust call to:

```rust
let garments = [
    ClothingInsulation::from_clo(0.25),
    ClothingInsulation::from_clo(0.15),
    ClothingInsulation::from_clo(0.10),
];
let rust_total = clo_intrinsic_insulation_ensemble(&garments);
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --features std --test python_comparison clo_intrinsic --no-run`
Expected: FAIL — `expected &[f64], found &[ClothingInsulation]`

- [ ] **Step 3: Change the signature**

In `src/utilities.rs`, change:

```rust
pub fn clo_intrinsic_insulation_ensemble(clo_garments: &[f64]) -> f64 {
```

to:

```rust
pub fn clo_intrinsic_insulation_ensemble(clo_garments: &[ClothingInsulation]) -> f64 {
```

and inside the body convert at the point of use, e.g. `let total: f64 = clo_garments.iter().map(|c| c.as_clo()).sum();` — read the existing body first and adapt rather than assuming its shape. Update the doctest above it to build `ClothingInsulation` values.

- [ ] **Step 4: Add the `AirPermeability` newtype and use it in `ireq`**

In `src/lib.rs`, next to `ClothingInsulation`, add:

```rust
/// Air permeability of clothing.
///
/// ISO 11079 expresses this in litres per square metre per second, the rate at
/// which air passes through the fabric. It is a distinct dimension from air
/// speed, so it gets its own type rather than reusing `Speed`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AirPermeability(f64);

impl AirPermeability {
    /// Construct from litres per square metre per second [l/(m²·s)]
    pub const fn from_l_per_m2_s(value: f64) -> Self {
        Self(value)
    }

    /// Value in litres per square metre per second [l/(m²·s)]
    pub const fn as_l_per_m2_s(self) -> f64 {
        self.0
    }
}
```

Add `AirPermeability` to the `pub use` list so it is reachable as `thermalcomfort::AirPermeability`.

In `src/models/ireq.rs`, change the parameter type from `p: f64` to `p: AirPermeability`, import it, and convert once at the top of `ireq` with `let p = p.as_l_per_m2_s();` so the internal `solve_criterion` signature is unchanged. Update the doctest and the module's unit tests to construct `AirPermeability::from_l_per_m2_s(50.0)`.

Update the `ireq` call sites in `tests/python_comparison.rs` and in the README example.

- [ ] **Step 5: Record the deferred items in the README**

In `README.md`, under `### Re-exported Types`, after the `Defined in this crate` list, add:

```markdown
Physical quantities are newtypes on **inputs**; result structs return plain `f64`.
Four input parameters remain untyped because no suitable type exists yet: solar
angles (`solar_gain`, `transpose_sharp_altitude` — `measurements::Angle` is not
re-exported), irradiance in W/m² (`esi`, `solar_gain`), and the blood-flow and
sweating caps in `use_fans_heatwaves`.
```

- [ ] **Step 6: Verify**

Run: `make verify`
Expected: lint clean, both configurations pass.

- [ ] **Step 7: Commit**

```bash
git add src/ tests/ README.md
git commit -m "Use newtypes for clothing ensembles and air permeability"
```

---

## Self-review notes

- **Spec coverage.** The two asks were (a) a plan for the randomised differential sweep and (b) a newtype audit. (a) is Tasks 1–11; (b) is Task 12 with the audit findings recorded inline so they survive a context reset.
- **Deliberate omissions.** `units='IP'` is not swept: the Rust API takes typed quantities, so unit conversion is the type system's job and has no parameter to vary. `airspeed_control` has no Rust counterpart. `units_converter` and `valid_range` have no Rust counterpart worth comparing.
- **Known risk.** Tasks 5–10 each say "fix every divergence", and the number of divergences is unknown — that is the point of the exercise. If a task uncovers more than about three real bugs, split the fixes into their own commits rather than one giant one, and consider pausing to report before continuing.
- **Do not** widen a tolerance to make a sweep pass without first confirming the difference is float-representation noise. That reflex would defeat the entire plan.
