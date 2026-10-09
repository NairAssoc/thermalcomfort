//! Self-tests for the differential sweep scaffolding.
//!
//! The harness has to be trustworthy before any divergence it reports means anything.

mod support;

use support::compare::{FieldCmp, NanPolicy, compare_field};
use support::domain::{Domain, draw};
use support::rng::Rng;
use support::sweep::{run_sweep, sweep_n};

#[test]
fn rng_is_deterministic_for_a_seed() {
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
    let (mut lo_seen, mut hi_seen) = (false, false);
    for _ in 0..10_000 {
        let x = r.uniform(-5.0, 5.0);
        assert!((-5.0..=5.0).contains(&x), "out of range: {x}");
        lo_seen |= x < -4.0;
        hi_seen |= x > 4.0;
    }
    assert!(
        lo_seen && hi_seen,
        "uniform should cover the whole interval"
    );
}

#[test]
fn compare_field_respects_tolerance_and_nan() {
    let f = FieldCmp::new("t_core", 0.05);
    assert!(compare_field(&f, 37.10, 37.14).is_ok());
    assert!(compare_field(&f, 37.10, 37.20).is_err());
    assert!(compare_field(&f, f64::NAN, f64::NAN).is_ok());

    let err = compare_field(&f, f64::NAN, 0.5).unwrap_err();
    assert!(err.contains("t_core"), "message must name the field: {err}");
    assert!(compare_field(&f, 0.5, f64::NAN).is_err());

    // The escape hatch is one-directional
    let lax = FieldCmp::new("pet", 0.1).nan(NanPolicy::RustMayBeNan);
    assert!(compare_field(&lax, f64::NAN, 25.0).is_ok());
    assert!(compare_field(&lax, 25.0, f64::NAN).is_err());
}

#[test]
fn draw_respects_axis_bounds_and_is_reproducible() {
    let domain = Domain::new()
        .real("tdb", 10.0, 40.0)
        .enumerated("posture", 3)
        .flag("limit_inputs");

    let mut r1 = Rng::new(99);
    let mut r2 = Rng::new(99);
    for _ in 0..200 {
        let s1 = draw(&domain, &mut r1);
        let s2 = draw(&domain, &mut r2);
        assert!((10.0..=40.0).contains(&s1.real("tdb")));
        assert!(s1.index("posture") < 3);
        assert_eq!(s1.real("tdb"), s2.real("tdb"), "same seed -> same sample");
        assert_eq!(s1.flag("limit_inputs"), s2.flag("limit_inputs"));
    }
}

#[test]
#[should_panic(expected = "unknown axis")]
fn unknown_axis_panics_rather_than_returning_zero() {
    let domain = Domain::new().real("tdb", 20.0, 21.0);
    let mut r = Rng::new(5);
    draw(&domain, &mut r).real("nope");
}

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
    run_sweep("shrinking", &domain, |s| {
        if s.real("x") > 0.0 {
            Err(format!("x was positive: {}", s.real("x")))
        } else {
            Ok(())
        }
    });
}
