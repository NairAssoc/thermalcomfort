//! Shared scaffolding for the differential parity sweep.
//!
//! Lives under `tests/` because it is `std`-only; the crate itself is `no_std`.
#![allow(dead_code)]

pub mod compare;
pub mod domain;
pub mod rng;
pub mod sweep;
