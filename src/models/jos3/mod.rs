//! JOS3 thermoregulation model.
//!
//! The full port of `pythermalcomfort.models.jos3` / `pythermalcomfort.jos3_functions`:
//! `parameters` (constant tables and default coefficients), `construction`
//! (body-parameter validation and the thermal conductance and capacity matrices),
//! `matrix` (the 85-node index layout and the blood-flow coefficient matrices),
//! `thermoregulation` (the physiology: basal metabolism, shivering, sweating,
//! vasomotion and the heat-transfer coefficients), and `jos3` (the simulation loop
//! and public surface — [`jos3::Jos3Builder`], [`jos3::Jos3Model`], and
//! [`jos3::Jos3Results`]).
//!
//! `matrix` owns the node layout — `IDICT` and `NUM_NODES` — and every other module
//! here reads it from there rather than keeping its own copy.

// The sub-modules are implementation detail. pythermalcomfort's public surface for this
// model is the `JOS3` class alone; `jos3_functions` is its internals. Exposing them here
// would also oblige every helper to carry a cross-library parity test of its own, which
// `make parity-coverage` correctly insisted on.
pub(crate) mod construction;
pub(crate) mod matrix;
pub(crate) mod parameters;
pub(crate) mod thermoregulation;

#[allow(clippy::module_inception)]
mod jos3;

pub use jos3::{
    Jos3BuildError, Jos3Builder, Jos3Conditions, Jos3Error, Jos3Model, Jos3Options, Jos3Results,
    PerBodyPart,
};
// `ClothingEnsemble`/`LOCAL_CLO_TYPICAL_ENSEMBLES` are data, not part of the simulation
// internals the rest of `parameters` holds, and upstream exposes them: unlike the
// helper functions kept `pub(crate)` above, this per-segment clothing-insulation table
// has no equivalent elsewhere in the crate (`utilities::CLO_TYPICAL_ENSEMBLES` is
// whole-body only), so it is re-exported here as real public surface.
pub use parameters::{ClothingEnsemble, LOCAL_CLO_TYPICAL_ENSEMBLES};
// These three error types are `pub` inside `pub(crate)` submodules, but callers of
// `Jos3Error`/`Jos3BuildError` need to match on them (not just `Display` them) to tell
// which validation failed -- `PerBodyPart::resolve` returns `BodyPartsInputError`
// directly, too. Re-exporting them here (rather than making `construction`/
// `thermoregulation` public) keeps the rest of those modules internal while making this
// part of the error surface actually nameable from outside the crate.
pub use construction::{BodyParameterError, BodyPartsInputError};
pub use thermoregulation::ThermoregulationError;
// `Jos3Conditions::posture`'s type. Named `Posture` inside `thermoregulation` (where it
// keys the convective/radiative coefficient tables) and re-exported under the
// model-qualified name the rest of the crate uses for a per-model posture enum, beside
// `models::pet::Posture` and `models::phs::PhsPosture`.
pub use thermoregulation::Posture as Jos3Posture;
