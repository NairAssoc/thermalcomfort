//! JOS3 thermoregulation model.
//!
//! This is a partial port. Ported so far: [`parameters`] (constant tables and default
//! coefficients), [`construction`] (body-parameter validation and the thermal
//! conductance and capacity matrices), and [`matrix`] (the 85-node index layout and the
//! blood-flow coefficient matrices), and [`thermoregulation`] (the physiology: basal
//! metabolism, shivering, sweating, vasomotion and the heat-transfer coefficients).
//! Still to come: the simulation loop and public surface in `jos3.py`.
//!
//! [`matrix`] owns the node layout — `IDICT` and `NUM_NODES` — and every other module
//! here reads it from there rather than keeping its own copy.

pub mod construction;
pub mod matrix;
pub mod parameters;
pub mod thermoregulation;
