//! JOS3 thermoregulation model.
//!
//! This is a partial port. Only [`parameters`] (constant tables and default
//! coefficients) has been ported so far; the simulation itself
//! (`jos3.py`, `thermoregulation.py`, `construction.py`, `matrix.py`) is not
//! yet implemented in Rust.

pub mod parameters;
