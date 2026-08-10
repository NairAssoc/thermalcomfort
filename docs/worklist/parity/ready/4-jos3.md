# Port JOS3

Scope CONFIRMED (user, this session): port it now. ~6,100 lines of Python.

Decisions, all closed:
- **Linear algebra**: nalgebra `DMatrix::lu().solve()`, unconditional dep on `alloc`+`libm`.
  Dense first — Python computes an explicit inverse and multiplies, so a sparse solve would
  legitimately differ in the last digits and make early parity failures ambiguous.
- **API shape** (user): builder + explicit step/advance. `Jos3Builder::new()...build()?`
  then `sim.advance(conditions, minutes)?`. State stays explicit; no property-mutation
  mirror of Python's class.
- **Feature gating** (user): none. Unconditional, one build configuration.

Sources: `models/jos3.py` 1650, `jos3_functions/parameters.py` 1627,
`thermoregulation.py` 1575, `construction.py` 789, `matrix.py` 452.
85×85 dense solve per timestep (`models/jos3.py:1010`, `:1046`); 85-element state vector.

Suggested order: `parameters.rs` (data tables, independent of API shape) → `construction.rs`
→ `thermoregulation.rs` → `matrix.rs` → the builder/advance surface.

Then: add `sweep_jos3` to `tests/differential_sweep.rs` comparing whole trajectories, and
remove the `PYTHON_NOT_PORTED` entry in `scripts/check_parity_coverage.py` — the checker
fails until it is removed, which is the intended ratchet.
