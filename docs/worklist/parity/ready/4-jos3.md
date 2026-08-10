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

Progress: `parameters.rs`, `construction.rs` and `matrix.rs` are ported and committed
(ade759b). `matrix.rs` owns the 85-node layout (`IDICT`, `NUM_NODES`); nothing else keeps
a copy. `thermoregulation.rs` is in flight. Remaining after that: the `jos3.py` surface.

**Surface design, from reading `models/jos3.py`.** Python is a class whose environment is
set through 8 settable properties, then advanced by
`simulate(times: int, dtime=60, output=True)`, which loops `_run(dtime)` and appends each
step to `_history`. `dict_results()` transposes that history; 25 read-only properties expose
the current state.

Mapping that onto the agreed builder + advance shape:

- `Jos3Builder` carries the constructor arguments — `height`, `weight`, `fat`, `age`, `sex`,
  `ci`, `bmr_equation`, `bsa_equation` — with upstream's defaults, and `build()` returns
  `Result`, running `validate_body_parameters` (height [0.5,3.0] m, weight [20,200] kg,
  age [5,100] y, fat [1,90] %). Those bounds already live on the `BodyFat` newtype and in
  `construction::validate_body_parameters`.
- `advance(&mut self, conditions: &Jos3Conditions, steps: u32, dtime: Duration)` replaces
  set-properties-then-`simulate`. Both `times` and `dtime` must be exposed: `times` is the
  loop count and `dtime` the seconds per step, and upstream lets them vary independently.
- `Jos3Conditions` holds the 8 settable properties (`tdb`, `tr`, `rh`, `v`, `clo`, `par`,
  `posture`, and `to`). Each is per-body-part upstream — `to_array_body_parts` broadcasts a
  scalar, a list or a by-name mapping to 17 elements — so each field needs to accept both a
  uniform value and a per-segment one. `construction.rs` already ports the three broadcast
  forms as separate typed functions.
- The step results accumulate into a history the caller can read, mirroring
  `dict_results()`. Same struct-of-trajectories shape settled on for the sleep model.

Open when implementing: whether `advance` returns the steps it just ran or only appends to
an internal history. Upstream does the latter and exposes `dict_results()`; returning them
is friendlier but diverges. Prefer upstream's shape per the governing principle.

Then: add `sweep_jos3` to `tests/differential_sweep.rs` comparing whole trajectories, and
remove the `PYTHON_NOT_PORTED` entry in `scripts/check_parity_coverage.py` — the checker
fails until it is removed, which is the intended ratchet.
