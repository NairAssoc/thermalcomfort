# Audit Rust signatures against Python parameter lists

For every ported function, does the Rust signature expose every parameter Python
accepts — including `**kwargs`-style optionals?

Found while deciding the sleep model's shape: `two_nodes_gagge_ji` hardcodes
`length_time_simulation = 120` (`src/models/two_nodes_gagge.rs:942`) where Python takes it
as a kwarg, and does not expose `body_weight`, `initial_skin_temp` or `initial_core_temp`
at all. A hardcoded constant standing in for a Python parameter is a parity gap: the
caller cannot ask a question Python answers, and no parity test can catch it because the
sweep never varies what the Rust API cannot express.

Governing principle (user, this branch): **match the basic Python API, modulo our newtypes,
for both input and output.** We are porting functionality, not redesigning it; parity
testing only works when the calls are the same. Ownership/borrowing is ours to choose.

Output: a table of function → Python params → Rust params → gap. Feeds item 2.

Done: 8 gaps found across 55 functions — `vertical_tmp_grad_ppd` and `solar_gain`
hardcoding `round_output`, `cooling_effect` missing `units`, `pmv_ppd_ashrae` missing
`airspeed_control`, and four `model` selectors (`pmv_ppd_ashrae`, `pmv_ppd_iso`,
`clo_dynamic_ashrae`, `clo_dynamic_iso`). Closed so far: `solar_gain`,
`vertical_tmp_grad_ppd`, `cooling_effect`. The rest are in flight with item 2.

## Still open: the audit only ran one direction

It asked "does Rust expose everything Python does". The reverse question — **does Rust
expose anything Python does NOT** — was never asked, and the first place anyone looked
turned one up: `cooling_effect` took `still_air_threshold`, `body_surface_area`, `p_atm`
and `posture`, all of which upstream fixes as private module constants
(`cooling_effect.py:18-22` and `:141`).

**Audit run, 2 found, both confirmed by hand against upstream. Both still to fix:**

1. **`two_nodes_gagge_ji.round_output` — a live parity bug in the DEFAULT path.**
   `two_nodes_gagge_ji.py` contains no `round` or `np.around` call anywhere; upstream never
   rounds this model. `GaggeTwoNodesJiOptions::default()` sets `round_output: true`, so the
   default Rust result differs from Python at every trajectory point. It survived because
   `sweep_two_nodes_gagge_ji` rounds *Python's* values to match before comparing when the
   flag is set — that is validating Rust against a fabricated reference, not against
   pythermalcomfort. **Delete the parameter and the rounding**, then the sweep can compare
   at 1e-9 unconditionally. Delete the sweep's `round_output` axis with it.

2. **`pet_steady.round_output`.** `pet_steady.py:474` rounds unconditionally:
   `return round(optimize.fsolve(f, pet_guess)[0], 2)`. There is no flag upstream, so
   `round_output: false` yields a value Python cannot produce. Make the rounding
   unconditional and drop the option.

Both are in files the API pass has yet to reach (`src/models/pet.rs`,
`src/models/two_nodes_gagge.rs`), so fold them into that work rather than doing them twice.
No default-value divergences were found on parameters that legitimately exist on both sides.

This direction is worse than a missing parameter, because it fails silently and *looks*
like generosity. A caller can request a configuration upstream cannot produce; there is no
Python call to compare against, so no parity test can be written for it; and the sweep
happily explores that space and proves nothing. Run the same audit the other way across all
55 functions, and fold the results into item 2. Where a knob turns out genuinely useful,
that is a deliberate extension and must be documented as one — not left looking like parity.
