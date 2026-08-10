# Port `two_nodes_gagge_sleep` to the Yan et al. (2022) model

Shape DECIDED (user, this session) — no open questions:

- **Inputs**: a `SleepInputs` struct whose fields are Python's six per-minute arrays,
  named identically — `tdb, tr, v, rh, clo, thickness_quilt` — as borrowed **slices**.
  Their common length is the simulation duration, exactly as Python defines it.
  Plus `wme` and `p_atm` and the `**kwargs` optionals (`ltime`, `height`, `weight`,
  `c_sw`, `c_dil`, …) in a defaulted `SleepOptions`.
- **Output**: `GaggeTwoNodesSleepResult`, a struct of 10 owned `alloc::Vec`s named exactly
  as Python's `GaggeTwoNodesSleep`: `set, t_core, t_skin, wet, t_sens, disc, e_skin,
  met_shivering, alfa, skin_blood_flow`. Newtyped per item 2. No fixed capacity — Python's
  arrays are unbounded, so a cap would be a Rust-invented limit.

The current signature returns `GaggeTwoNodesResult`: 18 scalar fields, **11 the sleep model
cannot produce** (`e_rsw, e_max, q_sensible, q_skin, q_res, m_bl, m_rsw, w_max, et,
pmv_gagge, pmv_set`), **missing 3 it does** (`met_shivering, alfa, skin_blood_flow`).

What is wrong today: `src/models/two_nodes_gagge.rs:624` delegates to the standard Gagge
model at fixed `met_sleep = 0.7` and computes `_f_a_cl_bedding` without using it, so
`quilt_thickness` has no effect. At tdb=25/tr=25/v=0.1/rh=50/clo=0.5 a 9 cm quilt gives
SET 20.88 in Python and 23.05 here; `disc` sign-flips.

Port: transcribe `_sleep_set` (`models/two_nodes_gagge_sleep.py`, 218 lines)
statement-for-statement as `two_nodes_gagge_ji` was, then add
`sweep_two_nodes_gagge_sleep` to `tests/differential_sweep.rs` comparing whole trajectories.
