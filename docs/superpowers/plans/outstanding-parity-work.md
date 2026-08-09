# Outstanding parity work

The randomised differential sweep is built and complete: `tests/differential_sweep.rs`
covers every public model and utility, `make sweep` runs it deep, and CI runs a short one
on every push. See `README.md` under "Differential sweep" for how to run and reproduce.
The plan that produced it, and the 24 defects it closed, are in git history.

Two items remain. **Both are blocked on a decision only the developer can make** — neither
is waiting on implementation effort, and neither can be resolved by reading more code.

---

## A. Port `two_nodes_gagge_sleep` to the Yan et al. (2022) model

**Blocked on: what shape should the Rust function return?**

Python returns *arrays* — one value per minute of the night. The Rust signature returns a
single `GaggeTwoNodesResult`. The options are:

1. Return the final minute only — smallest change, loses the trajectory.
2. Return a `Vec`/`heapless::Vec` trajectory — matches Python, breaks the current
   signature. This is what `two_nodes_gagge_ji` does, so there is precedent in the crate.
3. Take a duration and return a trajectory of that length — most flexible, largest API.

**Recommendation: option 2**, for consistency with the Ji model, which was ported to a
`heapless::Vec` trajectory and now matches Python exactly.

**What is wrong today.** `src/models/two_nodes_gagge.rs::two_nodes_gagge_sleep` delegates
to the standard Gagge model at a fixed `met_sleep = 0.7`, and computes
`_f_a_cl_bedding = 0.0308 * quilt_thickness + 0.7695` without ever using it. So
`quilt_thickness` has no effect: at tdb=25, tr=25, v=0.1, rh=50, clo=0.5 a 1 cm quilt
gives SET 23.05 in both implementations, but a 9 cm quilt gives **20.88 in Python and
still 23.05 here**. `e_skin` is 32.2 vs 6.91 and `disc` +0.9926 vs -0.23 (a sign flip).
The public doc comment warns users; the code is unchanged.

**What the port needs.** Python's `_sleep_set`
(`pythermalcomfort/models/two_nodes_gagge_sleep.py`, 218 lines) is a per-minute
simulation, driven by a loop that supplies for each minute `i`:

- metabolic rate from the Yan polynomial
  `-5.75e-13 x^5 + 7.85521e-10 x^4 - 3.9173563e-7 x^3 + 8.7620232151e-5 x^2
   - 8.801558913211e-3 x + 1.09952538864493` where `x = (i - 1) / 60`
- a *prescribed* core temperature `0.022234 x^2 - 0.27677 x + 37.02`
- carried-over state from the previous minute: `t_skin`, `e_skin`, `alfa`,
  `skin_blood_flow`, `met_shivering`

Inside `_sleep_set`: `f_a_cl = 0.0308 * thickness + 0.7695` (this is where the quilt
enters), `w_max = 0.38 v^-0.29` and `i_cl = 1` when `clo <= 0` else `0.59 v^-0.08` and
`0.45`, a clothing-temperature fixed point, then an `ltime`-step loop, and finally two
Newton solves against `_fnerre`/`_fnerrs` for SET. `met_shivering = 19.4 * cold_s * cold_c`
and `alfa = 0.0417737 + 0.7451833 / (skin_blood_flow + 0.585417)`.

**Once the shape is chosen**, the port itself is ordinary work: transcribe `_sleep_set`
statement-for-statement (as `two_nodes_gagge_ji` was), then add `sweep_two_nodes_gagge_sleep`
to `tests/differential_sweep.rs` following the Ji sweep, which compares whole trajectories.

---

## C. The `measurements::Temperature` round-trip

**Blocked on: should the public API carry raw Celsius instead of typed quantities?**

`Temperature::from_celsius(21.4).as_celsius()` differs from 21.4 by 2.1e-14 — the type
round-trips through Kelvin. Every public entry point takes `Temperature`, so the caller's
Celsius value is perturbed before any model sees it, which flips exact boundary
comparisons.

Concretely: `adaptive_ashrae(tdb=21.4, tr=21.8, t_running_mean=23.4, v=0.11,
limit_inputs=false)` gives `acceptability_80 = True` in Python and `False` here, because
the operative temperature lands either side of the band edge.

This is the only defect from the review that no amount of work inside the models can fix.
Closing it means the public API carries raw `f64` Celsius rather than
`measurements::Temperature`, which is a large break and contradicts the crate's
typed-quantity design.

**Scale:** 8 of 6000 swept adaptive acceptability cases. The differential sweep does not
currently fail on it because the adaptive sweeps compare the numeric band edges, where the
perturbation is far below tolerance; only the derived booleans flip.

**Options:**

1. Accept and document it as a known limitation — zero API churn, 8-in-6000 stays wrong.
2. Take raw `f64` Celsius on public entry points — fixes it, large break, loses the type
   safety the crate is built around.
3. Replace `measurements::Temperature` with a crate-local temperature newtype that stores
   Celsius directly — fixes it, keeps typed quantities, but is a breaking change to every
   signature's type name and drops the `measurements` interop.

**Recommendation: option 3** if a break is affordable, otherwise option 1.

**Constraint on timing:** this crate's version tracks pythermalcomfort's, so the port
cannot take a major bump on its own schedule. A breaking change is therefore cheap only
when upstream majors (4.x -> 5.x), or if the port deliberately departs from the
version-mirroring convention.

**Batch with it, if a break happens.** The same class of API item, recorded during the
2026-08-09 newtype audit and deliberately deferred:

- `SolarGainResult.delta_mrt` and the `f64` returned by `cooling_effect` are temperature
  *differences* and should be `TemperatureDelta`, but the crate's convention is that
  result structs return plain `f64`.
- Solar angles (`solar_gain`, `transpose_sharp_altitude`) are bare `f64` degrees;
  `measurements::Angle` exists but is not re-exported.
- Irradiance in W/m² (`esi`, `solar_gain`) has no type in `measurements`.
- The blood-flow and sweating caps in `use_fans_heatwaves` are bare dimensional rates.
