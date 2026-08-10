# Outstanding parity work

The randomised differential sweep is built: `tests/differential_sweep.rs` covers every
public model and utility except `two_nodes_gagge_sleep` (unported, below), `make sweep`
runs it deep, and CI runs a short one on every push. See `README.md` under "Differential sweep" for how to run and reproduce.
The plan that produced it, and the 24 defects it closed, are in git history.

Three items remain. **All are blocked on a decision only the developer can make** — none
is waiting on implementation effort, and none can be resolved by reading more code.

`make parity-coverage` now checks both directions, so a future upstream release growing a
model the port lacks will fail the build rather than pass silently. JOS3 below is recorded
in `PYTHON_NOT_PORTED` in `scripts/check_parity_coverage.py`: reported on every run,
non-fatal, so it cannot mask the *next* gap.

The upstream source is not in the repo. If `/tmp/ptc_diff` has been cleared, get it back
with `pip download pythermalcomfort==4.4.0 --no-deps --no-binary :all: -d /tmp/` and
`tar xzf`.

---

## Port `JOS3` — next up

**Blocked on: two architectural decisions, plus whether the size is worth it.**

pythermalcomfort's 17-segment whole-body thermoregulation model. It is the only upstream
model with no Rust counterpart at all, and it is the largest thing in the package by a
wide margin.

**Reconnaissance (2026-08-10), so it need not be redone:**

| | |
|---|---|
| Size | **~6,100 lines** of Python: `models/jos3.py` 1650, `jos3_functions/parameters.py` 1627, `thermoregulation.py` 1575, `construction.py` 789, `matrix.py` 452 |
| Dependencies | **numpy only — no scipy.** No root-finding, unlike PET and SET |
| Numerics | Solves an **85×85 dense linear system** each step: `np.linalg.inv(arr_a)` then `np.dot` (`models/jos3.py:1010` and `:1046`) |
| State | 85-element body-temperature vector; stateful across calls |
| API | A **class**, not a function: `JOS3(height, weight, fat, age, sex, ci, bmr_equation, bsa_equation)`, then `simulate()`, with 25 properties and `dict_results()` / `to_csv()` |

For scale: the entire existing port is ~76 public functions, and the largest single model
ported so far (the Ji two-node model) is a few hundred lines. JOS3 alone is comparable to
a significant fraction of the crate. **Worth confirming it is wanted before starting** —
"not ported" is a defensible permanent answer for a model this size, and is now recorded
honestly in the README rather than papered over.

**Decision 1 — `no_std` or `std`-gated?** The 85×85 inverse is the crux. The crate is
`no_std` by default and only pulls `nalgebra` under the `std` feature, for PET's 3×3 LU.
Options: gate JOS3 behind `std` entirely (simplest, but the crate's selling point is
`no_std`); hand-roll a dense LU with partial pivoting for `no_std` (85×85 is not hard, but
it is new numerical code that needs its own tests); or restructure to avoid the explicit
inverse — note Python inverts and multiplies, where solving directly is both faster and
better conditioned, so a Rust port need not reproduce the inversion.

**Decision 2 — what shape is the Rust API?** Python is a stateful object whose properties
are mutated between `simulate()` calls (set `tdb`, simulate 60 min, change it, simulate
again). This is the same class of question as the sleep model's return shape. Options: a
struct with `&mut self` methods mirroring Python; a builder plus an explicit step/advance
API; or a pure function taking a schedule of conditions and returning a trajectory.

**Recommendation:** settle Decision 1 first — if it lands on `std`-gated, `nalgebra`
already provides the solve and Decision 2 becomes the only real work. Use the
`superpowers:brainstorming` skill on Decision 2 before writing code; it is a genuine
design question, not a transcription.

**Once decided**, the porting routine is the established one: transcribe
statement-for-statement, then add `sweep_jos3` to `tests/differential_sweep.rs` comparing
whole trajectories (the Ji and ridge-regression sweeps are the pattern), and remove the
`PYTHON_NOT_PORTED` entry — the checker will fail until it is removed, which is the
intended ratchet.

---

## Port `two_nodes_gagge_sleep` to the Yan et al. (2022) model

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

## The `measurements::Temperature` round-trip

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
