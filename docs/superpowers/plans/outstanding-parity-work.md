# Outstanding parity work

The randomised differential sweep is built: `tests/differential_sweep.rs` covers every
public model and utility except `two_nodes_gagge_sleep` (unported, below), `make sweep`
runs it deep, and CI runs a short one on every push. See `README.md` under "Differential sweep" for how to run and reproduce.
The plan that produced it, and the 24 defects it closed, are in git history.

Three pieces of work remain, in the order they should be done. The API pass is **decided**
and ready to implement. JOS3 and the sleep model each carry one open decision, noted in
place.

`make parity-coverage` now checks both directions, so a future upstream release growing a
model the port lacks will fail the build rather than pass silently. JOS3 is recorded
in `PYTHON_NOT_PORTED` in `scripts/check_parity_coverage.py`: reported on every run,
non-fatal, so it cannot mask the *next* gap.

The upstream source is not in the repo. If `/tmp/ptc_diff` has been cleared, get it back
with `pip download pythermalcomfort==4.4.0 --no-deps --no-binary :all: -d /tmp/` and
`tar xzf`.

**This branch is a major bump — 3.9.8 to 4.4.0 — so breaking API changes are in scope and
this is the window for them.** Earlier notes in this repo said the opposite ("the crate
version tracks pythermalcomfort's and cannot take a major bump on its own schedule"). That
reasoning applies to a patch or minor release; it does not apply here, and the API pass
below depends on that.

---

---

## 1. API modernisation — decided 2026-08-10

Unit confusion and positional swapping are both ordinary API calling errors, and the crate
currently only defends against the first. Two orthogonal mechanisms, applied together:

**Rule 1 — every dimensioned or bounded value is a newtype, on inputs *and* outputs.**
Catches °C-vs-°F and isolates conversion from calculation: convert once at the boundary,
calculate in plain `f64`, construct the newtype at the end. The crate already does the
input half of this.

**Rule 2 — every model takes a named input struct.** Catches positional swapping, which
Rule 1 cannot: `adaptive_ashrae(tdb, tr, t_running_mean, v)` is fully newtyped today and
all three temperatures are silently interchangeable. Measured worst cases:

| Function | Swappable run |
|---|---|
| `solar_gain` | **7 consecutive `f64`** (5 are [0,1] fractions) |
| `adaptive_ashrae`, `adaptive_en` | 3 consecutive `Temperature` |
| `wbgt` | 2 consecutive `Temperature` (`twb`, `tg`) |
| `heat_index_rothfusz` | 2 consecutive `bool` (`round_output`, `limit_inputs`) |
| `use_fans_heatwaves` | 14 parameters |

Shape: a required-fields `XInputs` struct with **no `Default`** (so every field must be
named) plus the existing defaulted `XOptions`. Extends the `*Options` pattern already in
the crate rather than inventing a parallel one.

```rust
adaptive_ashrae(
    AdaptiveInputs {
        tdb: Temperature::from_celsius(25.0),
        tr: Temperature::from_celsius(25.0),
        t_running_mean: Temperature::from_celsius(20.0),
        v: Speed::from_meters_per_second(0.1),
    },
    Default::default(),
)
```

**Role newtypes are NOT needed.** `DryBulb`/`MeanRadiant`/`WetBulb` were considered for the
cross-model seams; once inputs are named structs, naming supplies the role distinction the
type would have encoded. Two mechanisms, not three.

### Output typing

Outputs are currently all plain `f64`, which reintroduces on the way out the confusion the
input types prevent. Concretely: `wbgt()` returns `f64` while `work_capacity_iso()` takes
`wbgt: Temperature`, so the natural pipeline forces `Temperature::from_celsius(result)` —
this crate's own tests do that re-wrap 8 times.

| Output kind | Becomes |
|---|---|
| Absolute temperatures (`pet`, `utci`, `set`, `t_re`, `tmp_cmf`, `wbgt`, …) | `Temperature` |
| Temperature *differences* (`SolarGainResult.delta_mrt`, `cooling_effect()`) | `TemperatureDelta` — **actively wrong today**, indistinguishable from absolute values |
| Heat flows in W/m² (`e_skin`, `q_sensible`, …) | `HeatFluxDensity` (new) |
| Genuinely dimensionless (`pmv`, `ppd`, `di`, risk levels, capacity %) | stay `f64` |

### New types required

`CardiacIndex` (L/(min·m²)), `HeatFluxDensity` (W/m²), `ActivityRatio` (PAR, dimensionless
multiple of BMR), `BodyFat` (percentage). The last two follow `WorkEfficiency`: bounded,
and an out-of-range value yields a plausible-looking wrong answer rather than failing.
`BsaFormula` already exists and covers JOS3's `bsa_equation`; `bmr_equation` needs a new
enum.

### Cost, honestly

~40 signatures and ~40 result structs, plus every test, doctest and example that touches
them. That is a larger diff than JOS3 itself. It is much cheaper before JOS3 than after,
because JOS3 adds 6,100 lines that would otherwise be written twice.

---

---

## 2. Port `JOS3`

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

**Decision 1 — linear algebra — SETTLED 2026-08-10.** nalgebra is now an unconditional
dependency built against `alloc` + `libm`, so `DMatrix::lu().solve()` is available in the
default `no_std` build and JOS3's 85×85 solve costs nothing extra. Verified on
`wasm32-unknown-unknown` and bare-metal `thumbv7em-none-eabihf`. The old framing — "gate
JOS3 behind `std`" — was based on a false premise: nalgebra has supported `no_std` since
0.15, and the crate's `std` feature never bought compatibility.

Note the matrix is **4.3% dense** (309 nonzeros of 7225, bordered block-diagonal: 17
per-segment blocks plus a central-blood node). A sparse solve is therefore possible later,
but is deliberately *not* the first pass: Python computes an explicit inverse and
multiplies, which is less accurate than a direct solve, so a sparse implementation would
legitimately differ in the last digits and make early parity failures ambiguous. Port
dense first, prove parity, then optimise with the sweep as the safety net. Performance is
not a motivation — an 85×85 dense LU is ~200k flops against a per-minute timestep.

**Decision 2 — what shape is the Rust API? STILL OPEN.** Python is a stateful object whose
properties are mutated between `simulate()` calls (set `tdb`, simulate 60 min, change it,
simulate again). Options: a struct with `&mut self` methods mirroring Python; a builder
plus an explicit step/advance API; or a pure function taking a schedule of conditions and
returning a trajectory. This is the same class of question as the sleep model's return
shape. Use the `superpowers:brainstorming` skill on it before writing code — it is a
genuine design question, not a transcription.

**Decision 3 — feature-gate JOS3? STILL OPEN.** ~6,100 lines and ~120 KB peak working RAM
(85×85 f64 matrix plus nalgebra's LU copy) is a lot to impose on someone who only wants
PMV. A default-off `jos3` feature would make that opt-in. Against: another build
configuration to test.

**Once decided**, the porting routine is the established one: transcribe
statement-for-statement, then add `sweep_jos3` to `tests/differential_sweep.rs` comparing
whole trajectories (the Ji and ridge-regression sweeps are the pattern), and remove the
`PYTHON_NOT_PORTED` entry — the checker will fail until it is removed, which is the
intended ratchet.

---

---

## 3. Port `two_nodes_gagge_sleep` to the Yan et al. (2022) model

**Blocked on: what shape should the Rust function return?**

Python returns *arrays* — one value per minute of the night. The Rust signature returns a
single `GaggeTwoNodesResult`. The options are:

1. Return the final minute only — smallest change, loses the trajectory.
2. Return a `Vec`/`heapless::Vec` trajectory — matches Python, breaks the current
   signature. This is what `two_nodes_gagge_ji` does, so there is precedent in the crate.
3. Take a duration and return a trajectory of that length — most flexible, largest API.

**Recommendation: option 2**, for consistency with the Ji model, which was ported to a
`heapless::Vec` trajectory and now matches Python exactly. The API pass above also settles
the surrounding shape: a named `SleepInputs` struct, and trajectory elements carrying
`Temperature`/`HeatFluxDensity` rather than bare `f64`. Note the current single-struct
signature returns values the model cannot actually produce, so keeping it is not the
conservative option it looks like.

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

---

## REOPENED AND FIXED: the `measurements::Temperature` round-trip

**This section previously read "Closed: not reproducible" and told you not to spend a
crate-local temperature type on it. That was wrong, and the way it was wrong is worth
keeping.**

The evidence behind the closure was real: 80,000 swept adaptive samples comparing
acceptability booleans found zero divergence, and the reasoning was that 1e-14 °C only
matters where a float feeds a discrete comparison, with nothing in the swept space landing
that close to a band edge. Both true. The flaw was the last clause — it described the models
that existed *then*.

JOS3 does land on a band edge, deliberately. Its
`_calculate_operative_temp_when_pmv_is_zero` is a damped fixed-point search that walks the
operative temperature up onto ISO 7730's 30 °C applicability limit and *depends on crossing
it* to get NaN back and take a retry branch. `Temperature` stores kelvin, so constructing one
from 30.00000000000001 and reading it back gives exactly 30.0: the limit check passed, the
retry never ran, and because set points are fixed at construction the subject's whole
simulation was wrong — `t_skin_mean` 33.88 against Python's 32.93. One subject in a 3000-way
constructor sweep, and an ordinary one.

Fixed in 28adfa8: `pmv_ppd_iso` now has a plain-`f64` core that the newtype version wraps,
and JOS3 calls the core. No crate-local temperature type was needed after all — the actual
rule is narrower and stricter:

> **Never re-wrap a plain `f64` into a newtype mid-calculation inside the crate.** Convert
> once at the public boundary and calculate in `f64` throughout. Every in-crate caller
> holding an `f64` must use the `*_celsius` entry point.

The transferable lesson is about the closure, not the bug: "swept N samples, found nothing"
bounds what the sweep *could* reach. A model added later can move the boundary the sweep was
implicitly assuming.
