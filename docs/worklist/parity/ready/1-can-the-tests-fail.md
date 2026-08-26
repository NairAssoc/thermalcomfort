# Prove the tests can fail

The question this exists to answer: *how do we know there are no more bugs of the kind this
branch kept finding?* Three of them shared one shape — **a check that could not fail**:

- `two_nodes_gagge_sleep` ignored `quilt_thickness` entirely; its parity test compared one
  steady-state point at `epsilon = 2.0` against a 2.17 °C error.
- `sweep_two_nodes_gagge_ji` rounded *Python's* values to match Rust's before comparing, so
  the reference was manufactured to agree.
- `sweep_pet_steady` decided convergence from process-global warning state, silently
  discarding ~10% of its samples in any full-suite run; the count went to an `eprintln!`,
  which `cargo test` discards for passing tests.

Each looked exactly like a passing test. **Green is not evidence a test can detect anything.**
The only way to know is to break the thing deliberately and confirm the test notices.

## What has been measured (2026-08-12)

Done, no action needed:

- **Every public function is in the differential sweep.** 74 tracked, 0 with a parity test
  but no sweep. So the ~70 parity assertions at `epsilon >= 0.1` (11 at 1.0, 2 at 2.0) are
  belt-and-braces — the sweep compares the same functions at 1e-9. Tightening them is
  cosmetic; do not spend time on it before the items below.

  **Correction (2026-08-26): "the sweep compares at 1e-9" is not true crate-wide.**
  `sweep_phs` compares at 0.06 absolute on the temperatures, 0.6 on the exposure limits
  and 1.1 on `sweat_loss_g`. Before relying on the 1e-9 figure for any given model, read
  that sweep's own `FieldCmp` list. Auditing which sweeps are actually loose, and why
  each one is, belongs with Gap 2.
- **Every result field is swept, except two** (see below).
- **Skip counts are now bounded and asserted**, not printed. `report_skipped` in
  `tests/support/sweep.rs` fails the sweep if the excluded share exceeds a documented
  ceiling, and each ceiling carries its measured rate.

## Gap 1 — CLOSED 2026-08-26, but not the way this item expected

Both fields are now compared in `sweep_pmv_ppd_iso` and `sweep_pmv_ppd_ashrae`, via
`compare_category` and a new `compare_optional_bool`/`py_optional_bool` pair.

**Adding them to the sweep was necessary and not sufficient, which is the finding.** Fault
injection showed the sweep catches a *mislabelled* band and an *inverted* compliance flag,
but not the faults that actually matter here:

| injection | sweep | fixture test |
| --- | --- | --- |
| mislabel `SlightlyWarm` as `"Warm"` | caught | caught |
| give ISO ASHRAE's right-closed bands (i.e. bug `b88bbc0`) | **missed** | caught |
| shift one band edge by 1e-9 | **missed** | caught |
| invert `compliance` | caught | caught |
| widen the comfort band by 0.01 | **missed** | caught |
| evaluate compliance on the rounded PMV | **missed** | caught |

The reason is uniform: a discrete output changes only within a window a few thousandths
wide around its edge, and randomised reals do not reliably land there. Closed by two
fixture tests — `test_pmv_tsv_band_edges_iso_versus_ashrae` (six inputs whose rounded PMV
is exactly an edge, where ISO and ASHRAE are one band apart) and
`test_pmv_compliance_band_edges` (four inputs either side of ±0.5, run with
`round_output` both ways).

**Generalise this before doing Gap 2.** Every other discrete output in the crate has the
same shape and the same likely hole: `UtciResult.stress_category`,
`heat_index_lu`/`heat_index_rothfusz` stress categories, the adaptive acceptability flags,
`sports_heat_stress_risk`'s risk bands, and `set_tmp`'s heat-strain verdicts. Being *in*
the sweep is not evidence its edges are tested. One documented non-goal: swapping strict
for inclusive comparisons on `compliance` is untestable, because the criterion runs on the
unrounded PMV and no input lands exactly on ±0.5.

## Original text of Gap 1 — `tsv` and `compliance` are never swept

`PmvPpdResult` has four fields. `pmv` and `ppd` are swept; `tsv` (thermal sensation
category) and `compliance` are **not** — 0 occurrences in `tests/differential_sweep.rs`.
They appear only in `tests/python_comparison.rs`, at a handful of fixed points.

This is the highest-value gap on the list because `tsv` is a **band categorisation**, and
band edges on this branch have already produced two bugs: `b88bbc0` (the ISO and ASHRAE
sensation bands were conflated) and `18bf5ff` (UTCI's stress-category edges, plus it should
have been an `Option`). A discrete output driven by a float is precisely where a 1e-14
difference becomes a wrong answer — the JOS3 boundary bug (`28adfa8`) was the same mechanism.

Add both to `sweep_pmv_ppd_iso` and `sweep_pmv_ppd_ashrae`, comparing the categorical value
against Python's. `UtciResult.stress_category` is already swept and shows how to compare an
`Option<enum>` against a Python string.

## Gap 2 — nobody has shown the other sweeps can fail

Two guards have been fault-injection tested, both ad hoc:

- the Ji conditioning gate (`442f859`): offsets of 1e-6, 4e-8, 2e-7, a 1e-8 relative
  scaling, and offsets applied only from minute 150 were all caught at sample 0. A constant
  +1e-8 is *not* caught — inside the 1e-9 relative bound at 30-40 °C, documented in the code
  as the price of that bound.
- `report_skipped` (`8ef95e8`): verified by tightening a ceiling below its measured rate and
  watching the assertion fire.

The other 45 sweeps have never been shown capable of failing. **Do this systematically:** for
each model, perturb its Rust output by a small constant and confirm its sweep fails, then
revert. A sweep that still passes is either not comparing that field, not reaching that code
path, or has a tolerance wider than the perturbation — all three are findings.

Mechanise it rather than hand-editing 45 times: a temporary `THERMALCOMFORT_FAULT` env var
read at the top of each model, adding a small offset to one output, is enough to drive the
whole matrix in one run. Delete the hook when the audit is done — do not ship it.

Suggested perturbation: 1e-6 absolute. Large enough to clear every documented tolerance,
small enough to be physically meaningless.

## Gap 3 — sweep domains may not reach every branch

A sweep only tests what its domain generates. `quilt_thickness` had no effect for as long as
it did partly because nothing varied it meaningfully; `two_nodes_gagge_ji`'s
`length_time_simulation` was hardcoded, so no sweep could vary it. Both are fixed, but the
class is not audited.

For each sweep, check that every parameter of the model appears as a domain axis, and that
enumerated axes cover every variant rather than the first two. `PET_MEASURE`-style coverage
counters, or simply asserting that each branch of a `match` is hit at least once across a
run, would make this checkable rather than eyeballed.

**This class produced a live miss on 2026-08-26, which is the argument for doing the audit
properly rather than by inspection.** Porting upstream 4.4.1's ISO 7933:2023 Annex E
minute-1 skin-temperature special case, every existing PHS test — the two 480-minute
comparisons and the 60-minute one — passed with the ported code *deleted*. The special
case perturbs `t_sk` on minute 1 only and the `exp(-1/3)` lag decays that difference to
~6e-10 °C by minute 60, so nothing running to 480 minutes could see it. Root cause:
`sweep_phs` never varied `duration` — it took the 480-minute default, so no sample was
short enough. Fixed by adding a 5-value `duration` axis (1/2/5/60/480), which now catches
the deletion, plus `test_phs_minute_one_skin_temperature_special_case` as a direct pin.

The general lesson: a parameter left at its default is not merely untested, it can hide a
whole time-domain of behaviour. `sweep_phs` still leaves `limit_inputs`, `f_r`, the `t_sk`/
`t_cr`/`t_re`/`t_cr_eq` initial conditions, `t_sk_t_cr_wg`, `sweat_rate_watt` and
`evap_load_wm2_min` at their defaults; PHS has the largest option surface in the crate and
is the obvious place to start this gap.

## What "confident" can honestly mean here

Not "there are no bugs". The reachable claim is: *every public function is compared against
upstream over a randomised domain at 1e-9, every output field is compared, and every guard
that can exclude a sample has been shown to fail when it should.* The first is true today,
the second is true except for `tsv`/`compliance`, and the third is true for two guards out of
roughly fifty. Gaps 1 and 2 are what close it.
