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

## Priority (user, 2026-08-26)

**Gap 2, Gap 3 and the remaining band-edge sites all land before any new functionality.**
No new models, no new ports, no API work until these three are done.

Order, and the reason for it: remaining band edges first (in flight, and the technique is
established), then Gap 3, then Gap 2. **Gap 3 must precede Gap 2.** A sweep whose domain
never reaches a branch cannot fail when that branch is perturbed, so fault-injecting an
unaudited domain measures the domain, not the guard — and returns a clean bill of health
for code the sweep never executed. That is the same false-confidence shape this whole item
exists to eliminate.

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

## Gap 1 — closed, but not the way this item expected

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

One documented non-goal: swapping strict for inclusive comparisons on `compliance` is
untestable, because the criterion runs on the unrounded PMV and no input lands exactly
on ±0.5.

### The generalisation — half done

Every discrete output in the crate has the same shape. Being *in* the sweep is not
evidence its edges are tested. Done on 2026-08-26 for the four banded indices
(`test_banded_indices_at_exact_edges`, `test_discomfort_index_bands_before_rounding`):

- `UtciResult.stress_category` — 9 edges
- `HumidexResult.discomfort` — 5 edges
- `HeatIndexResult.stress_category` (`heat_index_rothfusz`) — 4 edges
- `DiscomfortIndexResult.discomfort_condition` — the band-before-round order

Measured contrast, which is the reason these were worth adding:

| fault | Rust-only unit test | sweep | cross-library edge test |
| --- | --- | --- | --- |
| UTCI right-inclusive → right-open | caught | **missed** | caught |
| humidex right-inclusive → right-open | caught | — | caught |
| DI bands rounded not unrounded | **missed** | caught | caught |

The pre-existing `#[cfg(test)]` edge tests are *not* useless — they catch a later
regression in a rule. What they cannot catch is a rule that was wrong when transcribed,
since the expectation was written from the same reading as the code. `b88bbc0` was
exactly that, self-consistently wrong. Only a comparison against pythermalcomfort closes
it.

**Two facts found while doing it, both worth not rediscovering:** the round-then-band
order is *not* uniform — UTCI, humidex and `heat_index_rothfusz` round to 1 dp and then
band, while `discomfort_index` bands the unrounded value and rounds only what it reports
(Rust matches Python on all four). And UTCI categorises `utci_si`, not the possibly-IP-
rescaled `utci_approx` it returns; the port handles this and comments it, but it reads
like a bug.

**Still open**, and needing a different technique because their edges are on inputs or on
exact-equality caps rather than on a rounded output:

- ~~`AdaptiveAshraeResult.acceptability_80`/`_90` and `AdaptiveEnResult.
  acceptability_cat_i`/`_ii`/`_iii`~~ — DONE 2026-08-26,
  `test_adaptive_acceptability_at_band_bounds`. Technique: ask Python for the bounds and
  feed each back as `tdb = tr = bound`. Two things came out of it. The guard
  (`assert_is_really_the_edge`, requiring Python's own verdict to flip across the bound)
  fired immediately on EN and revealed that half the test was vacuous — EN rounds each
  bound *after* unit conversion, so the reported bound is not the one acceptability is
  evaluated against. And probing EN on the bound surfaced a real divergence from
  upstream that is nothing to do with banding: see
  `2-temperature-newtype-is-lossy-at-the-boundary.md`, which needs a decision.
- `SportsHeatStressRisk.recommendation` — edges at risk level 2.0/3.0/4.0, but the value
  is a brentq root, so landing on one exactly needs solving rather than scanning.
- `UseFansHeatwavesResult.heat_strain*` — driven by exact `==` against the caps
  (`m_bl == max_skin_blood_flow`, `w == w_max`, `m_rsw == max_sweating`), not a literal
  band edge.
- `IreqResult.dle_min`/`dle_neutral` — `Hours` vs `MoreThanEight` at `dle == 0.0` and
  `dle == 8.0`.

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
upstream over a randomised domain, every output field is compared, every discrete output is
compared at its band edges, and every guard that can exclude a sample has been shown to fail
when it should.*

As of 2026-08-26: the first is true, but at each sweep's own tolerance, not a uniform 1e-9
(see the correction above). The second is true. The third is true for `tsv`, `compliance`,
the four banded indices and the adaptive acceptability flags, and false for the four sites
still listed under "Still open". The fourth is true for two guards out of roughly fifty.

Gap 2 is the bulk of what remains, and the edge work above has changed what it should be.
A uniform 1e-6 perturbation per model is the right probe for a *continuous* output and the
wrong one for a discrete one — a 1e-6 shift in a float that feeds a band comparison changes
nothing unless the sample happens to sit within 1e-6 of an edge, which is exactly why the
sweep missed six of the eight faults injected above. Split Gap 2 in two: perturbation for
continuous fields, edge fixtures for discrete ones.
