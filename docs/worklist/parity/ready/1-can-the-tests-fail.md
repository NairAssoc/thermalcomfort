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

**All nine sites are now done** (2026-08-26). Each needed a different technique, because
the "edge" is a different thing each time: a rounded output, a computed bound, an
exact-equality cap, a reporting ceiling, a floor nudge.

**The headline number: of the nine, the differential sweep missed the discriminating fault
at seven.** That is the empirical answer to Gap 2's premise below — see the note at the end
of this document on why Gap 2 must be split rather than run as one uniform perturbation
pass.

- ~~`AdaptiveAshraeResult.acceptability_80`/`_90` and `AdaptiveEnResult.
  acceptability_cat_i`/`_ii`/`_iii`~~ — DONE 2026-08-26,
  `test_adaptive_acceptability_at_band_bounds`. Technique: ask Python for the bounds and
  feed each back as `tdb = tr = bound`. Two things came out of it. The guard
  (`assert_is_really_the_edge`, requiring Python's own verdict to flip across the bound)
  fired immediately on EN and revealed that half the test was vacuous — EN rounds each
  bound *after* unit conversion, so the reported bound is not the one acceptability is
  evaluated against. And probing EN on the bound surfaced the `Temperature`
  kelvin round-trip, now a documented divergence in the README's coverage table.
- ~~`SportsHeatStressRisk.recommendation`~~ — DONE 2026-08-26,
  `test_sports_risk_level_floor_nudge_and_ceiling`. The real hole was not the 2.0/3.0/4.0
  recommendation edges (a 0.1-wide target the sweep hits routinely) but the **`1e-9` nudge
  inside `min(floor((risk + 1e-9) * 10) / 10, 4.9)`, whose deletion the entire suite could
  not detect** — lib tests, sweep and parity all stayed green. It is reachable after all:
  `d(risk)/d(tdb)` ≈ 0.22, so the window is ~4.5e-9 in `tdb`, four orders of magnitude
  wider than the `Temperature` error. `tdb = 25.29999999` sits inside it. The 4.9 clamp is
  covered by a lib test but was uncovered cross-library; now pinned too.
- ~~`UseFansHeatwavesResult.heat_strain*`~~ — DONE 2026-08-26,
  `test_use_fans_heatwaves_strain_flags_at_the_caps`. The inverse of every other site: the
  values are *clamped* to their caps, so a saturated one is bit-identical and upstream uses
  `==`. The failure mode is a tolerance where equality was meant. A 1e-3 window is caught
  by the sweep and the new test; a **1e-4 window is missed by the sweep** and caught here.
  Also fixed a comment in `use_fans_heatwaves.rs` citing `tdb=38.1, tr=43.7, v=2.16` as the
  case a 1e-3 window got wrong — at 4.4.2 those inputs sit 0.032 from the cap, so the
  example had stopped demonstrating its own bug. Cite a test, not a tuple.
- ~~`IreqResult.dle_min`/`dle_neutral`~~ — DONE 2026-08-26,
  `test_ireq_dle_at_the_eight_hour_ceiling`. Moving the ceiling 8.0 → 8.1 is caught here
  and missed by `sweep_ireq`. Two things learned. Classifying the *rounded* `dle` instead
  of the unrounded one is **untestable**, not untested: the two orders differ only for a
  `dle` in `(8.0, 8.05]`, and `dle = -40 / storage` is discontinuous at the ceiling, so no
  input lands in that window — do not add cases chasing it. And the crossing sits between
  two *adjacent* f64s, which the `Temperature` newtype cannot deliver (it perturbs by
  ~1e-13, millions of ULPs at that magnitude), so the test brackets by 1e-4 instead. That
  is the sharpest instance of the `Temperature` kelvin round-trip (README, coverage
  table): near a discontinuity the crate cannot express the input at all.

## Gap 2 — closed 2026-08-26

**All 47 sweeps have now been shown capable of failing.** Driven by a
`THERMALCOMFORT_FAULT` hook in `tests/support/compare.rs` (and a matching tap in
`jos3_close`, which does not route through `compare_field`): set it to `*` or a field
name and every comparison is fed a deliberately corrupted Rust value.

**This item's prescribed method was wrong, and following it would have produced a
misleading result.** It called for a fixed 1e-6 perturbation, "large enough to clear every
documented tolerance". It is not: the 109 `FieldCmp` sites use tolerances from 1e-9 to 1.1,
and 63 of them are 1e-6 or looser, so a fixed 1e-6 would have reported those 63 as
undetectable — measuring the tolerances rather than the guards. The hook therefore scales
the offset to each field's own bound (`tol * 2`, and past the relative bound where one
exists), which asks the question that matters: *is this field compared at all, and does the
sweep reach the code that computes it?*

Two sweeps initially read as "cannot fail", and **both were artefacts of the instrument,
not defects** — worth recording, because reporting them as findings would have been exactly
the false result this document exists to prevent:

- `sweep_clo_lookup_tables` uses its own exhaustive `assert!` rather than `compare_field`.
  Perturbing one real table entry by 1e-7 fails it immediately.
- `sweep_jos3` uses `jos3_close`. With the hook wired in there too it fails as it should.

### The real finding: loose tolerances were being applied to unrounded samples

The loose bounds (0.06, 0.11, 0.6, 1.1) exist to absorb a single rounding step. But these
sweeps vary `round_output` as a domain axis, and the field set was built **once, outside
the sample closure** — so the half of the domain upstream never rounded was compared just
as loosely.

Measured: with `round_output = false`, every `gagge_fields` field agrees with upstream at
**1e-9**. That is seven orders of magnitude of unnecessary slack on half of every affected
sweep's samples. Demonstrated: a 1e-3 error injected into `set` **passes** the old flat
0.06 tolerance and **fails** the per-sample one.

Fixed with `rounding_aware(name, rounded_tol, round_output)` across the seven sweeps that
vary `round_output` and carry a loose bound: `two_nodes_gagge`, `pmv_ppd_iso`,
`pmv_ppd_ashrae`, `set_tmp`, `utci`, `use_fans_heatwaves`, `phs`. PHS keeps its `.rel()`
bounds — those cover error compounding over a 480-minute integration, which is a different
phenomenon from rounding and does not vanish when `round_output` is off.

### The hook is test-only and stays

It lives in `tests/support`, is inert unless the variable is set, and makes "can this
sweep still fail?" a repeatable check rather than a one-off. Keep it; use it whenever a
sweep's comparator or domain changes.

## Gap 3 — closed 2026-08-26

All 47 sweeps cross-referenced against their models' `Inputs`/`Options` fields. 43 were
complete; four left parameters at defaults, all now swept:
`two_nodes_gagge_sleep` (12), `phs` (`duration`, done earlier), `two_nodes_gagge`
(`w_max`, `calculate_ce`), `set_tmp` (`calculate_ce`).

**It found two real port bugs**, both in `two_nodes_gagge`'s `calculate_ce` path — an
upstream entry point that was unexercised because the flag sat at its `false` default.
Upstream reaches it via `_gagge_two_nodes_optimized_return_set`, whose signature stops at
`position`, so `max_skin_blood_flow`/`max_sweating`/`w_max` never arrive and fall back to
kernel defaults (90/500/computed); and the branch returns before the `if round_output:`
block, so it is never rounded. This port did the opposite on both counts. Invisible until
a cap binds — 0.077 °C of SET error at met=3.05. Pinned by
`test_two_nodes_gagge_calculate_ce_drops_caps_and_skips_rounding`.

**The generalisable lesson**, which is now three-for-three on this branch
(`quilt_thickness`, `duration`, `calculate_ce`): *a boolean option left at its default
does not merely go untested — it can hide an entire alternate code path, and in this crate
those paths reach different upstream functions with different signatures.* When adding a
sweep axis for a flag, check what upstream does on the other branch before assuming the
two are the same calculation with one value changed.

## What "confident" can honestly mean here

Not "there are no bugs". The reachable claim, and as of 2026-08-26 all four parts are true:

1. Every public function is compared against upstream over a randomised domain, at that
   sweep's own documented tolerance — and, where upstream did not round the sample, at
   1e-9.
2. Every output field is compared, including the discrete ones.
3. Every discrete output is compared at its band edges, against pythermalcomfort rather
   than against a transcribed constant.
4. Every one of the 47 sweeps has been shown to fail when its model is perturbed.

**What that still does not cover**, and should be the next item if this work continues:

- **Tolerances on the rounded half.** 43 `FieldCmp` sites remain looser than 0.011 for
  rounded samples. Each is defensible as "one rounding step", but no one has checked that
  each *is* one rounding step for that field's own rounding — `jos3_tolerance` does exactly
  this per field and is the model to copy.
- **Reachability.** Gap 3 established every *parameter* is varied; it did not establish
  every *branch* is entered. The `PET_MEASURE`-style coverage counters this document
  originally proposed would still be worth having.

