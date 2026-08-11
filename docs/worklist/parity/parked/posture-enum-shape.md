# DECISION NEEDED: what type should `Jos3Conditions.posture` be?

**One question, answerable in one line. Everything needed to answer it is below; nothing
needs to be looked up, run, or asked.**

> Should JOS3's public posture field stay `utilities::Posture` (7 variants, 2 rejected at
> runtime), or become a narrow `Jos3Posture` (3 variants, nothing to reject)?

**My recommendation: switch to `Jos3Posture`.** Reasons under "Why B" below. It is a ~40-line
change, fully specified, and I did not make it unilaterally only because this branch already
went the other way once and a second flip on my own judgement is not mine to make.

---

## State right now (committed, green, shippable as-is)

`Jos3Conditions.posture: crate::utilities::Posture` — the crate-wide 7-variant enum
(Standing, Sitting, Sedentary, Reclining, Lying, Supine, Crouching). JOS3 maps them exactly
as upstream's setter does (`jos3.py:1471-1491`):

    Standing            -> Standing
    Sitting | Sedentary -> Sitting
    Lying   | Supine    -> Lying
    Reclining, Crouching -> Err(Jos3Error::UnsupportedPosture)

This shape was adopted in 391a9d0 to fix a hard blocker: JOS3's internal 3-variant enum lived
in a `pub(crate)` module, so **no external caller could name a posture at all** and every
simulation built outside the crate was permanently stuck Standing. That fix was necessary and
is correct. The question is only whether the *type* it chose is the right long-term one.

## Option A — keep `utilities::Posture` (status quo, zero work)

- One posture vocabulary across the whole crate; callers do not learn a second name.
- Two of the seven variants are accepted by the type and rejected at runtime.
- `advance()` keeps a failure mode that exists purely because the type is wider than the
  model.

## Option B — narrow `Jos3Posture` (recommended, ~40 lines)

Mirrors the **existing precedent in this crate**: `PhsPosture` (`src/models/phs.rs:99`) is
exactly this — a narrow 3-variant enum for a model that accepts three postures, and Python's
own `_posture_to_code` *raises* for the rest.

**Blast radius, measured:** 6 `UnsupportedPosture` sites, 35 posture mentions in `jos3.rs`,
1 posture use in the test suite. Nothing outside `src/models/jos3/` and one sweep axis.

**Why B:**

1. **The variant sets genuinely differ, and neither is a subset.** PHS accepts
   Standing/Sitting/Crouching; JOS3 accepts Standing/Sitting/Lying. A shared enum cannot be
   right for both.
2. **"Standing" is not one constant.** The radiating-area ratio is 0.73 in base Gagge, 0.77
   in Gagge-JI and PHS; `f_eff` is 0.696 in PET and `solar_gain`; JOS3's is not a scalar at
   all but a 17-element per-segment array. A shared posture type invites sharing a constant
   that must not be shared — this crate already had a `Posture::radiation_area_ratio()`
   helper doing exactly that, deleted in f6d3d9a.
3. **It deletes a runtime error instead of documenting one.** `UnsupportedPosture` exists
   only because the type admits values the model does not. Narrowing moves that to compile
   time, which is what `PhsPosture` already does.
4. Nothing is lost: `Sedentary` and `Supine` are pure aliases upstream, so a 3-variant enum
   expresses everything pythermalcomfort's JOS3 can express.

**Against B:** two posture enums in the public API rather than one, and a caller converting
between models writes a `match`. (They already do for `PhsPosture`.)

## What happens on each answer

- **"A / leave it"** — nothing to do. The branch is already in this state and green.
- **"B / narrow it"** — mechanical and fully specified: add `pub enum Jos3Posture { Standing,
  Sitting, Lying }` beside the existing `PhsPosture` precedent, change
  `Jos3Conditions.posture`, delete `Jos3Error::UnsupportedPosture` and its 6 sites, drop the
  now-impossible error branch from `advance()`, and narrow the sweep's posture axis from 5
  values to 3. Est. ~40 lines, no numerics change — Standing/Sitting/Lying must stay
  bit-identical, and the two JOS3 regression tests
  (`operative_temp_search_crosses_the_iso_limit_like_python`,
  `sixty_steps_at_reference_environment_matches_python`) pin that.

## Not in scope for this decision

`utilities::Posture`, `pet::Posture` and `PhsPosture` were also considered for a merge. The
analysis concluded against it for the same reasons as (1) and (2) above, and no work is
pending on them. This item is only about JOS3's field.
