# API modernisation — named input structs + output newtypes

Decided in `docs/superpowers/plans/outstanding-parity-work.md` §1. ~40 signatures, ~40
result structs.

- Rule 1: every dimensioned/bounded value is a newtype, on inputs *and* outputs.
- Rule 2: every model takes a named `XInputs` struct (no `Default`) plus the existing
  defaulted `XOptions`.
- Outputs: absolute temps → `Temperature`; temp *differences* → `TemperatureDelta`
  (wrong today); W/m² → `HeatFluxDensity` (new); genuinely dimensionless → stay `f64`.
- New types: `HeatFluxDensity`, `CardiacIndex`, `ActivityRatio`, `BodyFat`, `BmrEquation`.
  `TemperatureDelta` already exists (`src/lib.rs:79`).

Must also close every gap item 1 finds — the input struct's fields are Python's parameter
list, so a missing parameter is a missing field.

## Consolidate the posture enums

The crate now has **four** posture types: `utilities::Posture` (Standing, Sitting,
Sedentary, Reclining, Lying, Supine), `pet::Posture`, `phs::PhsPosture`, and
`jos3::thermoregulation::Posture` (Standing, Sitting, Lying). The JOS3 one was added
following the precedent of the other two, which is a fair reading of the existing code —
the problem is the precedent, not that change.

`utilities::Posture` is a superset of every variant the others need, and upstream accepts
the same posture strings across all these models, so one enum is both simpler and closer to
Python. Collapsing the four is a breaking change to three public signatures, which is
exactly what this branch is for; it belongs here rather than as an ad-hoc fix, because it
has to be decided once and applied everywhere.

Check while doing it whether each model's *behavioural* collapsing survives: JOS3 treats
sedentary as sitting and supine as lying, and PET/PHS may differ. Where they differ, that
is a real distinction and the match arms must preserve it — do not flatten behaviour to
make the types line up.
