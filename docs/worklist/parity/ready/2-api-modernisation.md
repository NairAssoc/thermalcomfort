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
