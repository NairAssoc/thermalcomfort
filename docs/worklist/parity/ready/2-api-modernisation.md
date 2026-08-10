# API modernisation — named input structs + output newtypes

Decided in `docs/superpowers/plans/outstanding-parity-work.md` §1.

- Rule 1: every dimensioned/bounded value is a newtype, on inputs *and* outputs.
- Rule 2: every model takes a named `XInputs` struct (no `Default`) plus the existing
  defaulted `XOptions`.
- Outputs: absolute temps → `Temperature`; temp *differences* → `TemperatureDelta`;
  W/m² → `HeatFluxDensity`; genuinely dimensionless → stay `f64`.
- **Never re-wrap an `f64` into a newtype mid-calculation** — `Temperature` stores kelvin
  and the round trip loses the last ULP. That cost a 2 °C JOS3 bug (28adfa8). In-crate
  callers holding `f64` use a `*_celsius` entry point.

**Done:** `solar_gain`, `ankle_draft`, `vertical_tmp_grad_ppd`, `cooling_effect`, `wbgt`,
the PMV family, `clo_dynamic_ashrae`/`clo_dynamic_iso`.

A full per-function conversion plan (current signature, Python signature including body
`kwargs.pop`, proposed Inputs/Options fields, output newtypes, difficulty) was produced and
is batched below. Batches 1-5 do not share source files, but **all five touch
`tests/python_comparison.rs` and `tests/differential_sweep.rs`, so those two files must be
owned by one agent at a time.**

| Batch | Files | Difficulty |
|---|---|---|
| 1 | `two_nodes_gagge.rs`, `set_tmp.rs`, `use_fans_heatwaves.rs` (+ `cooling_effect.rs` call sites) | MODERATE → TRICKY |
| 2 | `phs.rs`, `sports_heat_stress_risk.rs`, `ireq.rs` | TRIVIAL → TRICKY |
| 3 | `psychrometrics.rs`, `adaptive.rs`, `thermal_indices.rs`, `heat_index_lu.rs` | TRIVIAL → MODERATE |
| 4 | `pet.rs`, `utci.rs`, `work_capacity.rs`, `ridge_regression.rs` | TRIVIAL → MODERATE |
| 5 | `utilities.rs`, `specialty.rs` (`f_svv`, `transpose_sharp_altitude`) | TRIVIAL → MODERATE |

## New defects the planning pass turned up

Fold each into its batch; none is a pure API reshape.

1. **`two_nodes_gagge_ji` takes `relative_humidity`; Python takes `vapor_pressure` (torr)
   directly.** Rust computes the vapour pressure internally, so a caller cannot supply one
   measured directly — the Rust API is *less* general than upstream and forces
   `vapor_pressure == rh·p_sat_torr(tdb)/100`. Same family as the invented-parameter
   findings, opposite sign.
2. **`two_nodes_gagge(calculate_ce=true)` diverges.** Python's `calculate_ce` branch
   (`two_nodes_gagge.py:129-142`) hardcodes `position=1` (standing) regardless of the
   caller's posture; Rust always honours `options.posture`. Masked today only because every
   caller passing `calculate_ce: true` also happens to pass `Standing`. Force Standing
   internally when `calculate_ce` is set.
3. **`heat_index_lu` has no `round_output`** and rounds unconditionally — the third
   instance of the hardcoded-rounding defect after `solar_gain` and `vertical_tmp_grad_ppd`.
4. **`units` is missing from `adaptive_ashrae`, `adaptive_en` and `utci`.** Typed inputs
   make the input half moot, but upstream *rounds in the output unit*, so the output half is
   a real divergence. Unlike `cooling_effect`'s IP branch this is a genuine affine
   conversion. Note `adaptive_ashrae` and `adaptive_en` round in a different order from each
   other — check each.
5. **`pet_steady` cannot express `"standing, forced convection"`** (`pet_steady.py:329-330`,
   `hc = 8.6·v^0.513`). A whole convection branch is unreachable. Add
   `forced_convection: bool` to `PetOptions` rather than a third `Posture` variant.
6. **`humidex_masterson` is a separate Rust function** where Python has
   `humidex(model="rana"|"masterson")`. Merge behind a `HumidexModel` option.
7. **Required-in-Rust, defaulted-in-Python:** `psy_ta_rh.p_atm`;
   `mean_radiant_temperature`'s `d`/`emissivity`/`use_iso`; `operative_temperature`'s
   `use_ashrae`; `body_surface_area.formula`; `running_mean_outdoor_temperature.alpha`
   (which also lacks `units`); `work_capacity_dunne`/`_hothaps`'s `work_intensity`.
8. **`Posture::radiation_area_ratio()` (`src/utilities.rs:70-77`) is dead code** — zero
   callers, and actively misleading, since it encodes one ratio where the crate needs three
   different ones. Delete.
9. **Doc-comment unit bug:** `GaggeTwoNodesResult.q_sensible`/`q_skin`/`q_res` and the same
   fields on `UseFansHeatwavesResult` are commented `(W)` but are W/m². Fix when they become
   `HeatFluxDensity`.
10. **`two_nodes_gagge_ji`'s `heapless::Vec<f64, 120>`** silently drops pushes past 120
    (`let _ = ...push(...)`). Becomes a live bug the moment `length_time_simulation` is
    exposed; move to `alloc::Vec` as `two_nodes_gagge_sleep` did.

## OPEN DECISION: the four posture enums

Earlier notes here said to collapse `utilities::Posture`, `pet::Posture`, `phs::PhsPosture`
and JOS3's into one. **The planning pass recommends the opposite, with evidence, and it is
persuasive:**

- The variant *sets* genuinely differ. PHS accepts Standing/Sitting/Crouching and its Python
  `_posture_to_code` *raises* for anything else; JOS3 accepts Standing/Sitting/Lying and its
  whitelist raises for `reclining`/`crouching`. Neither is a subset of the other.
- "Standing" is not one constant. `a_r_du`/radiating-area is **0.73** in the base Gagge
  model (`two_nodes_gagge.rs:296-303`), **0.77** in Gagge-JI (`:828-830`) and PHS
  (`phs.rs:322-326`), and `f_eff` is **0.696** in PET and `solar_gain`. JOS3's is not a
  scalar at all — it is a 17-element per-segment array.
- Narrow enums make illegal states unrepresentable; one shared enum reintroduces them and
  needs runtime rejection to match Python's `ValueError`, which is strictly weaker.

**This branch has already gone the other way for JOS3** (391a9d0): `Jos3Conditions.posture`
takes `utilities::Posture` and rejects Reclining/Crouching at runtime via
`Jos3Error::UnsupportedPosture`. That was done to fix a hard blocker — the internal enum was
unreachable from outside the crate, so posture could not be set at all — and it is correct
and tested, but it is the *shared-enum* shape the analysis argues against.

The alternative is to re-export JOS3's narrow three-variant enum as a public `Jos3Posture`,
exactly mirroring the existing public `PhsPosture`, and drop `UnsupportedPosture`. That is
more consistent with the crate as it stands and moves the error to compile time.

Both are defensible and both match Python behaviourally (upstream's `sedentary`/`supine` are
pure aliases, so nothing is lost by not spelling them). **This is a public-API shape call on
a major bump — worth a human decision rather than another flip.**
