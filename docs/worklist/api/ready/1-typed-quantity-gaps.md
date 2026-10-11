# DECISION NEEDED: the typed-quantity API has untyped corners

Found while writing one example per public function (`examples/`). None is a parity
defect; each is a place where the "physical quantities are newtypes on inputs and outputs"
promise in the README is not kept, so a caller has to re-wrap or guess units.

## Return types

- `clo_dynamic_ashrae` returns `ClothingInsulation`; `clo_dynamic_iso`,
  `clo_total_insulation`, `clo_insulation_air_layer`, `clo_intrinsic_insulation_ensemble`,
  `clo_tout`, `clo_typical_ensemble` and `clo_individual_garment` return bare `f64` clo.
  Feeding any of the seven into a `ClothingInsulation` input means re-wrapping by hand.

## Input fields

- `EsiInputs::sol_radiation_global` and `AtOptions::q` are `f64` W/m²;
  `SolarGainInputs::sol_radiation_dir` is `HeatFluxDensity`.
- `RidgeRegressionInputs::age` and `PetOptions::age` are `f64`; `Jos3Builder::age` is `i32`.
- `GaggeTwoNodesJiOptions::length_time_simulation` is `usize` minutes;
  `GaggeTwoNodesSleepOptions::ltime` is `u32`.
- JOS3's `Jos3Conditions` fields (`tdb`, `tr`, `rh`, `v`, `clo`) are `PerBodyPart<f64>`
  in °C, %, m/s and clo, with no newtype anywhere in the per-segment path.
- `transpose_sharp_altitude(sharp, altitude)` takes its two `Angle`s in the opposite order
  to `SolarGainInputs { sol_altitude, sharp, .. }`, so a swap compiles.

## Naming

- `Jos3Model::bmr()` is W/m², per its doc comment; the name reads as whole-body watts.

## Behaviour worth a doc note rather than a change

- `vertical_tmp_grad_ppd` returns a negative PPD for small gradients (-1.6 % at 3 K). It is
  the ASHRAE 55 offset-logistic fit and matches pythermalcomfort; a user will read it as a
  bug unless the result's doc says so.

## Options

- **A — do the consistency pass.** Return `ClothingInsulation` from all eight clothing
  functions, type the radiation and duration fields, pick one integer type for age, and
  consider a `PerBodyPart<Temperature>` for JOS3. Breaking, so it belongs to a major bump
  — the next port is one.
- **B — leave the surface, document the corners** in each function's rustdoc.

Recommendation: A, folded into the next upstream port, since that bump is breaking anyway.
