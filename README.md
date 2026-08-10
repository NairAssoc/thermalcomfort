# thermalcomfort

[![Crates.io](https://img.shields.io/crates/v/thermalcomfort.svg)](https://crates.io/crates/thermalcomfort)
[![Documentation](https://docs.rs/thermalcomfort/badge.svg)](https://docs.rs/thermalcomfort)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Rust port of the [pythermalcomfort](https://pypi.org/project/pythermalcomfort/) Python package (v4.4.0) for thermal comfort calculations. Every model, utility function and clothing database is implemented and verified against the Python reference, with two documented exceptions (see [Coverage](#coverage)).

This library is `no_std` compatible and can run in WASM environments, making it suitable for embedded systems, web applications, and resource-constrained environments.

For model documentation, parameters, and references, see the [pythermalcomfort documentation](https://pythermalcomfort.readthedocs.io/).

## Features

- **Near-complete coverage**: every pythermalcomfort v4.4.0 model except `JOS3`, with two documented gaps (see [Coverage](#coverage))
- **Identical Results**: verified against the Python reference by a randomised differential sweep over the full input space (see [Accuracy](#accuracy--validation) for the one `no_std` exception)
- **`no_std`**: one configuration, no std/no_std accuracy split. Verified on `wasm32-unknown-unknown` and bare-metal `thumbv7em-none-eabihf`
- **Rigorously Validated**: 302 tests (110 unit + 74 Python comparison + 65 doctests +
  46 differential sweeps + 7 harness self-tests). Every public function with a
  pythermalcomfort counterpart has a cross-library parity test, and all but one are also
  driven through the randomised sweep.
- **Type-safe**: All physical quantities use typed wrappers to prevent unit errors at compile time
- **Standards Compliant**: ISO 7730, ISO 7933, ASHRAE 55, EN 16798-1, ISO 9920

### Re-exported Types

The library re-exports the following types for convenience:

From the [`measurements`](https://crates.io/crates/measurements) crate:
- `Temperature` - Celsius, Fahrenheit, Kelvin, Rankine
- `Speed` - m/s, km/h, mph, knots, etc.
- `Humidity` - Relative humidity (0-100%)
- `Length` - meters, centimeters, feet, inches, etc.
- `Mass` - kilograms, pounds, etc.
- `Power` - watts, kilowatts, horsepower, etc.
- `Area` - m², ft², etc.
- `Pressure` - Pa, kPa, mmHg, atm, etc.

Defined in this crate:
- `TemperatureDelta` - A temperature *difference* (°C/K or °F). Distinct from `Temperature`,
  which is absolute: a change of 1 °C is a change of 1.8 °F, with no offset
- `AirPermeability` - Air permeability of clothing (l/(m²·s)), per ISO 11079
- `ClothingInsulation` - Clothing insulation (clo, tog, m²·K/W)
- `MetabolicRate` - Metabolic rate (met, W/m², Btu/(h·ft²))
- `Sex` - Biological sex for physiological models

All types support automatic unit conversion through the type system, preventing errors like passing Fahrenheit where Celsius is expected.

Physical quantities are newtypes on **inputs**; result structs return plain `f64`. Four input
parameters remain untyped because no suitable type exists yet: solar angles (`solar_gain`,
`transpose_sharp_altitude` — `measurements::Angle` is not re-exported), irradiance in W/m²
(`esi`, `solar_gain`), and the blood-flow and sweating caps in `use_fans_heatwaves`.

### The `std` feature (deprecated no-op)

There is no longer anything to enable. The crate has one configuration, and it is
`no_std`. `std` is retained as an empty feature so existing dependants that wrote
`features = ["std"]` keep building; it will go at the next upstream major.

Historically `std` swapped PET onto a nalgebra solver "for perfect accuracy in extreme
cold+wind". Two things ended that: the accuracy gap was closed by fixes to the solver
itself, so both implementations produced identical results; and nalgebra never needed
`std` in the first place — it has supported `no_std` since 0.15, and is now built against
`alloc` + `libm` like any other dependency here.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
thermalcomfort = "4.4.0"
```

## Usage

### Basic PMV/PPD Calculation

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};

fn main() {
    let tdb = Temperature::from_celsius(25.0);
    let tr = Temperature::from_celsius(25.0);
    let rh = Humidity::from_percent(50.0);
    let v = Speed::from_meters_per_second(0.1);
    let met = MetabolicRate::from_met(1.4);
    let clo = ClothingInsulation::from_clo(0.5);

    let vr = v_relative(v, met);

    let result = pmv_ppd_iso(tdb, tr, vr, rh, met, clo, Default::default());

    println!("PMV: {:.2}", result.pmv);  // ~0.17
    println!("PPD: {:.1}%", result.ppd); // ~5.6%
    println!("Thermal Sensation: {:?}", result.tsv);
}
```

### Sports Heat Stress Risk

```rust
use thermalcomfort::{Temperature, Speed, Humidity};
use thermalcomfort::models::sports_heat_stress_risk::{Sports, sports_heat_stress_risk};

fn main() {
    let result = sports_heat_stress_risk(
        Temperature::from_celsius(35.0),
        Temperature::from_celsius(35.0),
        Humidity::from_percent(40.0),
        Speed::from_meters_per_second(0.1),
        Sports::RUNNING,
    );

    println!("Risk level: {:.1}", result.risk_level_interpolated); // 2.1 (Moderate)
    println!("Recommendation: {}", result.recommendation);
}
```

### UTCI (Universal Thermal Climate Index)

```rust
use thermalcomfort::{Temperature, Speed, Humidity};
use thermalcomfort::models::utci;

fn main() {
    let result = utci(
        Temperature::from_celsius(25.0),
        Temperature::from_celsius(27.0),
        Speed::from_meters_per_second(1.0),
        Humidity::from_percent(50.0),
        Default::default()
    );
    println!("UTCI: {:.1}°C", result.utci);
    println!("Stress: {}", result.stress_category.as_str());
}
```

### PET (Physiological Equivalent Temperature)

```rust
use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
use thermalcomfort::models::pet_steady;

fn main() {
    let result = pet_steady(
        Temperature::from_celsius(25.0),
        Temperature::from_celsius(27.0),
        Speed::from_meters_per_second(1.0),
        Humidity::from_percent(50.0),
        MetabolicRate::from_met(1.5),
        ClothingInsulation::from_clo(1.0),
        Default::default()
    );
    println!("PET: {:.1}°C", result.pet);
}
```

### PHS (Predicted Heat Strain)

```rust
use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
use thermalcomfort::models::{phs, PhsPosture, PhsOptions};

fn main() {
    let result = phs(
        Temperature::from_celsius(40.0),
        Temperature::from_celsius(40.0),
        Speed::from_meters_per_second(0.3),
        Humidity::from_percent(33.85),
        MetabolicRate::from_met(2.5),
        ClothingInsulation::from_clo(0.5),
        PhsPosture::Standing,
        PhsOptions::default()
    );

    println!("Rectal temperature: {:.1}°C", result.t_re);
    println!("Max exposure (50%): {:.0} min", result.d_lim_loss_50);
    println!("Sweat loss: {:.0} g", result.sweat_loss_g);
}
```

### IREQ (Required Clothing Insulation, ISO 11079)

For cold environments: the clothing insulation required for thermal equilibrium, and how
long exposure can last when the clothing available is not enough.

```rust
use thermalcomfort::{Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
use thermalcomfort::AirPermeability;
use thermalcomfort::models::{ireq, IreqOptions, DurationLimitedExposure};

fn main() {
    let result = ireq(
        Temperature::from_celsius(-15.0),
        Temperature::from_celsius(-15.0),
        Speed::from_meters_per_second(2.0),
        Humidity::from_percent(55.0),
        MetabolicRate::from_met(175.0 / 58.15),
        ClothingInsulation::from_clo(2.8),
        AirPermeability::from_l_per_m2_s(50.0),
        Speed::from_meters_per_second(1.1),
        IreqOptions::default()
    );

    println!("Required insulation (minimal): {:.1} clo", result.ireq_min);
    println!("Required insulation (neutral): {:.1} clo", result.ireq_neutral);

    match result.dle_min {
        DurationLimitedExposure::Hours(h) => println!("Exposure limit: {h:.1} h"),
        DurationLimitedExposure::MoreThanEight => println!("Exposure limit: more than 8 h"),
        DurationLimitedExposure::NotApplicable => println!("Outside ISO 11079 limits"),
    }
}
```

### Unit Conversions

All measurement types support automatic unit conversion:

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};

fn main() {
    // Use any units - automatically converts internally
    let tdb = Temperature::from_fahrenheit(77.0);
    let tr = Temperature::from_celsius(25.0);
    let v = Speed::from_kilometers_per_hour(0.36);
    let rh = Humidity::from_percent(50.0);
    let met = MetabolicRate::from_met(1.4);
    let clo = ClothingInsulation::from_clo(0.5);

    let vr = v_relative(v, met);
    let result = pmv_ppd_iso(tdb, tr, vr, rh, met, clo, Default::default());
    println!("PMV: {:.2}", result.pmv);
}
```

### Clothing Insulation Lookups

```rust
use thermalcomfort::{clo_typical_ensemble, clo_individual_garment};
use thermalcomfort::utilities::clo_intrinsic_insulation_ensemble;

fn main() {
    let summer_clo = clo_typical_ensemble("Typical summer indoor clothing").unwrap();
    println!("Summer clothing: {} clo", summer_clo); // 0.5 clo

    let shirt = clo_individual_garment("Long-sleeve dress shirt").unwrap();
    let pants = clo_individual_garment("Thick trousers").unwrap();
    let underwear = clo_individual_garment("Men's underwear").unwrap();

    let garments = [shirt, pants, underwear];
    let total_clo = clo_intrinsic_insulation_ensemble(&garments);
    println!("Total ensemble: {:.2} clo", total_clo); // ~0.60 clo
}
```

## WASM Support

This library is `no_std` compatible and can be compiled to WebAssembly:

```bash
cargo build --target wasm32-unknown-unknown --release
```

## Accuracy & Validation

All models produce identical results to pythermalcomfort v4.4.0, in the one build
configuration the crate has. There is no accuracy trade-off to choose between.

Verification is a randomised differential sweep (see [Testing](#testing)) that drives
every model through pseudo-random input vectors covering its optional parameters, not
just its physical inputs, and compares every output field against Python. Divergences
found this way are fixed in the port; tolerances are only widened where the difference is
demonstrably float-representation noise, and the two places that needed a documented
exclusion say so in the test.

Earlier releases shipped a second, hand-written PET solver for `no_std` and documented it
as less accurate in extreme cold+wind. Both the second solver and the caveat are gone: the
underlying solver bugs were fixed, after which the two implementations agreed everywhere
measured, so the duplicate was deleted rather than kept as a choice.

## Coverage

Every public function is checked against pythermalcomfort by `make parity-coverage`, and
every one except `two_nodes_gagge_sleep` is additionally driven through the randomised
differential sweep. One gap is known:

| Gap | Status |
|-----|--------|
| `JOS3` | **Port in progress.** pythermalcomfort's 17-segment whole-body thermoregulation model; see `docs/worklist/parity/`. |

`two_nodes_gagge_sleep` is now a faithful port of the Yan et al. (2022) model: it simulates
the night minute by minute and takes a per-minute schedule for each driving variable, as
upstream does. It previously delegated to the standard Gagge model at a fixed 0.7 met, so
`quilt_thickness` had no effect at all. All ten output trajectories are compared against
pythermalcomfort to 1e-9 in `tests/python_comparison.rs`.

Outstanding work is tracked in `docs/worklist/parity/` and
`docs/superpowers/plans/outstanding-parity-work.md`.

## Testing

The parity tests compare this crate against the real `pythermalcomfort` package through
pyo3, so they need the **exact version this crate ports** to be importable. The crate
version is that version — they are kept in lockstep deliberately.

```bash
# One-time: create a venv holding pythermalcomfort==<crate version>
make setup-parity

# Run the whole suite, then confirm the crate still builds for a no_std target
make test

# Lint (fmt + clippy + parity coverage) followed by the full suite
make verify
```

`make` wires up the venv for you. To drive cargo directly, point `PYTHONPATH` at it:

```bash
export PYTHONPATH=$(ls -d .parity-venv/lib/python*/site-packages)

cargo test --lib                      # library tests
cargo test --doc                      # documentation tests
cargo test --test python_comparison   # hand-written Python parity tests
cargo test --test differential_sweep  # randomised differential sweep
```

Two guards keep the comparison honest:

- `test_pythermalcomfort_version_matches_crate` fails the suite if the importable
  `pythermalcomfort` is not the version being ported. Without it, a stale install makes
  every parity assertion silently meaningless — which is exactly what happened when CI
  sat pinned to 3.8.0 through four releases.
- `make parity-coverage` fails if any public function has no parity test. New functions
  must be compared against Python, not just unit-tested against transcribed constants.
  Functions with no Python counterpart go in `EXEMPT` in `scripts/check_parity_coverage.py`
  with a reason. `KNOWN_GAPS` is empty: adding to it is a regression, so write the test.

### Differential sweep

Beyond the hand-written parity cases, `tests/differential_sweep.rs` drives every model
through pseudo-random input vectors — including the optional parameters (`wme`, `p_atm`,
posture, blood-flow and sweating caps) that fixed cases leave at their defaults — and
compares every output field against Python.

```bash
make sweep                      # deep run, SWEEP_N=20000
SWEEP_N=500 cargo test --test differential_sweep
```

The sweep exists because "every function is called by a parity test" is not the same as
"every function is verified": a median hand-written case pinned 49 of 103 optional
parameters at their defaults, and real bugs lived in that residue.

Failures print the seed and a shrunk input vector. Reproduce with:

```bash
SWEEP_SEED=<seed> SWEEP_N=<n> cargo test --test differential_sweep -- --nocapture
```

Widening a tolerance to make a sweep pass is almost always wrong: the sweep exists to find
the differences that fixed cases miss.

When bumping to a new pythermalcomfort release, change the version in `Cargo.toml`, re-run
`make setup-parity`, and CI will follow automatically — it derives the pin from
`Cargo.toml` rather than hardcoding it.

## Standards Compliance

- **ISO 7730:2025** - PMV/PPD (formulae unchanged from ISO 7730:2005)
- **ISO 7933:2004/2023** - Predicted Heat Strain
- **ISO 11079:2007** - Required clothing insulation (IREQ) and duration limited exposure
- **ASHRAE 55** - Thermal Environmental Conditions for Human Occupancy
- **ISO 7726:1998** - Instruments for measuring physical quantities
- **ISO 9920:2007** - Clothing insulation estimation
- **EN 16798-1** - Adaptive comfort

## Credits

Rust port of [pythermalcomfort](https://github.com/pythermalcomfort/pythermalcomfort) (v4.4.0), developed by Federico Tartarini and Stefano Schiavon.

If you use this crate in your research, please cite the original work:

> Tartarini, F., Schiavon, S., 2020. pythermalcomfort: A Python package for thermal comfort research. SoftwareX 12, 100578. https://doi.org/10.1016/j.softx.2020.100578

## License

MIT License - see LICENSE file for details.
