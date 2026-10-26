# thermalcomfort

[![Crates.io](https://img.shields.io/crates/v/thermalcomfort.svg)](https://crates.io/crates/thermalcomfort)
[![Documentation](https://docs.rs/thermalcomfort/badge.svg)](https://docs.rs/thermalcomfort)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Rust port of the [pythermalcomfort](https://pypi.org/project/pythermalcomfort/) Python package (v4.6.2) for thermal comfort calculations. Every model, utility function and clothing database is implemented and verified against the Python reference (see [Coverage](#coverage) for the divergences that remain).

This library is `no_std` compatible and can run in WASM environments, making it suitable for embedded systems, web applications, and resource-constrained environments.

For model documentation, parameters, and references, see the [pythermalcomfort documentation](https://pythermalcomfort.readthedocs.io/).

## Features

- **Complete coverage**: every pythermalcomfort v4.6.2 model, `JOS3` included
- **Identical Results**: verified against the Python reference by a randomised differential sweep over the full input space (see [Coverage](#coverage) for the five edge-case divergences)
- **`no_std`**: one configuration, no std/no_std accuracy split. Verified on `wasm32-unknown-unknown` and bare-metal `thumbv7em-none-eabihf`
- **Rigorously Validated**: every public function with a pythermalcomfort counterpart has
  a cross-library parity test *and* is driven through the randomised differential sweep,
  and `make parity-coverage` fails the build if either stops being true. The code on this
  page is compiled by `cargo test --doc`, and every program under `examples/` is run by
  `make examples`, so neither can drift from the API.
- **Type-safe**: every input and result whose quantity has a type is that type —
  temperatures come back as `Temperature`, durations as `core::time::Duration`, clothing as
  `ClothingInsulation` — so a unit error is a compile error (see
  [Re-exported Types](#re-exported-types) for the rule)
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
- `Angle` - degrees, radians; used for solar geometry

Defined in this crate:
- `TemperatureDelta` - A temperature *difference* (°C/K or °F). Distinct from `Temperature`,
  which is absolute: a change of 1 °C is a change of 1.8 °F, with no offset
- `AirPermeability` - Air permeability of clothing (l/(m²·s)), per ISO 11079
- `ClothingInsulation` - Clothing insulation (clo, tog, m²·K/W)
- `MetabolicRate` - Metabolic rate (met, W/m², Btu/(h·ft²))
- `Age` - Age in years, shared by PET, the ridge-regression predictor and JOS3
- `Sex` - Biological sex for physiological models

All types support automatic unit conversion through the type system, preventing errors like passing Fahrenheit where Celsius is expected.

Also defined here and used on **results**, not just inputs:
- `HeatFluxDensity` - a density of heat flow rate (W/m²), distinct from `Power` over the
  whole body
- `CardiacIndex`, `ActivityRatio`, `BodyFat`, `BmrEquation` - JOS3's body parameters

The rule for what is typed, on inputs **and** outputs:

- A result or input is a newtype when a type for its quantity already exists in this crate
  or in `measurements`. A result is `Temperature` when it is a temperature — measured,
  equivalent or weighted: SET, PET, WBGT, apparent temperature, net effective temperature,
  wind-chill temperature, operative and mean radiant temperature. A result stays `f64` when
  it is a classification index whose value is read against bands — UTCI, humidex, heat
  index (both), discomfort index, THI, ESI — or is dimensionless (PMV, PPD, fractions,
  percentages, coefficients). UTCI additionally keeps its `Units` option because upstream
  rounds it in the output unit.
- Durations are `core::time::Duration`, inputs and outputs alike. Models that step in
  one-minute increments round the duration to whole minutes (at least one) and say so in
  the field's doc comment.
- Ages are `Age`.
- Quantities with no existing type (mass and volume flows such as kg/h/m² and L/h, thermal
  and evaporative resistances, specific enthalpy J/kg, W·min/m² accumulators, W/s rate
  limits) stay `f64` with the unit stated in the field's doc comment. There are no one-off
  newtypes for them.

Every model takes the parameters upstream requires followed by a defaulted `XOptions`
struct for the ones it defaults — mirroring pythermalcomfort's own split, so a call
transfers between the two libraries by reading. Required parameters are positional where
their types already tell them apart (`humidex(tdb, rh, options)`), and gathered into a
named `XInputs` struct, deliberately without a `Default`, only where two or more share a
type: `pmv_ppd_iso` takes two temperatures side by side, and `solar_gain` seven
consecutive `f64`, five of them fractions, which no type system distinguishes.

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
thermalcomfort = "4.6.2"
```

## Usage

The quickest start is PMV/PPD under ISO 7730:

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity, MetabolicRate, ClothingInsulation};
use thermalcomfort::models::pmv::PmvPpdInputs;

fn main() {
    let tdb = Temperature::from_celsius(25.0);
    let tr = Temperature::from_celsius(25.0);
    let rh = Humidity::from_percent(50.0);
    let v = Speed::from_meters_per_second(0.1);
    let met = MetabolicRate::from_met(1.4);
    let clo = ClothingInsulation::from_clo(0.5);

    let vr = v_relative(v, met);

    let result = pmv_ppd_iso(
        PmvPpdInputs {
            tdb,
            tr,
            vr,
            rh,
            met,
            clo,
        },
        Default::default(),
    );

    println!("PMV: {:.2}", result.pmv);  // ~0.17
    println!("PPD: {:.1}%", result.ppd); // ~5.6%
    println!("Thermal Sensation: {:?}", result.tsv);
}
```

Every public function is demonstrated in a runnable program under [`examples/`](examples/).
`make examples` builds and runs all of them, and `make parity-coverage` fails if a public
function appears in none, so they cannot drift from the API. Run one with
`cargo run --example <name>`:

| Example | Shows |
|---|---|
| `basic_pmv` | PMV/PPD under ISO 7730 and the relative air speed correction |
| `typed_api` | The typed-quantity API: inputs in any unit, results read back in any unit, ISO against ASHRAE PMV |
| `pmv_variants` | ASHRAE PMV with its options, the adaptive (`pmv_a`), expectancy (`pmv_e`) and ATHB variants, and the cooling effect of elevated air speed |
| `adaptive_comfort` | Adaptive comfort under ASHRAE 55 and EN 16798, from a running mean outdoor temperature |
| `two_node_models` | The Gagge two-node model, SET, the Ji variant, the sleep model's minute-by-minute night, and fan use in heatwaves |
| `jos3` | The 17-segment JOS3 model: build, advance through phases, read segment and whole-body results |
| `heat_strain` | PHS (ISO 7933), IREQ (ISO 11079), PET, and sports heat stress risk |
| `outdoor_comfort` | UTCI and WBGT, with the wet-bulb temperature they rest on |
| `stress_indices` | Heat index in three forms, humidex, discomfort index, apparent temperature, net effective temperature, THI, ESI, and wind chill in two forms |
| `work_capacity` | Work capacity under four models, and the ridge-regression skin and rectal temperature predictor |
| `solar_and_specialty` | Solar gain on the body, ankle draft, vertical temperature gradient, and sky-view fraction |
| `psychrometrics` | Saturation and partial pressures, wet bulb, dew point, enthalpy, body surface area |
| `environment` | A station wind scaled to pedestrian height over four terrains (and one where the profile is undefined), the running mean outdoor temperature, mean radiant and operative temperature, relative air speed, sky-vault view and sun position |

| `clothing` | Garment and ensemble lookups; intrinsic, total and dynamic insulation; air-layer and area factors; outdoor clothing estimate |

The modules mirror pythermalcomfort's: `models`, `environment`, `psychrometrics`, `clothing` and `utilities`. The names 4.6.0 moved out of `utilities` are still reachable there, deprecated, for two minor releases, as upstream's are.

## WASM Support

This library is `no_std` compatible and can be compiled to WebAssembly:

```bash
cargo build --target wasm32-unknown-unknown --release
```

## Accuracy & Validation

All models produce results identical to pythermalcomfort v4.6.2 across the swept input
space, in the one build configuration the crate has. There is no accuracy trade-off to
choose between. The five edge-case divergences that remain are listed under
[Coverage](#coverage); each concerns an input at the boundary of what the model can answer.

Verification is a randomised differential sweep (see [Testing](#testing)) that drives
every model through pseudo-random input vectors covering its optional parameters, not
just its physical inputs, and compares every output field against Python. Divergences
found this way are fixed in the port rather than tolerated: a tolerance is widened only
where the residue is demonstrably float-representation noise, and each such place says so
in the test and states what resolution was given up. Where a model rounds its own outputs
upstream, the sweep cannot see drift below that rounding step — which is why run lengths
are chosen so accumulated drift would exceed it.

Earlier releases shipped a second, hand-written PET solver for `no_std` and documented it
as less accurate in extreme cold+wind. Both the second solver and the caveat are gone: the
underlying solver bugs were fixed, after which the two implementations agreed everywhere
measured, so the duplicate was deleted rather than kept as a choice.

## Coverage

Every public function is checked against pythermalcomfort by `make parity-coverage`, and
every one is driven through the randomised differential sweep. The checker runs in both
directions, so an upstream release growing a model this port lacks fails the build.

Five divergences are known and deliberate. Each is a case where the two libraries disagree
about an input at the edge of what the model can answer; none affects ordinary results.

| Divergence | Detail |
|---|---|
| `pet_steady` returns `NaN` where scipy returns a number | Rust's 3-node Newton demands a 1e-5 residual. scipy's `fsolve` stops on step size and accepts points whose energy balance is still ~0.3 W/m² out. Where no root meets the stricter bar this port reports `NaN` rather than a wrong number. |
| `two_nodes_gagge_sleep` returns infinity where Python raises `OverflowError` | Hot, humid and heavily quilted, the model's own exponentials overflow. Confined to inputs upstream declines to answer at all. |
| `JOS3` drift below one rounding step is invisible to the sweep | Upstream rounds its own outputs (2 dp for most fields), so there is no unrounded reference to compare against. |
| `adaptive_en` acceptability flags can flip on an input sitting exactly on a comfort bound | `Temperature` stores kelvin, so a Celsius input that is not exactly representable comes back changed by up to 2.3e-14 °C (one ULP of the kelvin value; integers and half-integers are exact). EN compares the operative temperature directly against bounds computed from the running mean, so `tdb = tr = 24.1` at `t_running_mean = 10` is category I in Python and not here. Accepted as irrelevant at that magnitude. |
| About one sweep sample in 3000 is skipped as ill-conditioned in `JOS3`, `two_nodes_gagge_ji` and `two_nodes_gagge_sleep` | Upstream's own answer is not a stable function of the inputs there: `JOS3`'s chest segment can sit exactly on the wettedness saturation clip, and the two minute-by-minute Gagge integrations can pass through a locally unstable step. A sample is skipped only after re-running Python against itself with the inputs moved one ULP shows upstream's answer moving by an amount comparable to the Rust–Python difference, and each sweep fails if more than 0.5 % of samples are skipped. |

`two_nodes_gagge_sleep` is a faithful port of the Yan et al. (2022) model: it simulates the
night minute by minute and takes a per-minute schedule for each driving variable, as upstream
does. It previously delegated to the standard Gagge model at a fixed 0.7 met, so
`quilt_thickness` had no effect at all. All ten output trajectories are compared against
pythermalcomfort to 1e-9.

`JOS3` is ported in full — all 17 body segments, the 85-node thermal network and the
per-timestep dense solve — with every output field compared against upstream and driven
through the sweep.

## Testing

The parity tests compare this crate against the real `pythermalcomfort` package through
pyo3, so they need the **exact version this crate ports** to be importable. The crate
version is that version — they are kept in lockstep deliberately.

```bash
# One-time: create a venv holding pythermalcomfort==<crate version>, and install the
# no_std targets and beta toolchain that verify needs
make setup-parity
make setup-toolchain

# The quick loop: the full suite against the reference
make test

# Everything CI runs, in one command: lint, cargo check, the suite, the no_std and
# wasm builds, the differential sweep at CI's depth, and lint plus the suite again on
# the beta toolchain. Half an hour. A tree that passes here cannot fail CI for a
# reason the tree controls, because CI's jobs invoke these same make targets.
make verify
```

`make verify` fails rather than skips when a target or toolchain is missing: a silent
skip once let it pass locally on a tree CI rejected. Two CI conditions it cannot
reproduce: the operating system, for which it stands in by giving the parity-coverage
script the cp1252 console the Windows runners have, and the Python minor version, which is
whatever `python3` is locally and 3.11 on CI. Re-run `make setup-parity` now and then; it
upgrades the reference venv's dependencies to what a fresh CI runner would resolve.

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

Rust port of [pythermalcomfort](https://github.com/pythermalcomfort/pythermalcomfort) (v4.6.2), developed by Federico Tartarini and Stefano Schiavon.

If you use this crate in your research, please cite the original work:

> Tartarini, F., Schiavon, S., 2020. pythermalcomfort: A Python package for thermal comfort research. SoftwareX 12, 100578. https://doi.org/10.1016/j.softx.2020.100578

## License

MIT License - see LICENSE file for details.
