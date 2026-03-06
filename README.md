# thermalcomfort

[![Crates.io](https://img.shields.io/crates/v/thermalcomfort.svg)](https://crates.io/crates/thermalcomfort)
[![Documentation](https://docs.rs/thermalcomfort/badge.svg)](https://docs.rs/thermalcomfort)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Rust port of the [pythermalcomfort](https://pypi.org/project/pythermalcomfort/) Python package (v3.9.1) for thermal comfort calculations. All 38 core models, all utility functions, and all clothing databases are implemented with identical results to the Python reference.

This library is `no_std` compatible and can run in WASM environments, making it suitable for embedded systems, web applications, and resource-constrained environments.

For model documentation, parameters, and references, see the [pythermalcomfort documentation](https://pythermalcomfort.readthedocs.io/).

## Features

- **100% Feature Complete**: All 38 core models from pythermalcomfort v3.9.1
- **Identical Results**: Perfect accuracy compared to the Python reference for all models (see [Accuracy](#accuracy--validation) for the one `no_std` exception)
- **`no_std` compatible**: Works in embedded and WASM environments (default)
- **`std` feature**: Optional for perfect PET accuracy in extreme cold+wind conditions
- **Rigorously Validated**: 202 tests (88 unit + 56 Python comparison + 58 doctests)
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
- `Clo` - Clothing insulation (clo, tog, m²·K/W)
- `Met` - Metabolic rate (met, W/m², Btu/(h·ft²))
- `Sex` - Biological sex for physiological models

All types support automatic unit conversion through the type system, preventing errors like passing Fahrenheit where Celsius is expected.

### Optional `std` Feature

For applications requiring perfect Python accuracy matching in extreme PET conditions, enable the `std` feature:

```toml
[dependencies]
thermalcomfort = { version = "3.9.1", features = ["std"] }
```

This uses nalgebra for numerically stable linear algebra (LU decomposition), matching Python's scipy.optimize.fsolve. The trade-off is breaking `no_std` compatibility and a slightly larger binary (~100KB). Only needed when extreme cold+wind PET accuracy is critical (< 5°C, > 2 m/s).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
thermalcomfort = "3.9.1"
```

## Usage

### Basic PMV/PPD Calculation

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity, Met, Clo};

fn main() {
    let tdb = Temperature::from_celsius(25.0);
    let tr = Temperature::from_celsius(25.0);
    let rh = Humidity::from_percent(50.0);
    let v = Speed::from_meters_per_second(0.1);
    let met = Met::new(1.4);
    let clo = Clo::new(0.5);

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

    println!("Risk level: {:.1}", result.risk_level_interpolated); // 3.0 (Extreme)
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
use thermalcomfort::{Temperature, Speed, Humidity, Met, Clo};
use thermalcomfort::models::pet_steady;

fn main() {
    let result = pet_steady(
        Temperature::from_celsius(25.0),
        Temperature::from_celsius(27.0),
        Speed::from_meters_per_second(1.0),
        Humidity::from_percent(50.0),
        Met::new(1.5),
        Clo::new(1.0),
        Default::default()
    );
    println!("PET: {:.1}°C", result.pet);
}
```

### PHS (Predicted Heat Strain)

```rust
use thermalcomfort::{Temperature, Speed, Humidity, Met, Clo};
use thermalcomfort::models::{phs, PhsPosture, PhsOptions};

fn main() {
    let result = phs(
        Temperature::from_celsius(40.0),
        Temperature::from_celsius(40.0),
        Speed::from_meters_per_second(0.3),
        Humidity::from_percent(33.85),
        Met::new(2.5),
        Clo::new(0.5),
        PhsPosture::Standing,
        PhsOptions::default()
    );

    println!("Rectal temperature: {:.1}°C", result.t_re);
    println!("Max exposure (50%): {:.0} min", result.d_lim_loss_50);
    println!("Sweat loss: {:.0} g", result.sweat_loss_g);
}
```

### Unit Conversions

All measurement types support automatic unit conversion:

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity, Met, Clo};

fn main() {
    // Use any units - automatically converts internally
    let tdb = Temperature::from_fahrenheit(77.0);
    let tr = Temperature::from_celsius(25.0);
    let v = Speed::from_kilometers_per_hour(0.36);
    let rh = Humidity::from_percent(50.0);
    let met = Met::new(1.4);
    let clo = Clo::new(0.5);

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

All models produce identical results to pythermalcomfort v3.9.1. The only exception is the PET model under extreme cold+wind conditions when using the default `no_std` build:

| Condition | Python | Rust (`no_std`) | Rust (`std`) |
|-----------|--------|-----------------|--------------|
| Normal (25°C, 0.1 m/s, 50% RH) | 24.17°C | 24.17°C | 24.17°C |
| Hot (35°C, 1.0 m/s, 60% RH) | 36.26°C | 36.26°C | 36.26°C |
| Cold+wind (5°C, 2.0 m/s, 50% RH) | -0.46°C | 2.06°C | -0.46°C |

The `no_std` PET solver uses a custom Newton-Raphson method with a full 3x3 Jacobian, which is less numerically stable than Python's scipy HYBRD algorithm in extreme conditions. Enabling the `std` feature switches to a MINPACK-based HYBRD solver for perfect accuracy in all conditions.

All other models (PMV/PPD, UTCI, PHS, SET, Gagge variants, sports heat stress risk, etc.) produce identical results in both `no_std` and `std` builds.

## Testing

```bash
# Run all tests
cargo test

# Run only library tests (88 tests)
cargo test --lib

# Run documentation tests (58 tests)
cargo test --doc

# Run Python comparison tests (56 tests, requires pythermalcomfort)
cargo test --test python_comparison
```

## Standards Compliance

- **ISO 7730:2005** - PMV/PPD
- **ISO 7933:2004/2023** - Predicted Heat Strain
- **ASHRAE 55** - Thermal Environmental Conditions for Human Occupancy
- **ISO 7726:1998** - Instruments for measuring physical quantities
- **ISO 9920:2007** - Clothing insulation estimation
- **EN 16798-1** - Adaptive comfort

## Credits

Rust port of [pythermalcomfort](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort) (v3.9.1), developed by the Center for the Built Environment at UC Berkeley.

## License

MIT License - see LICENSE file for details.
