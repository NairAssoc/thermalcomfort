# thermalcomfort

[![Crates.io](https://img.shields.io/crates/v/thermalcomfort.svg)](https://crates.io/crates/thermalcomfort)
[![Documentation](https://docs.rs/thermalcomfort/badge.svg)](https://docs.rs/thermalcomfort)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Rust port of the [pythermalcomfort](https://pypi.org/project/pythermalcomfort/) Python package for thermal comfort calculations.

This library is `no_std` compatible and can run in WASM environments, making it suitable for embedded systems, web applications, and resource-constrained environments.

## Implementation Status

**33/37 core models implemented (89%)** from pythermalcomfort v3.8.0
**All utility functions and clothing databases implemented (100%)**

### Implemented Models ✅

All core thermal comfort models, heat/cold stress indices, psychrometric functions, and utility functions are complete:

- **PMV/PPD variants**: ISO 7730, ASHRAE 55, Adaptive (pmv_a), Expectancy (pmv_e), ATHB
- **Thermoregulation**: Two-node Gagge model, SET calculation, Two-node Gagge sleep variant (simplified)
- **Adaptive models**: ASHRAE 55, EN 16798-1
- **Outdoor comfort**: UTCI, WBGT
- **Heat stress**: Heat Index (Rothfusz, Lu & Romps), Humidex, THI, Discomfort Index, AT, NET, ESI, Work Capacity models, Use Fans Heatwaves
- **Cold stress**: Wind Chill Index, Wind Chill Temperature
- **Specialty models**: Solar gain, Ankle draft, Vertical temperature gradient
- **Utilities**: All psychrometric functions, all clothing insulation functions, clo_tout, clothing databases (9 typical ensembles, 56 individual garments)

### Partially Implemented

- **Two-nodes Gagge sleep**: Simplified steady-state version implemented. Full time-series simulation available in Python pythermalcomfort.

### Not Implemented (4 complex models)

The following models are not yet implemented due to their complexity (averaging 600+ lines each):

- **PHS**: Predicted Heat Strain ISO 7933 (715 lines)
- **PET Steady**: Physiological Equivalent Temperature (493 lines)
- **Ridge Regression**: ML-based rectal/skin temperature prediction (467 lines)
- **Two-nodes Gagge JI**: Ji variant for older individuals (453 lines)

These models can be added in future releases if needed.

## Features

- **`no_std` compatible**: Works in embedded and WASM environments
- **Comprehensive models**: PMV/PPD, psychrometrics, and more
- **Well-tested**: Includes comparison tests against the original Python implementation
- **Type-safe**: Leverages Rust's type system for safe thermal comfort calculations
- **Unit handling**: Uses the `measurements` crate for type-safe physical quantities with automatic unit conversion
- **Clear API**: Descriptive parameter names and strongly-typed interfaces

## Supported Models

### Core Thermal Comfort Models

- **PMV/PPD (Predicted Mean Vote / Predicted Percentage Dissatisfied)**
  - ISO 7730:2005 standard
  - ASHRAE 55 standard (with cooling effect correction)
- **SET (Standard Effective Temperature)**
  - Two-node Gagge thermoregulation model
  - SET calculation with standard applicability limits
- **Adaptive Comfort Models**
  - ASHRAE 55 adaptive model
  - EN 16798-1 adaptive model
  - Running mean outdoor temperature calculation
- **UTCI (Universal Thermal Climate Index)**
  - Comprehensive outdoor thermal comfort index
  - 6th-order polynomial regression model
  - Thermal stress categories from extreme cold to extreme heat

### Heat Stress Indices

- **WBGT (Wet Bulb Globe Temperature)** - ISO 7243:2017
- **Heat Index (Rothfusz)** - NWS heat index with stress categories
- **Humidex** - Canadian humidity index (Rana and Masterson models)
- **THI (Temperature-Humidity Index)**
- **Discomfort Index (DI)** - Effective temperature for warm environments

### Cold Stress Indices

- **Wind Chill Index (WCI)** - ASHRAE 2017
- **Wind Chill Temperature (WCT)** - North American standard

### Psychrometric Functions

- Saturation vapor pressure (Pa and torr)
- Wet bulb temperature
- Dew point temperature
- Mean radiant temperature (ISO and Mixed Convection methods)
- Operative temperature
- Air enthalpy

### Clothing Insulation Functions

- Dynamic clothing insulation (ASHRAE 55 and ISO 9920:2007)
- Clothing area factor
- Total insulation of clothing ensemble
- Boundary air layer insulation
- **Clothing databases**:
  - 9 typical ensembles (e.g., "Typical summer indoor clothing" → 0.5 clo)
  - 56 individual garments (e.g., "Long-sleeve dress shirt" → 0.25 clo)
  - Intrinsic insulation calculation from garment lists

### Utility Functions

- Relative air speed calculation
- Running mean outdoor temperature
- Body postures (standing, sitting, etc.)
- Temperature and unit conversions
- Numerical methods (Brent's root finding)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
thermalcomfort = "3.8.0"
```

## Usage

### Basic PMV/PPD Calculation

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity};

fn main() {
    let tdb = 25.0;  // dry bulb temperature [°C]
    let tr = 25.0;   // mean radiant temperature [°C]
    let rh = 50.0;   // relative humidity [%]
    let v = 0.1;     // air speed [m/s]
    let met = 1.4;   // metabolic rate [met]
    let clo = 0.5;   // clothing insulation [clo]

    // Calculate relative air speed (accounts for body movement)
    let vr = v_relative(v, met);

    // Calculate PMV and PPD
    let result = pmv_ppd_iso(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(vr),
        Humidity::from_percent(rh),
        met,
        clo,
        Default::default()
    );

    println!("PMV: {:.2}", result.pmv);  // ~0.17
    println!("PPD: {:.1}%", result.ppd); // ~5.6%
    println!("Thermal Sensation: {:?}", result.tsv);
}
```

### Psychrometric Calculations

```rust
use thermalcomfort::psychrometrics::psy_ta_rh;

fn main() {
    let tdb = 25.0;  // dry bulb temperature [°C]
    let rh = 50.0;   // relative humidity [%]
    let p_atm = 101325.0;  // atmospheric pressure [Pa]

    let psychro = psy_ta_rh(tdb, rh, p_atm);

    println!("Wet bulb temp: {:.1}°C", psychro.t_wb);  // ~17.7°C
    println!("Dew point: {:.1}°C", psychro.t_dp);      // ~13.9°C
    println!("Humidity ratio: {:.4}", psychro.hr);
}
```

### Custom PMV/PPD Options

```rust
use thermalcomfort::{pmv_ppd_iso, Temperature, Speed, Humidity};
use thermalcomfort::models::PmvPpdOptions;

fn main() {
    let options = PmvPpdOptions {
        wme: 0.0,              // external work [met]
        limit_inputs: false,   // don't limit to standard ranges
        round_output: true,    // round output values
    };

    let result = pmv_ppd_iso(
        Temperature::from_celsius(30.0),
        Temperature::from_celsius(30.0),
        Speed::from_meters_per_second(0.1),
        Humidity::from_percent(50.0),
        1.2,
        0.5,
        options
    );
    println!("PMV: {:.2}", result.pmv);
}
```

### Clothing Insulation Lookups

```rust
use thermalcomfort::{clo_typical_ensemble, clo_individual_garment};
use thermalcomfort::utilities::clo_intrinsic_insulation_ensemble;

fn main() {
    // Look up typical ensemble
    let summer_clo = clo_typical_ensemble("Typical summer indoor clothing").unwrap();
    println!("Summer clothing: {} clo", summer_clo); // 0.5 clo

    // Look up individual garments
    let shirt = clo_individual_garment("Long-sleeve dress shirt").unwrap();
    let pants = clo_individual_garment("Thick trousers").unwrap();
    let underwear = clo_individual_garment("Men's underwear").unwrap();

    // Calculate total ensemble insulation
    let garments = [shirt, pants, underwear];
    let total_clo = clo_intrinsic_insulation_ensemble(&garments);
    println!("Total ensemble: {:.2} clo", total_clo); // ~0.60 clo
}
```

### Standard Effective Temperature (SET)

```rust
use thermalcomfort::{Temperature, Speed, Humidity};
use thermalcomfort::models::set_tmp;

fn main() {
    let tdb = 25.0;  // dry bulb temperature [°C]
    let tr = 25.0;   // mean radiant temperature [°C]
    let v = 0.3;     // air speed [m/s]
    let rh = 50.0;   // relative humidity [%]
    let met = 1.2;   // metabolic rate [met]
    let clo = 0.5;   // clothing insulation [clo]

    let set = set_tmp(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(v),
        Humidity::from_percent(rh),
        met,
        clo,
        Default::default()
    );
    println!("SET: {:.1}°C", set);  // Standard Effective Temperature
}
```

### Cooling Effect

```rust
use thermalcomfort::{Temperature, Speed, Humidity};
use thermalcomfort::models::cooling_effect;

fn main() {
    let tdb = 28.0;  // dry bulb temperature [°C]
    let tr = 28.0;   // mean radiant temperature [°C]
    let vr = 0.8;    // relative air speed [m/s]
    let rh = 50.0;   // relative humidity [%]
    let met = 1.2;   // metabolic rate [met]
    let clo = 0.5;   // clothing insulation [clo]

    // Calculate temperature reduction equivalent to the elevated air speed
    let ce = cooling_effect(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(vr),
        Humidity::from_percent(rh),
        met,
        clo,
        Default::default()
    );
    println!("Cooling effect: {:.2}°C", ce);
}
```

### UTCI (Universal Thermal Climate Index)

```rust
use thermalcomfort::{Temperature, Speed, Humidity};
use thermalcomfort::models::utci;

fn main() {
    let tdb = 25.0;  // dry bulb temperature [°C]
    let tr = 27.0;   // mean radiant temperature [°C]
    let v = 1.0;     // wind speed at 10m [m/s]
    let rh = 50.0;   // relative humidity [%]

    let result = utci(
        Temperature::from_celsius(tdb),
        Temperature::from_celsius(tr),
        Speed::from_meters_per_second(v),
        Humidity::from_percent(rh),
        Default::default()
    );
    println!("UTCI: {:.1}°C", result.utci);
    println!("Stress: {}", result.stress_category.as_str());
    // Output: UTCI: 25.2°C, Stress: no thermal stress
}
```

### Unit Conversions with Measurements

The measurement types support automatic unit conversions:

```rust
use thermalcomfort::{pmv_ppd_iso, v_relative, Temperature, Speed, Humidity};

fn main() {
    // Use any temperature or speed units - automatically converts internally
    let tdb = Temperature::from_fahrenheit(77.0);  // Automatically converts to Celsius
    let tr = Temperature::from_celsius(25.0);
    let v = Speed::from_kilometers_per_hour(0.36); // Automatically converts to m/s
    let rh = Humidity::from_percent(50.0);
    let met = 1.4;
    let clo = 0.5;

    // Calculate relative air speed
    let vr_ms = v_relative(v.as_meters_per_second(), met);
    let vr = Speed::from_meters_per_second(vr_ms);

    // Type-safe calculation
    let result = pmv_ppd_iso(tdb, tr, vr, rh, met, clo, Default::default());
    println!("PMV: {:.2}", result.pmv);
}
```

The library re-exports the following measurement types for convenience:
- `Temperature` - Automatic conversion between Fahrenheit, Celsius, Kelvin
- `Speed` - Automatic conversion between m/s, km/h, mph, etc.
- `Humidity` - Relative humidity percentage (0-100%)
- `Area` - Automatic conversion between m², ft², etc. (used for body surface area)
- `Pressure` - Automatic conversion between Pa, kPa, mmHg, etc. (used for atmospheric pressure)

These types provide:
- Automatic unit conversion with compile-time type safety
- Prevention of unit errors through the type system
- Clear documentation of expected units

## WASM Support

This library is `no_std` compatible and can be compiled to WebAssembly:

```bash
cargo build --target wasm32-unknown-unknown --release
```

## Testing

The library includes comprehensive tests, including comparisons with the original Python implementation:

```bash
# Run all tests
cargo test

# Run only library tests
cargo test --lib

# Run with Python comparison (requires pythermalcomfort installed)
cargo test test_compare_with_python
```

## Standards Compliance

This library implements thermal comfort calculations according to:

- **ISO 7730:2005** - Ergonomics of the thermal environment
- **ASHRAE 55** - Thermal Environmental Conditions for Human Occupancy
- **ISO 7726:1998** - Ergonomics of the thermal environment - Instruments for measuring physical quantities
- **ISO 9920:2007** - Ergonomics of the thermal environment - Estimation of thermal insulation and water vapour resistance of a clothing ensemble

## Credits

This is a Rust port of [pythermalcomfort](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort) (v3.8.0).

Original Python package developed by the Center for the Built Environment at UC Berkeley.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! This port aims to maintain feature parity with pythermalcomfort while leveraging Rust's safety and performance benefits.


