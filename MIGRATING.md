# Migrating from 4.6.2 to 4.8.1

thermalcomfort 4.8.1 ports pythermalcomfort 4.7.0, 4.8.0 and 4.8.1, and it applies one
typing rule throughout the public API: **every value with a physical unit is a newtype**.
Dimensionless numbers (PMV, PPD, fractions, percentages, dimensionless coefficients) stay
`f64`.

The changes fall into three groups:

1. [Code that no longer compiles](#1-code-that-no-longer-compiles). The compiler finds
   every site. Each section shows the 4.6.2 code and its 4.8.1 replacement.
2. [Results that change with the same code](#2-results-that-change-with-the-same-code).
   These follow upstream's corrections, so the new values match pythermalcomfort 4.8.1.
   Re-check any thresholds or stored baselines that depend on them.
3. [New API](#3-new-api) you may want to adopt.

The 4.8.1 code blocks below are compiled as doctests.

## 1. Code that no longer compiles

### Reading quantities in SI or IP: import `UnitSystem`

Every quantity type implements the [`UnitSystem`] trait: `from_si`/`as_si`,
`from_ip`/`as_ip` (IP is inch-pound), and `from_units`/`in_units` for a `Units` value. The
methods are trait methods, so bring the trait into scope:

```rust
use thermalcomfort::{Speed, Temperature, UnitSystem};

let t = Temperature::from_ip(77.0); // °F
let v = Speed::from_si(0.2); // m/s
assert!((t.as_si() - 25.0).abs() < 1e-9); // °C
assert!((v.as_ip() - 0.2 * 3.281).abs() < 1e-12); // ft/s
```

Each type's documentation names the exact SI and IP unit `as_si` and `as_ip` read in, and
any other accessors. For example, `HumidityRatio` reads kg/kg through `as_si`, and g/kg
through `as_grams_per_kilogram`. The factors, and the four that are not exact (notably air
speed's 3.281 ft/s per m/s, which is pythermalcomfort's), are listed in the
[`unit_system`] module documentation.

### Yes/no outputs are `Option<bool>`

| 4.6.2 | 4.8.1 |
|---|---|
| `AdaptiveAshraeResult::{acceptability_80, acceptability_90}: bool` | `Option<bool>` |
| `AdaptiveEnResult::{acceptability_cat_i, _cat_ii, _cat_iii}: bool` | `Option<bool>` |
| `ankle_draft(..) -> (f64, bool)` | `(f64, Option<bool>)` |
| `vertical_tmp_grad_ppd(..) -> (f64, bool)` | `(f64, Option<bool>)` |

`None` is pythermalcomfort's NaN. It means an input is NaN or outside the model's
applicability limits, and the model gives no verdict. The `use_fans_heatwaves` heat-strain
fields were already `Option<bool>`; they are now also `None` for a NaN input.

```rust
use thermalcomfort::models::{AdaptiveInputs, adaptive_ashrae};
use thermalcomfort::{Speed, Temperature};

let r = adaptive_ashrae(
    AdaptiveInputs {
        tdb: Temperature::from_celsius(25.0),
        tr: Temperature::from_celsius(25.0),
        t_running_mean: Temperature::from_celsius(20.0),
        v: Speed::from_meters_per_second(0.1),
    },
    Default::default(),
);
// 4.6.2: if r.acceptability_80 { .. }
match r.acceptability_80 {
    Some(true) => { /* acceptable */ }
    Some(false) => { /* not acceptable */ }
    None => { /* outside the model's applicability */ }
}
// or, treating "no verdict" as not acceptable:
let ok = r.acceptability_80 == Some(true);
# let _ = ok;
```

### Flows, enthalpy, humidity ratio, resistances and rates are newtypes

Construct with the named constructor in the unit you used before. The number is the
same, so wrapping and unwrapping changes no value. Read the value back with the matching
accessor or with `as_si()`.

| Model | Field | 4.6.2 | 4.8.1 type | Constructor / accessor (old unit) |
|---|---|---|---|---|
| `two_nodes_gagge`, `use_fans_heatwaves` | `m_bl` (result), `max_skin_blood_flow` (option) | `f64` | `VolumeFluxDensity` | `from_/as_liters_per_hour_per_square_meter` |
| `two_nodes_gagge`, `use_fans_heatwaves` | `m_rsw` (result), `max_sweating` (option) | `f64` | `MassFluxDensity` | `from_/as_grams_per_hour_per_square_meter` |
| `two_nodes_gagge_sleep` | `skin_blood_flow` (option; result `Vec`) | `f64` | `VolumeFluxDensity` | `from_/as_liters_per_hour_per_square_meter` |
| `two_nodes_gagge_sleep` | `c_sw` | `f64` | `SweatingCoefficient` | `from_/as_grams_per_square_meter_per_hour_per_kelvin` |
| `two_nodes_gagge_sleep` | `c_dil` | `f64` | `VasodilationCoefficient` | `from_/as_liters_per_square_meter_per_hour_per_kelvin` |
| `two_nodes_gagge_sleep` | `c_str` | `f64` | `VasoconstrictionCoefficient` | `from_/as_per_kelvin` |
| JOS3 | `cardiac_output`, `bf_core`, `bf_muscle`, `bf_fat`, `bf_skin`, `bf_ava_hand`, `bf_ava_foot` | `f64` | `VolumeFlow` | `as_liters_per_hour` |
| JOS3 | `weight_loss_by_evap_and_res` | `f64` | `MassFlow` | `as_grams_per_second` |
| JOS3 | `r_t` | `f64` | `ThermalResistance` | `as_square_meter_kelvins_per_watt` |
| JOS3 | `r_et` | `f64` | `EvaporativeResistance` | `as_square_meter_kilopascals_per_watt` |
| JOS3 | `limit_dshiv_dt` (option) | `Option<f64>` | `Option<PowerRate>` | `from_watts_per_second` |
| `phs` | `evap_load_wm2_min` (option and result) | `f64` | `HeatPerArea` | `from_/as_watt_minutes_per_square_meter` |
| psychrometrics | `PsychrometricValues::hr`; `enthalpy_air` and `hr_to_rh` argument | `f64` | `HumidityRatio` | `from_/as_kilograms_per_kilogram` |
| psychrometrics | `PsychrometricValues::h`; `enthalpy_air` return | `f64` | `SpecificEnthalpy` | `as_joules_per_kilogram` |

JOS3's array and `Vec` shapes are unchanged: `Vec<[f64; 17]>` becomes
`Vec<[VolumeFlow; 17]>`, and so on.

Sweat rate is in **g/(h·m²)**, which is pythermalcomfort's mL/(h·m²). The 4.6.2
documentation labelled it kg/h/m², but the numbers it held were already g/(h·m²). Skin
blood flow is in L/(h·m²); upstream labels it kg/(h·m²), the same number at the 1 kg/L its
heat balance assumes.

```rust
use thermalcomfort::models::{GaggeTwoNodesInputs, GaggeTwoNodesOptions, two_nodes_gagge};
use thermalcomfort::psychrometrics::enthalpy_air;
use thermalcomfort::{
    ClothingInsulation, Humidity, HumidityRatio, MassFluxDensity, MetabolicRate, Speed,
    Temperature, UnitSystem, VolumeFluxDensity,
};

// 4.6.2: max_skin_blood_flow: 90.0, max_sweating: 500.0
let options = GaggeTwoNodesOptions {
    max_skin_blood_flow: VolumeFluxDensity::from_liters_per_hour_per_square_meter(90.0),
    max_sweating: MassFluxDensity::from_grams_per_hour_per_square_meter(500.0),
    ..Default::default()
};
let r = two_nodes_gagge(
    GaggeTwoNodesInputs {
        tdb: Temperature::from_celsius(30.0),
        tr: Temperature::from_celsius(30.0),
        v: Speed::from_meters_per_second(0.1),
        rh: Humidity::from_percent(50.0),
        met: MetabolicRate::from_met(1.2),
        clo: ClothingInsulation::from_clo(0.5),
    },
    options,
);
// 4.6.2: let sweat: f64 = r.m_rsw;
let sweat_si = r.m_rsw.as_grams_per_hour_per_square_meter(); // g/(h·m²)
let sweat_ip = r.m_rsw.as_ip(); // lb/(h·ft²)
# let _ = (sweat_si, sweat_ip);

// 4.6.2: let h: f64 = enthalpy_air(tdb, 0.01);
let h = enthalpy_air(
    Temperature::from_celsius(25.0),
    HumidityRatio::from_kilograms_per_kilogram(0.01),
);
let h_j_per_kg = h.as_joules_per_kilogram();
# let _ = h_j_per_kg;
```

### UTCI and the heat index are `Temperature`

| 4.6.2 | 4.8.1 |
|---|---|
| `UtciResult::utci: f64` | `Temperature` |
| `HeatIndexResult::hi: f64` (`heat_index_rothfusz`, `heat_index_schoen`, `heat_index_lu`) | `Temperature` |
| `StressCategory::from_utci_opt(f64)` | `from_utci_opt(Temperature)` |
| `HeatIndexStress::from_hi(f64)`, `from_hi_opt(f64)` | take `Temperature` |

`UtciOptions::units` still selects the unit upstream rounds the UTCI in. In 4.6.2,
`Units::IP` made `utci` an `f64` in °F. In 4.8.1 it is a `Temperature` rounded to 0.1 °F;
read it in the system you asked for, to within storage round-off (a few 1e-14). The stress
category is always computed from the Celsius value, as upstream does.

```rust
use thermalcomfort::models::{UtciInputs, UtciOptions, heat_index_rothfusz, utci};
use thermalcomfort::{Humidity, Speed, Temperature, UnitSystem, Units};

let units = Units::IP;
let r = utci(
    UtciInputs {
        tdb: Temperature::from_fahrenheit(86.0),
        tr: Temperature::from_fahrenheit(86.0),
        v: Speed::from_ip(3.0),
        rh: Humidity::from_percent(50.0),
    },
    UtciOptions { units, ..Default::default() },
);
// 4.6.2: let utci_f: f64 = r.utci;
let utci_f = r.utci.in_units(units); // °F, rounded to 0.1 °F
let utci_c = r.utci.as_celsius(); // the same temperature in °C, unrounded
# let _ = (utci_f, utci_c, r.stress_category);

let hi = heat_index_rothfusz(
    Temperature::from_celsius(35.0),
    Humidity::from_percent(60.0),
    Default::default(),
);
// 4.6.2: let hi_c: f64 = hi.hi;
let hi_c = hi.hi.as_celsius();
let hi_f = hi.hi.as_fahrenheit();
# let _ = (hi_c, hi_f);
```

Humidex, the discomfort index, THI and ESI are unchanged (`f64`). Their bands exist in one
scale only.

### Maximum skin wettedness is a `MaxSkinWettedness`

`GaggeTwoNodesOptions::w_max` changes from `Option<f64>` to `Option<MaxSkinWettedness>`.
The constructor returns `None` outside (0.06, 1], which pythermalcomfort rejects.

```rust
use thermalcomfort::MaxSkinWettedness;
use thermalcomfort::models::GaggeTwoNodesOptions;

// 4.6.2: w_max: Some(0.85)
let options = GaggeTwoNodesOptions {
    w_max: Some(MaxSkinWettedness::new(0.85).expect("in (0.06, 1]")),
    ..Default::default()
};
# let _ = options;
assert!(MaxSkinWettedness::new(0.05).is_none());
```

### Removed items

| 4.6.2 | 4.8.1 replacement |
|---|---|
| `thermalcomfort::constants::*` | Crate-private. For 1 met in W/m², use `MetabolicRate::MET_TO_W_M2` or `MetabolicRate::from_met(1.0).as_w_per_m2()`. Other values were implementation details of the models. |
| `utilities::temp_to_celsius(t)` | `t.as_celsius()` |
| `utilities::celsius_to_temp(c)` | `Temperature::from_celsius(c)` |
| `utilities::speed_to_ms(v)` | `v.as_meters_per_second()` |
| `utilities::ms_to_speed(x)` | `Speed::from_meters_per_second(x)` |
| `GaggeTwoNodesSleepOptions::temp_core_neutral` | Removed upstream; delete the field from your options. The model derives the neutral core temperature itself. |

## 2. Results that change with the same code

Each change follows pythermalcomfort's own fix, so 4.8.1 matches upstream 4.8.1.

- **`cooling_effect` in IP**: the result is the Celsius cooling effect × 1.8, a temperature
  difference in °F (upstream #473). 4.6.2 computed `/ 1.8 × 3.28`, which was wrong. SI
  results are unchanged.
- **`adaptive_ashrae`**:
  - The comfort bounds and acceptability use unrounded values. Only the reported
    temperatures are rounded, after the °F conversion when `Units::IP` is set.
  - Reported temperatures can differ in the last digit, mostly in IP.
  - Acceptability can differ for an operative temperature within one rounding step of a
    bound.
- **`adaptive_en`**: with `limit_inputs` (the default), the running mean outdoor
  temperature must lie in 10–30 °C. The previous range was 10–33.5 °C; outside the new
  range the result is NaN and the flags are `None`.
- **Both adaptive models**: a NaN operative temperature gets no air-speed cooling
  allowance.
- **`two_nodes_gagge`, `set_tmp`, `use_fans_heatwaves`, the sleep model**:
  - At the skin-wettedness cap, regulatory sweating is `(w_max − 0.06) / 0.94` of the
    maximum (previously `w_max / 0.94`). Results change only when the cap is reached, in
    hot, humid or high-activity conditions.
  - `two_nodes_gagge` returns NaN for a non-finite input instead of panicking.
- **`two_nodes_gagge_sleep`**:
  - It guards a non-positive maximum evaporative capacity.
  - It caps the thermal discomfort `disc` at 6.
  - Hot nights no longer run away.
- **`solar_gain`, supine posture**:
  - The reflected term uses the solar altitude.
  - The projected-area lookup uses unrounded rotated angles.
  - An angle outside the lookup table gives NaN instead of the first table entry.
- **`transpose_sharp_altitude`**: returns unrounded angles. It previously rounded them to
  three decimals.

## 3. New API

- **`UnitSystem`** (`thermalcomfort::UnitSystem`): SI/IP accessors on every quantity type.
  See the `unit_systems` example.
- **Quantity types**: `MassFluxDensity`, `VolumeFluxDensity`, `VolumeFlow`, `MassFlow`,
  `SpecificEnthalpy`, `HumidityRatio`, `HeatPerArea`, `ThermalResistance`,
  `EvaporativeResistance`, `PowerRate`, `SweatingCoefficient`, `VasodilationCoefficient`,
  `VasoconstrictionCoefficient` and `MaxSkinWettedness`.
- **`Units`** is re-exported at the crate root.
- **JOS3**: `set_t_core` and `set_t_skin` set the initial core and skin temperatures.

[`UnitSystem`]: https://docs.rs/thermalcomfort/4.8.1/thermalcomfort/unit_system/trait.UnitSystem.html
[`unit_system`]: https://docs.rs/thermalcomfort/4.8.1/thermalcomfort/unit_system/index.html
