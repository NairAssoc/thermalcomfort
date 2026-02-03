# PyThermalComfort to Rust Conversion Plan

This document tracks the comprehensive port of all functions from pythermalcomfort (v3.8.0) to Rust.

## Summary

- **Total Models**: 37
- **Total Utilities**: 25
- **Completed**: 2 models + 5 utilities
- **Remaining**: 35 models + 20 utilities

## Status Legend

- ✅ **Complete**: Fully implemented and tested
- 🚧 **In Progress**: Partially implemented
- 📋 **Planned**: Not yet started
- ⏸️ **Deferred**: Complex, will implement later

---

## Models Status

### Thermal Comfort Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `pmv_ppd_iso` | ✅ | Critical | ISO 7730:2005 standard |
| `pmv_ppd_ashrae` | ✅ | Critical | ASHRAE 55 (complete with cooling_effect) |
| `pmv_a` | 📋 | High | PMV with adaptive clothing |
| `pmv_athb` | 📋 | Medium | PMV with adaptive thermal heat balance |
| `pmv_e` | 📋 | Medium | PMV for elevated air speeds |
| `set_tmp` | ✅ | High | Standard Effective Temperature |
| `utci` | ✅ | High | Universal Thermal Climate Index |
| `two_nodes_gagge` | ✅ | Medium | Two-node Gagge model |

### Adaptive Comfort Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `adaptive_ashrae` | ✅ | High | ASHRAE 55 adaptive model |
| `adaptive_en` | ✅ | High | EN 16798-1 adaptive model |

### Heat Stress Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `wbgt` | ✅ | High | Wet Bulb Globe Temperature |
| `phs` | 📋 | Medium | Predicted Heat Strain (ISO 7933) |
| `heat_index_rothfusz` | ✅ | Medium | Heat index (Rothfusz equation) |
| `heat_index_lu` | 📋 | Medium | Heat index (Lu & Romps 2022) |
| `humidex` | ✅ | Low | Canadian humidex |
| `thi` | ✅ | Low | Temperature Humidity Index |
| `discomfort_index` | ✅ | Low | Discomfort index |
| `at` | 📋 | Low | Apparent temperature |
| `net` | 📋 | Low | Normal Effective Temperature |
| `esi` | 📋 | Low | Environmental Stress Index |

### Cold Stress Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `wci` | ✅ | Medium | Wind Chill Index |
| `wind_chill_temperature` | ✅ | Medium | Wind chill temperature |

### Work Capacity Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `work_capacity_iso` | 📋 | Low | ISO 7243 work capacity |
| `work_capacity_dunne` | 📋 | Low | Dunne work capacity |
| `work_capacity_hothaps` | 📋 | Low | HOTHAPS work capacity |
| `work_capacity_niosh` | 📋 | Low | NIOSH work capacity |

### Advanced Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `JOS3` | ⏸️ | Low | Complex thermoregulation model (class) |
| `two_nodes_gagge` | 📋 | Medium | Two-node Gagge model |
| `two_nodes_gagge_ji` | 📋 | Medium | Gagge-Ji variant |
| `two_nodes_gagge_sleep` | 📋 | Low | Gagge model for sleep |
| `pet_steady` | 📋 | Medium | Physiological Equivalent Temperature |
| `ridge_regression_predict_t_re_t_sk` | 📋 | Low | ML-based temperature prediction |

### Auxiliary Models

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `cooling_effect` | ✅ | High | Cooling effect for elevated air speed |
| `solar_gain` | 📋 | Medium | Solar radiation effects |
| `ankle_draft` | 📋 | Low | Ankle draft discomfort |
| `vertical_tmp_grad_ppd` | 📋 | Low | Vertical temperature gradient PPD |
| `use_fans_heatwaves` | 📋 | Low | Fan usage during heatwaves |
| `clo_tout` | 📋 | Low | Clothing vs outdoor temperature |

---

## Utilities Status

### Psychrometric Functions

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `psy_ta_rh` | ✅ | Critical | Psychrometric calculations |
| `wet_bulb_tmp` | ✅ | Critical | Wet bulb temperature |
| `dew_point_tmp` | ✅ | Critical | Dew point temperature |
| `p_sat` | ✅ | Critical | Saturation vapor pressure |
| `p_sat_torr` | 📋 | Low | Saturation pressure in torr |
| `enthalpy_air` | ✅ | High | Air enthalpy |
| `antoine` | 📋 | Medium | Antoine equation |

### Mean Radiant Temperature

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `mean_radiant_tmp` | ✅ | High | MRT from globe temperature |

### Operative Temperature

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `operative_tmp` | ✅ | High | Operative temperature |

### Clothing Functions

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `clo_area_factor` | ✅ | High | Clothing area factor |
| `clo_dynamic_ashrae` | ✅ | High | Dynamic clo (ASHRAE) |
| `clo_dynamic_iso` | ✅ | High | Dynamic clo (ISO 9920) |
| `clo_intrinsic_insulation_ensemble` | 📋 | Medium | Ensemble insulation |
| `clo_insulation_air_layer` | ✅ | Medium | Air layer insulation |
| `clo_total_insulation` | ✅ | Medium | Total insulation |
| `clo_correction_factor_environment` | 📋 | Medium | Environmental correction |

### Velocity Functions

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `v_relative` | ✅ | Critical | Relative air speed |

### Body Surface Area

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `body_surface_area` | 📋 | Medium | BSA calculation (DuBois, etc.) |

### Miscellaneous

| Function | Status | Priority | Notes |
|----------|--------|----------|-------|
| `running_mean_outdoor_temperature` | ✅ | Medium | For adaptive comfort |
| `f_svv` | 📋 | Low | Sky-vault view fraction |
| `transpose_sharp_altitude` | 📋 | Low | Solar position |
| `units_converter` | 📋 | High | Unit conversion utility |
| `valid_range` | ✅ | High | Input validation |

---

## Implementation Phases

### Phase 1: Core Foundations ✅ (COMPLETE)
- [x] Project setup & build system
- [x] Core types (Met, Clo, PMV, PPD, etc.)
- [x] Basic constants
- [x] PMV/PPD ISO & ASHRAE
- [x] Basic psychrometrics
- [x] Python comparison test framework

### Phase 2: Essential Models (Priority: High)
1. **cooling_effect** - Required for ASHRAE PMV
2. **set_tmp** - Standard Effective Temperature
3. **utci** - Universal Thermal Climate Index
4. **adaptive_ashrae** - ASHRAE 55 adaptive
5. **adaptive_en** - EN 16798-1 adaptive
6. **wbgt** - Wet Bulb Globe Temperature

### Phase 3: Extended Utilities (Priority: High)
1. **clo_dynamic_iso** - ISO 9920 dynamic clothing
2. **units_converter** - Complete unit conversion
3. **running_mean_outdoor_temperature** - For adaptive models
4. **body_surface_area** - All equations
5. Complete clothing insulation functions

### Phase 4: Heat & Cold Stress (Priority: Medium)
1. **phs** - Predicted Heat Strain
2. **heat_index_rothfusz** - Heat index
3. **wci** - Wind Chill Index
4. **wind_chill_temperature**
5. **two_nodes_gagge** family

### Phase 5: Specialized Models (Priority: Low-Medium)
1. **pet_steady** - Physiological Equivalent Temp
2. **pmv_a**, **pmv_athb**, **pmv_e** - PMV variants
3. **solar_gain** - Solar radiation
4. **ankle_draft**, **vertical_tmp_grad_ppd**
5. Work capacity models

### Phase 6: Advanced & Optional (Priority: Low)
1. **JOS3** - Complex thermoregulation (class-based)
2. **ridge_regression_predict_t_re_t_sk** - ML model
3. Remaining auxiliary models
4. Complete test coverage

---

## Testing Strategy

Each implemented function must have:
1. ✅ Unit tests in Rust
2. ✅ Python comparison tests via PyO3
3. ✅ Documentation with examples
4. 📋 Edge case & error handling tests

Current test coverage:
- **Unit tests**: 23 passing
- **Integration tests**: 9 passing
- **Doc tests**: 12 passing
- **Total**: 44 tests

Target: 200+ tests covering all models

---

## Known Limitations & TODO

### Current Limitations
1. **ASHRAE cooling effect**: Not implemented for vr > 0.1
2. **Array operations**: Currently scalar only (Python supports arrays)
3. **JOS3 model**: Complex class-based model needs design

### Design Decisions Needed
1. Should we support array/batch operations?
2. How to handle the JOS3 class (state machine)?
3. ML model serialization for ridge regression?

---

## Dependencies

### Current
- `libm` (0.2) - no_std math
- `measurements` (0.11) - typed measurements
- `pyo3` (0.23) - Python interop for testing
- `approx` (0.5) - floating point comparisons

### May Need
- `ndarray` - for array operations (optional feature?)
- `serde` - for serialization (optional feature?)

---

## Progress Tracking

Last Updated: 2026-02-01 (Session 2)

- **Models Completed**: 15/37 (41%) - PMV ISO/ASHRAE, SET, Two-node Gagge, Cooling Effect, UTCI, Adaptive ASHRAE/EN, WBGT, WCI, WCT, Humidex, THI, DI, Heat Index
- **Utilities Completed**: 15/25 (60%) - Added running_mean, clothing insulation functions (ISO), correction factors
- **Overall Progress**: 30/62 (48%)

Recent additions (Session 1):
- ✅ two_nodes_gagge - Complete two-node thermoregulation model
- ✅ set_tmp - Standard Effective Temperature wrapper
- ✅ cooling_effect - Cooling effect calculation using Brent's method
- ✅ utci - Universal Thermal Climate Index (6th-order polynomial)
- ✅ Brent's method root finder for numerical optimization

Recent additions (Session 2):
- ✅ adaptive_ashrae - ASHRAE 55 adaptive comfort model
- ✅ adaptive_en - EN 16798-1 adaptive comfort model
- ✅ running_mean_outdoor_temperature - Exponentially weighted mean
- ✅ wbgt - Wet Bulb Globe Temperature (ISO 7243:2017)
- ✅ clo_dynamic_iso - ISO 9920:2007 dynamic clothing
- ✅ clo_total_insulation - Total ensemble insulation
- ✅ clo_insulation_air_layer - Boundary air layer insulation
- ✅ wci - Wind Chill Index (ASHRAE 2017)
- ✅ wind_chill_temperature - Wind chill temperature
- ✅ humidex - Canadian humidity index (Rana model)
- ✅ humidex_masterson - Humidex (Masterson model)
- ✅ thi - Temperature-Humidity Index
- ✅ discomfort_index - Discomfort Index
- ✅ heat_index_rothfusz - Heat Index (Rothfusz 1990)

Progress: 48% complete (30/62 functions)
Next milestone: 32 remaining functions
Estimated completion: 70-80% with next batch

---

## How to Contribute

When implementing a new function:
1. Check this plan and mark status as 🚧
2. Implement in appropriate module
3. Add Python comparison test
4. Add documentation with examples
5. Update this plan to ✅
6. Create PR with test results

---

## References

- [pythermalcomfort Documentation](https://pythermalcomfort.readthedocs.io/)
- [ISO 7730:2005](https://www.iso.org/standard/39155.html)
- [ASHRAE 55-2023](https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy)
- [ISO 7933:2023](https://www.iso.org/standard/74622.html) - Heat Stress
- [ISO 7726:1998](https://www.iso.org/standard/14562.html) - Instruments
