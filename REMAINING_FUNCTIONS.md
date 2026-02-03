# Remaining Python Functions to Implement

## Summary

**Total Functions in pythermalcomfort**: 62
**Implemented**: 16 (26%)
**Remaining**: 46 (74%)

---

## High Priority Models (11 remaining)

These are the most important models for thermal comfort assessment:

### Adaptive Comfort Models (2)
- ✅ `adaptive_ashrae` - ASHRAE 55 adaptive comfort model
- ✅ `adaptive_en` - EN 16798-1 adaptive comfort model

### PMV Variants (1)
- ✅ `pmv_a` - PMV with adaptive clothing

### Heat Stress Models (1)
- ✅ `wbgt` - Wet Bulb Globe Temperature

### Clothing Functions (1)
- ✅ `clo_dynamic_iso` - Dynamic clothing insulation (ISO 9920)

### Supporting Utilities (1)
- ✅ `running_mean_outdoor_temperature` - Required for adaptive comfort models

---

## Medium Priority Models (16 remaining)

### PMV Variants (2)
- ✅ `pmv_athb` - PMV with adaptive thermal heat balance
- ✅ `pmv_e` - PMV for elevated air speeds

### Heat/Cold Stress Models (5)
- ✅ `phs` - Predicted Heat Strain (ISO 7933)
- ✅ `heat_index_rothfusz` - Heat index (Rothfusz equation)
- ✅ `heat_index_lu` - Heat index (Lu & Romps 2022)
- ✅ `wci` - Wind Chill Index
- ✅ `wind_chill_temperature` - Wind chill temperature

### Advanced Models (3)
- ✅ `two_nodes_gagge_ji` - Gagge-Ji variant
- ✅ `pet_steady` - Physiological Equivalent Temperature
- ✅ `solar_gain` - Solar radiation effects

### Clothing Functions (4)
- ✅ `clo_intrinsic_insulation_ensemble` - Ensemble insulation
- ✅ `clo_insulation_air_layer` - Air layer insulation
- ✅ `clo_total_insulation` - Total insulation
- ✅ `clo_correction_factor_environment` - Environmental correction

### Psychrometric/Utilities (2)
- ✅ `antoine` - Antoine equation for vapor pressure
- ✅ `body_surface_area` - BSA calculation (DuBois, Takahira, Fujimoto, Kurazumi methods)

---

## Low Priority Models (19 remaining)

### Simple Heat/Cold Indices (7)
- ✅ `humidex` - Canadian humidex
- ✅ `thi` - Temperature Humidity Index
- ✅ `discomfort_index` - Discomfort index
- ✅ `at` - Apparent temperature
- ✅ `net` - Normal Effective Temperature
- ✅ `esi` - Environmental Stress Index

### Work Capacity Models (4)
- ✅ `work_capacity_iso` - ISO 7243 work capacity
- ✅ `work_capacity_dunne` - Dunne work capacity
- ✅ `work_capacity_hothaps` - HOTHAPS work capacity
- ✅ `work_capacity_niosh` - NIOSH work capacity

### Specialized Models (4)
- ✅ `two_nodes_gagge_sleep` - Gagge model for sleep
- ✅ `ridge_regression_predict_t_re_t_sk` - ML-based temperature prediction
- ✅ `ankle_draft` - Ankle draft discomfort
- ✅ `vertical_tmp_grad_ppd` - Vertical temperature gradient PPD

### Miscellaneous (4)
- ✅ `use_fans_heatwaves` - Fan usage during heatwaves
- ✅ `clo_tout` - Clothing vs outdoor temperature
- ✅ `f_svv` - Sky-vault view fraction
- ✅ `transpose_sharp_altitude` - Solar position

---

## Already Implemented ✅ (16 total)

### Core Models (6)
1. `pmv_ppd_iso` - PMV/PPD ISO 7730
2. `pmv_ppd_ashrae` - PMV/PPD ASHRAE 55
3. `set_tmp` - Standard Effective Temperature
4. `two_nodes_gagge` - Two-node Gagge thermoregulation
5. `cooling_effect` - Cooling effect for elevated air speeds
6. `utci` - Universal Thermal Climate Index

### Psychrometric Functions (5)
7. `psy_ta_rh` - Complete psychrometric calculations
8. `wet_bulb_tmp` - Wet bulb temperature
9. `dew_point_tmp` - Dew point temperature
10. `p_sat` - Saturation vapor pressure (Pa)
11. `p_sat_torr` - Saturation vapor pressure (torr)
12. `enthalpy_air` - Air enthalpy

### Utilities (4)
13. `mean_radiant_tmp` - Mean radiant temperature
14. `operative_tmp` - Operative temperature
15. `clo_area_factor` - Clothing area factor
16. `clo_dynamic_ashrae` - Dynamic clothing (ASHRAE)
17. `v_relative` - Relative air speed
18. `valid_range` - Input validation

### Numerical Methods (1)
19. Brent's method root finder (custom implementation)

---

## Recommended Implementation Order

### Phase 2 Completion (High Priority - Next Steps)

1. **`adaptive_ashrae`** - ASHRAE 55 adaptive comfort
   - Required for completing ASHRAE 55 standard support
   - Needs: running_mean_outdoor_temperature

2. **`adaptive_en`** - EN 16798-1 adaptive comfort
   - Required for European standard support
   - Needs: running_mean_outdoor_temperature

3. **`running_mean_outdoor_temperature`** - Utility function
   - Required by both adaptive models
   - Simple exponentially weighted calculation

4. **`wbgt`** - Wet Bulb Globe Temperature
   - Important heat stress index
   - Uses existing wet_bulb_tmp

5. **`clo_dynamic_iso`** - ISO 9920 dynamic clothing
   - Completes clothing insulation functions
   - Similar to existing clo_dynamic_ashrae

6. **`pmv_a`** - PMV with adaptive clothing
   - Extends existing PMV implementation

### Phase 3 (Medium Priority)

7. **`phs`** - Predicted Heat Strain (ISO 7933)
   - Important heat stress standard
   - Complex model but well-documented

8. **`heat_index_rothfusz`** - Heat index
   - Widely used, simple polynomial

9. **`wci`** / **`wind_chill_temperature`** - Cold stress
   - Simple calculations

10. **`pet_steady`** - Physiological Equivalent Temperature
    - Important outdoor comfort index

### Phase 4 (Simple Indices - Low Priority)

11-17. Simple heat indices: humidex, thi, discomfort_index, at, net, esi
18. Work capacity models (4 variants)
19. Specialized models

---

## Complexity Estimates

### Simple (1-2 hours each)
- humidex, thi, discomfort_index, at, net, esi
- wci, wind_chill_temperature
- heat_index_rothfusz
- running_mean_outdoor_temperature
- clo_dynamic_iso
- body_surface_area

### Medium (3-6 hours each)
- adaptive_ashrae, adaptive_en
- wbgt
- pmv_a, pmv_e, pmv_athb
- heat_index_lu
- antoine
- solar_gain
- Clothing insulation functions

### Complex (6-12 hours each)
- phs (Predicted Heat Strain)
- pet_steady
- two_nodes_gagge_ji
- ridge_regression_predict_t_re_t_sk (requires ML model data)

### Very Complex (12+ hours)
- JOS3 thermoregulation model (deferred - requires class/state machine design)

---

## Dependencies

Some functions depend on others being implemented first:

- `adaptive_ashrae`, `adaptive_en` → need `running_mean_outdoor_temperature`
- `pmv_a` → extends existing `pmv_ppd_iso`
- `wbgt` → uses existing `wet_bulb_tmp`
- Work capacity models → may need `wbgt` or other heat stress indices
- `pet_steady` → complex model, may need additional utilities

---

## Special Considerations

### JOS3 Model (Deferred)
The JOS3 model is a complex class-based thermoregulation model (Python class with state). In Rust, this would require:
- State machine design
- Multiple method calls over time
- Possibly a builder pattern
- Significant API design work

Recommendation: Defer until other models complete, then design proper Rust API.

### Machine Learning Models
`ridge_regression_predict_t_re_t_sk` requires:
- Pre-trained model coefficients
- Input data for the model
- Possibly serialization/deserialization

May need additional dependencies or data files.

---

## Current Progress Summary

- **Implemented**: 16/62 (26%)
- **High Priority Remaining**: 6 models
- **Medium Priority Remaining**: 16 models
- **Low Priority Remaining**: 19 models
- **Deferred (Complex)**: 1 model (JOS3)

**Estimated Time to Complete Phase 2**: 20-30 hours
**Estimated Time to 90% Completion**: 80-120 hours
