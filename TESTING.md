# Testing Documentation

This document describes the comprehensive test suite for the `thermalcomfort` Rust library, which validates correctness against the original `pythermalcomfort` Python package.

## Test Structure

The test suite is organized into three main components:

### 1. Unit Tests (`src/*/tests`)

Located within each module's source file, these tests verify individual function behavior:

- **`src/models/pmv.rs`**: Tests PMV/PPD calculations, thermal sensation mapping, and input validation
- **`src/psychrometrics.rs`**: Tests psychrometric functions (wet bulb, dew point, operative temperature)
- **`src/utilities.rs`**: Tests utility functions (relative air speed, conversions, validation)

**Total:** 12 unit tests

### 2. Python Comparison Tests (`tests/python_comparison.rs`)

These integration tests use PyO3 to call the original Python implementation and verify that the Rust port produces identical results.

#### Test Coverage

- **`test_pmv_ppd_iso_standard_conditions`**: Tests PMV/PPD ISO calculations across 5 standard conditions
- **`test_pmv_ppd_iso_extreme_conditions`**: Tests PMV/PPD with `limit_inputs=false` for extreme conditions
- **`test_pmv_ppd_ashrae`**: Tests ASHRAE 55 calculations (vr ≤ 0.1)
- **`test_v_relative`**: Tests relative air speed calculation across 6 scenarios
- **`test_wet_bulb_temperature`**: Tests wet bulb temperature across 6 conditions
- **`test_dew_point_temperature`**: Tests dew point temperature across 5 conditions
- **`test_psychrometrics`**: Tests complete psychrometric calculations across 4 scenarios
- **`test_pmv_ppd_iso_outside_limits`**: Tests proper NaN handling for out-of-range inputs
- **`test_pmv_sequential_scenarios`**: Tests sequential PMV calculations

**Total:** 9 integration tests

### 3. Documentation Tests

Rust automatically tests all code examples in documentation comments.

**Total:** 4 documentation tests

## Running Tests

### Run All Tests

```bash
cargo test
```

### Run Only Unit Tests

```bash
cargo test --lib
```

### Run Only Python Comparison Tests

```bash
cargo test --test python_comparison
```

### Run Specific Test

```bash
cargo test test_pmv_ppd_iso_standard_conditions
```

### Run with Output

```bash
cargo test -- --nocapture
```

## Test Requirements

### Prerequisites

Python comparison tests require:
- Python 3.7+
- `pythermalcomfort` package installed: `pip install pythermalcomfort`
- PyO3 (automatically handled by Cargo)

### Running Without Python

If you don't have Python/pythermalcomfort installed, you can still run unit and doc tests:

```bash
cargo test --lib
```

## Test Results Summary

As of version 3.8.0:

```
✅ Unit tests:          12/12 passed
✅ Integration tests:    9/9  passed
✅ Documentation tests:  4/4  passed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total:              25/25 passed
```

## Accuracy and Tolerance

### Comparison Tolerances

Tests comparing Rust vs Python implementations use the following tolerances:

| Value | Tolerance | Reason |
|-------|-----------|--------|
| PMV | ±0.02 | Acceptable for thermal comfort predictions |
| PPD | ±0.2% | Acceptable percentage point difference |
| Temperatures | ±0.1°C | Acceptable for engineering calculations |
| Relative air speed | ±0.001 m/s | High precision for velocity |
| Humidity ratio | ±0.0001 kg/kg | Acceptable for psychrometrics |
| Pressures | ±1 Pa | Acceptable for pressure calculations |

### Known Limitations

#### ASHRAE Cooling Effect

The ASHRAE 55 standard applies a cooling effect correction when `vr > 0.1 m/s`. This involves:

1. Calculating the cooling effect using the SET model
2. Subtracting the cooling effect from both `tdb` and `tr`
3. Setting `vr` to 0.1 m/s for the PMV calculation

**Status:** Not yet implemented in the Rust port

**Impact:** ASHRAE calculations with `vr > 0.1` will differ from Python implementation

**Workaround:** Use `pmv_ppd_iso` for general PMV calculations, or ensure `vr ≤ 0.1` when using `pmv_ppd_ashrae`

## Continuous Integration

To integrate these tests into CI/CD:

```yaml
# Example GitHub Actions
- name: Run Rust tests
  run: cargo test --lib

- name: Setup Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.10'

- name: Install Python dependencies
  run: pip install pythermalcomfort

- name: Run Python comparison tests
  run: cargo test --test python_comparison
```

## Adding New Tests

### For New Functions

1. Add unit tests in the same file as the function
2. Add Python comparison test in `tests/python_comparison.rs`
3. Add documentation example that will be tested

### Example Template

```rust
#[test]
fn test_new_function() {
    Python::with_gil(|py| {
        let pythermal = PyModule::import(py, "pythermalcomfort.models")
            .expect("Failed to import pythermalcomfort");

        let test_cases = vec![
            // (input1, input2, expected_output),
        ];

        for (input1, input2) in test_cases {
            let py_result = pythermal
                .getattr("function_name").unwrap()
                .call1((input1, input2)).unwrap()
                .extract::<f64>().unwrap();

            let rust_result = function_name(input1, input2);

            assert_abs_diff_eq!(rust_result, py_result, epsilon = 0.02);
        }
    });
}
```

## Performance

Test execution times (on typical development machine):

- Unit tests: ~0.8s
- Python comparison tests: ~0.8s
- Documentation tests: ~0.2s
- **Total: ~1.8s**

## References

- [pythermalcomfort GitHub](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort)
- [PyO3 Documentation](https://pyo3.rs/)
- [approx crate](https://docs.rs/approx/)
