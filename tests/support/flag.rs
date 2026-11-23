//! Reading pythermalcomfort's yes/no outputs.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat};

/// Read a pythermalcomfort yes/no output as the port's `Option<bool>`.
///
/// Every such output upstream is a float, scalar or 0-d array: 1.0 where the condition
/// holds, 0.0 where it does not, and NaN where the inputs are outside the model's
/// applicability limits, which the port spells `None`. Anything else -- a bool, an int,
/// or a float other than those three -- is an error rather than a value to coerce, because
/// a lenient reader turns a real comparison into a vacuous one.
pub fn flag_value(value: &Bound<'_, PyAny>) -> Result<Option<bool>, String> {
    // numpy scalars and 0-d arrays unwrap to the Python object they hold
    let item = match value.call_method0("item") {
        Ok(item) => item,
        Err(_) => value.clone(),
    };
    if item.is_instance_of::<PyBool>() || !item.is_instance_of::<PyFloat>() {
        return Err(format!("expected a 1.0/0.0/NaN float flag, got {value:?}"));
    }
    let x: f64 = item.extract().map_err(|e: PyErr| e.to_string())?;
    match x {
        1.0 => Ok(Some(true)),
        0.0 => Ok(Some(false)),
        x if x.is_nan() => Ok(None),
        _ => Err(format!("expected a 1.0/0.0/NaN float flag, got {x}")),
    }
}

/// [`flag_value`] of the attribute `field` of a Python result.
pub fn py_flag(result: &Bound<'_, PyAny>, field: &str) -> Result<Option<bool>, String> {
    let attr = result
        .getattr(field)
        .map_err(|e| format!("{field}: missing on Python result: {e}"))?;
    flag_value(&attr).map_err(|e| format!("{field}: {e}"))
}
