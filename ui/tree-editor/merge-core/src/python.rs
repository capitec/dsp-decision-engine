//! Python binding: `_native.call(name, args_json) -> result_json`.

use crate::Error;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
fn call(py: Python<'_>, name: &str, args: &str) -> PyResult<String> {
    py.allow_threads(|| crate::call(name, args)).map_err(|e| match e {
        Error::Value(m) => PyValueError::new_err(m),
        Error::Key(m) => PyKeyError::new_err(m),
        Error::Type(m) => PyTypeError::new_err(m),
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(call, m)?)
}
