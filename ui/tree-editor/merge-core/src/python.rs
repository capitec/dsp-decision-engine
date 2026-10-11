//! Python binding: `_native.call(name, args_json) -> result_json`.

use crate::Error;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
fn call(py: Python<'_>, name: &str, args: &str) -> PyResult<String> {
    py.allow_threads(|| crate::call(name, args)).map_err(to_py)
}

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Value(m) => PyValueError::new_err(m),
        Error::Key(m) => PyKeyError::new_err(m),
        Error::Type(m) => PyTypeError::new_err(m),
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "direct-python")]
    m.add_function(wrap_pyfunction!(direct::call_direct, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)
}

#[cfg(feature = "direct-python")]
mod direct {
    use super::*;
    use pyo3::types::PyAny;

    #[pyfunction]
    pub fn call_direct<'py>(py: Python<'py>, name: &str, args: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let args: Vec<serde_json::Value> = pythonize::depythonize(args)?;
        let out = crate::call_value(name, args).map_err(super::to_py)?;
        Ok(pythonize::pythonize(py, &out)?)
    }
}
