//! Browser/node binding: `call(name, argsJson) -> resultJson`; errors are thrown as `ValueError: ...` etc.

use crate::Error;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn call(name: &str, args: &str) -> Result<String, JsError> {
    crate::call(name, args).map_err(|e| match e {
        Error::Value(m) => JsError::new(&format!("ValueError: {m}")),
        Error::Key(m) => JsError::new(&format!("KeyError: {m}")),
        Error::Type(m) => JsError::new(&format!("TypeError: {m}")),
    })
}
