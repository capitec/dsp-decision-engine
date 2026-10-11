//! Browser/node binding: `call(name, argsJson) -> resultJson`; errors are thrown as `ValueError: ...` etc.

use crate::Error;
use wasm_bindgen::prelude::*;
#[cfg(feature = "direct-wasm")]
use serde::Serialize;

#[wasm_bindgen]
pub fn call(name: &str, args: &str) -> Result<String, JsError> {
    crate::call(name, args).map_err(|e| match e {
        Error::Value(m) => JsError::new(&format!("ValueError: {m}")),
        Error::Key(m) => JsError::new(&format!("KeyError: {m}")),
        Error::Type(m) => JsError::new(&format!("TypeError: {m}")),
    })
}

#[cfg(feature = "direct-wasm")]
#[wasm_bindgen(js_name = callDirect)]
pub fn call_direct(name: &str, args: JsValue) -> Result<JsValue, JsError> {
    let args: Vec<serde_json::Value> = serde_wasm_bindgen::from_value(args)?;
    let out = crate::call_value(name, &args).map_err(|e| JsError::new(&format!("{e:?}")))?;
    Ok(out.serialize(&serde_wasm_bindgen::Serializer::json_compatible())?)
}
