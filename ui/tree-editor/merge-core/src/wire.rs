//! One entry point for every binding: a function name and its arguments as a JSON array, the result as JSON.

use crate::{Error, Res};
use serde_json::Value;

fn list<'a>(args: &'a [Value], i: usize) -> Res<&'a [Value]> {
    match args.get(i) {
        Some(Value::Array(l)) => Ok(l),
        other => Err(Error::Type(format!("argument {i} is not a list: {other:?}"))),
    }
}

fn keys(args: &[Value], i: usize) -> Res<Vec<String>> {
    list(args, i)?
        .iter()
        .map(|k| k.as_str().map(String::from).ok_or_else(|| Error::Type(format!("not a key: {k}"))))
        .collect()
}

fn arg(args: &[Value], i: usize) -> Res<&Value> {
    args.get(i).ok_or_else(|| Error::Type(format!("missing argument {i}")))
}

/// Call `name` (`diff`, `apply`, `stale_ops`, `conflicts`, `rebase_orders`, `is_related`, `touched`, `reconcile`)
/// with `args`; `apply` and `stale_ops` answer with indices into the ops that were passed.
pub fn call_value(name: &str, mut owned: Vec<Value>) -> Res<Value> {
    if name == "apply" {
        // The document is the caller's own parse, so it is edited in place rather than copied.
        let [doc, ops] = owned.as_mut_slice() else { return Err(Error::Type("apply takes a document and ops".into())) };
        let Value::Array(ops) = &*ops else { return Err(Error::Type(format!("ops is not a list: {ops}"))) };
        let dropped = crate::apply_in_place(doc, ops);
        return Ok(Value::Array(vec![std::mem::take(doc), dropped.into()]));
    }
    let args = &owned[..];
    Ok(match name {
        "diff" => crate::diff(arg(args, 0)?, arg(args, 1)?).into(),
        "stale_ops" => crate::stale_ops(list(args, 0)?, list(args, 1)?)?.into(),
        "conflicts" => crate::conflicts(list(args, 0)?, list(args, 1)?, arg(args, 2)?)?.into(),
        "rebase_orders" => crate::rebase_orders(list(args, 0)?, arg(args, 1)?, arg(args, 2)?)?.into(),
        "is_related" => crate::is_related(arg(args, 0)?, arg(args, 1)?)?.into(),
        "touched" => crate::touched(list(args, 0)?)?.into(),
        "reconcile" => crate::reconcile(&keys(args, 0)?, &keys(args, 1)?).into(),
        _ => return Err(Error::Key(format!("no function {name}"))),
    })
}

/// `call_value` over JSON text: `args` is a JSON array, the result a JSON value.
pub fn call(name: &str, args: &str) -> Res<String> {
    let Value::Array(args) = serde_json::from_str(args).map_err(|e| Error::Value(e.to_string()))? else {
        return Err(Error::Type("args is not a JSON array".into()));
    };
    Ok(call_value(name, args)?.to_string())
}
