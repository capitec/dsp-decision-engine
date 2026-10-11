//! Runs the op-level vectors of the shared bench (`../merge-core-bench/vectors/op_level.json`) against the core.

use merge_core::{apply, conflicts, diff, is_related, reconcile, rebase_orders, same, stale_ops, touched, Error};
use serde_json::Value;

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn ops(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}

fn pick(all: &[Value], idx: Vec<usize>) -> Value {
    idx.into_iter().map(|i| all[i].clone()).collect()
}

fn check(c: &Value) -> Result<(), String> {
    let kind = c["kind"].as_str().unwrap();
    let ok = |cond: bool, what: &str| if cond { Ok(()) } else { Err(what.to_string()) };
    match kind {
        "diff" => {
            let got = diff(&c["old"], &c["new"]);
            ok(same(&Value::Array(got.clone()), &c["ops"]), &format!("diff gave {}", Value::Array(got.clone())))?;
            let (doc, dropped) = apply(&c["old"], &got);
            ok(same(&doc, &c["new"]) && dropped.is_empty(), "apply of the diff")
        }
        "apply" => {
            let (doc, dropped) = apply(&c["doc"], ops(&c["ops"]));
            ok(same(&doc, &c["expected"]), &format!("gave {doc}"))?;
            ok(same(&pick(ops(&c["ops"]), dropped), &c["dropped"]), "dropped")?;
            match c.get("keys") {
                Some(k) => ok(doc.as_object().unwrap().keys().cloned().collect::<Vec<_>>() == strings(k), "key order"),
                None => Ok(()),
            }
        }
        "stale" => {
            let got = stale_ops(ops(&c["mine"]), ops(&c["concurrent"])).map_err(|e| format!("{e:?}"))?;
            ok(same(&pick(ops(&c["mine"]), got), &c["dropped"]), "stale")
        }
        "related" => {
            let (a, b) = (&c["a"], &c["b"]);
            ok(is_related(a, b).unwrap() == c["related"] && is_related(b, a).unwrap() == c["related"], "related")
        }
        "conflicts" => {
            let got = conflicts(ops(&c["mine"]), ops(&c["theirs"]), &c["base"]).map_err(|e| format!("{e:?}"))?;
            ok(got == strings(&c["conflicts"]), &format!("gave {got:?}"))
        }
        "rebase" => {
            let got = rebase_orders(ops(&c["mine"]), &c["base"], &c["current"]).map_err(|e| format!("{e:?}"))?;
            ok(same(&Value::Array(got), &c["expected"]), "rebase")
        }
        "reconcile" => ok(reconcile(&strings(&c["current"]), &strings(&c["given"])) == strings(&c["expected"]), "reconcile"),
        "touched" => ok(touched(ops(&c["ops"])).unwrap() == strings(&c["expected"]), "touched"),
        "raises" => {
            let args = ops(&c["args"]);
            let got = match c["fn"].as_str().unwrap() {
                "is_related" => is_related(&args[0], &args[1]).map(|_| ()),
                "stale_ops" => stale_ops(ops(&args[0]), ops(&args[1])).map(|_| ()),
                "conflicts" => conflicts(ops(&args[0]), ops(&args[1]), &args[2]).map(|_| ()),
                "touched" => touched(ops(&args[0])).map(|_| ()),
                f => return Err(format!("unknown fn {f}")),
            };
            let name = match got {
                Err(Error::Value(_)) => "ValueError",
                Err(Error::Key(_)) => "KeyError",
                Err(Error::Type(_)) => "TypeError",
                Ok(()) => "nothing",
            };
            ok(name == c["error"].as_str().unwrap(), &format!("raised {name}"))
        }
        k => Err(format!("unknown kind {k}")),
    }
}

#[test]
fn op_level_vectors() {
    let data: Value = serde_json::from_str(include_str!("../../merge-core-bench/vectors/op_level.json")).unwrap();
    let failures: Vec<String> = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| check(c).err().map(|e| format!("{} [{}]: {e}", c["name"], c["semantics"])))
        .collect();
    assert!(failures.is_empty(), "{} failed:\n{}", failures.len(), failures.join("\n"));
}
