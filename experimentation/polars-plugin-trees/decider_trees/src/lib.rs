//! A decision tree as a polars expression.
//!
//! The tree arrives as data (pickled kwargs, one flat node list); the
//! features arrive as real `Series` with their real dtypes. Nothing is cast:
//! a node declares which dtype it tests, and a column of any other dtype is
//! an error, not a coercion.
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, PrimitiveArray, Utf8ViewArray};
use polars_arrow::bitmap::Bitmap;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;

/// One node on the wire. A leaf has `leaf` (and optionally `value`); a test
/// node has `col`, `op`, exactly one of `f`/`i`/`b`/`s`, and `then`/`else`.
#[derive(Deserialize, Debug)]
struct WireNode {
    leaf: Option<i32>,
    value: Option<f64>,
    col: Option<usize>,
    op: Option<String>,
    f: Option<f64>,
    i: Option<i64>,
    b: Option<bool>,
    s: Option<String>,
    then: Option<usize>,
    #[serde(rename = "else")]
    otherwise: Option<usize>,
}

#[derive(Deserialize)]
struct TreeKwargs {
    nodes: Vec<WireNode>,
    #[serde(default)]
    parallel: bool,
}

#[derive(Clone, Copy)]
enum Cmp { Lt, Le, Gt, Ge, Eq, Ne }

impl Cmp {
    fn parse(op: &str) -> PolarsResult<Cmp> {
        Ok(match op {
            "<" => Cmp::Lt, "<=" => Cmp::Le, ">" => Cmp::Gt,
            ">=" => Cmp::Ge, "==" => Cmp::Eq, "!=" => Cmp::Ne,
            _ => polars_bail!(ComputeError: "decider_trees: unknown numeric op {:?}", op),
        })
    }
    #[inline(always)]
    fn eval<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self { Cmp::Lt => a < b, Cmp::Le => a <= b, Cmp::Gt => a > b,
                     Cmp::Ge => a >= b, Cmp::Eq => a == b, Cmp::Ne => a != b }
    }
}

enum Test {
    F64(usize, Cmp, f64),
    I64(usize, Cmp, i64),
    Bool(usize, bool),
    StrEq(usize, String),
    StrPrefix(usize, String),
    StrRegex(usize, Regex),
}

enum Node {
    Test { test: Test, then: usize, otherwise: usize },
    Leaf { idx: i32, value: f64 },
}

/// A single-chunk, typed view of one input column. Built once per call.
enum Col<'a> {
    F64(&'a [f64], Option<&'a Bitmap>),
    I64(&'a [i64], Option<&'a Bitmap>),
    Bool(&'a BooleanArray),
    Str(&'a Utf8ViewArray),
}

#[inline(always)]
fn valid(v: Option<&Bitmap>, i: usize) -> bool { v.map_or(true, |b| b.get_bit(i)) }

/// `None` means a null was met at a tested node: the answer is null.
#[inline(always)]
fn eval_test(t: &Test, cols: &[Col], i: usize) -> Option<bool> {
    match t {
        Test::F64(c, op, v) => match cols[*c] { Col::F64(xs, va) => valid(va, i).then(|| op.eval(xs[i], *v)), _ => unreachable!() },
        Test::I64(c, op, v) => match cols[*c] { Col::I64(xs, va) => valid(va, i).then(|| op.eval(xs[i], *v)), _ => unreachable!() },
        Test::Bool(c, want) => match cols[*c] { Col::Bool(a) => (!a.is_null(i)).then(|| a.value(i) == *want), _ => unreachable!() },
        Test::StrEq(c, v) => match cols[*c] { Col::Str(a) => (!a.is_null(i)).then(|| a.value(i) == v.as_str()), _ => unreachable!() },
        Test::StrPrefix(c, v) => match cols[*c] { Col::Str(a) => (!a.is_null(i)).then(|| a.value(i).starts_with(v.as_str())), _ => unreachable!() },
        Test::StrRegex(c, re) => match cols[*c] { Col::Str(a) => (!a.is_null(i)).then(|| re.is_match(a.value(i))), _ => unreachable!() },
    }
}

#[inline(always)]
fn walk_row(nodes: &[Node], cols: &[Col], i: usize) -> Option<(i32, f64)> {
    let mut n = 0usize;
    loop {
        match &nodes[n] {
            Node::Leaf { idx, value } => return Some((*idx, *value)),
            Node::Test { test, then, otherwise } => {
                n = if eval_test(test, cols, i)? { *then } else { *otherwise };
            }
        }
    }
}

/// Turn the wire nodes into typed nodes, checking every column reference
/// against the real dtype of the Series it names. This is where "honest about
/// types" lives: a mismatch is an error naming the node, the column and both
/// dtypes; it is never a cast.
fn compile(kw: &TreeKwargs, inputs: &[Series]) -> PolarsResult<Vec<Node>> {
    let n = kw.nodes.len();
    polars_ensure!(n > 0, ComputeError: "decider_trees: tree has no nodes");
    let mut out = Vec::with_capacity(n);
    for (k, w) in kw.nodes.iter().enumerate() {
        if let Some(idx) = w.leaf {
            out.push(Node::Leaf { idx, value: w.value.unwrap_or(f64::NAN) });
            continue;
        }
        let col = w.col.ok_or_else(|| polars_err!(ComputeError: "decider_trees: node {k} is neither a leaf nor a test (no `leaf`, no `col`)"))?;
        polars_ensure!(col < inputs.len(), ComputeError:
            "decider_trees: node {k} tests column #{col} but only {} columns were passed", inputs.len());
        let then = w.then.ok_or_else(|| polars_err!(ComputeError: "decider_trees: node {k} has no `then`"))?;
        let otherwise = w.otherwise.ok_or_else(|| polars_err!(ComputeError: "decider_trees: node {k} has no `else`"))?;
        polars_ensure!(then < n && otherwise < n, ComputeError:
            "decider_trees: node {k} points at node {then}/{otherwise}, tree has {n} nodes");
        let s = &inputs[col];
        let op = w.op.as_deref().unwrap_or("==");
        let dt = s.dtype();
        let name = s.name();
        let test = match (w.f, w.i, w.b, &w.s) {
            (Some(v), None, None, None) => {
                polars_ensure!(matches!(dt, DataType::Float64), SchemaMismatch:
                    "decider_trees: node {k} tests '{name}' as f64 {op} {v}, but the column is {dt}; cast it explicitly if that is what you mean");
                Test::F64(col, Cmp::parse(op)?, v)
            }
            (None, Some(v), None, None) => {
                polars_ensure!(matches!(dt, DataType::Int64), SchemaMismatch:
                    "decider_trees: node {k} tests '{name}' as i64 {op} {v}, but the column is {dt}; cast it explicitly if that is what you mean");
                Test::I64(col, Cmp::parse(op)?, v)
            }
            (None, None, Some(v), None) => {
                polars_ensure!(matches!(dt, DataType::Boolean), SchemaMismatch:
                    "decider_trees: node {k} tests '{name}' as bool, but the column is {dt}");
                Test::Bool(col, v)
            }
            (None, None, None, Some(v)) => {
                polars_ensure!(matches!(dt, DataType::String), SchemaMismatch:
                    "decider_trees: node {k} tests '{name}' as a string ({op} {v:?}), but the column is {dt}; no dictionary encoding is done for you");
                match op {
                    "==" => Test::StrEq(col, v.clone()),
                    "prefix" => Test::StrPrefix(col, v.clone()),
                    "regex" => Test::StrRegex(col, Regex::new(v).map_err(|e| polars_err!(ComputeError: "decider_trees: node {k} regex {v:?}: {e}"))?),
                    _ => polars_bail!(ComputeError: "decider_trees: node {k}: unknown string op {op:?} (want ==, prefix, regex)"),
                }
            }
            _ => polars_bail!(ComputeError: "decider_trees: node {k} must carry exactly one of f/i/b/s"),
        };
        out.push(Node::Test { test, then, otherwise });
    }
    Ok(out)
}

/// One chunk per input, borrowed. `rechunk` is a no-op on already-contiguous
/// input and a copy otherwise; it is the one materialisation this plugin does.
fn views<'a>(single: &'a [Series]) -> PolarsResult<Vec<Col<'a>>> {
    single.iter().map(|s| Ok(match s.dtype() {
        DataType::Float64 => { let a: &PrimitiveArray<f64> = s.f64()?.downcast_iter().next().unwrap(); Col::F64(a.values().as_slice(), a.validity()) }
        DataType::Int64 => { let a: &PrimitiveArray<i64> = s.i64()?.downcast_iter().next().unwrap(); Col::I64(a.values().as_slice(), a.validity()) }
        DataType::Boolean => Col::Bool(s.bool()?.downcast_iter().next().unwrap()),
        DataType::String => Col::Str(s.str()?.downcast_iter().next().unwrap()),
        dt => polars_bail!(SchemaMismatch: "decider_trees: column '{}' is {dt}; only f64, i64, bool and str are understood", s.name()),
    })).collect()
}

fn run<T: Send + Copy>(inputs: &[Series], kw: &TreeKwargs, pick: impl Fn(Option<(i32, f64)>) -> Option<T> + Sync) -> PolarsResult<Vec<Option<T>>> {
    let nodes = compile(kw, inputs)?;
    let single: Vec<Series> = inputs.iter().map(|s| s.rechunk()).collect();
    let cols = views(&single)?;
    let n = single.first().map_or(0, |s| s.len());
    polars_ensure!(single.iter().all(|s| s.len() == n), ComputeError:
        "decider_trees: pass columns of one length, not scalar literals (got lengths {:?})",
        single.iter().map(|s| s.len()).collect::<Vec<_>>());
    let row = |i: usize| pick(walk_row(&nodes, &cols, i));
    Ok(if kw.parallel && n >= 8192 {
        polars_core::runtime::THREAD_POOL.install(|| (0..n).into_par_iter().with_min_len(4096).map(row).collect())
    } else {
        (0..n).map(row).collect()
    })
}

/// Leaf index reached, as Int32 (null if a tested feature was null).
#[polars_expr(output_type=Int32)]
fn walk(inputs: &[Series], kwargs: TreeKwargs) -> PolarsResult<Series> {
    let v = run(inputs, &kwargs, |r| r.map(|(i, _)| i))?;
    Ok(Int32Chunked::from_iter_options(PlSmallStr::from_static("leaf"), v.into_iter()).into_series())
}

/// The reached leaf's `value`, as Float64.
#[polars_expr(output_type=Float64)]
fn walk_value(inputs: &[Series], kwargs: TreeKwargs) -> PolarsResult<Series> {
    let v = run(inputs, &kwargs, |r| r.map(|(_, x)| x))?;
    Ok(Float64Chunked::from_iter_options(PlSmallStr::from_static("value"), v.into_iter()).into_series())
}

/// Deliberately panics — to show what an unprotected `panic!` inside a
/// plugin does to the calling process (RESULTS.md §4).
#[polars_expr(output_type=Int32)]
fn panic_demo(inputs: &[Series]) -> PolarsResult<Series> {
    let xs: Vec<i64> = Vec::new();
    let _ = xs[inputs.len() + 10]; // index out of bounds -> panic
    unreachable!()
}

/// Zero-work expression: the floor for "one plugin call".
#[polars_expr(output_type=Int32)]
fn noop(inputs: &[Series]) -> PolarsResult<Series> {
    Ok(Int32Chunked::full(PlSmallStr::from_static("noop"), 0, inputs[0].len()).into_series())
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
