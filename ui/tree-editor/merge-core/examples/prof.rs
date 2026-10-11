use std::time::Instant;
use serde_json::Value;

fn med(mut f: impl FnMut()) -> f64 {
    let mut t = vec![];
    for _ in 0..30 {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64() * 1000.0);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[15]
}

fn main() {
    let all: Value = serde_json::from_str(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap()).unwrap();
    let base = all["generated-521"]["base"].clone();
    let text = base.to_string();
    let batch = &all["generated-521"]["batch50"];
    println!("bytes {}", text.len());
    println!("parse {:.3}", med(|| { let _: Value = serde_json::from_str(&text).unwrap(); }));
    println!("clone {:.3}", med(|| { let _ = base.clone(); }));
    println!("serialize {:.3}", med(|| { let _ = base.to_string(); }));
    println!("diff base->batch50 {:.3}", med(|| { let _ = merge_core::diff(&base, batch); }));
    let ops = merge_core::diff(&base, batch);
    println!("ops {}", ops.len());
    println!("apply {:.3}", med(|| { let _ = merge_core::apply(&base, &ops); }));
    println!("conflicts {:.3}", med(|| { let _ = merge_core::conflicts(&ops, &ops, &base); }));
}
