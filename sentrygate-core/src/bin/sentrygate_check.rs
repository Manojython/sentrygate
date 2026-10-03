//! Validate the Rust Scanner against the Python-generated golden vectors.
//! Usage: sentrygate_check <tokenizer.json> <laya.onnx> <golden_injection.json>

use sentrygate_core::Scanner;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    label: bool,
    max: f32,
    is_attack: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: sentrygate_check <tokenizer.json> <laya.onnx> <golden_injection.json>");
        std::process::exit(2);
    }
    let scanner = Scanner::injection(&args[1], &args[2])?;
    let golden: Golden = serde_json::from_reader(std::fs::File::open(&args[3])?)?;

    let mut max_diff = 0.0f32;
    let mut fails = 0;
    println!("{:<50} {:<6} {:>9} {:>9}", "case", "label", "rust", "verdict");
    for c in &golden.cases {
        let f = scanner.scan(&c.text)?;
        let d = (f.score - c.max).abs();
        max_diff = max_diff.max(d);
        let ok = f.is_attack == c.is_attack && d < 0.02;
        if !ok {
            fails += 1;
        }
        let head: String = c.text.chars().take(48).collect();
        println!(
            "{:<50} {:<6} {:>9.3} {:>9}{}",
            head,
            c.label,
            f.score,
            if f.is_attack { "ATTACK" } else { "ok" },
            if ok { "" } else { "  <-- MISMATCH" }
        );
    }
    println!("\nmax |rust - py| MAX-score = {max_diff:.4}");
    if fails == 0 && max_diff < 0.02 {
        println!("ALL GOLDEN CHECKS PASSED");
        Ok(())
    } else {
        eprintln!("FAILED: {fails} verdict mismatch(es)");
        std::process::exit(1);
    }
}
