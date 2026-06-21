//! `slha-audit` — run the SLHA v2 self-audit and emit a report.
//!
//! ```text
//! slha-audit              # human-readable Markdown to stdout
//! slha-audit --json       # compact JSON (machine-readable / CI / agents)
//! slha-audit --pretty     # indented JSON
//! slha-audit --out FILE   # also write the rendered report to FILE
//! slha-audit --diff PRIOR # run now, diff against a prior JSON report
//! ```
//!
//! Exit code: `0` if the audit verdict is ok (and, with `--diff`, no changes);
//! `1` if a check failed or a regression was found; `2` on I/O / parse errors.

use scirust::audit;
use scirust::json::Json;
use std::process::exit;

const HELP: &str = "\
slha-audit — SLHA v2 self-audit + report

USAGE:
    slha-audit [--json | --pretty] [--out FILE] [--diff PRIOR.json]

OPTIONS:
    --json         Emit compact JSON instead of Markdown
    --pretty       Emit indented JSON
    --out FILE     Also write the rendered report to FILE
    --diff PRIOR   Run now and diff against a prior JSON report (regression check)
    -h, --help     Show this help

EXIT CODE:
    0  audit ok (no regressions);  1  a check failed / report changed;  2  I/O error
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |f: &str| args.iter().any(|a| a == f);
    let val = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    if has("-h") || has("--help") {
        print!("{HELP}");
        return;
    }

    let report = audit::run();
    let ok = report
        .get("verdict")
        .and_then(|v| v.get("ok"))
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    let rendered = if has("--json") {
        report.to_compact()
    } else if has("--pretty") {
        report.to_pretty()
    } else {
        audit::to_markdown(&report)
    };

    if let Some(path) = val("--out") {
        if let Err(e) = std::fs::write(&path, &rendered) {
            eprintln!("slha-audit: cannot write {path}: {e}");
            exit(2);
        }
        eprintln!("slha-audit: wrote {path}");
    }

    // --diff: compare against a prior JSON report and report changes.
    if let Some(prior_path) = val("--diff") {
        let txt = match std::fs::read_to_string(&prior_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("slha-audit: cannot read {prior_path}: {e}");
                exit(2);
            }
        };
        let prior = match Json::parse(&txt) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("slha-audit: cannot parse {prior_path}: {e}");
                exit(2);
            }
        };
        let changes = audit::diff(&prior, &report);
        if changes.is_empty() {
            println!("slha-audit: no changes vs {prior_path}");
        } else {
            println!("slha-audit: {} change(s) vs {prior_path}:", changes.len());
            for line in &changes {
                println!("  {line}");
            }
        }
        exit(if ok && changes.is_empty() { 0 } else { 1 });
    }

    print!("{rendered}");
    if !rendered.ends_with('\n') {
        println!();
    }
    exit(if ok { 0 } else { 1 });
}
