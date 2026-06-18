#!/usr/bin/env bash
# SLHA v2 — on-device benchmark kit.
#
# Run this on each target to get comparable, honestly-labelled numbers:
#   * the x86-64 server baseline, and
#   * the AArch64 edge device (e.g. Jetson Thor AGX 128, Neoverse-V3AE).
#
# It runs the cross-platform capability/throughput report plus the seeded
# examples behind the paper's §7, and tees everything to results_<arch>.txt.
#
# Usage:   ./scripts/bench_device.sh
# Tip:     on the Thor, build natively (the AArch64/NEON path is selected at
#          runtime); `RUSTFLAGS="-C target-cpu=native"` is optional.
set -euo pipefail

cd "$(dirname "$0")/.."
ARCH="$(uname -m)"
OUT="results_${ARCH}.txt"
RUN="cargo run -q --release -p scirust --example"

echo "SLHA v2 device benchmark — arch=${ARCH} — $(date -u +%FT%TZ)" | tee "$OUT"
echo "rustc: $(rustc --version)" | tee -a "$OUT"
echo | tee -a "$OUT"

echo "### platform_report (capabilities + throughput) ###" | tee -a "$OUT"
$RUN platform_report 2>&1 | tee -a "$OUT"
echo | tee -a "$OUT"

# Reproduce the paper's §7 numbers on this device (all seeded, portable).
for ex in measure measure_learned attention_fidelity bench_vs_fp16 calibrate_lambda ccos_softpaging salient_outliers; do
  echo "### ${ex} ###" | tee -a "$OUT"
  $RUN "$ex" 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
done

# x86-only TSC view (prints a notice and exits on AArch64).
echo "### cycles (x86-only TSC) ###" | tee -a "$OUT"
$RUN cycles 2>&1 | tee -a "$OUT" || true

echo
echo "Wrote ${OUT}. Paste the platform_report '§7.4 paste line' and the relevant"
echo "example tables into paper/slhav2.tex (label the arch; keep x86 as baseline)."
