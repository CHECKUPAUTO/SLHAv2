//! "Freeze" the λ calibration (see `examples/calibrate_lambda.rs`).
//!
//! The fused score is `coarse + λ·r`. The spec sets `λ = σ_E·√(π/(2·d_s))`.
//! Against an FP reference, the closed-form optimal multiplier on that λ is
//! `α* = Σ rt·(λr) / Σ (λr)²` with `rt = ⟨Q,K⟩ − coarse`.
//!
//! This test pins two empirical facts (regression guard for the kernel /
//! quantiser / λ formula):
//!   1. `α*` is ~constant across `rho`  ⇒  the `λ ∝ σ_E` *shape* is validated.
//!   2. `α*` sits in a measured band (~4.2)  ⇒  the analytical *constant*
//!      `√(π/(2·d_s))` under-weights the residual by ~4.2× on this benchmark
//!      (corrected constant `C_emp ≈ 0.33` at `d_s = 256`).

use scirust::attention::slha_v2::FLAG_WARM;
use scirust::metrics::dot;
use scirust::scenario::{build_tile, generate, Projection};

/// Closed-form optimal multiplier on the formula's λ, for one residual energy.
fn alpha_star(proj: &Projection, rho: f32) -> f32 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for qi in 0..8u64 {
        let (q, toks) = generate(1000 + qi, 256, rho);
        let q_sign = proj.sign_bits(&q);
        for (i, tok) in toks.iter().enumerate() {
            let hot = build_tile(proj, tok, i as u32, false);
            let mut warm = hot.clone();
            warm.flags |= FLAG_WARM;
            let coarse = warm.compute_score(&q, &q_sign);
            let lamr = hot.compute_score(&q, &q_sign) - coarse; // λ · r
            let rt = dot(&q, &tok.k_real) - coarse;
            num += (rt * lamr) as f64;
            den += (lamr * lamr) as f64;
        }
    }
    (num / den) as f32
}

#[test]
fn lambda_calibration_is_stable_and_pinned() {
    let proj = Projection::new(0xCA11B);
    let a2 = alpha_star(&proj, 0.2);
    let a5 = alpha_star(&proj, 0.5);

    // (1) Shape λ ∝ σ_E validated: optimal multiplier ~constant across rho.
    assert!(
        (a2 - a5).abs() < 0.5,
        "α* not stable across rho: {a2} vs {a5}"
    );

    // (2) Constant correction over √(π/(2·d_s)) pinned (~4.2 on this benchmark).
    for a in [a2, a5] {
        assert!(
            (3.0..=5.5).contains(&a),
            "α* {a} outside expected band ~4.2"
        );
    }
}
