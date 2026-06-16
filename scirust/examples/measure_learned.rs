//! Measurement prototype with a **learned** (PCA) low-rank projection.
//!
//! Run with:  `cargo run --example measure_learned --release`
//!
//! Unlike `measure` (which hand-sets the residual energy `rho`), here the
//! projection is learned by PCA on sample keys, so the captured energy — and
//! hence the effective `rho` — is *determined by the data spectrum* at rank
//! `D_C = 128`. PCA is the optimal linear rank-`D_C` reconstruction, so this is
//! the honest "with a trained low-rank base" picture.

use scirust::attention::slha_v2::{SciRustSlhaTile, FLAG_WARM};
use scirust::learned::{gen_keys, LearnedModel};
use scirust::metrics::{dot, spearman, topk_overlap};

/// Returns (HOT Spearman, HOT top16, WARM Spearman, WARM top16).
fn evaluate(model: &LearnedModel, decay: f32, grouped: bool) -> (f32, f32, f32, f32) {
    let d = model.d;
    let eval = gen_keys(20, 512, d, 256, decay, 0.02);
    let q = &gen_keys(30, 1, d, 256, decay, 0.02)[0];
    let q_coarse = model.query_coarse(q);
    let q_sign = model.sign_bits(q);

    let mut s_true = Vec::new();
    let mut s_hot = Vec::new();
    let mut s_warm = Vec::new();
    for (i, key) in eval.iter().enumerate() {
        s_true.push(dot(q, key));
        let hot: SciRustSlhaTile = model.encode_with(key, i as u32, false, grouped);
        let mut warm = hot.clone();
        warm.flags |= FLAG_WARM;
        s_hot.push(hot.compute_score(&q_coarse, &q_sign));
        s_warm.push(warm.compute_score(&q_coarse, &q_sign));
    }
    (
        spearman(&s_hot, &s_true),
        topk_overlap(&s_true, &s_hot, 16),
        spearman(&s_warm, &s_true),
        topk_overlap(&s_true, &s_warm, 16),
    )
}

fn main() {
    let d = 256;
    let r = 256;
    let n_train = 1024;

    println!("== SLHA v2 — mesure avec projection APPRISE (PCA) ==\n");
    println!("  d_model = {d}, latent D_C = 128, résidu d_s = 256 bits");
    println!(
        "  P = top-128 vecteurs propres de la covariance des clés (PCA), latent INT4 non whitené\n"
    );

    println!(
        "  {:>6} {:>9} {:>6} | {:^17} | {:^17}",
        "", "énergie", "", "HOT (latent+résidu)", "WARM (latent seul)"
    );
    println!(
        "  {:>6} {:>9} {:>6} | {:>8} {:>8} | {:>8} {:>8}",
        "decay", "captée", "rho~", "Spear", "top16", "Spear", "top16"
    );
    println!("  {}", "-".repeat(62));

    for &decay in &[0.99f32, 0.97, 0.93, 0.85] {
        let train = gen_keys(10, n_train, d, r, decay, 0.02);
        let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);
        let rho_eff = (1.0 - model.captured_energy).max(0.0).sqrt();
        let (h_sp, h_tk, w_sp, w_tk) = evaluate(&model, decay, true);
        println!(
            "  {:>6.2} {:>8.1}% {:>6.2} | {:>8.3} {:>8.3} | {:>8.3} {:>8.3}",
            decay,
            model.captured_energy * 100.0,
            rho_eff,
            h_sp,
            h_tk,
            w_sp,
            w_tk
        );
    }

    // --- INT4 latent : échelle unique vs micro-échelles par groupe -----------
    println!("\n  Quantification du latent INT4 (decay = 0.93) :");
    let train = gen_keys(10, n_train, d, r, 0.93, 0.02);
    let model = LearnedModel::fit(&train, d, 0xC0FFEE, false);
    for (label, grouped) in [("échelle unique", false), ("8 groupes (MX)", true)] {
        let (h_sp, h_tk, w_sp, _w_tk) = evaluate(&model, 0.93, grouped);
        println!("    {label:<16} -> HOT Spearman {h_sp:.3} (top16 {h_tk:.3})   WARM {w_sp:.3}");
    }

    println!(
        "\n  Lecture : la PCA capte ~97–99,7 % de l'énergie ; le goulot devient alors\n  \
         l'INT4 du latent. Les micro-échelles par groupe (8×u8, logées dans les 8\n  \
         octets jadis « reserved ») donnent à la queue de faible variance sa propre\n  \
         échelle et récupèrent de la fidélité, sans agrandir la tuile. Le résidu\n  \
         1-bit aide partout (HOT > WARM). (Le whitening, lui, dégrade — cf. test.)"
    );
}
