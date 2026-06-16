//! Randomized **property / fuzz** tests — dependency-free, driven by the crate's
//! own deterministic RNG (so failures are reproducible). They assert invariants
//! over thousands of random inputs rather than a single hand-picked case:
//!
//! - SIMD paths (AVX2 / AVX-512) ≡ scalar on random INT4 tiles;
//! - `compute_score` never produces NaN/inf on bounded-random INT4 **and** NF4
//!   tiles (robustness fuzz);
//! - WARM == coarse-only;
//! - softmax is a probability distribution;
//! - INT4(MX) and NF4 dequantisation stay within their per-group error bound.

use scirust::attention::slha_v2::{
    quantize_latent_grouped, quantize_latent_nf4, SciRustSlhaTile, D_C, FLAG_NF4, FLAG_WARM,
    GROUP_DIM, LATENT_BYTES, N_GROUPS, RESIDUAL_WORDS,
};
use scirust::metrics::softmax_into;
use scirust::rng::Rng;

fn rand_q(rng: &mut Rng) -> [f32; D_C] {
    let mut q = [0.0f32; D_C];
    rng.fill_gaussian(&mut q);
    q
}

fn rand_q_sign(rng: &mut Rng) -> [u64; RESIDUAL_WORDS] {
    let mut s = [0u64; RESIDUAL_WORDS];
    for w in s.iter_mut() {
        *w = rng.next_u64();
    }
    s
}

/// A random *uniform-INT4* tile with random per-group scales; HOT or WARM.
fn rand_int4_tile(rng: &mut Rng, warm: bool) -> SciRustSlhaTile {
    let mut latent_kv = [0u8; LATENT_BYTES];
    for b in latent_kv.iter_mut() {
        *b = rng.next_u64() as u8;
    }
    let mut group_scales = [0u8; N_GROUPS];
    for g in group_scales.iter_mut() {
        *g = (rng.next_u64() as u8) | 1; // 1..=255, never a zero scale
    }
    SciRustSlhaTile {
        latent_kv,
        residual_bitmap: rand_q_sign(rng),
        scale: rng.next_unit() * 2.0 + 0.01,
        dynamic_lambda: rng.next_gaussian() * 0.1,
        residual_sigma: rng.next_unit(),
        token_id: 0,
        position: 0,
        head_id: 0,
        flags: if warm { FLAG_WARM } else { 0 },
        group_scales,
    }
}

#[test]
fn prop_simd_paths_equal_scalar() {
    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = std::is_x86_feature_detected!("avx2");
        let has_avx512 = std::is_x86_feature_detected!("avx512f");
        if !has_avx2 && !has_avx512 {
            return;
        }
        let mut rng = Rng::new(0xF00D);
        for it in 0..1000 {
            let tile = rand_int4_tile(&mut rng, it % 3 == 0);
            let q = rand_q(&mut rng);
            let qs = rand_q_sign(&mut rng);
            let s = tile.compute_score_scalar(&q, &qs);
            let tol = 1e-3 * (1.0 + s.abs());
            if has_avx2 {
                let a = unsafe { tile.compute_score_avx2(&q, &qs) };
                assert!((s - a).abs() <= tol, "avx2 iter {it}: {s} vs {a}");
            }
            if has_avx512 {
                let a = unsafe { tile.compute_score_avx512(&q, &qs) };
                assert!((s - a).abs() <= tol, "avx512 iter {it}: {s} vs {a}");
            }
        }
    }
}

#[test]
fn prop_score_is_always_finite() {
    // Robustness fuzz over INT4 and NF4 tiles: a valid tile never yields NaN/inf.
    let mut rng = Rng::new(0xBEEF);
    for it in 0..3000 {
        let mut tile = rand_int4_tile(&mut rng, it % 5 == 0);
        if it % 2 == 0 {
            tile.flags |= FLAG_NF4;
        }
        let s = tile.compute_score(&rand_q(&mut rng), &rand_q_sign(&mut rng));
        assert!(s.is_finite(), "non-finite score at iter {it}: {s}");
    }
}

#[test]
fn prop_warm_equals_coarse_only() {
    let mut rng = Rng::new(7);
    for _ in 0..500 {
        let hot = rand_int4_tile(&mut rng, false);
        let q = rand_q(&mut rng);
        let qs = rand_q_sign(&mut rng);
        let mut warm = hot.clone();
        warm.flags |= FLAG_WARM;
        let mut hot_lambda0 = hot.clone();
        hot_lambda0.dynamic_lambda = 0.0;
        let sw = warm.compute_score(&q, &qs);
        let s0 = hot_lambda0.compute_score(&q, &qs);
        assert!((sw - s0).abs() <= 1e-4 * (1.0 + sw.abs()), "{sw} vs {s0}");
    }
}

#[test]
fn prop_softmax_is_a_distribution() {
    let mut rng = Rng::new(11);
    for _ in 0..400 {
        let n = 1 + (rng.next_u64() as usize % 64);
        let mut s = vec![0.0f32; n];
        for x in s.iter_mut() {
            *x = rng.next_gaussian() * (rng.next_unit() * 20.0); // varied magnitudes
        }
        let mut w = vec![0.0f32; n];
        softmax_into(&s, rng.next_unit() * 2.0, &mut w);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax sum = {sum}");
        assert!(
            w.iter().all(|&p| (0.0..=1.0).contains(&p)),
            "softmax weight out of [0,1]"
        );
    }
}

#[test]
fn prop_dequant_error_within_bound() {
    let mut rng = Rng::new(99);
    for _ in 0..300 {
        let mut v = [0.0f32; D_C];
        for x in v.iter_mut() {
            *x = rng.next_gaussian() * (rng.next_unit() * 5.0 + 0.1);
        }

        // Uniform INT4 (MX): error ≤ one quantisation step (the per-group scale).
        let (latent_kv, scale, group_scales) = quantize_latent_grouped(&v);
        let t = SciRustSlhaTile {
            latent_kv,
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: 0,
            group_scales,
        };
        let dq = t.dequant_latent();
        for d in 0..D_C {
            let eff = scale * (group_scales[d / GROUP_DIM] as f32 / 255.0);
            assert!((dq[d] - v[d]).abs() <= eff + 1e-5, "int4 dim {d}");
        }

        // NF4: error ≤ ~half the widest codebook gap (~0.15) times the group absmax.
        let (lk, sc, gs) = quantize_latent_nf4(&v);
        let mut t2 = t.clone();
        t2.latent_kv = lk;
        t2.scale = sc;
        t2.group_scales = gs;
        t2.flags |= FLAG_NF4;
        let dq2 = t2.dequant_latent();
        for d in 0..D_C {
            let eff = sc * (gs[d / GROUP_DIM] as f32 / 255.0);
            assert!((dq2[d] - v[d]).abs() <= 0.16 * eff + 1e-5, "nf4 dim {d}");
        }
    }
}
