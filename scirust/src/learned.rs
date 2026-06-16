//! Learned low-rank projection for SLHA v2.
//!
//! The synthetic prototype in [`crate::scenario`] *assumes* the low-rank base
//! is captured ideally. This module removes that assumption: it **learns** the
//! projection by PCA (the optimal linear rank-`D_C` reconstruction, by
//! Eckart–Young) on sample keys, then measures how much energy a real rank-`D_C`
//! projection actually keeps.
//!
//! ## Latent whitening (INT4-friendliness)
//! PCA latent components have wildly different variances (the eigenvalue
//! spectrum). A single per-tile INT4 scale is then dominated by the top
//! component and crushes the small ones. We therefore optionally **whiten** the
//! latent — store `h_k = (e_k·K) / s_k` with `s_k = sqrt(λ_k)` so every
//! component has ~unit variance — and **de-whiten the query** — `q_k =
//! (e_k·Q)·s_k`. The product `Σ q_k h_k = Σ (e_k·Q)(e_k·K)` is **identical** to
//! the un-whitened score, so whitening is mathematically neutral on the score
//! while making the INT4 latent far more accurate.
//!
//! Dimensions: keys live in `R^d` (`d > D_C`); the latent is `R^{D_C}`; the
//! sign-LSH residual `Z` maps `R^d -> D_S` bits. Everything still feeds the
//! *unchanged* fixed-size tile and kernel.

use crate::attention::slha_v2::{
    quantize_latent, quantize_latent_grouped, SciRustSlhaTile, D_C, D_S, FLAG_WARM, N_GROUPS,
    RESIDUAL_WORDS,
};
use crate::linalg::jacobi_eigh;
use crate::rng::Rng;

/// A PCA-learned projection plus a fixed random sign-LSH `Z` (`D_S × d`).
pub struct LearnedModel {
    pub d: usize,
    /// Top-`D_C` principal eigenvectors, row-major `D_C × d`.
    evec: Vec<f32>,
    /// Per-component scale `s_k` (`sqrt(λ_k)` if whitening, else `1.0`).
    scale: Vec<f32>,
    pub z: Vec<f32>,
    /// Fraction of total variance retained by the top-`D_C` subspace.
    pub captured_energy: f32,
    pub whiten: bool,
}

impl LearnedModel {
    /// Fit by PCA on `train` keys (each length `d`). `Z` is a fixed random
    /// Johnson–Lindenstrauss projection. With `whiten`, latent components are
    /// normalised to unit variance (score-preserving — see module docs).
    pub fn fit(train: &[Vec<f32>], d: usize, seed: u64, whiten: bool) -> Self {
        assert!(d > D_C, "need d > D_C for a non-trivial residual");
        let n = train.len().max(1);

        // Empirical (uncentered) covariance, row-major d×d.
        let mut cov = vec![0.0f32; d * d];
        for key in train {
            for i in 0..d {
                let ki = key[i];
                let row = i * d;
                for j in 0..d {
                    cov[row + j] += ki * key[j];
                }
            }
        }
        let inv = 1.0 / n as f32;
        for c in cov.iter_mut() {
            *c *= inv;
        }

        let (eigvals, eigvecs) = jacobi_eigh(&cov, d);

        let mut idx: Vec<usize> = (0..d).collect();
        idx.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap());

        let total: f64 = eigvals.iter().map(|&x| x.max(0.0)).sum();
        let kept: f64 = idx[..D_C].iter().map(|&i| eigvals[i].max(0.0)).sum();
        let captured_energy = if total > 0.0 {
            (kept / total) as f32
        } else {
            1.0
        };

        // Floor the whitening scale so near-zero eigenvalues don't blow up
        // pure-noise components (cap the dynamic range at ~sqrt(1e3)).
        let lambda_top = eigvals[idx[0]].max(1e-12);
        let floor = lambda_top * 1e-3;

        let mut evec = vec![0.0f32; D_C * d];
        let mut scale = vec![1.0f32; D_C];
        for (k, &ei) in idx[..D_C].iter().enumerate() {
            for i in 0..d {
                evec[k * d + i] = eigvecs[i * d + ei] as f32;
            }
            if whiten {
                scale[k] = (eigvals[ei].max(floor) as f32).sqrt();
            }
        }

        let mut rng = Rng::new(seed);
        let mut z = vec![0.0f32; D_S * d];
        rng.fill_gaussian(&mut z);

        LearnedModel {
            d,
            evec,
            scale,
            z,
            captured_energy,
            whiten,
        }
    }

    /// Build a model from an arbitrary (e.g. SGD-learned) projection `p`
    /// (`D_C × d`, row-major), with no per-component whitening. `Z` is seeded
    /// like [`Self::fit`], so two models built with the same `seed` share the
    /// residual projection and differ only in `P`.
    pub fn from_projection(p: Vec<f32>, d: usize, seed: u64) -> Self {
        assert_eq!(p.len(), D_C * d, "projection must be D_C×d");
        let mut rng = Rng::new(seed);
        let mut z = vec![0.0f32; D_S * d];
        rng.fill_gaussian(&mut z);
        LearnedModel {
            d,
            evec: p,
            scale: vec![1.0f32; D_C],
            z,
            captured_energy: f32::NAN, // not the projector of a single covariance
            whiten: false,
        }
    }

    /// The projection matrix `P` (`D_C × d`, row-major) — e.g. to warm-start
    /// [`train_projection`] from this model's PCA solution.
    pub fn projection(&self) -> &[f32] {
        &self.evec
    }

    #[inline]
    fn evec_dot(&self, k: usize, v: &[f32]) -> f32 {
        let row = &self.evec[k * self.d..(k + 1) * self.d];
        let mut acc = 0.0f32;
        for i in 0..self.d {
            acc += row[i] * v[i];
        }
        acc
    }

    /// Whitened latent `h_k = (e_k·key) / s_k` (length `D_C`).
    pub fn latent(&self, key: &[f32]) -> [f32; D_C] {
        let mut h = [0.0f32; D_C];
        for (k, hk) in h.iter_mut().enumerate() {
            *hk = self.evec_dot(k, key) / self.scale[k];
        }
        h
    }

    /// De-whitened query coarse `q_k = (e_k·Q)·s_k` (length `D_C`), consumed
    /// directly by `compute_score`.
    pub fn query_coarse(&self, q: &[f32]) -> [f32; D_C] {
        let mut h = [0.0f32; D_C];
        for (k, hk) in h.iter_mut().enumerate() {
            *hk = self.evec_dot(k, q) * self.scale[k];
        }
        h
    }

    /// Reconstruction `K_coarse_i = Σ_k h_k · s_k · e_k[i]` (length `d`).
    pub fn reconstruct(&self, h: &[f32; D_C]) -> Vec<f32> {
        let mut r = vec![0.0f32; self.d];
        for k in 0..D_C {
            let w = h[k] * self.scale[k];
            let row = &self.evec[k * self.d..(k + 1) * self.d];
            for i in 0..self.d {
                r[i] += w * row[i];
            }
        }
        r
    }

    /// Packed sign bits of `Z · v` (`v` length `d`).
    pub fn sign_bits(&self, v: &[f32]) -> [u64; RESIDUAL_WORDS] {
        let mut out = [0u64; RESIDUAL_WORDS];
        for s in 0..D_S {
            let row = &self.z[s * self.d..(s + 1) * self.d];
            let mut acc = 0.0f32;
            for i in 0..self.d {
                acc += row[i] * v[i];
            }
            if acc < 0.0 {
                out[s >> 6] |= 1u64 << (s & 63);
            }
        }
        out
    }

    /// Encode a key into a tile. The residual `E = key - reconstruct(latent)` is
    /// computed from the *un-quantised* latent; the INT4 error is a separate,
    /// smaller approximation folded into the coarse term. `grouped` selects
    /// per-group micro-scaling (recommended) vs a single global INT4 scale.
    pub fn encode_with(&self, key: &[f32], pos: u32, warm: bool, grouped: bool) -> SciRustSlhaTile {
        let h = self.latent(key);
        let (latent, scale, group_scales) = if grouped {
            quantize_latent_grouped(&h)
        } else {
            let (l, s) = quantize_latent(&h);
            (l, s, [255u8; N_GROUPS])
        };
        let recon = self.reconstruct(&h);
        let mut e = vec![0.0f32; self.d];
        for i in 0..self.d {
            e[i] = key[i] - recon[i];
        }
        let bitmap = self.sign_bits(&e);
        let sigma_e = (e.iter().map(|x| x * x).sum::<f32>() / self.d as f32).sqrt();
        let lambda = sigma_e * (std::f32::consts::PI / (2.0 * D_S as f32)).sqrt();
        SciRustSlhaTile {
            latent_kv: latent,
            residual_bitmap: bitmap,
            scale,
            dynamic_lambda: lambda,
            residual_sigma: sigma_e,
            token_id: pos,
            position: pos,
            head_id: 0,
            flags: if warm { FLAG_WARM } else { 0 },
            group_scales,
        }
    }

    /// Encode a key into a tile (per-group micro-scaled INT4 latent).
    pub fn encode(&self, key: &[f32], pos: u32, warm: bool) -> SciRustSlhaTile {
        self.encode_with(key, pos, warm, true)
    }
}

/// Generate `n` keys in `R^d` from a factor model: `r` random unit factor
/// directions with geometrically decaying strength `decay^i`, plus isotropic
/// `noise`. Lower `decay` / higher `noise` ⇒ flatter spectrum ⇒ a rank-`D_C`
/// projection captures less ⇒ larger residual.
pub fn gen_keys(seed: u64, n: usize, d: usize, r: usize, decay: f32, noise: f32) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);

    let mut factors = vec![vec![0.0f32; d]; r];
    for f in factors.iter_mut() {
        rng.fill_gaussian(f);
        let nrm = f.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in f.iter_mut() {
            *x /= nrm;
        }
    }
    let strengths: Vec<f32> = (0..r).map(|i| decay.powi(i as i32)).collect();

    let mut keys = Vec::with_capacity(n);
    for _ in 0..n {
        let mut k = vec![0.0f32; d];
        for fi in 0..r {
            let g = rng.next_gaussian() * strengths[fi];
            let fv = &factors[fi];
            for i in 0..d {
                k[i] += g * fv[i];
            }
        }
        for i in 0..d {
            k[i] += noise * rng.next_gaussian();
        }
        keys.push(k);
    }
    keys
}

/// Train a projection `P` (`D_C × d`) by SGD to minimise the **score** error
/// `E_{Q,K}[(⟨Q,K⟩ − ⟨PQ,PK⟩)²]` over *independent* query/key samples — a
/// task-aware objective, unlike PCA which only minimises key reconstruction and
/// ignores the query distribution.
///
/// Gradient (closed form, per sample): with `a = Pq`, `b = Pk`,
/// `r = ⟨q,k⟩ − ⟨a,b⟩`, then `∂r²/∂P = −2r (b qᵀ + a kᵀ)`.
///
/// Returns `(P, loss_history)` (mean per-sample loss per epoch). Warm-start
/// `init_p` from a PCA fit to measure the task-aware improvement on top of it.
pub fn train_projection(
    qs: &[Vec<f32>],
    ks: &[Vec<f32>],
    init_p: Vec<f32>,
    epochs: usize,
    lr: f32,
    batch: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        init_p.len().is_multiple_of(D_C),
        "projection must be D_C×d (row-major)"
    );
    let d = init_p.len() / D_C;
    let mut p = init_p;
    let mut rng = Rng::new(seed);
    let (nq, nk) = (qs.len(), ks.len());
    let mut a = vec![0.0f32; D_C];
    let mut b = vec![0.0f32; D_C];
    let mut grad = vec![0.0f32; D_C * d];
    let steps = (nq.max(nk) / batch).max(1);
    let mut history = Vec::with_capacity(epochs);

    for ep in 0..epochs {
        // Linear learning-rate decay to 0 — stabilises the late epochs of this
        // non-convex (quartic-in-P) objective.
        let cur_lr = lr * (1.0 - ep as f32 / epochs as f32);
        let mut epoch_loss = 0.0f64;
        for _ in 0..steps {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
            let mut batch_loss = 0.0f32;
            for _ in 0..batch {
                let q = &qs[(rng.next_u64() as usize) % nq];
                let k = &ks[(rng.next_u64() as usize) % nk];
                let mut qk = 0.0f32;
                for j in 0..d {
                    qk += q[j] * k[j];
                }
                for row in 0..D_C {
                    let pr = &p[row * d..(row + 1) * d];
                    let mut sa = 0.0f32;
                    let mut sb = 0.0f32;
                    for j in 0..d {
                        sa += pr[j] * q[j];
                        sb += pr[j] * k[j];
                    }
                    a[row] = sa;
                    b[row] = sb;
                }
                let mut ab = 0.0f32;
                for row in 0..D_C {
                    ab += a[row] * b[row];
                }
                let r = qk - ab;
                batch_loss += r * r;
                let c = -2.0 * r;
                for row in 0..D_C {
                    let gr = &mut grad[row * d..(row + 1) * d];
                    let (ai, bi) = (a[row], b[row]);
                    for j in 0..d {
                        gr[j] += c * (bi * q[j] + ai * k[j]);
                    }
                }
            }
            let step = cur_lr / batch as f32;
            for (pi, gi) in p.iter_mut().zip(&grad) {
                *pi -= step * gi;
            }
            epoch_loss += batch_loss as f64;
        }
        history.push((epoch_loss / (steps * batch) as f64) as f32);
    }
    (p, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{dot, spearman};

    fn run(decay: f32, whiten: bool) -> (f32, f32, f32) {
        let d = 160;
        let train = gen_keys(1, 600, d, 200, decay, 0.02);
        let model = LearnedModel::fit(&train, d, 7, whiten);

        let eval = gen_keys(2, 256, d, 200, decay, 0.02);
        let q = &gen_keys(3, 1, d, 200, decay, 0.02)[0];
        let q_coarse = model.query_coarse(q);
        let q_sign = model.sign_bits(q);

        let mut s_true = Vec::new();
        let mut s_hot = Vec::new();
        let mut s_warm = Vec::new();
        for (i, key) in eval.iter().enumerate() {
            s_true.push(dot(q, key));
            let hot = model.encode(key, i as u32, false);
            let mut warm = hot.clone();
            warm.flags |= FLAG_WARM;
            s_hot.push(hot.compute_score(&q_coarse, &q_sign));
            s_warm.push(warm.compute_score(&q_coarse, &q_sign));
        }
        (
            model.captured_energy,
            spearman(&s_hot, &s_true),
            spearman(&s_warm, &s_true),
        )
    }

    #[test]
    fn pca_captures_dominant_subspace_and_hot_beats_warm() {
        // Default (non-whitened) latent — see `whitening_does_not_help`.
        let (captured, sp_hot, sp_warm) = run(0.95, false);
        assert!(captured > 0.8, "captured energy {captured} too low");
        assert!(sp_hot > 0.8, "HOT Spearman {sp_hot} too low with learned P");
        assert!(sp_hot + 0.03 >= sp_warm, "HOT {sp_hot} << WARM {sp_warm}");
    }

    #[test]
    fn whitening_does_not_help_score_preserving_int4() {
        // Empirical finding: whitening the PCA latent does NOT improve ranking
        // (it hurts). A single-scale INT4 already allocates resolution to the
        // high-variance components that dominate the score; whitening then
        // re-amplifies their quantisation error through the de-whitened query.
        // The score itself is identical either way (whitening is neutral on it).
        let (_, hot_whiten, _) = run(0.9, true);
        let (_, hot_naive, _) = run(0.9, false);
        assert!(
            hot_naive + 0.02 >= hot_whiten,
            "whitening unexpectedly helped: naive {hot_naive} vs whiten {hot_whiten}"
        );
    }

    #[test]
    fn grouped_int4_does_not_hurt_end_to_end() {
        // Per-group MX scaling halves latent reconstruction error (see the
        // slha_v2 unit test); end-to-end it must be at least as good as a single
        // scale. (The end-to-end gain is small here because the score is
        // dominated by high-variance components a single scale already handles.)
        let d = 160;
        let train = gen_keys(1, 600, d, 200, 0.93, 0.02);
        let model = LearnedModel::fit(&train, d, 7, false);
        let eval = gen_keys(2, 256, d, 200, 0.93, 0.02);
        let q = &gen_keys(3, 1, d, 200, 0.93, 0.02)[0];
        let q_coarse = model.query_coarse(q);
        let q_sign = model.sign_bits(q);

        let hot_sp = |grouped: bool| {
            let mut st = Vec::new();
            let mut sh = Vec::new();
            for (i, key) in eval.iter().enumerate() {
                st.push(dot(q, key));
                sh.push(
                    model
                        .encode_with(key, i as u32, false, grouped)
                        .compute_score(&q_coarse, &q_sign),
                );
            }
            spearman(&sh, &st)
        };
        let single = hot_sp(false);
        let grouped = hot_sp(true);
        assert!(
            grouped + 0.02 >= single,
            "grouped {grouped} worse than single {single}"
        );
    }

    #[test]
    fn attention_output_is_high_fidelity_and_hot_ge_warm() {
        // The softmax-weighted value output is far more robust to score error
        // than raw ranking. Cosine(out_true, out_hot) must be high, HOT >= WARM.
        use crate::attention::slha_v2::FLAG_WARM;
        use crate::metrics::{cosine, softmax_into};
        use crate::rng::Rng;

        let (d, dv, n) = (160usize, 48usize, 200usize);
        let train = gen_keys(1, 600, d, 200, 0.9, 0.02);
        let model = LearnedModel::fit(&train, d, 7, false);
        let keys = gen_keys(2, n, d, 200, 0.9, 0.02);
        let mut rng = Rng::new(5);
        let values: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let mut v = vec![0.0f32; dv];
                rng.fill_gaussian(&mut v);
                v
            })
            .collect();
        let tiles: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| model.encode(k, i as u32, false))
            .collect();

        let mut q = vec![0.0f32; d];
        rng.fill_gaussian(&mut q);
        let qc = model.query_coarse(&q);
        let qs = model.sign_bits(&q);
        let scale = 1.0 / (d as f32).sqrt();

        let s_true: Vec<f32> = keys.iter().map(|k| dot(&q, k)).collect();
        let s_hot: Vec<f32> = tiles.iter().map(|t| t.compute_score(&qc, &qs)).collect();
        let s_warm: Vec<f32> = tiles
            .iter()
            .map(|t| {
                let mut w = t.clone();
                w.flags |= FLAG_WARM;
                w.compute_score(&qc, &qs)
            })
            .collect();

        let agg = |s: &[f32]| -> Vec<f32> {
            let mut w = vec![0.0f32; n];
            softmax_into(s, scale, &mut w);
            let mut o = vec![0.0f32; dv];
            for (wi, v) in w.iter().zip(&values) {
                for j in 0..dv {
                    o[j] += wi * v[j];
                }
            }
            o
        };
        let ot = agg(&s_true);
        let ch = cosine(&ot, &agg(&s_hot));
        let cw = cosine(&ot, &agg(&s_warm));
        assert!(ch > 0.9, "HOT attention-output cosine {ch} too low");
        assert!(ch + 0.02 >= cw, "HOT {ch} < WARM {cw}");
    }

    #[test]
    fn train_projection_reduces_score_loss() {
        // Optimiser sanity: from a deliberately bad (random) projection, the
        // task-aware SGD must substantially reduce E[(⟨Q,K⟩ − ⟨PQ,PK⟩)²].
        // (The decisive "learned beats PCA" result — ~0.16 vs ~0.86 WARM
        // Spearman on clean data — is shown by the `learn_projection` example.)
        let d = 160;
        let gen = |seed: u64, n: usize| -> Vec<Vec<f32>> {
            let mut rng = Rng::new(seed);
            (0..n)
                .map(|_| {
                    let mut v = vec![0.0f32; d];
                    rng.fill_gaussian(&mut v);
                    v
                })
                .collect()
        };
        let qs = gen(1, 256);
        let ks = gen(2, 256);

        let mut rng = Rng::new(3);
        let mut p = vec![0.0f32; D_C * d];
        for x in p.iter_mut() {
            *x = rng.next_gaussian() * 0.05; // deliberately poor starting point
        }

        // Kept light (it runs in debug under CI); a wrong-sign / no-op gradient
        // would fail this even with a lenient threshold.
        let (_p, hist) = train_projection(&qs, &ks, p, 40, 2.0e-3, 64, 7);
        let (l0, l1) = (hist[0], hist[hist.len() - 1]);
        assert!(
            l1 < 0.85 * l0,
            "score loss {l0:.2} -> {l1:.2} did not drop enough"
        );
    }
}
