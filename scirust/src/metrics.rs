//! Small statistics helpers for validating approximation quality.
//! No external dependencies.

use std::collections::HashSet;

/// Plain dot product — the full-precision ground-truth attention score.
///
/// ```
/// use scirust::metrics::dot;
/// assert!((dot(&[1.0, 2.0, 3.0], &[1.0, 0.0, -1.0]) + 2.0).abs() < 1e-6);
/// ```
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Pearson correlation coefficient (linear / magnitude fidelity).
pub fn pearson(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Fractional ranks (0-based), ties broken arbitrarily. Inputs are assumed
/// continuous (no ties) for the synthetic data used here.
pub fn ranks(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0f32; v.len()];
    for (rank, &i) in idx.iter().enumerate() {
        r[i] = rank as f32;
    }
    r
}

/// Spearman rank correlation (ordering fidelity — what attention top-k cares
/// about most).
pub fn spearman(x: &[f32], y: &[f32]) -> f32 {
    pearson(&ranks(x), &ranks(y))
}

/// Fraction of the top-`k` indices shared between two score vectors.
pub fn topk_overlap(truth: &[f32], approx: &[f32], k: usize) -> f32 {
    let top = |v: &[f32]| -> HashSet<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        idx.truncate(k);
        idx.into_iter().collect()
    };
    let a = top(truth);
    let b = top(approx);
    a.intersection(&b).count() as f32 / k as f32
}

/// Root-mean-square value of a slice (used as a per-tile sigma_E estimate).
#[inline]
pub fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

/// Cosine similarity between two vectors.
///
/// ```
/// use scirust::metrics::cosine;
/// assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
/// assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
/// ```
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = dot(a, a).sqrt();
    let nb = dot(b, b).sqrt();
    if na * nb == 0.0 {
        0.0
    } else {
        dot(a, b) / (na * nb)
    }
}

/// Relative L2 error `||a - b|| / ||a||` (a is the reference).
pub fn rel_l2(reference: &[f32], approx: &[f32]) -> f32 {
    let mut nd = 0.0f32;
    let mut na = 0.0f32;
    for i in 0..reference.len() {
        nd += (reference[i] - approx[i]).powi(2);
        na += reference[i] * reference[i];
    }
    (nd / na.max(1e-12)).sqrt()
}

/// Numerically stable softmax of `scores * scale` into `out`.
///
/// ```
/// use scirust::metrics::softmax_into;
/// let mut w = [0.0f32; 3];
/// softmax_into(&[0.0, 0.0, 0.0], 1.0, &mut w);
/// assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// assert!((w[0] - 1.0 / 3.0).abs() < 1e-6);
/// ```
pub fn softmax_into(scores: &[f32], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(scores.len(), out.len());
    let mut m = f32::NEG_INFINITY;
    for &x in scores {
        m = m.max(x * scale);
    }
    let mut sum = 0.0f32;
    for (o, &x) in out.iter_mut().zip(scores) {
        let e = (x * scale - m).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}
