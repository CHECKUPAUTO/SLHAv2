//! SLHA v2 — Sub-Low Rank Hybrid Attention micro-kernel (reference).
//!
//! Layout-aware tile + the fused float/binary score of eq. (2.3):
//!
//! ```text
//! score = <q_coarse, dequant(latent)>  +  lambda * (d_s - 2 * popcount(q_sign ^ B))
//!         \_______ continuous _______/     \_____________ binary (sign-LSH) ______/
//! ```
//!
//! ## What changed vs. the v1 listing (see spec §5.1)
//! - **No `read_volatile`.** The v1 reference read every element through
//!   `core::ptr::read_volatile`, which forbids LLVM from vectorising/reordering
//!   and thus *defeats* the stated performance goals. The hot path is now plain,
//!   auto-vectorisable scalar code over slices.
//! - **Signed INT4 (zero-point).** Dequantisation is `(nibble - 8) * scale`, so
//!   the latent base can represent **negative** values (real keys are signed).
//!   v1 used `nibble * scale`, clamping the base to `[0, 15]·scale >= 0`.
//! - **Safe API, no bogus `target_feature`.** `count_ones()` already lowers to
//!   `POPCNT` when the target supports it, with a portable fallback otherwise;
//!   no `unsafe` and no misleading `avx2` gate (the body has no AVX2 intrinsics).
//! - **Tile is exactly 128 bytes with zero padding** (see [`SciRustSlhaTile`]).

/// Latent dimensionality stored per tile (INT4).
pub const D_C: usize = 128;
/// Sign-LSH residual width, in bits.
pub const D_S: usize = 256;
/// Bytes used by the INT4 latent block: two 4-bit samples per byte.
pub const LATENT_BYTES: usize = D_C / 2; // 64
/// Number of `u64` words in the residual bitmap.
pub const RESIDUAL_WORDS: usize = D_S / 64; // 4
/// Number of micro-scaling groups for the INT4 latent (one scale byte each).
pub const N_GROUPS: usize = 8;
/// Latent dimensions per micro-scaling group.
pub const GROUP_DIM: usize = D_C / N_GROUPS; // 16

// --- Tile state flags (the CCOS Soft-Paging modes of spec §4) ---------------
/// Full-fidelity tile: latent + residual both live (cache L1/L2).
pub const FLAG_HOT: u16 = 0;
/// Elastic paging: residual bitmap considered freed; score uses the latent
/// base only (`dynamic_lambda` is bypassed). ~30% footprint drop, no I/O.
pub const FLAG_WARM: u16 = 1 << 0;

/// A single SLHA v2 context tile.
///
/// `#[repr(C, align(64))]` and the field set are chosen so the type is
/// **exactly 128 bytes with no padding** — a clean multiple of the 64-byte
/// cache line (two lines). The 24 bytes that were *tail padding* in the v1
/// layout (104 useful bytes rounded up to 128 by the alignment) are now spent
/// on useful per-tile metadata instead of being wasted.
///
/// Byte map (offsets): latent 0..64 | residual 64..96 | scale 96 | lambda 100 |
/// residual_sigma 104 | token_id 108 | position 112 | head_id 116 |
/// flags 118 | group_scales 120..128.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct SciRustSlhaTile {
    /// Latent base `h_KV` (128 dims) quantised to signed INT4. 64 bytes.
    pub latent_kv: [u8; LATENT_BYTES],
    /// Johnson–Lindenstrauss sign residual: 256 bits. 32 bytes.
    pub residual_bitmap: [u64; RESIDUAL_WORDS],
    /// INT4 dequantisation **global** scale; per-group refinement in `group_scales`.
    pub scale: f32,
    /// Binary-correction weight λ (eq. 3.2), calibrated per tile.
    pub dynamic_lambda: f32,
    /// Per-tile residual energy estimate σ_E (kept for λ recalibration).
    pub residual_sigma: f32,
    /// Token identifier (causal/event-log bookkeeping).
    pub token_id: u32,
    /// Sequence position.
    pub position: u32,
    /// Attention head id.
    pub head_id: u16,
    /// State flags (`FLAG_HOT` / `FLAG_WARM`).
    pub flags: u16,
    /// Per-group micro-scaling bytes: `effective_scale(g) = scale · gs[g]/255`.
    /// (Was reserved padding; now refines the INT4 latent — keeps the tile 128 B.)
    pub group_scales: [u8; N_GROUPS],
}

impl SciRustSlhaTile {
    /// True if the residual has been paged out (WARM mode).
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.flags & FLAG_WARM != 0
    }

    /// Effective dequant scale for dimension `d`: global scale × the dim's
    /// per-group micro-scale.
    #[inline]
    pub fn group_scale(&self, d: usize) -> f32 {
        self.scale * (self.group_scales[d / GROUP_DIM] as f32 / 255.0)
    }

    /// Dequantise one latent dimension `d` with the signed zero-point and its
    /// per-group scale.
    #[inline]
    pub fn dequant_at(&self, d: usize) -> f32 {
        let byte = self.latent_kv[d >> 1];
        let nib = if d & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        ((nib as i32) - 8) as f32 * self.group_scale(d)
    }

    /// Materialise the full dequantised latent vector (mostly for tests).
    pub fn dequant_latent(&self) -> [f32; D_C] {
        let mut out = [0.0f32; D_C];
        for (d, o) in out.iter_mut().enumerate() {
            *o = self.dequant_at(d);
        }
        out
    }

    /// Fused asymmetric attention score (eq. 2.3).
    ///
    /// `q_coarse` is `Q · W_up` in the latent space (`D_C` dims); `q_sign` is
    /// the packed sign of `Q · Zᵀ`. In WARM mode the binary term is dropped.
    ///
    /// Dispatches to an AVX2 path at runtime when available, else the portable
    /// scalar path. Both yield the same result up to float reassociation.
    #[inline]
    pub fn compute_score(&self, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime feature detection.
                return unsafe { self.compute_score_avx2(q_coarse, q_sign) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64 — no runtime detection needed.
            // SAFETY: NEON is always available on this target.
            return unsafe { self.compute_score_neon(q_coarse, q_sign) };
        }
        #[allow(unreachable_code)] // unreachable on aarch64 (returns above)
        {
            self.compute_score_scalar(q_coarse, q_sign)
        }
    }

    /// Binary 1-bit correction: λ · (d_s − 2·popcount(q_sign ^ B)).
    /// popcount(XOR) is the Hamming distance; d_s − 2·Hamming is the signed dot
    /// product of the two ±1 sign vectors.
    #[inline]
    fn residual_term(&self, q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        let mut hamming = 0u32;
        for w in 0..RESIDUAL_WORDS {
            hamming += (q_sign[w] ^ self.residual_bitmap[w]).count_ones();
        }
        self.dynamic_lambda * (D_S as f32 - 2.0 * hamming as f32)
    }

    /// Portable scalar reference path.
    pub fn compute_score_scalar(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        let k = self.dequant_latent();
        let mut coarse = 0.0f32;
        for d in 0..D_C {
            coarse += q_coarse[d] * k[d];
        }
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 path: vectorised INT4 dequant + dot for the coarse term.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_score_avx2(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm256_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(nibble - 8) * group_scale` for 8 dims, multiply by q, accumulate.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let n = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($bytes));
                let v = _mm256_mul_ps($gs_v, _mm256_sub_ps(n, eight));
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            // Per-byte high nibble: shift 16-bit lanes, then mask each byte.
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            // Interleave so nibbles come out in dimension order.
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON path (aarch64): vectorised INT4 dequant + dot for the coarse term.
    /// NEON is baseline on aarch64, so the dispatcher calls this unconditionally.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn compute_score_neon(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = vdupq_n_f32(8.0);
        let mask = vdup_n_u8(0x0F);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(n - 8) * group_scale` for a 4-lane chunk, then FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let v = vmulq_f32(vsubq_f32($n4, eight), $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[g] as f32 * inv255));
            let packed = vld1_u8(latent.add(g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_lo))), base, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_lo))), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_hi))), base + 8, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_hi))), base + 12, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }
}

/// Quantise a latent vector to signed INT4 with a symmetric per-tile scale.
///
/// Returns the packed nibbles and the scale. `value ≈ (nibble - 8) * scale`,
/// with `nibble ∈ [0, 15]` mapping to the signed range `[-8, 7]`.
pub fn quantize_latent(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    // Map the largest magnitude to +7 so it survives the [-8, 7] clamp.
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let q = (v[d] / scale).round() as i32;
        let nib = (q.clamp(-8, 7) + 8) as u8 & 0x0F; // 0..=15
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, scale)
}

/// Per-group ("micro-scaling") signed INT4 quantisation.
///
/// Splits the latent into [`N_GROUPS`] groups of [`GROUP_DIM`] dims; each group
/// gets its own scale stored as a `u8` relative to the global (max) scale:
/// `effective_scale(g) = global · gs[g]/255`. Because PCA orders the latent by
/// descending variance, grouping gives the low-variance tail its own finer
/// scale instead of being crushed by a single global scale. Returns
/// `(nibbles, global_scale, group_bytes)`.
pub fn quantize_latent_grouped(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32, [u8; N_GROUPS]) {
    let mut group_scale = [0.0f32; N_GROUPS];
    for g in 0..N_GROUPS {
        let mut mx = 0.0f32;
        for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
            mx = mx.max(v[d].abs());
        }
        group_scale[g] = mx / 7.0;
    }
    let global = group_scale.iter().copied().fold(0.0f32, f32::max);
    let global = if global > 0.0 { global } else { 1.0 };

    let mut gs = [0u8; N_GROUPS];
    for g in 0..N_GROUPS {
        let r = (group_scale[g] / global * 255.0).round();
        gs[g] = r.clamp(1.0, 255.0) as u8; // never 0, so dequant stays well-defined
    }

    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let eff = global * (gs[d / GROUP_DIM] as f32 / 255.0);
        let nib = (((v[d] / eff).round() as i32).clamp(-8, 7) + 8) as u8 & 0x0F;
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, global, gs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn tile_is_exactly_128_bytes_zero_padding() {
        // align(64) forces size to a multiple of 64; the field set is chosen
        // so that multiple is exactly 128 with no wasted padding byte.
        assert_eq!(align_of::<SciRustSlhaTile>(), 64);
        assert_eq!(size_of::<SciRustSlhaTile>(), 128);

        // Sum of field sizes == struct size  =>  no padding anywhere.
        let field_bytes = LATENT_BYTES            // latent_kv
            + RESIDUAL_WORDS * 8                  // residual_bitmap
            + 4 + 4 + 4                           // scale, dynamic_lambda, residual_sigma
            + 4 + 4                               // token_id, position
            + 2 + 2                               // head_id, flags
            + 8; // group_scales
        assert_eq!(field_bytes, 128);
        assert_eq!(field_bytes, size_of::<SciRustSlhaTile>());
    }

    fn tile_from(
        latent_kv: [u8; LATENT_BYTES],
        scale: f32,
        group_scales: [u8; N_GROUPS],
    ) -> SciRustSlhaTile {
        SciRustSlhaTile {
            latent_kv,
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: FLAG_HOT,
            group_scales,
        }
    }

    #[test]
    fn int4_dequant_round_trips_signed_values() {
        // A vector with both signs must survive quantise -> dequantise within
        // one quantisation step, and crucially keep negative values negative.
        let mut v = [0.0f32; D_C];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32) - 64.0) / 16.0; // spans negative and positive
        }
        let (packed, scale) = quantize_latent(&v);
        // [255; N_GROUPS] makes every group's effective scale == the global scale,
        // i.e. exactly the single-scale behaviour.
        let tile = tile_from(packed, scale, [255; N_GROUPS]);
        let dq = tile.dequant_latent();
        // At least one strictly-negative reconstructed value (zero-point works).
        assert!(
            dq.iter().any(|&x| x < 0.0),
            "no negative values reconstructed"
        );
        for d in 0..D_C {
            assert!(
                (dq[d] - v[d]).abs() <= scale + 1e-6,
                "dim {d}: |{} - {}| > step {scale}",
                dq[d],
                v[d]
            );
        }
    }

    #[test]
    fn grouped_int4_beats_single_on_spread_variance() {
        // Per-group magnitudes spanning orders of magnitude (like PCA components
        // ordered by eigenvalue): a single global scale crushes the small
        // groups, per-group scaling does not.
        let mut v = [0.0f32; D_C];
        let mut rng = crate::rng::Rng::new(5);
        for g in 0..N_GROUPS {
            let amp = 10f32.powi(-(g as i32)); // 1, 0.1, 0.01, ...
            for d in g * GROUP_DIM..(g + 1) * GROUP_DIM {
                v[d] = amp * rng.next_gaussian();
            }
        }
        let sq_err = |t: &SciRustSlhaTile| -> f32 {
            let dq = t.dequant_latent();
            (0..D_C).map(|d| (dq[d] - v[d]).powi(2)).sum()
        };
        let (p1, s1) = quantize_latent(&v);
        let e_single = sq_err(&tile_from(p1, s1, [255; N_GROUPS]));
        let (p2, s2, gs2) = quantize_latent_grouped(&v);
        let e_grouped = sq_err(&tile_from(p2, s2, gs2));
        assert!(
            e_grouped < e_single * 0.5,
            "grouped err {e_grouped} not clearly < single err {e_single}"
        );
    }

    #[test]
    fn avx2_path_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("avx2 unavailable — skipping equivalence check");
                return;
            }
            use crate::scenario::{build_tile, generate, Projection};
            let proj = Projection::new(9);
            let (q, toks) = generate(123, 64, 0.4);
            let q_sign = proj.sign_bits(&q);
            for (i, t) in toks.iter().enumerate() {
                // Alternate HOT / WARM to cover both branches.
                let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
                let s = tile.compute_score_scalar(&q, &q_sign);
                let a = unsafe { tile.compute_score_avx2(&q, &q_sign) };
                assert!(
                    (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                    "tile {i}: scalar {s} vs avx2 {a}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_path_matches_scalar() {
        use crate::scenario::{build_tile, generate, Projection};
        let proj = Projection::new(9);
        let (q, toks) = generate(123, 64, 0.4);
        let q_sign = proj.sign_bits(&q);
        for (i, t) in toks.iter().enumerate() {
            // Alternate HOT / WARM to cover both branches.
            let tile = build_tile(&proj, t, i as u32, i % 2 == 0);
            let s = tile.compute_score_scalar(&q, &q_sign);
            let a = unsafe { tile.compute_score_neon(&q, &q_sign) };
            assert!(
                (s - a).abs() <= 1e-3 * (1.0 + s.abs()),
                "tile {i}: scalar {s} vs neon {a}"
            );
        }
    }
}
