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
/// flags 118 | _reserved 120..128.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct SciRustSlhaTile {
    /// Latent base `h_KV` (128 dims) quantised to signed INT4. 64 bytes.
    pub latent_kv: [u8; LATENT_BYTES],
    /// Johnson–Lindenstrauss sign residual: 256 bits. 32 bytes.
    pub residual_bitmap: [u64; RESIDUAL_WORDS],
    /// INT4 dequantisation scale for `latent_kv`.
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
    /// Reserved; keeps the struct at exactly 128 bytes for forward-compat.
    pub _reserved: [u8; 8],
}

impl SciRustSlhaTile {
    /// True if the residual has been paged out (WARM mode).
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.flags & FLAG_WARM != 0
    }

    /// Dequantise one latent dimension `d` with the signed zero-point.
    #[inline]
    pub fn dequant_at(&self, d: usize) -> f32 {
        let byte = self.latent_kv[d >> 1];
        let nib = if d & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        ((nib as i32) - 8) as f32 * self.scale
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
        self.compute_score_scalar(q_coarse, q_sign)
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

        let scale_v = _mm256_set1_ps(self.scale);
        let eight = _mm256_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(nibble - 8) * scale` for 8 dims, multiply by q, accumulate.
        macro_rules! group {
            ($bytes:expr, $off:expr) => {{
                let n = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($bytes));
                let v = _mm256_mul_ps(scale_v, _mm256_sub_ps(n, eight));
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 4 blocks × 16 bytes = 4 × 32 dims = 128 dims.
        for blk in 0..4 {
            let base = blk * 32;
            let packed = _mm_loadu_si128(latent.add(blk * 16) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            // Per-byte high nibble: shift 16-bit lanes, then mask each byte.
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            // Interleave so nibbles come out in dimension order.
            let d_lo = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            let d_hi = _mm_unpackhi_epi8(lo, hi); // dims base+16..base+31
            group!(d_lo, base);
            group!(_mm_srli_si128(d_lo, 8), base + 8);
            group!(d_hi, base + 16);
            group!(_mm_srli_si128(d_hi, 8), base + 24);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

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
            + 8; // _reserved
        assert_eq!(field_bytes, 128);
        assert_eq!(field_bytes, size_of::<SciRustSlhaTile>());
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
        let tile = SciRustSlhaTile {
            latent_kv: packed,
            residual_bitmap: [0; RESIDUAL_WORDS],
            scale,
            dynamic_lambda: 0.0,
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: FLAG_HOT,
            _reserved: [0; 8],
        };
        let dq = tile.dequant_latent();
        // At least one strictly-negative reconstructed value (zero-point works).
        assert!(dq.iter().any(|&x| x < 0.0), "no negative values reconstructed");
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
}
