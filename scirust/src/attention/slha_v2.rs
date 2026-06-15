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
    pub fn compute_score(&self, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        // 1. Continuous low-fidelity term: <q_coarse, dequant(latent)>.
        //    Materialise the latent first so the dot product is a clean,
        //    auto-vectorisable loop over two contiguous slices.
        let k = self.dequant_latent();
        let mut coarse = 0.0f32;
        for d in 0..D_C {
            coarse += q_coarse[d] * k[d];
        }

        // 2. WARM: residual paged out -> latent base only.
        if self.is_warm() {
            return coarse;
        }

        // 3. Binary 1-bit correction: λ · (d_s - 2·popcount(q_sign ^ B)).
        //    popcount(XOR) is the Hamming distance; d_s - 2·Hamming is the
        //    signed dot product of the two ±1 sign vectors.
        let mut hamming = 0u32;
        for w in 0..RESIDUAL_WORDS {
            hamming += (q_sign[w] ^ self.residual_bitmap[w]).count_ones();
        }
        let residual_score = D_S as f32 - 2.0 * hamming as f32;
        coarse + self.dynamic_lambda * residual_score
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
}
