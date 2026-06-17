//! CCOS elastic KV-cache manager — drives the §4 "Soft-Paging" policy over a
//! **contiguous arena** of tiles, with three states:
//!
//! - **HOT**  — full tile (latent + residual), 128 B.
//! - **WARM** — residual bitmap masked/freed (`FLAG_WARM`, `λ = 0`); the score
//!   falls back to the coarse term. ~32 B reclaimed (logical footprint 96 B).
//! - **COLD** — evicted from the active set; its slot is recycled on the next
//!   `insert` (no I/O here; a real CCOS would snapshot it to the EventLog).
//!
//! `enforce_budget()` keeps the **logical** footprint under a byte budget by
//! paging HOT→WARM (per [`PageOutPolicy`]) and, if needed, evicting →COLD.
//!
//! Note: tiles physically remain 128 B in the arena `Vec`; `live_bytes()` is the
//! *elastic* accounting (HOT 128 / WARM 96 / COLD 0) — i.e. what a packed WARM
//! store would occupy. Masking is O(1) (zero 32 B + flip a flag), no allocation.

use crate::attention::slha_v2::{SciRustSlhaTile, D_C, FLAG_WARM, RESIDUAL_WORDS};

/// Logical footprint of a full (HOT) tile.
pub const HOT_BYTES: usize = 128;
/// Logical footprint of a WARM tile (residual's 32 B reclaimed).
pub const WARM_BYTES: usize = HOT_BYTES - RESIDUAL_WORDS * 8; // 96

/// Soft-Paging state of a slot (spec §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileState {
    Hot,
    Warm,
    Cold,
}

/// Order in which HOT tiles are paged out (HOT→WARM) under memory pressure.
///
/// Note this only governs the **paging** phase. Eviction (WARM/HOT→COLD), which
/// only kicks in once paging the whole working set is not enough, is **always**
/// causal (oldest-inserted first) — dropping a token entirely is a harder loss
/// than freeing its residual, so it should hit the most causally-distant context
/// regardless of `σ_E`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PageOutPolicy {
    /// **Hybrid (recommended, default).** Page out the lowest-`σ_E` tiles first
    /// — their 1-bit residual matters least, so WARM is near-lossless there
    /// (cf. §7.2) — then, if still over budget, evict the oldest by age. Best of
    /// both: free residuals where they hurt least, drop tokens where they matter
    /// least causally.
    #[default]
    LowestImpactFirst,
    /// Pure causal: page out the oldest-inserted tiles first (causal distance,
    /// §4). Eviction order is unchanged (also oldest-first).
    OldestFirst,
}

/// Elastic KV-cache over a contiguous arena of [`SciRustSlhaTile`].
pub struct ElasticKvCache {
    tiles: Vec<SciRustSlhaTile>,
    state: Vec<TileState>,
    seq: Vec<u64>, // insertion order (paging tie-break + eviction), survives reuse
    free: Vec<usize>,
    budget_bytes: usize,
    policy: PageOutPolicy,
    next_seq: u64,
}

impl ElasticKvCache {
    pub fn new(budget_bytes: usize, policy: PageOutPolicy) -> Self {
        Self {
            tiles: Vec::new(),
            state: Vec::new(),
            seq: Vec::new(),
            free: Vec::new(),
            budget_bytes,
            policy,
            next_seq: 0,
        }
    }

    /// Convenience constructor with the recommended default policy (the hybrid
    /// [`PageOutPolicy::LowestImpactFirst`]: page by `σ_E`, evict by age).
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self::new(budget_bytes, PageOutPolicy::default())
    }

    /// Insert a HOT tile, reusing a recycled (COLD) slot when available. Returns
    /// the slot id.
    pub fn insert(&mut self, tile: SciRustSlhaTile) -> usize {
        let slot = if let Some(s) = self.free.pop() {
            self.tiles[s] = tile;
            self.state[s] = TileState::Hot;
            self.seq[s] = self.next_seq;
            s
        } else {
            self.tiles.push(tile);
            self.state.push(TileState::Hot);
            self.seq.push(self.next_seq);
            self.tiles.len() - 1
        };
        self.next_seq += 1;
        slot
    }

    /// Fused attention score for a (non-COLD) slot. WARM slots return the coarse
    /// term only (the kernel honours `FLAG_WARM`).
    pub fn score(&self, slot: usize, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        self.tiles[slot].compute_score(q_coarse, q_sign)
    }

    /// Score the query against every live (non-COLD) tile: `(slot, score)`.
    pub fn score_all(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> Vec<(usize, f32)> {
        (0..self.tiles.len())
            .filter(|&s| self.state[s] != TileState::Cold)
            .map(|s| (s, self.tiles[s].compute_score(q_coarse, q_sign)))
            .collect()
    }

    pub fn state(&self, slot: usize) -> TileState {
        self.state[slot]
    }

    /// HOT → WARM: mask/free the 32-byte residual bitmap (zero it, drop λ, set
    /// the flag). No I/O, no allocation.
    pub fn page_out(&mut self, slot: usize) {
        if self.state[slot] == TileState::Hot {
            self.tiles[slot].residual_bitmap = [0u64; RESIDUAL_WORDS];
            self.tiles[slot].dynamic_lambda = 0.0;
            self.tiles[slot].flags |= FLAG_WARM;
            self.state[slot] = TileState::Warm;
        }
    }

    /// Evict a slot (→ COLD) and recycle it for a future `insert`.
    pub fn evict(&mut self, slot: usize) {
        if self.state[slot] != TileState::Cold {
            self.state[slot] = TileState::Cold;
            self.free.push(slot);
        }
    }

    /// Elastic logical footprint: Σ over live slots (HOT 128, WARM 96, COLD 0).
    pub fn live_bytes(&self) -> usize {
        self.state
            .iter()
            .map(|s| match s {
                TileState::Hot => HOT_BYTES,
                TileState::Warm => WARM_BYTES,
                TileState::Cold => 0,
            })
            .sum()
    }

    /// `(hot, warm, cold)` slot counts.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut c = (0usize, 0usize, 0usize);
        for s in &self.state {
            match s {
                TileState::Hot => c.0 += 1,
                TileState::Warm => c.1 += 1,
                TileState::Cold => c.2 += 1,
            }
        }
        c
    }

    /// Bring the logical footprint under `budget_bytes` in two phases:
    ///
    /// 1. **Page** HOT→WARM in [`PageOutPolicy`] order (default hybrid: lowest
    ///    `σ_E` first — free the residual where it hurts least).
    /// 2. If still over budget, **evict** live tiles →COLD, **always oldest
    ///    first** (causal distance) — dropping a token is the harder loss, so it
    ///    targets the most distant context regardless of `σ_E`.
    pub fn enforce_budget(&mut self) {
        if self.live_bytes() <= self.budget_bytes {
            return;
        }

        let mut hot: Vec<usize> = (0..self.tiles.len())
            .filter(|&s| self.state[s] == TileState::Hot)
            .collect();
        match self.policy {
            PageOutPolicy::LowestImpactFirst => hot.sort_by(|&a, &b| {
                self.tiles[a]
                    .residual_sigma
                    .partial_cmp(&self.tiles[b].residual_sigma)
                    .unwrap_or(core::cmp::Ordering::Equal)
            }),
            PageOutPolicy::OldestFirst => hot.sort_by_key(|&s| self.seq[s]),
        }
        for s in hot {
            if self.live_bytes() <= self.budget_bytes {
                return;
            }
            self.page_out(s);
        }

        // Still over budget: evict oldest live tiles entirely.
        let mut live: Vec<usize> = (0..self.tiles.len())
            .filter(|&s| self.state[s] != TileState::Cold)
            .collect();
        live.sort_by_key(|&s| self.seq[s]);
        for s in live {
            if self.live_bytes() <= self.budget_bytes {
                return;
            }
            self.evict(s);
        }
    }
}
