//! Integration tests for the CCOS elastic KV-cache manager (§4 Soft-Paging).

use scirust::ccos::{ElasticKvCache, PageOutPolicy, TileState, HOT_BYTES, WARM_BYTES};
use scirust::rng::Rng;
use scirust::scenario::{build_tile, generate, Projection};

/// Invariants that must hold after every `enforce_budget`.
fn assert_invariants(cache: &ElasticKvCache, budget: usize) {
    // (1) The elastic footprint is within budget — eviction can always reach 0,
    //     so this holds for *any* budget ≥ 0.
    assert!(
        cache.live_bytes() <= budget,
        "footprint {} exceeds budget {budget}",
        cache.live_bytes()
    );
    // (2) Byte accounting is consistent with the HOT/WARM/COLD counts.
    let (h, w, _c) = cache.counts();
    assert_eq!(
        h * HOT_BYTES + w * WARM_BYTES,
        cache.live_bytes(),
        "counts ({h} HOT, {w} WARM) inconsistent with live_bytes"
    );
}

/// Randomised: across many (size, budget, policy) configurations, every
/// `enforce_budget` leaves the cache within budget with consistent accounting,
/// and COLD slots are recycled (the slot vector never exceeds the high-water
/// mark of simultaneously-live tiles).
#[test]
fn prop_enforce_budget_respects_budget_and_recycles() {
    let proj = Projection::new(0xB0D);
    for trial in 0..300u64 {
        let mut rng = Rng::new(trial);
        let n = 1 + (rng.next_u64() % 48) as usize;
        // Budget from 0 up to slightly above all-HOT (covers page-only, evict,
        // and evict-everything regimes).
        let budget = (rng.next_u64() % (n as u64 * HOT_BYTES as u64 + 1)) as usize;
        let policy = if trial.is_multiple_of(2) {
            PageOutPolicy::LowestImpactFirst
        } else {
            PageOutPolicy::OldestFirst
        };
        let mut cache = ElasticKvCache::new(budget, policy);
        let (_q, toks) = generate(trial + 1, n, 0.3);

        for (i, t) in toks.iter().enumerate() {
            cache.insert(build_tile(&proj, t, i as u32, false));
            if rng.next_u64().is_multiple_of(3) {
                cache.enforce_budget();
                assert_invariants(&cache, budget);
            }
        }
        cache.enforce_budget();
        assert_invariants(&cache, budget);

        // Slot recycling: total slots ever allocated ≤ tiles inserted, and COLD
        // slots are reusable — re-inserting fills them without growing the arena.
        let (h0, w0, c0) = cache.counts();
        let total_slots = h0 + w0 + c0;
        for (i, t) in toks.iter().enumerate() {
            cache.insert(build_tile(&proj, t, i as u32, false));
        }
        let (h1, w1, c1) = cache.counts();
        assert!(
            h1 + w1 + c1 <= total_slots + n,
            "arena grew past the live high-water mark (no recycling)"
        );
    }
}

#[test]
fn page_out_masks_residual_and_falls_back_to_coarse() {
    let proj = Projection::new(1);
    let (q, toks) = generate(2, 1, 0.5);
    let q_sign = proj.sign_bits(&q);

    let mut cache = ElasticKvCache::new(usize::MAX, PageOutPolicy::LowestImpactFirst);
    let slot = cache.insert(build_tile(&proj, &toks[0], 0, false));
    let hot = cache.score(slot, &q, &q_sign);

    cache.page_out(slot);
    assert_eq!(cache.state(slot), TileState::Warm);
    let warm = cache.score(slot, &q, &q_sign);

    // WARM == the coarse-only score of a freshly-built WARM tile, and differs
    // from HOT (the residual did contribute).
    let warm_ref = build_tile(&proj, &toks[0], 0, true).compute_score(&q, &q_sign);
    assert!(
        (warm - warm_ref).abs() <= 1e-4 * (1.0 + warm.abs()),
        "{warm} vs {warm_ref}"
    );
    assert!((hot - warm).abs() > 0.0, "residual contributed nothing");
}

#[test]
fn enforce_budget_bounds_live_bytes() {
    let proj = Projection::new(3);
    let budget = 50 * HOT_BYTES;
    let mut cache = ElasticKvCache::new(budget, PageOutPolicy::LowestImpactFirst);
    let (_q, toks) = generate(42, 100, 0.3);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    assert!(cache.live_bytes() > budget);
    cache.enforce_budget();
    assert!(
        cache.live_bytes() <= budget,
        "live={} > budget={budget}",
        cache.live_bytes()
    );
    let (_hot, warm, cold) = cache.counts();
    assert!(warm + cold > 0, "nothing was paged/evicted");
}

#[test]
fn evict_recycles_slots() {
    let proj = Projection::new(4);
    let (_q, toks) = generate(5, 4, 0.3);
    let mut cache = ElasticKvCache::new(usize::MAX, PageOutPolicy::OldestFirst);
    let s0 = cache.insert(build_tile(&proj, &toks[0], 0, false));
    let _s1 = cache.insert(build_tile(&proj, &toks[1], 1, false));
    cache.evict(s0);
    assert_eq!(cache.state(s0), TileState::Cold);
    let s2 = cache.insert(build_tile(&proj, &toks[2], 2, false));
    assert_eq!(s2, s0, "the freed COLD slot should be recycled");
    assert_eq!(cache.state(s2), TileState::Hot);
}

#[test]
fn page_out_policies_differ() {
    let proj = Projection::new(6);
    // σ_E decreasing with slot order (rho decreasing) -> slot 4 is lowest-impact.
    let rhos = [0.9f32, 0.7, 0.5, 0.3, 0.1];
    let n = rhos.len();
    let fill = |c: &mut ElasticKvCache| {
        for (i, &r) in rhos.iter().enumerate() {
            let (_q, toks) = generate(100 + i as u64, 1, r);
            c.insert(build_tile(&proj, &toks[0], i as u32, false));
        }
    };
    let budget = n * HOT_BYTES - 1; // forces exactly one page-out

    let mut lo = ElasticKvCache::new(budget, PageOutPolicy::LowestImpactFirst);
    fill(&mut lo);
    lo.enforce_budget();

    let mut old = ElasticKvCache::new(budget, PageOutPolicy::OldestFirst);
    fill(&mut old);
    old.enforce_budget();

    // OldestFirst pages slot 0; LowestImpactFirst pages the lowest-σ_E (last) slot.
    assert_eq!(old.state(0), TileState::Warm);
    assert_eq!(old.state(n - 1), TileState::Hot);
    assert_eq!(lo.state(n - 1), TileState::Warm);
    assert_eq!(lo.state(0), TileState::Hot);
}

#[test]
fn hybrid_evicts_oldest_not_lowest_impact() {
    // The default hybrid uses *two different keys*: it pages HOT→WARM by σ_E
    // (pinned by `page_out_policies_differ`), but evicts →COLD strictly by age.
    // Under a budget so tight that everything is paged and one tile must still
    // be dropped, the evicted tile must be the OLDEST — not the lowest-σ_E.
    let proj = Projection::new(11);
    let rhos = [0.9f32, 0.7, 0.5, 0.3, 0.1]; // slot 0 oldest & highest σ_E; slot 4 lowest σ_E
    let n = rhos.len();
    let mut cache = ElasticKvCache::with_budget((n - 1) * 96); // all-WARM(480) - one tile
    for (i, &r) in rhos.iter().enumerate() {
        let (_q, toks) = generate(200 + i as u64, 1, r);
        cache.insert(build_tile(&proj, &toks[0], i as u32, false));
    }
    assert_eq!(cache.state(0), TileState::Hot);
    cache.enforce_budget();

    // Exactly one eviction, and it is the oldest (slot 0) — NOT the lowest-σ_E
    // (slot 4), which only got paged to WARM.
    let (_h, _w, cold) = cache.counts();
    assert_eq!(cold, 1, "expected exactly one eviction");
    assert_eq!(cache.state(0), TileState::Cold, "oldest should be evicted");
    assert_eq!(
        cache.state(n - 1),
        TileState::Warm,
        "lowest-σ_E tile should be paged, not evicted"
    );
}

#[test]
fn score_all_skips_cold() {
    let proj = Projection::new(7);
    let (q, toks) = generate(8, 6, 0.3);
    let q_sign = proj.sign_bits(&q);
    let mut cache = ElasticKvCache::new(usize::MAX, PageOutPolicy::OldestFirst);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    cache.evict(2);
    let scored = cache.score_all(&q, &q_sign);
    assert_eq!(scored.len(), 5);
    assert!(scored.iter().all(|&(s, _)| s != 2));
}
