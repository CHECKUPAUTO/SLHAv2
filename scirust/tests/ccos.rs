//! Integration tests for the CCOS elastic KV-cache manager (§4 Soft-Paging).

use scirust::ccos::{ElasticKvCache, PageOutPolicy, TileState, HOT_BYTES};
use scirust::scenario::{build_tile, generate, Projection};

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
