//! Criterion micro-benchmarks for the SLHA v2 score kernel.
//!
//! Run with:  `cargo bench`
//!
//! Benchmarks the per-token `compute_score` over a batch of 1024 tiles, for the
//! scalar / AVX2 / AVX-512 paths and the runtime dispatcher. Throughput is
//! reported in elements/s (= scores/s). The library stays dependency-free;
//! criterion is a dev-dependency only.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use scirust::attention::slha_v2::SciRustSlhaTile;
use scirust::scenario::{build_tile, generate, Projection};

fn score_batch(c: &mut Criterion) {
    let proj = Projection::new(1);
    let (q, toks) = generate(1, 1024, 0.3);
    let q_sign = proj.sign_bits(&q);
    let tiles: Vec<SciRustSlhaTile> = toks
        .iter()
        .enumerate()
        .map(|(i, t)| build_tile(&proj, t, i as u32, false))
        .collect();

    let mut g = c.benchmark_group("compute_score/1024_tiles");
    g.throughput(Throughput::Elements(tiles.len() as u64));

    g.bench_function("scalar", |b| {
        b.iter(|| {
            let mut s = 0.0f32;
            for t in &tiles {
                s += t.compute_score_scalar(black_box(&q), black_box(&q_sign));
            }
            black_box(s)
        })
    });

    g.bench_function("dispatch", |b| {
        b.iter(|| {
            let mut s = 0.0f32;
            for t in &tiles {
                s += t.compute_score(black_box(&q), black_box(&q_sign));
            }
            black_box(s)
        })
    });

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            g.bench_function("avx2", |b| {
                b.iter(|| {
                    let mut s = 0.0f32;
                    for t in &tiles {
                        s += unsafe { t.compute_score_avx2(black_box(&q), black_box(&q_sign)) };
                    }
                    black_box(s)
                })
            });
        }
        if std::is_x86_feature_detected!("avx512f") {
            g.bench_function("avx512", |b| {
                b.iter(|| {
                    let mut s = 0.0f32;
                    for t in &tiles {
                        s += unsafe { t.compute_score_avx512(black_box(&q), black_box(&q_sign)) };
                    }
                    black_box(s)
                })
            });
        }
    }

    g.finish();
}

criterion_group!(benches, score_batch);
criterion_main!(benches);
