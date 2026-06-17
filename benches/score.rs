use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scirust::attention::slha_v2::SciRustSlhaTile;

fn bench_tile_score(c: &mut Criterion) {
    let mut latent = [0u8; 64];
    for (i, byte) in latent.iter_mut().enumerate() {
        *byte = ((i as u8) << 4) | (i as u8 & 0x0F);
    }
    let bitmap = [0x1234_5678_9ABC_DEF0u64; 4];
    let tile = SciRustSlhaTile::new(latent, bitmap, 1.0, 0.5);

    let q_coarse: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let q_residual = [0xABCD_EF01_2345_6789u64; 4];

    c.bench_function("compute_tile_score", |b| {
        b.iter(|| unsafe {
            tile.compute_tile_score_unchecked(
                black_box(q_coarse.as_ptr()),
                black_box(q_residual.as_ptr()),
            )
        })
    });
}

fn bench_tile_score_safe(c: &mut Criterion) {
    let mut latent = [0u8; 64];
    for (i, byte) in latent.iter_mut().enumerate() {
        *byte = ((i as u8) << 4) | (i as u8 & 0x0F);
    }
    let bitmap = [0x1234_5678_9ABC_DEF0u64; 4];
    let tile = SciRustSlhaTile::new(latent, bitmap, 1.0, 0.5);

    let q_coarse: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let q_residual = [0xABCD_EF01_2345_6789u64; 4];

    c.bench_function("compute_tile_score_safe", |b| {
        b.iter(|| tile.score_safe(black_box(&q_coarse), black_box(&q_residual)))
    });
}

fn bench_enforce_paging(c: &mut Criterion) {
    let mut latent = [0u8; 64];
    for (i, byte) in latent.iter_mut().enumerate() {
        *byte = ((i as u8) << 4) | (i as u8 & 0x0F);
    }
    let bitmap = [0x1234_5678_9ABC_DEF0u64; 4];

    c.bench_function("enforce_paging", |b| {
        b.iter_with_large_drop(|| {
            let mut tile = SciRustSlhaTile::new(
                black_box(latent),
                black_box(bitmap),
                1.0,
                0.5,
            );
            tile.enforce_paging();
            tile
        })
    });
}

criterion_group!(
    benches,
    bench_tile_score,
    bench_tile_score_safe,
    bench_enforce_paging
);
criterion_main!(benches);
