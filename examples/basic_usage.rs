//! SLHA v2 Examples
//!
//! This crate provides executable examples for the SLHA v2 attention scorer.

use scirust::attention::slha_v2::SciRustSlhaTile;

/// Basic usage example: create a tile and compute a score
fn main() {
    // Create a tile with sample data
    let mut latent = [0u8; 64];
    // Pack nibbles: value 0x77 → nibbles (7,7) → centered (0,0)
    // Pack nibbles: value 0x88 → nibbles (8,8) → centered (1,1)
    latent[0] = 0x77; // dims 0,1 = (0.0, 0.0)
    latent[1] = 0x88; // dims 2,3 = (1.0, 1.0)

    let tile = SciRustSlhaTile::new(latent, [0u64; 4], 1.0, 0.5);

    // Create a query with dimension 2 active
    let mut q_coarse = vec![0.0f32; 128];
    q_coarse[2] = 3.0; // Activates dimension 2 at intensity 3.0

    let q_residual = [0u64; 4];

    // Safe API
    match tile.score_safe(&q_coarse, &q_residual) {
        Ok(score) => println!("Score (safe): {:.6}", score),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Unsafe API for maximum performance
    let score_unsafe = unsafe {
        tile.compute_tile_score_unchecked(q_coarse.as_ptr(), q_residual.as_ptr())
    };
    println!("Score (unsafe): {:.6}", score_unsafe);

    // Enforce paging (HOT → WARM)
    let mut warm_tile = SciRustSlhaTile::new(latent, [0xFFFF_FFFF_FFFF_FFFFu64; 4], 1.0, 0.5);
    println!("State before paging: {:?}", warm_tile.state()); // Hot
    warm_tile.enforce_paging();
    println!("State after paging:  {:?}", warm_tile.state()); // Warm
}
