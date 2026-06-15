//! SciRust — reference implementation of the SLHA v2 attention mechanism.
//!
//! SLHA v2 (Sub-Low Rank Hybrid Attention) splits the key/value representation
//! into two asymmetric components:
//!
//! 1. a **low-rank latent base** (`d_c` dims) stored as INT4, and
//! 2. a **1-bit sign-LSH residual** (`d_s` bits) capturing the high-frequency
//!    correction the low-rank base misses.
//!
//! The unnormalised attention score fuses a continuous dot product with a
//! binary (popcount-based) term — see [`attention::slha_v2`] and eq. (2.3) of
//! the spec.
//!
//! This crate is a **correctness-first reference**: the hot path is portable,
//! `unsafe`-free scalar code with a safe, testable API. SIMD specialisation is
//! left as future work (and is now *unblocked*, since the previous reference
//! used `read_volatile`, which forbids auto-vectorisation).

pub mod attention;
pub mod learned;
pub mod linalg;
pub mod metrics;
pub mod rng;
pub mod scenario;

pub use attention::slha_v2::{SciRustSlhaTile, D_C, D_S, LATENT_BYTES, RESIDUAL_WORDS};
