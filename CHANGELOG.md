# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-17

### Added
- Core micro-kernel `SciRustSlhaTile` with 104-byte tile layout
- AVX2-accelerated XOR for binary residual scoring
- Hardware POPCNT for Hamming distance computation
- Scalar fallback for non-AVX2 platforms
- Signed 4-bit dequantization with centered nibble convention
- `TileState` enum (Hot/Warm/Cold) for CCOS soft-paging
- `enforce_paging()` for HOT→WARM state transition
- `score_safe()` safe wrapper with error handling
- Compile-time tile size assertion (104 bytes)
- Unit tests for dequantization, popcount, paging, and edge cases
- Criterion benchmarks for `compute_tile_score`, `score_safe`, and `enforce_paging`
- CI workflow (test, clippy, bench)
- README with quick start guide (English)
- CHANGELOG

### Fixed
- **Critical**: 4-bit dequantization now centers nibbles `[-7, +8]` instead of unsigned `[0, 15]`
- Struct layout: `#[repr(C)]` without alignment padding, verified at compile time
- Removed unused `std::arch::x86_64::*` wildcard import
- Removed `read_volatile` that prevented compiler optimizations
- Added `# Safety` documentation on all unsafe functions
- Added `debug_assert!` null-pointer checks
