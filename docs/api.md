# SLHA v2 — API Reference

Référence de l'API **réelle** du crate `scirust` (vérifiée contre le code et les
tests). Pour la spécification et les mesures, voir [`../SLHAv2.md`](../SLHAv2.md)
et [`../FINDINGS.md`](../FINDINGS.md).

> ⚠️ Une version antérieure de ce fichier décrivait une API inexistante
> (`new`, `score_safe`, `enforce_paging`, `TileState`, `TileError`, …) et une
> tuile de « 104 octets ». **C'est faux** : voir l'API ci-dessous (tuile de
> **128 octets**, score via `compute_score`). Corrigé.

## Modules (`scirust::…`)

| Module | Rôle |
|---|---|
| `attention::slha_v2` | Tuile + kernel de score fusionné (eq. 2.3), quantizers INT4/NF4 |
| `metrics` | `dot`, `cosine`, `rel_l2`, `pearson`, `spearman`, `topk_overlap`, `rms`, `softmax_into` |
| `rng` | PRNG déterministe `Rng` (SplitMix64) + gaussien |
| `linalg` | `jacobi_eigh` (décomposition propre symétrique, pour la PCA) |
| `learned` | `LearnedModel` (PCA + SGD task-aware), `train_projection`, `gen_keys` |
| `scenario` | `Projection` (sign-LSH), `build_tile`, `generate` (données synthétiques) |

## Constantes (`attention::slha_v2`)

```rust
pub const D_C: usize = 128;            // dim latente (INT4)
pub const D_S: usize = 256;            // bits de résidu sign-LSH
pub const LATENT_BYTES: usize = 64;    // D_C / 2
pub const RESIDUAL_WORDS: usize = 4;   // D_S / 64
pub const N_GROUPS: usize = 8;         // groupes de micro-échelles
pub const GROUP_DIM: usize = 16;       // D_C / N_GROUPS

pub const FLAG_HOT: u16 = 0;           // tuile complète (latent + résidu)
pub const FLAG_WARM: u16 = 1 << 0;     // résidu paginé : score = latent seul
pub const FLAG_NF4: u16 = 1 << 1;      // latent codé en NF4 (sinon INT4 uniforme)

pub const NF4_CODEBOOK: [f32; 16];     // 16 quantiles N(0,1) normalisés à [-1, 1]
```

## `SciRustSlhaTile` — **128 octets, align 64, zéro padding**

```rust
#[repr(C, align(64))]
pub struct SciRustSlhaTile {
    pub latent_kv: [u8; 64],          // 64  base h_KV : 128 dims en INT4/NF4 (2/octet)
    pub residual_bitmap: [u64; 4],    // 32  résidu sign-LSH (256 bits)
    pub scale: f32,                   //  4  échelle globale de déquantification
    pub dynamic_lambda: f32,          //  4  poids de correction binaire λ (eq. 3.2)
    pub residual_sigma: f32,          //  4  σ_E par tuile (recalibrage de λ)
    pub token_id: u32,                //  4
    pub position: u32,                //  4
    pub head_id: u16,                 //  2
    pub flags: u16,                   //  2  HOT / WARM / NF4
    pub group_scales: [u8; 8],        //  8  micro-échelles MX : eff(g) = scale·gs[g]/255
}
```

Le test `tile_is_exactly_128_bytes_zero_padding` vérifie `size_of == 128`,
`align_of == 64`, et l'absence de padding.

### Méthodes

```rust
impl SciRustSlhaTile {
    pub fn is_warm(&self) -> bool;     // résidu paginé ?
    pub fn is_nf4(&self) -> bool;      // latent en NF4 ?
    pub fn group_scale(&self, d: usize) -> f32;   // scale·gs[d/16]/255
    pub fn dequant_at(&self, d: usize) -> f32;     // (nibble−8)·eff (INT4) ou NF4[nibble]·eff
    pub fn dequant_latent(&self) -> [f32; 128];

    /// Score fusionné (eq. 2.3) :
    ///   <q_coarse, dequant(latent)> + λ·(d_s − 2·popcount(q_sign ⊕ B))
    /// Dispatch runtime : AVX-512 > AVX2 > scalaire (x86_64), NEON (aarch64).
    /// Les tuiles NF4 passent par le chemin scalaire. En WARM, le terme binaire
    /// est supprimé.
    pub fn compute_score(&self, q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;

    pub fn compute_score_scalar(&self, q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;

    #[cfg(target_arch = "x86_64")]  // # Safety: nécessite la feature CPU correspondante
    pub unsafe fn compute_score_avx2(&self,  q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn compute_score_avx512(&self, q_coarse: &[f32; 128], q_sign: &[u64; 4]) -> f32;
    // (aarch64) compute_score_neon : interne, appelée par le dispatcher.
}
```

> Il n'y a **pas** de constructeur `new`, de `score_safe`/`Result`, ni
> d'`enforce_paging`/`TileState`. Une tuile se construit par littéral de struct
> (cf. exemple) ou via `learned::LearnedModel::encode` / `scenario::build_tile`.
> Pour passer en WARM : `tile.flags |= FLAG_WARM`.

## Quantizers (`attention::slha_v2`)

```rust
// INT4 uniforme signé, une échelle globale. value ≈ (nibble−8)·scale.
pub fn quantize_latent(v: &[f32; 128]) -> ([u8; 64], f32);

// INT4 « micro-scaling » : une échelle par groupe de 16 dims (dans group_scales).
pub fn quantize_latent_grouped(v: &[f32; 128]) -> ([u8; 64], f32, [u8; 8]);

// NF4 (codebook normal) par groupe — même tuile, flag FLAG_NF4 requis au scoring.
pub fn quantize_latent_nf4(v: &[f32; 128]) -> ([u8; 64], f32, [u8; 8]);
```

`LatentCodec { Int4Single, Int4Grouped, Nf4 }` sélectionne le codec via
`LearnedModel::encode_with(key, pos, warm, codec)`.

## Features Cargo

Le crate **n'a pas** de features gating la compilation des chemins SIMD : la
sélection est **à l'exécution** (`std::is_x86_feature_detected!`), avec repli
scalaire portable. (Les anciennes features `avx2/popcnt/neon = []` étaient des
no-op trompeuses — supprimées.) La bibliothèque est **sans dépendance** ;
`criterion` n'est qu'une dev-dependency pour `cargo bench`.

## Performance (mesurée, x86_64 Xeon, banc partagé)

| Chemin | Débit (1024 tuiles) | Rapport |
|---|---|---|
| Scalaire | ~3,0 M scores/s | 1× |
| AVX2 | ~34–38 M scores/s | ~×11,5 |
| AVX-512 | ~40–42 M scores/s | ~×14,1 |

- **Mémoire :** tuile 128 o/token contre 256 o pour une clé bf16 → **~2,5×**
  plus de tokens/s à débit comparable (§7.5).
- **Fidélité :** la sortie d'attention (`softmax·V`) reste à **cosinus
  0,95–0,997** vs FP malgré un score approché (§7.6).

(Voir `SLHAv2.md` §7 pour la méthodologie et les réserves — projections
synthétiques, `perf`/perplexité hors banc.)

## Exemple minimal

```rust
use scirust::attention::slha_v2::{quantize_latent, SciRustSlhaTile, FLAG_HOT};

let mut v = [0.0f32; 128];
for (i, x) in v.iter_mut().enumerate() { *x = ((i as f32) - 64.0) / 16.0; }
let (latent_kv, scale) = quantize_latent(&v);

let tile = SciRustSlhaTile {
    latent_kv,
    residual_bitmap: [0; 4],
    scale,
    dynamic_lambda: 0.5,
    residual_sigma: 0.0,
    token_id: 0, position: 0, head_id: 0,
    flags: FLAG_HOT,
    group_scales: [255; 8], // [255;8] == échelle unique (équiv. INT4 simple)
};

let q_coarse = [0.0f32; 128];
let q_sign = [0u64; 4];
let score = tile.compute_score(&q_coarse, &q_sign); // dispatch SIMD auto
```

Voir aussi `scirust/examples/basic_usage.rs` (exemple exécutable identique).

## Build / test / bench (depuis la racine, workspace)

```sh
cargo test                 # 30 tests (unitaires + intégration + property/fuzz + doctests)
cargo bench                # micro-benchs criterion (scalaire / AVX2 / AVX-512)
cargo run -p scirust --example basic_usage
cargo build --workspace --all-targets   # compile lib + tests + benches + exemples
```

---
*SLHA v2 — référence d'API alignée sur le code (`scirust` v0.2.0).*
