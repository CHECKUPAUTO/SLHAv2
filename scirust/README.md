# SciRust — noyau de référence SLHA v2

Implémentation de référence du mécanisme d'attention **SLHA v2** (Sub-Low Rank
Hybrid Attention) décrit dans [`../SLHAv2.md`](../SLHAv2.md).

## Idée

- Base latente **bas-rang** (`d_c = 128`) stockée en **INT4 signé par groupe** (MX, 64 o + 8 octets d'échelles).
- Résidu de correction **1-bit** via sign-LSH Johnson–Lindenstrauss
  (`d_s = 256` bits, 32 o).
- Score fusionné continu + binaire (`popcount`), eq. (2.3) du paper.
- Tuile alignée cache : **128 octets exacts, zéro padding** (garanti par test).

## Build / test / mesure

```sh
cargo test                                       # 16 tests : Hamming, layout 128 o, zero-point INT4, WARM,
                                                 # sign-LSH, Jacobi, PCA, INT4 groupé (MX), NF4, sortie d'attention,
                                                 # SGD task-aware, AVX2≡scalaire (+ NEON sur ARM)
cargo run --example measure --release            # rho fixé : fidélité, HOT vs WARM, débit scalaire vs AVX2
cargo run --example measure_learned --release    # base apprise par PCA + INT4 groupé (MX)
cargo run --example bench_vs_fp16 --release       # SLHA 128 o vs clé bf16 256 o : débit & trafic mémoire
cargo run --example attention_fidelity --release  # fidélité de la sortie softmax·V (proxy perplexité)
cargo run --example learn_projection --release    # projection apprise (task-aware) vs PCA
```

**Zéro dépendance externe** : le crate compile et se teste entièrement
hors-ligne (PRNG déterministe maison, pas de `rand`/`criterion`).

## Statut (voir §5.1 et §7 du paper)

API sûre (pas de `read_volatile`), sémantique exacte, avec des **chemins SIMD
AVX2 (x86_64) et NEON (aarch64)** dispatché à l'exécution + repli scalaire
portable, chacun avec un test d'équivalence ≡ scalaire. AVX2 ~×13 vs scalaire ;
NEON **vérifié par cross-compilation** (non chronométré, pas d'ARM sur le banc).
**AVX-512** reste à écrire.

Le prototype de mesure utilise des projections **aléatoires** (non apprises) :
il valide la machinerie *quantification INT4 + résidu 1-bit + ranking*, **pas**
la qualité d'une projection bas-rang apprise (qui ne peut qu'améliorer les
chiffres). Résultat clé : HOT ≥ WARM partout, Soft-Paging quasi sans perte à
faible énergie résiduelle, gains du résidu 1-bit modérés à `d_s = 256`.

## Organisation (`src/`)

| Fichier | Rôle |
|---|---|
| `attention/slha_v2.rs` | Tuile `SciRustSlhaTile` (128 o), kernel `compute_score` (scalaire + AVX2), codecs latents INT4 (MX) / NF4 |
| `linalg.rs` | Décomposition propre symétrique (Jacobi) pour la PCA |
| `learned.rs` | Projection bas-rang : PCA + **SGD task-aware** (`train_projection`) + génération de clés |
| `scenario.rs` | Projection sign-LSH, génération de contexte à énergie résiduelle `rho` contrôlable |
| `metrics.rs` | `dot`, Pearson, Spearman, top-k overlap |
| `rng.rs` | PRNG déterministe (SplitMix64) + échantillonneur gaussien |
| `../tests/slha.rs` | Tests d'intégration (preuves) |
| `../examples/measure.rs` | Prototype de mesure (`rho` fixé) |
| `../examples/measure_learned.rs` | Prototype avec base apprise (PCA) + INT4 groupé (MX) |
| `../examples/bench_vs_fp16.rs` | Débit / trafic mémoire : SLHA (128 o) vs clé bf16 (256 o) |
| `../examples/attention_fidelity.rs` | Fidélité de la sortie `softmax·V` (proxy de perplexité) |
| `../examples/learn_projection.rs` | Projection apprise (task-aware) vs PCA |
