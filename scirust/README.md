# SciRust — noyau de référence SLHA v2

Implémentation de référence du mécanisme d'attention **SLHA v2** (Sub-Low Rank
Hybrid Attention) décrit dans [`../SLHAv2.md`](../SLHAv2.md).

## Idée

- Base latente **bas-rang** (`d_c = 128`) stockée en **INT4 signé** (64 o).
- Résidu de correction **1-bit** via sign-LSH Johnson–Lindenstrauss
  (`d_s = 256` bits, 32 o).
- Score fusionné continu + binaire (`popcount`), eq. (2.3) du paper.
- Tuile alignée cache : **128 octets exacts, zéro padding** (garanti par test).

## Build / test / mesure

```sh
cargo test                                    # 11 tests : Hamming (vs réf. brute), layout 128 o,
                                              # zero-point INT4, WARM, sign-LSH, Jacobi, PCA, AVX2≡scalaire
cargo run --example measure --release         # rho fixé : fidélité, HOT vs WARM, débit scalaire vs AVX2
cargo run --example measure_learned --release # base apprise par PCA : HOT vs WARM par spectre
```

**Zéro dépendance externe** : le crate compile et se teste entièrement
hors-ligne (PRNG déterministe maison, pas de `rand`/`criterion`).

## Statut (voir §5.1 et §7 du paper)

API sûre (pas de `read_volatile`), sémantique exacte, avec un **chemin AVX2**
(dispatch à l'exécution + repli scalaire portable, équivalence testée, ~×13 vs
scalaire sur le banc partagé). **NEON** (ARM) et **AVX-512** restent à écrire.

Le prototype de mesure utilise des projections **aléatoires** (non apprises) :
il valide la machinerie *quantification INT4 + résidu 1-bit + ranking*, **pas**
la qualité d'une projection bas-rang apprise (qui ne peut qu'améliorer les
chiffres). Résultat clé : HOT ≥ WARM partout, Soft-Paging quasi sans perte à
faible énergie résiduelle, gains du résidu 1-bit modérés à `d_s = 256`.

## Organisation (`src/`)

| Fichier | Rôle |
|---|---|
| `attention/slha_v2.rs` | Tuile `SciRustSlhaTile`, kernel `compute_score` (scalaire + AVX2), quantification INT4 |
| `linalg.rs` | Décomposition propre symétrique (Jacobi) pour la PCA |
| `learned.rs` | Projection bas-rang **apprise** par PCA + génération de clés à spectre contrôlable |
| `scenario.rs` | Projection sign-LSH, génération de contexte à énergie résiduelle `rho` contrôlable |
| `metrics.rs` | `dot`, Pearson, Spearman, top-k overlap |
| `rng.rs` | PRNG déterministe (SplitMix64) + échantillonneur gaussien |
| `../tests/slha.rs` | Tests d'intégration (preuves) |
| `../examples/measure.rs` | Prototype de mesure (`rho` fixé) |
| `../examples/measure_learned.rs` | Prototype avec base apprise (PCA) |
