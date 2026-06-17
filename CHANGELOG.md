# Changelog

Format basé sur [Keep a Changelog](https://keepachangelog.com/) ; versioning
[SemVer](https://semver.org/). Ce fichier décrit l'état **réel** du code.

## [Unreleased]

### Added
- **Couche d'interfaçage CCOS** (`src/ccos.rs`, §4) : `ElasticKvCache`, un cache
  KV élastique sur **arène contiguë** qui pilote le *Soft-Paging*. Trois états
  HOT (128 o) / WARM (96 o, résidu masqué + `λ = 0`) / COLD (évincé, slot
  recyclé) ; `page_out()` masque/libère les 32 o de `residual_bitmap` en **O(1)**
  sans I/O ni allocation ; `enforce_budget()` borne l'empreinte logique sous un
  budget en octets (`PageOutPolicy::LowestImpactFirst` — plus faible `σ_E`
  d'abord — ou `OldestFirst`) puis évince si nécessaire ; `evict()` recycle le
  slot via free-list. La politique par défaut est l'**hybride** (`Default` :
  pagination par `σ_E`, éviction par ancienneté) ; `with_budget()` la construit.
  Exemple `examples/ccos_softpaging.rs` + 6 tests d'intégration
  (`tests/ccos.rs`). Mesure : pager **la moitié** des tuiles HOT→WARM laisse la
  sortie d'attention à **cos ≈ 0,9995** vs tout-HOT.
- **Calibration de λ** (`examples/calibrate_lambda.rs` + test
  `tests/calibration.rs`, §7.9) : confronte le poids du résidu à une attention
  FP de référence. La forme `λ ∝ σ_E` est **validée** (α* stable sur `rho`) ;
  la constante `√(π/(2·d_s))` **sous-pondère ~4,2×** → constante calibrée
  `C_emp ≈ 0,33` (d_s = 256). La formule analytique reste le défaut conservateur.
- **Coût en cycles** (`examples/cycles.rs`, via `rdtsc`) : ~942 cyc/tuile
  scalaire, ~89 AVX2, ~71 AVX-512 ; balayage de working-set (signal cache
  indirect — compteurs `perf` indisponibles). Complète le bench criterion (ns).

### Fixed
- **Doc & packaging.** Remplacement d'un second crate `scirust` déclaré à la
  racine dont le bench (`benches/score.rs`), la doc (`docs/api.md`) et ce
  changelog décrivaient une **API inexistante** *portée par la tuile*
  (`SciRustSlhaTile::new`, `score_safe`, `enforce_paging`, `TileState`/`TileError`)
  et une tuile de « 104 octets » (à ne pas confondre avec le gestionnaire réel
  `ccos::ElasticKvCache` ajouté ci-dessus, distinct de la tuile). La racine est
  désormais un **workspace Cargo** autour de
  l'unique crate `scirust` ; `docs/api.md` documente l'**API réelle** (tuile de
  **128 octets**, score via `compute_score`) ; le bench cassé est supprimé
  (`scirust/benches/kernel.rs`, fonctionnel, est conservé). Suppression des
  features `avx2/popcnt/neon = []` no-op (la sélection SIMD est *runtime*).

## [0.2.0] - 2026-06-16

### Added
- `SciRustSlhaTile` : tuile **128 octets**, alignée 64, **zéro padding** (latent
  64 o + résidu 32 o + métadonnées 32 o), vérifié par test.
- `compute_score` (eq. 2.3) avec dispatch à l'exécution **AVX-512 > AVX2 >
  scalaire** (x86_64) et **NEON** (aarch64) ; équivalences SIMD ≡ scalaire
  testées (property/fuzz inclus).
- Codecs latents : INT4 **signé** (zero-point), INT4 **par groupe (MX)**, **NF4**
  (codebook normal) — même tuile 128 o.
- Résidu 1-bit sign-LSH + cœur `popcount` (identité de Hamming prouvée vs réf.).
- `learned` : projection **PCA** (`jacobi_eigh`) et projection **apprise
  task-aware** par SGD (`train_projection`), qui bat la PCA sous décalage Q/K.
- Exemples : `measure`, `measure_learned`, `bench_vs_fp16`, `attention_fidelity`,
  `learn_projection`, `basic_usage`.
- Tests : unitaires + intégration + **property/fuzz** + **doctests** (30 au total).
- **criterion** benches (dev-dependency allégée, lib sans dépendance) ; **CI**
  (fmt + clippy `-D warnings` + tests + benches + cross-compile NEON).

### Fixed (par rapport au paper v1)
- Tuile : **128 octets** et non « 104 » (`align(64)` arrondit la taille ; vérifié
  empiriquement `size_of = 128`).
- Déquantification INT4 **signée** `(nibble − 8)·scale` (et non `[0, 15]·scale`).
- Retrait du `read_volatile` (qui bloquait la vectorisation) et de l'import /
  `target_feature(avx2)` trompeurs.

## [0.1.0] - 2026

### Added
- Spécification SLHA v2 (`SLHAv2.md`) et micro-noyau de référence initial.
