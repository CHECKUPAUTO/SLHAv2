# Changelog

Format basé sur [Keep a Changelog](https://keepachangelog.com/) ; versioning
[SemVer](https://semver.org/). Ce fichier décrit l'état **réel** du code.

## [Unreleased]

### Added
- **Kit de mesure multi-plateforme** (`examples/platform_report.rs` +
  `scripts/bench_device.sh`) : binaire portable (x86-64 **et** AArch64) qui
  détecte les features SIMD (AVX2/AVX-512/VPOPCNTDQ ou NEON/dotprod/SVE/SVE2),
  **liste tous les niveaux de cache et leur taille de ligne**, vérifie la
  taille/alignement de tuile vs la ligne, affiche le chemin kernel dispatché,
  et mesure le débit (scalaire vs SIMD) en temps mur. A servi à produire les
  **chiffres ARM réels** sur **Jetson Thor AGX 128** (NEON 17,1 M/s vs 3,0
  scalaire = 5,7× ; toutes lignes de cache à 64 o ; `sve2` présent) ; x86 reste
  la baseline serveur.

### Changed / Corrected
- **Alignement de tuile ramené à `align(64)` universel** (`SciRustSlhaTile`,
  §3.1). On avait introduit un `align(128)` conditionnel sur `aarch64` en
  supposant une ligne de cache de 128 o sur le Jetson Thor ; **la mesure de
  l'appareil l'a réfuté** (L1d/L1i/L2 = 64 o — le « 128 » d'*AGX 128* = les
  128 Go LPDDR5X). `align(64)` est correct et optimal sur les deux cibles
  (tuile = 2 lignes de 64 o). Un `align(128)` ne sert que sur les puces à ligne
  de 128 o (p. ex. Apple Silicon) → détection hôte en `build.rs` (roadmap).
- **Popcount résidu vectorisé AVX-512 VPOPCNTDQ** (`hamming_distance`, eq. 2.3) :
  chemin x86-64 *branchless* qui plie les 256 bits du résidu en un seul `vpopcntq`
  (`_mm256_popcnt_epi64`), sélectionné à l'exécution (`avx512vpopcntdq`+`vl`) avec
  repli `count_ones()` (→`POPCNT`/`CNT`). Équivalence bit-à-bit garantie
  (`vpopcntdq_hamming_matches_scalar` sur CPU compatible, compile-checked sinon ;
  `hamming_distance_matches_bruteforce` partout).
- **Constante λ calibrée exposée** (`scenario::LAMBDA_C_CALIBRATED` ≈ 0,33,
  `calibrated_lambda`, `analytic_lambda_c`, §7.9) : option pour le poids du
  résidu corrigé du facteur ~4,2×. `build_tile` garde la constante **analytique**
  par défaut (conservatrice : la calibration optimise la *magnitude*, pas le
  *ranking*). Test `calibrated_lambda_needs_no_further_multiplier` (α\* ≈ 1).
- **Property-tests CCOS randomisés** (`tests/ccos.rs`) :
  `prop_enforce_budget_respects_budget_and_recycles` (300 configs) épingle les
  invariants `live_bytes ≤ budget`, cohérence octets/compteurs, et recyclage des
  slots COLD, sur les deux politiques.
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
