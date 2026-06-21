# Changelog

Format basé sur [Keep a Changelog](https://keepachangelog.com/) ; versioning
[SemVer](https://semver.org/). Ce fichier décrit l'état **réel** du code.

## [Unreleased]

### Added
- **Serveur MCP `slha-mcp`** (nouveau crate du workspace, **zéro dépendance
  externe** — réutilise `scirust::json`) : serveur Model Context Protocol sur
  **stdio** (JSON-RPC 2.0 délimité par lignes) qui expose le noyau et l'auto-audit
  SLHA comme **outils appelables par un agent** (Claude Code / Desktop, ou tout
  client MCP). 5 outils : `slha.audit`, `slha.explain`, `slha.compress`,
  `slha.score`, `slha.benchmark`. Branchement :
  `claude mcp add slha -- .../target/release/slha-mcp`. Guide complet
  `docs/MCP.md`. +7 tests de dispatch → **57 tests** (workspace : 50 scirust + 7
  slha-mcp).
- **Outil d'auto-audit `slha-audit`** (bin) + modules `scirust::audit` et
  `scirust::json` (JSON **sans dépendance** : valeur + sérialiseur + parseur).
  L'audit exécute tous les invariants à l'exécution — layout de tuile (128 o,
  zéro padding, alignement), **équivalence SIMD ≡ scalaire** *live*, features
  CPU + niveaux de cache, **fidélité de sortie** vs attention complète,
  **invariant de budget CCOS**, déterminisme — et rend un rapport **Markdown**
  ou **JSON** (`--json`/`--pretty`/`--out FILE`), avec **diff vs un rapport
  antérieur** (`--diff PRIOR.json`, exit ≠ 0 sur régression). Code de sortie ≠ 0
  si un contrôle échoue. +9 tests (JSON 5, audit 4) → **50 tests** (scirust).
  Réutilisé par le serveur `slha-mcp` (ci-dessus).
- **Prêt pour crates.io / docs.rs** : métadonnées de publication sur `scirust`
  (`keywords`, `categories`, `readme`, `documentation`, `rust-version`) ;
  `cargo publish -p scirust --dry-run` passe (35 fichiers, sans avertissement).
  `slha-mcp` reçoit aussi les métadonnées et une dépendance `scirust` versionnée
  (publiable une fois `scirust` sur crates.io). **MSRV = 1.89** (intrinsèques
  AVX-512 stabilisées en 1.89 ; `usize::is_multiple_of` en 1.87).
- **Fichiers de licence** `LICENSE-MIT` + `LICENSE-APACHE` à la racine (le crate
  déclarait `MIT OR Apache-2.0` sans fournir les textes ; lien `LICENSE` du
  README désormais valide). Conformité double-licence façon écosystème Rust.
- **Harnais de test massif** `scripts/stress_test.sh` : exécute la barrière
  qualité complète (fmt, clippy `-D warnings`, build debug+release, tests
  debug+release, doc, benches, cross-compile aarch64), **lance les 11 exemples**,
  vérifie le **déterminisme** de sortie, propose un mode **soak**, et **génère un
  rapport Markdown + JSON horodaté** sous `target/stress/` (auditable). Lance
  aussi `slha-audit` ; suite à **50 tests** verts.
- **Alignement adaptatif à l'hôte via `build.rs`** (`SciRustSlhaTile`, §3.1) :
  script de build sans dépendance qui sonde la **taille de ligne L1d réelle de
  l'hôte** sur une *build native* (triplet hôte == cible ; `sysfs` Linux ou
  `sysctl` macOS) et émet `cfg(cache_line_128)` pour porter la tuile à
  `align(128)` **uniquement** sur une puce à ligne de 128 o (p. ex. Apple
  Silicon). En cross-compilation, la ligne de l'hôte n'a pas de rapport avec la
  cible : le défaut sûr `align(64)` est conservé. Raffinement de portabilité,
  pas de correctness — la tuile reste **128 o sans padding** dans les deux cas,
  et sur **toutes nos cibles** (x86-64, Thor) le résultat est inchangé
  (`align(64)`). Remplace l'hypothèse « AArch64 ⇒ 128 » retirée ; test
  `tile_is_exactly_128_bytes_zero_padding` rendu cfg-aware. `build.rs` émet
  aussi `rustc-check-cfg` (zéro warning `unexpected_cfgs`).
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
- **CI durcie** : ajout de `cargo doc` (warnings = erreurs), exécution
  bout-en-bout de **`slha-audit`**, `cargo publish -p scirust --dry-run`, et un
  **job MSRV (Rust 1.89)** qui vérifie tout le workspace `--all-targets`.
- **Durcissement NaN des tris flottants** (`metrics::ranks`/`topk_overlap`,
  `learned::fit`, exemple `salient_outliers`) : `partial_cmp().unwrap()` →
  `f32/f64::total_cmp` (ordre total sans panique). Comportement identique sur
  données finies ; supprime un risque de panique sur l'API publique en cas de
  `NaN`/`Inf` en entrée. Repéré par l'audit code.
- **Statut toolchain SVE2 documenté précisément** (roadmap #1 ; paper Future
  Work, `SLHAv2.md` §7.4, `FINDINGS.md`). Vérifié sur `rustc 1.94.1` : la
  **détection** runtime `is_aarch64_feature_detected!("sve2")` est *stable*,
  mais les **intrinsèques** SVE2 (`svdot_s32`…) sont **absentes du
  `core::arch::aarch64` stable** (nightly-only, comme `std::simd`) ; la seule
  voie stable (`asm!` manuel) *compile* mais reste **invérifiable sans appareil
  SVE2** (CI x86 ; la cross-compilation ne type-checke pas la sémantique de
  l'`asm!`). On garde donc **NEON + `cnt`** comme chemin livré, mesuré et
  correct ; SVE2 reste sur la roadmap (défer *justifié par le toolchain*, pas un
  oubli). Aucun changement de code.
- **Alignement de tuile ramené à `align(64)` universel** (`SciRustSlhaTile`,
  §3.1). On avait introduit un `align(128)` conditionnel sur `aarch64` en
  supposant une ligne de cache de 128 o sur le Jetson Thor ; **la mesure de
  l'appareil l'a réfuté** (L1d/L1i/L2 = 64 o — le « 128 » d'*AGX 128* = les
  128 Go de **mémoire unifiée CPU/GPU** LPDDR5X, pas la ligne de cache).
  `align(64)` est correct et optimal sur les deux cibles
  (tuile = 2 lignes de 64 o). Un `align(128)` ne sert que sur les puces à ligne
  de 128 o (p. ex. Apple Silicon) → détection hôte en `build.rs` (**désormais
  implémenté**, cf. *Added* ci-dessus).
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
