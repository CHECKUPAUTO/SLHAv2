# Plan d'action — Lever les limitations de SLHA v2

## Phase 1 — Intégration LLM réel

### 1.1 Intégration dans llama.cpp (modèle GGUF)

| Étape | Description | Composants | Durée |
|---|---|---|---|
| **1.1.1** | Forker llama.cpp et ajouter SLHAv2 en sous-module | `extern/SLHAv2` | 1h |
| **1.1.2** | Écrire le backend C d'encodage KV : `slha_encode_key()` | `slha-c/src/lib.rs` | 3-5j |
| **1.1.3** | Remplacer le stockage KV par un tableau de `SlhaTile` | `llama.cpp` → `llama_kv_cache` | 2-3j |
| **1.1.4** | Modifier la boucle d'attention : remplacer K·Q par `slha_process_tile()` | `llama.cpp` → `llama_build_attention()` | 3-5j |
| **1.1.5** | Ajouter le flag `--kv-cache slha` dans la CLI | `common/common.cpp` | 1j |
| **1.1.6** | Mesurer la perplexité sur WikiText-2, PTB, PG-19 | `perplexity` benchmark | 2-3j |
| **1.1.7** | Mesurer le débit tokens/s de 1K à 128K tokens | `benchmark` | 1-2j |
| **1.1.8** | Mesurer les cache misses via `perf stat` | `perf stat -e L1-dcache-load-misses,...` | 1j |

### 1.2 Intégration dans vLLM (modèle PyTorch/HuggingFace)

| Étape | Description | Durée |
|---|---|---|
| **1.2.1** | Implémenter un kernel CUDA/Triton SLHA pour le scoring GPU | 5-10j |
| **1.2.2** | Remplacer PagedAttention par SLHAAttention avec stockage en tuiles | 3-5j |
| **1.2.3** | Mesurer throughput et perplexité vs attention FP16 | 2-3j |

### 1.3 Intégration dans un moteur Rust custom

| Étape | Description | Durée |
|---|---|---|
| **1.3.1** | Écrire `integration_llm.rs` avec candle/burn + remplacement KV-cache | 5-10j |
| **1.3.2** | Ajouter le pipeline encode→compress→score via `LearnedModel` | 2-3j |
| **1.3.3** | Comparer distribution des scores sur activations réelles | 1-2j |

## Phase 2 — Entraînement conjoint des projections

### 2.1 Pipeline d'entraînement

| Étape | Description | Composants | Durée |
|---|---|---|---|
| **2.1.1** | Transformer `train_projection()` en module autonome exportable | `scirust/src/learned.rs` | 2j |
| **2.1.2** | Exemple `train_on_real_activations.rs` avec candle/hf | `scirust/examples/` | 3-5j |
| **2.1.3** | Ajouter la perte de perplexité comme objectif (pas seulement le score) | `scirust/src/learned.rs` ou nouveau module | 5j |
| **2.1.4** | Entraîner sur vrai modèle une projection pré-RoPE (axe A1) | Expérience reproductible | 5-10j |
| **2.1.5** | Créer un format de fichier pour les projections (P, Z, config) | Nouveau module `scirust/src/weights.rs` | 2j |

### 2.2 Calibration automatique de λ

| Étape | Description | Durée |
|---|---|---|
| **2.2.1** | Implémenter la calibration λ sur modèle réel (recherche de ligne) | 2-3j |
| **2.2.2** | Capturer σ_E sur activations réelles et ajuster la formule | 1-2j |

## Phase 3 — Performance et portabilité

### 3.1 Chemin SIMD SVE2 (ARM)

| Étape | Description | Durée |
|---|---|---|
| **3.1.1** | Diagnostic nightly Rust, sinon écrire le kernel en `asm!` inline | 1j + 5j |
| **3.1.2** | Mesurer sur appareil SVE2 (Jetson Thor, Graviton 4) vs NEON | 2-3j |
| **3.1.3** | Ajouter le dispatch runtime SVE2 → NEON | 1j |

### 3.2 Compression des valeurs V

| Étape | Description | Durée |
|---|---|---|
| **3.2.1** | Étendre la tuile pour inclure V compressé (INT4/NF4) | 5-10j |
| **3.2.2** | Mesurer l'impact sur la fidélité de la sortie | 2j |

### 3.3 Benchmark standardisé multi-arch

| Étape | Description | Durée |
|---|---|---|
| **3.3.1** | Configurer CI multi-arch (self-hosted ARM + x86) | 2-3j |
| **3.3.2** | Automatiser `slha-audit --perf` dans la CI | 1j |

### 3.4 Bindings additionnels

| Étape | Description | Durée |
|---|---|---|
| **3.4.1** | Binding WebAssembly (`slha-wasm`) via wasm-pack | 5j |
| **3.4.2** | Binding C++ header-only (`slha.hpp`) | 2j |
| **3.4.3** | Binding Java (JNI) pour Android/JVM | 5j |

## Phase 4 — Améliorations algorithmiques

### 4.1 Validation des compteurs de cache (perf)

| Étape | Description | Durée |
|---|---|---|
| **4.1.1** | Instrumenter `run_bench()` avec `perf_event_open()` direct | 2-3j |
| **4.1.2** | Mesurer les cache misses L1/L2/LLC sur Xeon + ARM | 2j |

### 4.2 Sparsification / attention sinks

| Étape | Description | Durée |
|---|---|---|
| **4.2.1** | Pruning de têtes basé sur la masse d'attention cumulée | 5j |
| **4.2.2** | Window attention : N derniers tokens HOT, reste WARM/COLD | 3j |

### 4.3 Résidu multi-bit intégré

| Étape | Description | Durée |
|---|---|---|
| **4.3.1** | Intégrer QuantResidual (A4) dans le pipeline d'encodage | 3j |
| **4.3.2** | Mesurer le gain sur modèle réel (1-bit vs 2-bit vs 4-bit) | 3-5j |

## Phase 5 — Outillage et déploiement

| Étape | Description | Durée |
|---|---|---|
| **5.1** | Emballage Debian (.deb) avec lib + en-tête C + binaires | 2j |
| **5.2** | Image Docker `checkupauto/slha-v2` | 1j |
| **5.3** | Extension MCP : outils `slha.train_projection`, `slha.page_out`, `slha.evict` | 3j |
| **5.4** | Publication sur crates.io (`cargo publish -p scirust`) | 1j |
| **5.5** | Publication du package Python sur PyPI (`pip install slha-core`) | 1j |

## Priorités

| Priorité | Tâche | Impact |
|---|---|---|
| **P0** | 1.1 Intégration llama.cpp | Première mesure réelle de perplexité et débit |
| **P0** | 2.1 Entraînement conjoint des projections | Levier #1 : meilleure projection apprise |
| **P1** | 1.2 Intégration vLLM | Validation sur GPU / PyTorch |
| **P1** | 3.1 Chemin SVE2 | Portabilité ARM complète |
| **P1** | 4.1 Valider compteurs cache | Prouver la thèse du mur de bande passante |
| **P2** | 3.2 Compression des V | Diviser la mémoire KV par 2 supplémentaires |
| **P2** | 4.3 Résidu multi-bit intégré | Amélioration de la fidélité à ρ élevé |
| **P2** | 5.4–5.5 Publication | Mise à disposition de la communauté |
