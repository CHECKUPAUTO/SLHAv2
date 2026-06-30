# SLHA v2 — Faites tourner une IA locale sans carte graphique

[![CI](https://github.com/CHECKUPAUTO/SLHAv2/actions/workflows/ci.yml/badge.svg)](https://github.com/CHECKUPAUTO/SLHAv2/actions)
[![Rust](https://img.shields.io/badge/rust-2021+-blue.svg)](https://rust-lang.org)

---

Faire tourner une IA en contexte long chez soi exige normalement une carte
graphique hors de prix : à chaque mot généré, l'IA doit se souvenir de tout ce
qui précède, et ce « souvenir » — le **KV-cache** — grossit sans cesse jusqu'à
saturer la VRAM.

**SLHA v2** compresse ce KV-cache en tuiles de **128 octets** alignées
cache-line — l'équivalent d'une ligne de texte par token, au lieu de plusieurs
kilo-octets. Deux lignes de cache de 64 octets, conçues pour rester proches du
processeur (caches L1/L2/L3) plutôt que de dépendre d'un GPU.

### Ce que vous pouvez faire

- **Compresser** chaque clé d'attention : latent bas-rang INT4 (codecs MX par
  groupe, INT4 uniforme ou NF4) + résidu de correction 1-bit (sign-LSH), le tout
  dans une tuile de 128 o (`512 o FP32 → 128 o`, facteur ×4).
- **Scorer** vite : score hybride (produit scalaire continu + `popcount`
  binaire), avec dispatch SIMD à l'exécution — **AVX2 / AVX-512** sur x86_64,
  **NEON** sur aarch64, et **repli scalaire portable** partout ailleurs. Chaque
  chemin SIMD est **vérifié équivalent au scalaire** par test (à 1e-3 près —
  FMA / accumulation réordonnée ; le sous-terme `popcount`, lui, est exact).
- **Piloter la mémoire sous budget** : cache KV élastique CCOS (Soft-Paging) qui
  bascule les tuiles entre les états HOT / WARM / COLD selon l'énergie résiduelle
  et l'ancienneté.
- **Auditer** : le binaire `slha-audit` vérifie à l'exécution le layout des
  tuiles, l'équivalence SIMD / scalaire, la fidélité de sortie, l'invariant de
  budget CCOS et le déterminisme — rapport Markdown ou JSON, avec diff de
  régression (`--diff`).
- **Brancher un agent** : le serveur **MCP** `slha-mcp` (stdio, JSON-RPC 2.0)
  expose 5 outils — `slha.audit`, `slha.explain`, `slha.compress`, `slha.score`,
  `slha.benchmark` — appelables depuis tout client MCP (Claude Code / Desktop).

### Pourquoi ça change la donne
**SLHA v2** compresse la mémoire des IA conversationnelles pour qu'elles tiennent
dans le cache de votre processeur, et pas seulement dans une carte graphique
hors de prix.

> **Concrètement (projection) :** en compressant le KV-cache, un LLM qui
> nécessite ~8 Go de VRAM pourrait tenir sur un CPU avec ~4 Go de RAM. C'est
> l'objectif du projet — **à valider sur un modèle réel** : aucune mesure de bout
> en bout n'existe encore (voir les *Réserves d'honnêteté* plus bas).

---

## Comment ça marche (en 30 secondes)

Quand une IA génère du texte, elle doit se souvenir de tout ce qui a été dit
avant. Ce « souvenir » (le **KV-cache**) grossit à chaque mot et sature la
mémoire.

SLHA v2 compresse chaque souvenir en une **tuile de 128 octets** — l'équivalent
d'une ligne de texte — au lieu de plusieurs kilo-octets normalement.

| Sans SLHA v2 | Avec SLHA v2 |
|---|---|
| ~500 Mo pour 32k tokens¹ | ~4 Mo pour 32k tokens¹ |
| Obligé d'avoir un GPU | Fonctionne sur CPU |
| RAM saturée rapidement | Cache L1/L2/L3 utilisé intelligemment |

### Des fondations vérifiables

**58 tests** au total (51 `scirust` + 7 `slha-mcp`), `clippy -D warnings`,
`cargo doc` propre (warnings = erreurs), MSRV **Rust 1.89**, double licence
**MIT OR Apache-2.0**, et `cargo publish -p scirust --dry-run` qui passe sans
avertissement. Les accélérations SIMD sont mesurées à titre indicatif (banc
partagé, dépendantes du matériel), pas annoncées comme des garanties.

### Périmètre

Workspace Cargo de deux crates **zéro-dépendance externe** (v0.2.0) :
**`scirust`** (noyau de référence, qui fournit aussi le binaire `slha-audit`) et
**`slha-mcp`** (serveur MCP, qui réutilise `scirust::json`). Importé comme
bibliothèque, le noyau n'ajoute **rien** à votre arbre de dépendances.
> ¹ *Projection* par tuile de 128 o/token (basée sur une clé non compressée
> ~15,6 ko/token). Le ratio **mesuré** au niveau kernel est 128 o vs 256 o pour
> une clé bf16 = **2× moins d'octets/token** (§7.5) ; le facteur de bout en bout
> sur un LLM réel reste à mesurer.

> Le dépôt est un **workspace Cargo** : toutes les commandes ci-dessous se
> lancent depuis la racine.

---

## Installation (30 secondes)

```bash
# Option 1 : One-click installer
curl -sSL https://raw.githubusercontent.com/CHECKUPAUTO/SLHAv2/master/install.sh | bash

# Option 2 : Manuel
git clone https://github.com/CHECKUPAUTO/SLHAv2.git
cd SLHAv2
cargo build --release
```

**Prérequis :** [Rust](https://rustup.rs) (si pas installé, le script le fait pour vous).

---

## Premier essai

```bash
# Voir ce que ça donne sur votre machine
cargo run --example basic_usage

# Mesure de performance complète
cargo run --example measure --release
```

Sortie de `basic_usage` :
```
Score: -8.000000
Tile is in HOT mode (full fidelity)
Dequantized latent[0..4]: [-4.0, -4.0, -4.0, -4.0]
```

---

## Auditer le système (`slha-audit`)

Un outil dédié vérifie que tout est sain et **génère un rapport** (Markdown ou
JSON) — pratique en CI, pour un agent, ou comme trace d'audit :

```bash
cargo run --bin slha-audit                              # rapport Markdown lisible
cargo run --bin slha-audit -- --json --out audit.json   # rapport machine (JSON)
cargo run --bin slha-audit -- --diff audit.json         # diff vs un rapport antérieur (régression)
```

Il contrôle, **à l'exécution** : le layout de tuile (128 o, zéro padding,
alignement), l'**équivalence SIMD ≡ scalaire** (le chemin AVX-512/NEON rend le
même score que la référence), les features CPU + niveaux de cache, la **fidélité
de sortie** vs attention complète, l'**invariant de budget CCOS**, et le
**déterminisme**. Code de sortie ≠ 0 si un contrôle échoue.

Pour éprouver tout le produit d'un coup (gate qualité + tous les exemples +
rapport horodaté) : `./scripts/stress_test.sh`.

---

## Connecter un agent / LLM (MCP)

Un serveur **MCP** (`slha-mcp`, **zéro dépendance**) expose le noyau et l'audit
SLHA comme **outils appelables par un agent** (Claude Code / Desktop, ou tout
client MCP) :

```bash
cargo build --release -p slha-mcp
claude mcp add slha -- "$(pwd)/target/release/slha-mcp"
```

L'agent dispose alors de 5 outils : `slha.audit`, `slha.explain`,
`slha.compress`, `slha.score`, `slha.benchmark`. Config Claude Desktop, schéma
des outils et exemple de session : [`docs/MCP.md`](docs/MCP.md).

---

## Intégrer SLHA v2 dans mon projet

### Projet Rust

Ajoutez à votre `Cargo.toml` :

```toml
[dependencies]
scirust = { git = "https://github.com/CHECKUPAUTO/SLHAv2" }
```

Puis dans votre code :

```rust
use scirust::attention::slha_v2;

// Compresser un vecteur de clé (128 dims -> 64 octets INT4)
let mon_vecteur = [0.5f32; 128];
let (packed, scale) = slha_v2::quantize_latent(&mon_vecteur);

// Créer une tuile compressée (128 octets)
let tuile = slha_v2::SciRustSlhaTile {
    latent_kv: packed,
    residual_bitmap: [0u64; 4],
    scale,
    dynamic_lambda: 0.5,
    residual_sigma: 0.0,
    token_id: 0,
    position: 0,
    head_id: 0,
    flags: slha_v2::FLAG_HOT,
    group_scales: [255u8; 8],
};

// Calculer le score d'attention (dispatch SIMD automatique)
let q_coarse = [0.0f32; 128];
let q_sign = [0u64; 4];
let score = tuile.compute_score(&q_coarse, &q_sign);
```

### Intégration avec llama.cpp / Ollama / vLLM

Voir le [guide d'intégration](docs/INTEGRATION.md) — **esquisse de conception**
(pseudo-code), pas une intégration livrée.

---

## Documentation

| Document | Pour qui | Contenu |
|---|---|---|
| [**Premiers pas**](docs/GETTING_STARTED.md) | Débutants | Installation, premier essai, concepts |
| [**Connexion MCP**](docs/MCP.md) | Agents / LLM | Brancher un agent sur les outils SLHA (audit, score, benchmark) |
| [**Guide d'intégration**](docs/INTEGRATION.md) | Développeurs | Esquisse pour llama.cpp, Ollama, vLLM |
| [**Spécification**](SLHAv2.md) | Chercheurs | Maths complètes + résultats §7 |
| [**Résultats**](FINDINGS.md) | Curieux | Ce qu'on a mesuré, ce qui marche, ce qui reste |
| [**API Reference**](docs/api.md) | Développeurs | Documentation technique (API réelle) |
| [**Détails du crate**](scirust/README.md) | Développeurs | Organisation de `scirust/` |

---

## État du projet

- ✅ **Mécanisme validé** : **85 tests** workspace (78 `scirust` + 7 `slha-mcp` : unitaires + intégration + property/fuzz + doctests + calibration λ + CCOS), clippy `-D warnings` clean, CI
- ✅ **Performance** : x86 **AVX2 ~×11,5 / AVX-512 ~×14,1** (banc Xeon partagé) ; ARM **NEON ~×5,7** (Jetson Thor AGX 128) — vs scalaire. _Ratios **indicatifs**, dépendants du CPU et de l'auto-vectorisation ; mesurez les vôtres : `cargo run --example cycles --release`._
- ✅ **Multi-plateforme** : x86_64 (AVX2/AVX-512/VPOPCNTDQ) + ARM AArch64 (NEON, **mesuré sur Jetson Thor** ; `sve2` détecté) — kit `examples/platform_report`
- ✅ **Fidélité** : cosinus 0,95–0,997 vs attention complète (sortie `softmax·V`)
- ✅ **Soft-Paging** : cache KV élastique (`ccos::ElasticKvCache`) — pager la moitié des tuiles HOT→WARM laisse la sortie à **cos 0,9995** (`examples/ccos_softpaging`, §4)
- ✅ **Auto-audit + accès agent** : outil `slha-audit` (rapports JSON/Markdown) et serveur **MCP** `slha-mcp` (5 outils, zéro dépendance) — [`docs/MCP.md`](docs/MCP.md)
- 🟡 **Intégration LLM réel** : *esquisse* — guide de conception + croquis pour llama.cpp/vLLM disponibles ([`docs/INTEGRATION.md`](docs/INTEGRATION.md)), **non intégrée** dans un moteur d'inférence ; perplexité non mesurée.

> Réserves d'honnêteté (projections synthétiques, `perf`/perplexité hors banc) :
> voir [`FINDINGS.md`](FINDINGS.md) et `SLHAv2.md` §6–7.

---

## Contribuer

Les contributions sont les bienvenues — voir [`CONTRIBUTING.md`](CONTRIBUTING.md)
et les [issues](https://github.com/CHECKUPAUTO/SLHAv2/issues).

```bash
git clone https://github.com/CHECKUPAUTO/SLHAv2.git
cd SLHAv2
cargo test                              # 85 tests (workspace), doivent passer
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Licence

Dual-licensed: [PolyForm Noncommercial 1.0.0](LICENSE.md) for noncommercial and personal
use; commercial license required for any commercial use.
See [LICENSING.md](LICENSING.md).

— [Forge CHECKUPAUTO](https://github.com/CHECKUPAUTO)
