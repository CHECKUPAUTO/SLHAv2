# SLHA v2 — Faites tourner une IA locale sans carte graphique

[![CI](https://github.com/CHECKUPAUTO/SLHAv2/actions/workflows/ci.yml/badge.svg)](https://github.com/CHECKUPAUTO/SLHAv2/actions)
[![Rust](https://img.shields.io/badge/rust-2021+-blue.svg)](https://rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green.svg)](#licence)

---

**SLHA v2** compresse la mémoire des IA conversationnelles pour qu'elles tiennent
dans le cache de votre processeur, et pas seulement dans une carte graphique
hors de prix.

> **Concrètement :** un LLM qui a normalement besoin de 8 Go de VRAM peut tourner
> avec SLHA v2 sur un PC portable avec 4 Go de RAM, sans ralentissement.

---

## Comment ça marche (en 30 secondes)

Quand une IA génère du texte, elle doit se souvenir de tout ce qui a été dit
avant. Ce « souvenir » (le **KV-cache**) grossit à chaque mot et sature la
mémoire.

SLHA v2 compresse chaque souvenir en une **tuile de 128 octets** — l'équivalent
d'une ligne de texte — au lieu de plusieurs kilo-octets normalement.

| Sans SLHA v2 | Avec SLHA v2 |
|---|---|
| ~500 Mo pour 32k tokens | ~4 Mo pour 32k tokens |
| Obligé d'avoir un GPU | Fonctionne sur CPU |
| RAM saturée rapidement | Cache L1/L2/L3 utilisé intelligemment |

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

- ✅ **Mécanisme validé** : **50 tests** (unitaires + intégration + property/fuzz + doctests + calibration λ + CCOS), clippy `-D warnings` clean, CI
- ✅ **Performance** : x86 **AVX2 ~×11,5 / AVX-512 ~×14,1** ; ARM **NEON ~×5,7** (mesuré sur Jetson Thor AGX 128) — vs scalaire
- ✅ **Multi-plateforme** : x86_64 (AVX2/AVX-512/VPOPCNTDQ) + ARM AArch64 (NEON, **mesuré sur Jetson Thor** ; `sve2` détecté) — kit `examples/platform_report`
- ✅ **Fidélité** : cosinus 0,95–0,997 vs attention complète (sortie `softmax·V`)
- ✅ **Soft-Paging** : cache KV élastique (`ccos::ElasticKvCache`) — pager la moitié des tuiles HOT→WARM laisse la sortie à **cos 0,9995** (`examples/ccos_softpaging`, §4)
- ✅ **Auto-audit + accès agent** : outil `slha-audit` (rapports JSON/Markdown) et serveur **MCP** `slha-mcp` (5 outils, zéro dépendance) — [`docs/MCP.md`](docs/MCP.md)
- ⏳ **Intégration LLM réel** (greffon KV-cache llama.cpp/vLLM) + perplexité : à venir (hors banc actuel)

> Réserves d'honnêteté (projections synthétiques, `perf`/perplexité hors banc) :
> voir [`FINDINGS.md`](FINDINGS.md) et `SLHAv2.md` §6–7.

---

## Contribuer

Les contributions sont les bienvenues — voir [`CONTRIBUTING.md`](CONTRIBUTING.md)
et les [issues](https://github.com/CHECKUPAUTO/SLHAv2/issues).

```bash
git clone https://github.com/CHECKUPAUTO/SLHAv2.git
cd SLHAv2
cargo test                              # 50 tests, doivent passer
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Licence

Distribué sous **double licence**, au choix :

- **MIT** — [`LICENSE-MIT`](LICENSE-MIT)
- **Apache 2.0** — [`LICENSE-APACHE`](LICENSE-APACHE)

Sauf mention contraire, toute contribution soumise sera couverte par cette
double licence, sans condition supplémentaire (cf. Apache-2.0 §5).

— [Forge CHECKUPAUTO](https://github.com/CHECKUPAUTO)
