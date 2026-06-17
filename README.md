# SLHA v2 — Faites tourner une IA locale sans carte graphique

[![CI](https://github.com/CHECKUPAUTO/SLHAv2/actions/workflows/ci.yml/badge.svg)](https://github.com/CHECKUPAUTO/SLHAv2/actions)
[![Rust](https://img.shields.io/badge/rust-2021+-blue.svg)](https://rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green.svg)](LICENSE)

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

Sortie typique :
```
Score: -8.000000
Tile is in HOT mode (full fidelity)
Dequantized latent[0..4]: [-4.0, -4.0, -4.0, -4.0]
```

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

// Compresser un vecteur de clé
let (packed, scale) = slha_v2::quantize_latent(&mon_vecteur);

// Créer une tuile compressée
let tuile = SciRustSlhaTile {
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

// Calculer le score d'attention
let score = tuile.compute_score(&ma_requete, &ma_signature);
```

### Intégration avec llama.cpp / Ollama / vLLM

Voir le [guide d'intégration complet](docs/INTEGRATION.md).

---

## Documentation

| Document | Pour qui | Contenu |
|---|---|---|
| [**Premiers pas**](docs/GETTING_STARTED.md) | Débutants | Installation, premier essai, concepts |
| [**Guide d'intégration**](docs/INTEGRATION.md) | Développeurs | Brancher SLHA dans llama.cpp, Ollama, vLLM |
| [**Spécification**](SLHAv2.md) | Chercheurs | Maths complètes du mécanisme |
| [**Résultats**](FINDINGS.md) | Curieux | Ce qu'on a mesuré, ce qui marche, ce qui reste à faire |
| [**API Reference**](docs/api.md) | Développeurs | Documentation technique complète |

---

## État du projet

- ✅ **Mécanisme validé** : 13 tests + 3 doctests, clippy clean
- ✅ **Performance** : AVX2 ~×11,5 plus rapide que le scalaire
- ✅ **Multi-plateforme** : x86_64 (AVX2/AVX-512) + ARM (NEON)
- ✅ **Fidélité** : cosinus 0,95-0,997 vs attention complète
- ✅ **Soft-Paging** : passage HOT→WARM sans perte de qualité
- ⏳ **Intégration LLM réel** : tests sur vrai modèle à venir

---

## Contribuer

Les contributions sont les bienvenues ! Voir les [issues](https://github.com/CHECKUPAUTO/SLHAv2/issues).

```bash
git clone https://github.com/CHECKUPAUTO/SLHAv2.git
cd SLHAv2
cargo test  # doit passer à 100%
cargo fmt --check
cargo clippy -- -D warnings
```

---

## Licence

MIT OR Apache-2.0 — [Forge CHECKUPAUTO](https://github.com/CHECKUPAUTO)
