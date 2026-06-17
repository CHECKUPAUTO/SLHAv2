# SLHA v2 — Sub-Low Rank Hybrid Attention

Mécanisme d'attention asymétrique pour l'inférence LLM **CPU-bound** : une base
latente bas-rang en **INT4** + un résidu de correction **1-bit** (sign-LSH),
empaquetés dans une **tuile de 128 octets** alignée sur la ligne de cache.

- **Spécification & résultats :** [`SLHAv2.md`](SLHAv2.md)
- **Synthèse des findings (court) :** [`FINDINGS.md`](FINDINGS.md)
- **Implémentation de référence** (Rust, lib **sans dépendance**) : [`scirust/`](scirust/) — détails dans [`scirust/README.md`](scirust/README.md)

## En bref

- Le score fusionne un produit scalaire continu (latent INT4 déquantifié) et un
  terme binaire `d_s − 2·popcount(q_sign ⊕ B)` (eq. 2.3 du paper).
- Tuile **128 o exacts, zéro padding** ; codecs latents INT4 **par groupe (MX)** et **NF4**.
- Kernels : scalaire portable + **AVX2 (~×11,5) / AVX-512 (~×14,1)** (x86_64) +
  **NEON** (aarch64, cross-compilé), avec tests d'équivalence SIMD ≡ scalaire.
- Modes CCOS **HOT / WARM** (« Soft-Paging » du résidu).

## Démarrer

Le dépôt est un **workspace Cargo** : les commandes se lancent depuis la racine.

```sh
cargo test                            # 30 tests (unitaires + intégration + property/fuzz + doctests)
cargo bench                           # micro-benchs criterion (scalaire / AVX2 / AVX-512)
cargo build --workspace --all-targets # lib + tests + benches + exemples

# Exemples :  cargo run -p scirust --release --example <nom>
#   basic_usage · measure · measure_learned · bench_vs_fp16 · attention_fidelity · learn_projection
```

API : [`docs/api.md`](docs/api.md) — détails du crate : [`scirust/README.md`](scirust/README.md).

## État (résumé des mesures — voir §7 du paper)

- **Sortie d'attention : cosinus 0,95–0,997** vs FP (le softmax absorbe
  l'essentiel de l'erreur de score — métrique la plus proche de la perplexité).
- **Débit : ~2,5× plus de tokens/s** qu'une référence bf16 (2× moins
  d'octets/token).
- **Honnêteté :** les projections `Z` sont aléatoires (non apprises) ; les
  compteurs de cache matériels (§6.1) et la perplexité réelle (§6.3) ne sont pas
  mesurables hors d'un vrai banc / modèle — réserves explicites dans le paper.
