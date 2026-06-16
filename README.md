# SLHA v2 — Sub-Low Rank Hybrid Attention

Mécanisme d'attention asymétrique pour l'inférence LLM **CPU-bound** : une base
latente bas-rang en **INT4** + un résidu de correction **1-bit** (sign-LSH),
empaquetés dans une **tuile de 128 octets** alignée sur la ligne de cache.

- **Spécification & résultats :** [`SLHAv2.md`](SLHAv2.md)
- **Implémentation de référence** (Rust, **zéro dépendance**) : [`scirust/`](scirust/) — détails dans [`scirust/README.md`](scirust/README.md)

## En bref

- Le score fusionne un produit scalaire continu (latent INT4 déquantifié) et un
  terme binaire `d_s − 2·popcount(q_sign ⊕ B)` (eq. 2.3 du paper).
- Tuile **128 o exacts, zéro padding** ; quantification INT4 **par groupe (MX)**.
- Kernels : scalaire portable + **AVX2** (~×13) + **NEON** (vérifié par
  cross-compilation), avec tests d'équivalence SIMD ≡ scalaire.
- Modes CCOS **HOT / WARM** (« Soft-Paging » du résidu).

## Démarrer

```sh
cd scirust
cargo test                                       # 14 tests (preuves + équivalences SIMD)
cargo run --example measure --release            # fidélité & débit (rho fixé)
cargo run --example measure_learned --release    # base apprise par PCA + INT4 groupé
cargo run --example bench_vs_fp16 --release      # trafic mémoire vs clé bf16
cargo run --example attention_fidelity --release # fidélité de la sortie softmax·V
```

## État (résumé des mesures — voir §7 du paper)

- **Sortie d'attention : cosinus 0,95–0,997** vs FP (le softmax absorbe
  l'essentiel de l'erreur de score — métrique la plus proche de la perplexité).
- **Débit : ~2,5× plus de tokens/s** qu'une référence bf16 (2× moins
  d'octets/token).
- **Honnêteté :** les projections `Z` sont aléatoires (non apprises) ; les
  compteurs de cache matériels (§6.1) et la perplexité réelle (§6.3) ne sont pas
  mesurables hors d'un vrai banc / modèle — réserves explicites dans le paper.
