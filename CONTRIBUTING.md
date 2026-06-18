# Contribuer à SLHA v2

Merci de votre intérêt ! Le dépôt est un **workspace Cargo** (le crate vit dans
`scirust/`). Toutes les commandes se lancent depuis la racine.

## Avant d'ouvrir une PR

La CI exige que ces commandes passent — lancez-les en local :

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace          # 41 tests (unitaires + intégration + property/fuzz + doctests + calibration λ + CCOS)
cargo build --workspace --all-targets
cargo bench --workspace --no-run
```

Pour le chemin NEON (ARM), vérifiez la cross-compilation :

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build -p scirust --lib --target aarch64-unknown-linux-gnu
```

## Principes du projet

- **Mesurer, pas affirmer.** Tout chiffre de performance/fidélité doit venir
  d'un test ou d'un exemple reproductible (graines fixes), pas d'une estimation.
  Les réserves d'honnêteté sont explicites (`FINDINGS.md`, `SLHAv2.md` §6–7).
- **La bibliothèque reste sans dépendance.** `criterion` est une dev-dependency
  (benches) uniquement.
- **Tout nouveau chemin SIMD** doit avoir un test d'équivalence `≡ scalaire`.
- **La doc doit décrire l'API réelle.** Pas d'API « plausible mais inexistante ».

## Style

`rustfmt` par défaut ; `clippy` sans warning (`-D warnings`). Les boucles
numériques peuvent utiliser l'indexation (`#![allow(clippy::needless_range_loop)]`
au niveau du crate).
