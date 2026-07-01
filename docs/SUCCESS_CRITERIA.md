# Critères de succès — SLHA v2 (à figer AVANT l'intégration)

> **But de ce document (plan §1).** Sans seuil, on ne peut pas conclure. On fixe
> ici — **avant** d'écrire une ligne d'intégration llama.cpp — ce qui compte
> comme GO. Les cibles sont **pré-enregistrées** : elles sont posées maintenant
> pour qu'on ne les ajuste pas *a posteriori* pour valider un résultat.
>
> Garde-fou du projet : **mesurer, pas affirmer.** Les cibles ci-dessous sont des
> *seuils à atteindre*, pas des mesures. Ce qui est déjà mesuré (hors LLM) est
> marqué comme **proxy** et renvoie à un exemple reproductible (graine fixe).

---

## 1. Les trois critères

| # | Critère | Cible (GO) | État HOT / WARM / COLD |
|---|---|---|---|
| **F** | **Fidélité** — Δ-perplexité relative vs FP16, sur le contexte cible | **≤ 1 % (HOT)**, **≤ 3 % (WARM)** ; COLD = borne documentée, non gated | à mesurer (Phase 2) |
| **M** | **Mémoire** — empreinte du cache KV vs baseline FP16 | **≥ 2× de réduction** à fidélité égale (HOT = 128 o/tuile) | proxy mesuré : ~2× moins d'octets/token |
| **D** | **Débit** — tokens/s de bout en bout, contexte long (≥ 32 K) | **≥ 1× (aucune régression)** vs FP16 ; l'avantage bande passante est un bonus, pas le critère | proxy mesuré : ~2,5× (Xeon AVX2), ~1,3× (scalaire) |

La **fidélité (F)** est le critère qui décide GO/NO-GO. M et D qualifient
l'intérêt de la technique une fois F acquise ; ils ne rachètent jamais un échec
de F.

---

## 2. Proxies offline déjà en place (le GO/NO-GO peu coûteux)

La vraie Δ-perplexité exige la Phase 2. En attendant, `examples/offline_validation.rs`
mesure deux proxies couche-isolée qui **doivent** passer avant d'intégrer :

| Proxy | Seuil GO | Ce qu'il approxime | Constante |
|---|---|---|---|
| Cosinus de sortie d'attention vs FP32 | **≥ 0,98 (HOT)** | l'erreur que le softmax laisse passer en aval (F) | `GO_COSINE` |
| KL(w_true ‖ w_slha) des poids softmax | **≤ 0,03 (HOT)** | le déplacement de la distribution d'attention ≈ ΔNLL (F) | `GO_KL` |

Ces seuils sont **codés** dans le harnais (`GO_COSINE`, `GO_KL`) : ils sont la
porte, pas une décoration. Ils sont **nécessaires mais non suffisants** — un
proxy couche-isolée qui passe ne garantit pas encore la perplexité de bout en
bout ; l'inverse (proxy qui échoue) suffit à conclure NO-GO sans payer la
Phase 2.

**Honnêteté du proxy.** Fitter la projection sur les clés qu'on score ensuite est
optimiste. `offline_validation --weights proj.slhw` charge une projection
**tenue à l'écart** (entraînée ailleurs, cf. `train_on_real_activations`) pour le
chiffre non-optimiste — c'est celui qui compte pour la décision.

---

## 3. Le chiffre qui manque, et comment l'obtenir

Tout le reste du harnais offline existe ; il ne manque que des **activations
réelles**. Sur une machine équipée de PyTorch/transformers :

```bash
# 1) Dumper Q/K/V d'une couche d'attention — deux corpus DISJOINTS (train ≠ test)
python scripts/dump_activations.py --model gpt2 --layer 0 --out /tmp/train --file train.txt
python scripts/dump_activations.py --model gpt2 --layer 0 --out /tmp/test  --file test.txt

# 2) Apprendre la projection sur le corpus d'entraînement, la sauver
cargo run --release --example train_on_real_activations -- --dump /tmp/train --out proj.slhw

# 3) GO/NO-GO offline sur le corpus de TEST, projection tenue à l'écart (chiffre honnête)
cargo run --release --example offline_validation -- --dump /tmp/test --weights proj.slhw
```

> La séparation train/test est le point : un même corpus des deux côtés
> re-mesure l'optimisme que `--weights` sert précisément à éliminer.

Si ce GO/NO-GO offline est **GO** → on engage la Phase 2 (perplexité réelle
llama.cpp) contre les cibles F/M/D ci-dessus. S'il est **NO-GO** → on ajuste
l'algo **ici** (RHT, résidu multi-bit, `d_c/d_s`, λ — chaque axe a déjà son
exemple), pas après trois semaines d'intégration.

---

## 4. Ce qui est déjà établi (proxies, hors LLM)

Reproductible, graines fixes ; synthèse et tableaux dans
[`../FINDINGS.md`](../FINDINGS.md) et [`../SLHAv2.md`](../SLHAv2.md) §7.

- Sortie d'attention à **cosinus 0,95–0,997** vs FP (le proxy le plus proche de
  la perplexité accessible hors LLM).
- Soft-Paging : pager la moitié des tuiles HOT→WARM laisse la sortie à
  **cos ≈ 0,9995**.
- **~2× moins d'octets/token** → **~2,5× tokens/s** (Xeon AVX2, borné bande
  passante) ; **~1,3×** sur CPU scalaire.
- Tuile **128 o, 0 padding** (prouvé par test), kernels scalaire/SIMD
  **équivalents** (≤ 1e-3).

Ces résultats valident la **mécanique** et cadrent les cibles ci-dessus ; ils ne
remplacent pas la mesure de perplexité sur un modèle réel (Phase 2).

---

*Portée : ce document fige l'intention. Il sera complété — pas réécrit — par le
rapport chiffré de la Phase 0 sur activations réelles, puis par la perplexité de
la Phase 2.*
