# SLHA v2 — Rapport de synthèse des findings

Synthèse honnête de ce que l'implémentation de référence (`scirust/`) et ses
mesures ont **réellement** établi. Toutes les valeurs sont reproductibles
(graines fixes) ; détails et tableaux complets dans [`SLHAv2.md`](SLHAv2.md) §7.

> **Cadre.** Mesures sur données **synthétiques**, projections `Z` (sign-LSH)
> **aléatoires** ; sauf au §7.7, la base bas-rang est une PCA (non entraînée
> conjointement à un modèle). Pas de vrai LLM, pas de compteurs `perf` (sandbox).
> Ces résultats valident la **mécanique**, pas (encore) la qualité sur un modèle réel.

## Tableau de bord

| Question | Résultat mesuré | Réf. |
|---|---|---|
| Tuile alignée cache exacte ? | **128 o, 0 padding** (prouvé par test) | §3.1 |
| Identité de Hamming du cœur binaire ? | exacte vs référence brute | §7.1 |
| Soft-Paging HOT→WARM à faible `rho` ? | quasi sans perte (Spearman ~0,98) | §7.2 |
| Le résidu 1-bit aide-t-il ? | **HOT ≥ WARM partout**, parfois +0,28 | §7.2–7.3 |
| Fidélité de la **sortie** d'attention ? | **cosinus 0,95–0,997** vs FP | §7.6 |
| Trafic mémoire vs bf16 ? | **2× moins d'octets/token → ~2,5× tokens/s** | §7.5 |
| Débit SIMD (vs scalaire) ? | AVX2 **×11,5**, AVX-512 **×14,1** | §7.4 |
| Projection apprise vs PCA (Q≠K) ? | WARM **0,16 → 0,86** | §7.7 |

## 1. Ce qui est validé

- **Le mécanisme est correct et implémentable.** Tuile 128 o sans gaspillage,
  score fusionné conforme à l'éq. (2.3), kernels scalaire/AVX2/AVX-512/NEON
  **prouvés équivalents** (22 tests dont property/fuzz, clippy `-D warnings`, CI).
- **Le « Soft-Paging » tient.** À faible énergie résiduelle, libérer le résidu
  1-bit (WARM) est quasi sans perte ; le résidu redevient utile quand la base
  bas-rang laisse passer de l'énergie. C'est exactement la politique HOT/WARM.
- **La sortie d'attention est robuste** — le résultat le plus important. Même
  quand le ranking des scores plafonne (Spearman 0,79–0,90), la sortie
  `softmax·V` reste à **cosinus 0,95–0,997** : le softmax absorbe l'erreur de
  score. C'est le proxy le plus proche de la perplexité accessible hors LLM.
- **L'argument « mur de bande passante » tient au niveau kernel.** 128 o/token
  contre 256 o pour une clé bf16 → **~2,5× plus de tokens/s** à débit GB/s
  comparable.

## 2. Les leviers réels (et les faux leviers)

- **Levier #1 — la projection bas-rang.** Une projection **apprise task-aware**
  (SGD minimisant l'erreur de *score*, pas la reconstruction) bat nettement la
  PCA quand requêtes et clés diffèrent (WARM 0,16 → **0,86**). La PCA optimise la
  reconstruction des clés et **ignore la distribution des requêtes**.
- **Levier #2 — le résidu 1-bit.** Il récupère une grande part de ce que le
  terme coarse rate (HOT ≫ WARM à `rho` élevé).
- **Faux levier — la largeur de bits du latent.** Une **référence INT8** ne fait
  pas mieux qu'INT4 au terme coarse (~0,61) : **la quantification n'est pas le
  goulot**, c'est la projection. NF4 et le groupage MX réduisent l'erreur de
  reconstruction mais ne déplacent quasiment pas le ranking end-to-end.
- **Largeur SIMD ≠ levier majeur ici.** AVX-512 n'ajoute que **~+23 %** sur AVX2 :
  le kernel (128 dims) est limité par le dénibblage/chargement, pas la largeur FMA.

## 3. Résultats négatifs assumés

- **Whitening du latent : dégrade** (HOT 0,859 → 0,750). L'échelle INT4 unique
  alloue mieux sa résolution non whitenée.
- **Groupage MX / NF4 : gain end-to-end marginal** malgré une meilleure
  reconstruction (le score est dominé par les composantes de forte variance).
- **INT8 : n'élève pas le plafond du coarse** — corrige une hypothèse initiale
  (§7.3) qui attribuait à tort ce plafond à l'INT4.

## 4. Honnêteté & limites

- Le **paper v1** contenait des affirmations fausses (tuile « 104 o » alors que
  `align(64)` ⇒ 128 o ; `read_volatile` et `avx2` contradictoires) — corrigées.
- Une de **mes propres** conclusions (§7.3, « goulot = INT4 ») a été **réfutée**
  par la mesure INT8 et corrigée (§7.8). C'est l'intérêt de mesurer.
- **Non mesurable dans ce sandbox** : compteurs de cache `perf` (§6.1,
  `perf_event_paranoid=2`), perplexité d'un vrai modèle (§6.3), entraînement
  conjoint des projections.

## 5. Prochaines étapes (hors périmètre sandbox)

1. **Entraîner conjointement** `W_down`/`W_up` avec un vrai modèle (le §7.7 en
   établit le principe sur données synthétiques).
2. **Validation matérielle réelle** : `perf stat` (cache misses) et perplexité
   sur un modèle + jeu de données réels.
3. Intégration dans une vraie pile d'inférence pour mesurer le gain **de bout en
   bout** (et non au seul niveau kernel).

---
*Réf. : crate `scirust/` (30 tests dont property/fuzz + doctests, criterion, CI), paper `SLHAv2.md` §1–8.*
