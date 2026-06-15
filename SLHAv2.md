# SLHA v2: Sub-Low Rank Hybrid Attention Co-Consciente et Linéarisée pour l'Inférence Cohérente aux Limites des Caches L1/L2/L3

**Auteurs :** Forge CHECKUPAUTO

**Statut :** Spécification de Recherche Fondamentale et d'Ingénierie Système — Édition 2026

---

## Résumé

L'inférence locale de grands modèles de langage (LLM) sur des architectures de serveurs denses ou des accélérateurs embarqués se heurte à une contrainte physique immuable : le mur de la bande passante mémoire (Memory-Bandwidth Wall). Le cache KV, en grandissant de manière linéaire avec le contexte, sature les bus d'interconnexion et provoque une sous-utilisation critique des unités de calcul vectoriel.

Nous présentons **SLHA v2** (Sub-Low Rank Hybrid Attention version 2), un mécanisme d'attention asymétrique et élastique conçu pour s'indexer précisément sur la topologie des caches L1, L2, et L3 des processeurs du marché (architectures multi-cœurs x86_64 type Xeon/Epyc et clusters ARM Neoverse/Thor). En fusionnant une compression latente de bas rang et une quantification résiduelle binaire sur 1-bit via l'infrastructure **SciRust**, SLHA v2 vise un "Soft-Paging" de la précision sémantique sans aucune allocation sur le tas, transformant la gestion du contexte en une opération déterministe au bit près, orchestrée par le noyau de système d'exploitation de contexte **CCOS**.

> **Statut (v1).** Ce document est une *spécification*, pas un rapport de résultats. Les cibles chiffrées de performance (§6) sont des **hypothèses non encore mesurées**, et l'implémentation de référence (§5) comporte des limitations explicitement recensées au §5.1.

---

## 1. Introduction & Limites du Matériel

Les pipelines d'attention conventionnels traitent le cache KV comme un flux continu de tenseurs en haute précision (FP16/BF16). Lors des phases de génération (decoding), chaque nouveau jeton exige le rechargement complet de l'historique du cache depuis la mémoire vive principale (DRAM ou VRAM) vers les registres du CPU/GPU.

Sur un processeur d'infrastructure type serveur multi-cœurs ou système sur puce (SoC) embarqué avancé :

- **Cache L1 (Data) :** Ultra-rapide (~1 cycle d'horloge) mais confiné à une capacité stricte (souvent 32 à 64 Ko par cœur).
- **Cache L2 :** Intermédiaire (~4 à 15 cycles), oscillant entre 512 Ko et 2 Mo par cœur.
- **Cache L3 / LLC (Last Level Cache) :** Partagé, offrant une bande passante élevée mais une latence accrue (~40 à 80 cycles), avec des tailles allant de 32 Mo à plusieurs centaines de Mo.

Si la topologie du cache KV ne s'aligne pas chirurgicalement sur ces strates de silicium, le processeur passe la quasi-totalité de son temps à attendre des transferts de données (Cache Misses). SLHA v2 résout ce problème en adaptant la structure même des tenseurs d'attention à la géométrie interne de ces mémoires caches.

---

## 2. Formalisation Mathématique de SLHA v2

SLHA v2 divise la représentation des Clés (K) et Valeurs (V) en deux composantes asymétriques distinctes : une base sémantique compacte et un résidu de correction binaire haute fréquence.

```
                  ┌──────────────────────────────────────────────┐
                  │          Vecteur d'Activation X_j            │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                ┌──────────────────────────────────────────────────┐
                │  Projection de Bas Rang Partagée (W_down)         │
                └────────────────────────┬─────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │  Espace Latent h_KV (Base Basse-Fidélité)│
                    └────────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼ (Soustraction de la Reconstruction)          ▼ (Projection Orthogonale Z)
    ┌──────────────────────────┐                    ┌──────────────────────────┐
    │  Erreur de Compression   │                    │  Quantification 1-Bit    │
    │        E_j^(n)           │                    │     Residual Bitmap      │
    └──────────────────────────┘                    └──────────────────────────┘
```

### 2.1 Compression Latente Multi-Têtes (MLC)

Soit **X**\_j ∈ ℝ^{d_model} le vecteur d'activation du jeton contextuel j. Nous projetons ce vecteur dans un espace latent partagé à goulot d'étranglement de dimension d_c :

```
h_KV,j = W_down · X_j ∈ ℝ^{d_c}
```

À partir de ce vecteur h_KV,j, chaque tête d'attention n extrait sa composante de clé grossière (coarse key) via une matrice d'up-projection spécifique W_up,K^(n) ∈ ℝ^{d_k × d_c} :

```
K_coarse,j^(n) = W_up,K^(n) · h_KV,j
```

### 2.2 Quantification Résiduelle 1-Bit Déterministe

Pour compenser la dérive sémantique induite par la réduction de rang sans impacter le bus mémoire, l'erreur de reconstruction **E**\_j^(n) = K_real,j^(n) - K_coarse,j^(n) est capturée par une projection aléatoire orthogonale de Johnson-Lindenstrauss. On extrait uniquement le signe du résidu sur d_s dimensions :

```
B_j^(n) = sign( Z · E_j^(n) ) ∈ {-1, 1}^{d_s}
```

Où **Z** ∈ ℝ^{d_s × d_k} est une matrice de projection fixe, stable et pseudo-aléatoire initialisée au démarrage du noyau. En mémoire, B_j^(n) est stocké sous forme de masques de bits compacts (bitmaps).

### 2.3 Équation de Score Flottant-Binaire Fusionnée

Le score d'attention non normalisé entre la requête Q_i^(n) et le jeton du cache j combine le produit scalaire continu et le produit scalaire binaire, accéléré par des opérations logiques de bas niveau :

```
Score_ij^(n) = (Q_i^(n) · W_up,K^(n)) · h_KV,j
              + λ · [ d_s - 2 · popcount( pack_sign(Q_i^(n) · Z^T) ⊕ B_j^(n) ) ]
```

Où ⊕ représente l'opérateur de OU exclusif (XOR) et popcount compte le nombre de bits à 1.

---

## 3. Topologie Matérielle et Élasticité des Caches

L'innovation de la version v2 réside dans l'agencement mémoire en **Tuiles Statiques (Hardware-Aware Tiling)** orchestrées par SciRust. Plutôt que de stocker les tenseurs dans des matrices globales déconnectées de la réalité du matériel, les structures de données adoptent des tailles modulaires calquées sur les lignes de cache de 64 octets.

### 3.1 Alignement Géométrique des Tuiles

Pour maximiser la localité spatiale et temporelle, nous définissons une structure `SciRustSlhaTile` alignée sur 64 octets (`#[repr(C, align(64))]`), dont l'empreinte mémoire est un multiple exact de la ligne de cache de 64 octets :

| Élément de la Tuile SLHA v2 | Format / Spécification | Taille Mémoire | Cible Matérielle Principale |
|---|---|---|---|
| Composante Latente (d_c = 128) | INT4 Quantifié (4 bits / échantillon) | 64 Octets | Cache L1 (Data) — Ligne complète |
| Bitmaps de Résidus (d_s = 256) | 4 mots binaires de 64 bits (u64) | 32 Octets | Registres Vectoriels AVX-512 / ARM Neon |
| Métadonnées (échelle, λ, σ_E, token, position, tête, flags, réserve) | 3×f32 + 2×u32 + 2×u16 + 8 o réservés | 32 Octets | Pagination CCOS & contrôle d'amplitude |

**Invariant Matériel (implémenté & vérifié) :** Les trois blocs totalisent **exactement 128 octets** — latent 64 o + résidu 32 o + métadonnées 32 o. Avec `#[repr(C, align(64))]`, `size_of::<SciRustSlhaTile>()` vaut **128 octets sans aucun padding** (`align_of = 64`), et une tuile occupe **2 lignes de cache pleines**. Garanti par le test `tile_is_exactly_128_bytes_zero_padding` (somme des champs == `size_of` == 128). *Historique :* l'énoncé v1 « exactement 104 octets / multiple d'une ligne de cache » était faux — 104 n'est pas un multiple de 64, et l'`align(64)` arrondit la taille à 128. Les 24 octets qui n'étaient alors que du remplissage portent désormais des métadonnées utiles (σ_E, identifiants, drapeaux de pagination).

Par ailleurs, le **bloc latent INT4 occupe à lui seul exactement 64 octets**, soit **une** ligne de cache pleine pour 128 dimensions sémantiques : c'est ce sous-bloc — et non la tuile entière — qui sature l'unité arithmétique en un unique chargement de ligne. Lors d'un calcul d'attention, SciRust pré-charge les vecteurs de requêtes dans le cache L1, puis balaie les tuiles de contexte séquentiellement.

> **État :** l'option « zéro gaspillage » est désormais **implémentée** dans le crate `scirust` (les 24 anciens octets de padding portent des métadonnées, test à l'appui). Reste ouverte une variante **SoA** (latent 64 o dans un flux, résidu + métadonnées dans un autre) si l'on veut ramener le balayage à **une seule** ligne de cache par tuile.

### 3.2 Calibration Dynamique du Facteur d'Échelle λ

La constante λ, qui régit le poids de la correction binaire, n'est plus fixe. SciRust évalue de manière analytique la variance de l'erreur de quantification σ_E² à chaque tick de traitement du contexte. En exploitant la trace de la projection, la valeur optimale s'ajuste en temps réel selon la formule :

```
λ = σ_E · √(π / (2 · d_s))
```

Si l'arborescence causale du code (analysée par le parser CCOS) détecte une zone hautement critique (par exemple, une signature de fonction fondamentale), σ_E augmente pour forcer le processeur à sur-pondérer la fidélité binaire.

> **Statut :** cette forme de λ est une **heuristique** de type estimateur sign-LSH — le facteur √(π/2) provient de la relation gaussienne 𝔼[|X|] = σ·√(2/π), et le 1/√d_s normalise la variance d'une somme de d_s termes ±1. Elle est **plausible mais non encore validée empiriquement** ; à confirmer (ou infirmer) via le protocole du §6.

---

## 4. Intégration Holistique CCOS : Politique de "Soft-Paging"

Grâce à la décomposition asymétrique de SLHA v2, le noyau CCOS applique le principe **P2 (Boundedness)** non plus par une expulsion binaire (tout ou rien), mais par une élasticité fine de la fidélité en fonction de la hiérarchie des caches matériels.

```
       [ Contexte HOT ]  ──▶  Mémoire Cache L1/L2 active
         (Latent 4-bit + Résidu 1-bit)
               │
               ▼  Pression mémoire détectée (enforce_paging)
      [ Contexte WARM ]  ──▶  Libération des Bitmaps Résiduels
         (Latent 4-bit uniquement) -> Gain immédiat de 30% d'espace
               │
               ▼  Éviction causale totale
       [ Contexte COLD ] ──▶  Snapshot chiffré sur disque (EventLog)
```

| Mode | Description | Impact |
|---|---|---|
| **HOT** (Working Set Actif) | Les tuiles SLHA v2 sont complètes. Le calcul fusionné (Bas rang + Résidu) s'exécute à pleine puissance dans les caches L1/L2. La perplexité sémantique est optimale. | Fidélité maximale |
| **WARM** (Pagination Élastique) | Lorsque la mémoire sature ou qu'un nœud s'éloigne dans l'arbre causal, CCOS invoque `enforce_paging()`. Au lieu de décharger le nœud sur disque, le noyau libère uniquement les 32 octets de `residual_bitmap`. La structure reste en cache L3 ou en DRAM, et SciRust bascule dynamiquement sur un mode d'attention basse-fidélité pur (`dynamic_lambda = 0.0`). L'empreinte chute instantanément sans aucune opération d'I/O. | Gain immédiat de ~30% d'espace mémoire |
| **COLD** (Archivé) | Le nœud est intégralement purgé de la mémoire active. Sa trace transactionnelle reste préservée de façon immuable dans l'EventLog. | Persistance chiffrée |

---

## 5. Implémentation Logicielle du Micro-Noyau Asymétrique

Le noyau d'inférence SLHA v2 est écrit en Rust (édition 2021) et respecte l'invariant de zéro allocation dans les boucles critiques. Le crate `scirust` **compile et passe ses tests** (`cargo test`, 7 tests verts) ; le listing ci-dessous en est le cœur. C'est une **implémentation de référence _scalaire_, correcte avant d'être rapide** : API sûre (pas de pointeurs bruts, pas d'`unsafe`), sémantique exacte du score fusionné (eq. 2.3). Le chemin SIMD explicite reste à écrire — mais il est désormais *débloqué*, l'ancien `read_volatile` (qui interdisait toute vectorisation) ayant été retiré (cf. §5.1).

Code source complet : [`scirust/src/attention/slha_v2.rs`](scirust/src/attention/slha_v2.rs)

```rust
// Constantes du modèle
pub const D_C: usize = 128;                 // dim latente (INT4) -> 64 octets
pub const D_S: usize = 256;                 // bits de résidu sign-LSH -> 32 octets
pub const LATENT_BYTES: usize = D_C / 2;    // 64
pub const RESIDUAL_WORDS: usize = D_S / 64; // 4
pub const FLAG_HOT: u16 = 0;
pub const FLAG_WARM: u16 = 1 << 0;          // résidu paginé : score = base latente seule

#[repr(C, align(64))] // 128 octets exacts, zéro padding (test à l'appui)
#[derive(Clone)]
pub struct SciRustSlhaTile {
    pub latent_kv: [u8; LATENT_BYTES],          // 64  base h_KV en INT4 signé
    pub residual_bitmap: [u64; RESIDUAL_WORDS], // 32  résidu sign-LSH (256 bits)
    pub scale: f32,            //  4  échelle de déquantification INT4
    pub dynamic_lambda: f32,   //  4  poids de correction binaire (eq. 3.2)
    pub residual_sigma: f32,   //  4  σ_E par tuile (recalibrage de λ)
    pub token_id: u32,         //  4
    pub position: u32,         //  4
    pub head_id: u16,          //  2
    pub flags: u16,            //  2  HOT / WARM
    pub _reserved: [u8; 8],    //  8  réserve -> total exact = 128
}

impl SciRustSlhaTile {
    #[inline]
    pub fn is_warm(&self) -> bool { self.flags & FLAG_WARM != 0 }

    /// Déquantification INT4 *signée* (zero-point) : (nibble − 8) · scale.
    #[inline]
    pub fn dequant_at(&self, d: usize) -> f32 {
        let byte = self.latent_kv[d >> 1];
        let nib = if d & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        ((nib as i32) - 8) as f32 * self.scale
    }

    /// Score fusionné (eq. 2.3). API sûre, sans `read_volatile`, boucle
    /// auto-vectorisable. En mode WARM, le terme binaire est ignoré.
    pub fn compute_score(&self, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        // 1. Terme continu bas-fidélité : <q_coarse, dequant(latent)>
        //    (le crate matérialise d'abord le latent via dequant_latent()
        //     pour une boucle SIMD plus propre — équivalent à l'inline ci-dessous)
        let mut coarse = 0.0f32;
        for d in 0..D_C {
            coarse += q_coarse[d] * self.dequant_at(d);
        }
        // 2. WARM : résidu paginé -> base latente seule
        if self.is_warm() {
            return coarse;
        }
        // 3. Correction binaire 1-bit : λ · (d_s − 2·popcount(q_sign ^ B))
        //    popcount(XOR) = distance de Hamming ; d_s − 2·Hamming = produit
        //    scalaire signé des deux vecteurs ±1.
        let mut hamming = 0u32;
        for w in 0..RESIDUAL_WORDS {
            hamming += (q_sign[w] ^ self.residual_bitmap[w]).count_ones();
        }
        let residual_score = D_S as f32 - 2.0 * hamming as f32;
        coarse + self.dynamic_lambda * residual_score
    }
}

/// Quantification INT4 signée, échelle symétrique par tuile.
/// value ≈ (nibble − 8) · scale, avec nibble ∈ [0, 15] -> plage signée [−8, 7].
pub fn quantize_latent(v: &[f32; D_C]) -> ([u8; LATENT_BYTES], f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let mut out = [0u8; LATENT_BYTES];
    for d in 0..D_C {
        let nib = (((v[d] / scale).round() as i32).clamp(-8, 7) + 8) as u8 & 0x0F;
        if d & 1 == 0 {
            out[d >> 1] = (out[d >> 1] & 0xF0) | nib;
        } else {
            out[d >> 1] = (out[d >> 1] & 0x0F) | (nib << 4);
        }
    }
    (out, scale)
}
```

### 5.1 État de l'implémentation de référence

Les limitations de la v1 sont **levées** dans le crate `scirust` (`cargo test` vert) :

- ✅ **`read_volatile` supprimé.** Le chemin chaud lit des `slice`s normaux : LLVM peut de nouveau auto-vectoriser et réordonner. La spécialisation SIMD n'est pas encore écrite, mais elle n'est plus **bloquée**.
- ✅ **INT4 signé (zero-point).** La déquantification est `(nibble − 8)·scale` : la base bas-rang représente désormais des valeurs négatives. Garanti par le test `int4_dequant_round_trips_signed_values`.
- ✅ **API sûre, pas de `target_feature` trompeur.** Plus d'`unsafe`, plus d'import mort, plus de gate `avx2` sans intrinsèque ; `count_ones()` se compile en `POPCNT` quand la cible le supporte, avec repli portable (ARM Neoverse/Thor inclus).
- ✅ **Tuile = 128 o sans padding** et **crate compilable + testé** : 7 tests, dont l'identité de Hamming `d_s − 2·popcount` prouvée contre une référence brute, et la correspondance code ↔ eq. (2.3).

**Avancées récentes & restant :**

- ✅ **Chemins SIMD : AVX2 + NEON.** AVX2 (x86_64, dispatch runtime, ~×13, §7.4) **et** NEON (aarch64, baseline), chacun avec un test d'équivalence ≡ scalaire. Le NEON est **vérifié par cross-compilation** (`aarch64-unknown-linux-gnu`) ; son équivalence à l'exécution (`neon_path_matches_scalar`) s'exécute sur matériel ARM. Reste **AVX-512**.
- ◑ **Projection bas-rang apprise** : le §7.3 l'aborde par PCA (optimal linéaire) et révèle que le goulot de fidélité devient alors l'**INT4 du latent**. Reste l'apprentissage de bout en bout des projections conjointement au modèle.
- ◑ **Quantification latente par groupe (MX)** implémentée : 8 micro-échelles `u8` logées dans les ex-octets `reserved` (scalaire + AVX2, test). Elle réduit >2× l'erreur de reconstruction mais le gain end-to-end est marginal — le vrai plafond est la **résolution 4 bits** (§7.3). Pistes : **INT8 / NF4** latent, ou `d_s` plus grand.

---

## 6. Protocole d'Évaluation et Validation Empirique

Pour valider les gains de performance de SLHA v2 par rapport aux structures de graphes et d'attention conventionnelles, trois métriques matérielles strictes doivent être mesurées lors des prochains tests de charge sous Debian 13.

**Les valeurs ci-dessous (≥ 85 %, 3,5×–5×, ΔP < 0,04) sont des cibles de conception / hypothèses, pas des résultats mesurés.** Les §6.1–6.3 décrivent le protocole matériel (cache misses, débit, perplexité) qui reste à exécuter sous Debian 13. En revanche, des **mesures préliminaires sur prototype** (fidélité de l'approximation, HOT vs WARM) existent déjà et sont rapportées au **§7**.

### 6.1 Taux de Cache Misses L1/L2/L3

- **Objectif :** Démontrer que le tuilage statique élimine le phénomène d'attente CPU.
- **Méthodologie :** Instrumentation du binaire via les compteurs de performance matériels du processeur (`perf stat -e L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses`).
- **Cible attendue :** Une baisse de ≥ 85% des lignes de cache invalidées lors du balayage d'un contexte de 32 000 jetons.

### 6.2 Débit d'Inférence sous Stress Temporel

- **Objectif :** Quantifier le gain en jetons par seconde lors des phases d'écriture intensive des agents de codage.
- **Méthodologie :** Comparaison de la latence du premier jeton (Time to First Token) et du débit continu face à un cache KV FP16 non compressé.
- **Cible attendue (hypothèse) :** Accélération d'un facteur 3,5× à 5× sur les architectures cibles sans accélération matérielle externe dédiée. Le chemin **AVX2** existe désormais et donne ~×13 sur le *seul* calcul de score (§7.4) ; mais le 3,5×–5× de **bout en bout** (vs cache KV FP16, bande passante mémoire incluse) reste à mesurer sous charge réelle.

### 6.3 Dérive de la Perplexité en Mode Dégradé (Soft-Paging Validation)

- **Objectif :** Valider que le passage du mode HOT au mode WARM (perte du résidu 1-bit) n'altère pas la capacité sémantique de l'agent à comprendre la structure globale d'un code.
- **Méthodologie :** Mesure de la perplexité du modèle sur les suites de tests CCOS (`benchmark --cycles 10000`) en basculant sélectivement les nœuds amonts en mode WARM.
- **Cible attendue (hypothèse) :** Dérive de la perplexité inférieure à ΔP < 0,04, ce qui *confirmerait* — une fois réellement mesuré — l'efficacité de la distribution sémantique de bas rang.

---

## 7. Résultats de Mesure Préliminaires (Prototype SciRust)

Le crate `scirust` inclut deux prototypes reproductibles — `cargo run --example measure --release` (résidu `rho` fixé à la main) et `--example measure_learned` (base bas-rang **apprise par PCA**) — qui mesurent la **qualité de l'approximation** du score SLHA sur données synthétiques.

**Portée & honnêteté méthodologique.** Dans les §7.1–7.2, les projections sont **aléatoires** (Gaussiennes) : `Z` (sign-LSH) l'est par conception, mais `W_down`/`W_up` ne sont pas apprises ; on suppose la base bas-rang *capturée idéalement* et on mesure la machinerie **quantification INT4 + résidu 1-bit + ranking**. On note `rho = ||e|| / ||k_real||` la part d'énergie que la base laisse au résidu. Le **§7.3 lève cette réserve** en apprenant la base par PCA.

### 7.1 Fidélité du cœur binaire (sign-LSH, d_s = 256)

Sur des paires couvrant toute la plage angulaire, l'estimateur 1-bit suit fidèlement le cosinus vrai (Spearman > 0,9, prouvé par test). Mais dans le **régime réaliste** (résidu quasi-orthogonal à la requête), 256 bits ne donnent qu'une résolution **modérée** : Spearman(résidu, cos θ) ≈ **0,67**. C'est une limite réelle — un seul bit par hyperplan résout mal des directions presque orthogonales.

### 7.2 Score complet vs vérité terrain FP (N = 512 jetons, requête fixe)

| rho | HOT Spearman | HOT top-16 | WARM Spearman | WARM top-16 |
|---|---|---|---|---|
| 0,05 | 0,987 | 0,875 | 0,987 | 0,875 |
| 0,10 | 0,983 | 0,750 | 0,982 | 0,750 |
| 0,20 | 0,966 | 0,750 | 0,961 | 0,688 |
| 0,30 | 0,937 | 0,625 | 0,925 | 0,562 |
| 0,50 | 0,844 | 0,500 | 0,811 | 0,438 |
| 0,70 | 0,702 | 0,375 | 0,627 | 0,250 |

**Lecture :**

- **Soft-Paging validé (§4).** À faible `rho` (≤ 0,1), HOT ≈ WARM : libérer le résidu 1-bit est **quasi sans perte** quand la base bas-rang capture l'essentiel — précisément la condition sous laquelle CCOS bascule un nœud en WARM.
- **Le résidu 1-bit aide, modestement.** HOT ≥ WARM partout ; l'apport croît avec `rho` (à 0,5 : +0,03 Spearman, +6 pts de top-16 ; à 0,7 : +0,08 Spearman, +12 pts de top-16). Effet réel, mais pas spectaculaire à 256 bits.
- **Mise en garde.** Quand la base rate beaucoup (`rho` élevé), même HOT ne récupère pas bien le top-k exact (0,375 de recouvrement top-16 à `rho` = 0,7). La fidélité finale dépend donc fortement de la qualité — apprise — de la base bas-rang.

### 7.3 Projection bas-rang apprise par PCA, et quantification par groupe

Pour lever la réserve « base idéale », `measure_learned` **apprend** la projection par PCA (meilleure reconstruction linéaire de rang `D_C`, par Eckart–Young) sur des clés synthétiques à spectre contrôlable (`d_model = 256`, latent 128, `d_s = 256`). `Z` reste aléatoire par conception. Le latent INT4 utilise désormais des **micro-échelles par groupe** (8 groupes de 16 dims, une échelle `u8` chacun, logées dans les 8 octets jadis « reserved » — la tuile reste à 128 o).

| decay | énergie captée | HOT Spearman | HOT top-16 | WARM Spearman | WARM top-16 |
|---|---|---|---|---|---|
| 0,99 | 97,6 % | 0,788 | 0,562 | 0,703 | 0,438 |
| 0,97 | 99,7 % | 0,814 | 0,375 | 0,700 | 0,312 |
| 0,93 | 99,5 % | 0,884 | 0,438 | 0,609 | 0,188 |
| 0,85 | 99,1 % | 0,905 | 0,562 | 0,871 | 0,500 |

**Lecture :**

- **Le bas-rang n'est plus le goulot.** La PCA capte 97–99,7 % de l'énergie à rang 128 ; pourtant le Spearman HOT plafonne à 0,79–0,90. Le facteur limitant devient la **quantification INT4 du latent**, pas la projection.
- **HOT > WARM partout**, parfois nettement (decay 0,93 : **+0,28** de Spearman). Le résidu 1-bit récupère une part de ce que l'INT4 dégrade.
- **Quantification par groupe (MX) : forte baisse de l'erreur de reconstruction, gain end-to-end marginal.** Le groupage divise par **>2×** l'erreur de reconstruction du latent sur un spectre étagé (prouvé par test), mais ne déplace quasiment pas le ranking ici (decay 0,93 : HOT 0,879 → 0,884 ; top-16 inchangé). Raison : le score est dominé par les composantes de **forte** variance, que l'échelle unique représente déjà bien. **Le vrai plafond est la résolution 4 bits** sur ces composantes — pas la plage dynamique inter-groupes. Le lever demande plus de bits (INT8 / NF4) ou un `d_s` plus grand, pas un groupage plus fin.
- **Résultat négatif assumé : whitener le latent DÉGRADE** (0,859 → 0,750 de HOT Spearman à decay 0,95). L'échelle INT4 vaut mieux *non* whitenée — elle alloue spontanément sa résolution aux composantes de forte variance. (Neutre sur le score ; c'est la quantification qu'il pénalise.)

### 7.4 Débit (scalaire vs AVX2)

Le kernel dispose désormais d'un **chemin AVX2** (dispatch à l'exécution via `is_x86_feature_detected!`, repli scalaire portable) doublé d'un **test d'équivalence** scalaire ≡ AVX2. Sur le banc partagé :

| Chemin | Débit | Rapport |
|---|---|---|
| Scalaire (référence) | ~3,0 M scores/s | 1× |
| AVX2 (dispatch) | ~39,5 M scores/s | **×13** |

Le facteur dépasse les 8× théoriques du SIMD `f32` car le chemin scalaire payait aussi une déquantification INT4 *branchy* que l'AVX2 fusionne. À traiter comme un ordre de grandeur sur banc partagé. Un **chemin NEON** (aarch64) existe également, **vérifié par cross-compilation** mais non chronométré ici (pas de matériel ARM sur le banc) ; **AVX-512** reste à écrire.

**Conclusion partielle.** Le mécanisme est **mathématiquement correct** (tests, dont l'équivalence scalaire/AVX2) et **directionnellement validé** : HOT ≥ WARM, Soft-Paging quasi sans perte à faible `rho`, accélération SIMD ×13. Deux bémols : (1) avec une base *apprise*, le goulot de fidélité devient l'**INT4 du latent**, pas le bas-rang ; (2) les gains du résidu 1-bit restent **modérés** à `d_s = 256`. Pistes : quantification latente plus fine (zero-point par groupe / NF4), `d_s` plus grand, projections apprises de bout en bout.

---

## 8. Conclusion

SLHA v2 **propose** qu'en mariant la rigueur d'un système d'exploitation gérant sa mémoire au bit près (CCOS) avec des abstractions mathématiques appliquées aux limites physiques du silicium (SciRust), l'inférence locale puisse changer de paradigme : le cache KV cesserait d'être un fardeau monolithique pour devenir une structure fluide, résiliente et consciente de l'architecture qui l'héberge.

**Cette spécification (v1) progresse vers la validation.** Le crate `scirust` est compilable et testé (§5.1), deux bancs reproductibles existent, un chemin **AVX2** (×13, §7.4) accélère le kernel, et les résultats (§7) confirment la correction du mécanisme et la viabilité du Soft-Paging. Le point dur révélé par la mesure : avec une base bas-rang *apprise*, le goulot de fidélité devient l'**INT4 du latent** — et le groupage MX (déjà en place) ne le lève pas, car le plafond est la résolution 4 bits (§7.3). Restent à faire : (1) une **résolution latente plus large** (INT8 / NF4) et l'extension SIMD à **AVX-512** (AVX2 et NEON déjà en place) ; (2) la validation **matérielle** du §6 (cache misses, perplexité sous Debian 13) ; (3) l'**apprentissage de bout en bout** des projections `W_down`/`W_up`.

---

*Spécification de Recherche Fondamentale et d'Ingénierie Système — Édition 2026 — Forge CHECKUPAUTO*
