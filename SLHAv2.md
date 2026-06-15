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
| Métadonnées de Scaling & λ | 2 Flottants simple précision (f32) | 8 Octets | Alignement et contrôle d'amplitude |

**Invariant Matériel (corrigé) :** Les champs *utiles* totalisent **104 octets** (64 + 32 + 8). Mais l'attribut `#[repr(C, align(64))]` impose que la taille du type soit un multiple de son alignement : `size_of::<SciRustSlhaTile>()` vaut donc **128 octets**, dont **24 octets de remplissage (padding)**, et chaque tuile occupe **2 lignes de cache de 64 octets — pas une seule**. (Vérifié sous `rustc 1.94` : `size_of = 128`, `align_of = 64`.) L'ancienne affirmation « occupe exactement 104 octets » *et* « multiple exact d'une ligne de cache » était auto-contradictoire, 104 n'étant pas un multiple de 64.

En revanche, le **bloc latent INT4 occupe à lui seul exactement 64 octets**, soit **une** ligne de cache pleine pour 128 dimensions sémantiques : c'est ce sous-bloc — et non la tuile entière — qui sature l'unité arithmétique en un unique chargement de ligne. Lors d'un calcul d'attention, SciRust pré-charge les vecteurs de requêtes dans le cache L1, puis balaie les tuiles de contexte séquentiellement.

> **Piste de conception (v2.1, hors périmètre de cette révision documentaire).** Pour rendre l'invariant « zéro gaspillage » réellement vrai, deux options : **(a)** exploiter les 24 octets de padding pour des métadonnées utiles (échelle du résidu, identifiant de jeton/position, tête, drapeaux) — ce qui justifie pleinement une tuile de 128 octets ; **(b)** un découpage **SoA** (latent 64 o dans un flux, résidu + métadonnées dans un autre) pour ne toucher qu'une seule ligne par flux. La structure du §5 reste inchangée dans la présente révision.

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

Le noyau d'inférence SLHA v2 est écrit en Rust (édition 2021) et respecte l'invariant de zéro allocation dans les boucles critiques. **Le listing ci-dessous est une implémentation de référence _scalaire_** : il fixe la *sémantique* exacte du score fusionné (déquantification INT4 en ligne + cœur binaire `popcount`), conforme à l'équation (2.3). Il n'est **pas** encore vectorisé et ne doit pas être lu comme du code optimisé prêt pour la production — ses limitations connues sont recensées au §5.1.

Code source complet : [`scirust/src/attention/slha_v2.rs`](scirust/src/attention/slha_v2.rs)

```rust
use std::arch::x86_64::*;

#[repr(C, align(64))]
pub struct SciRustSlhaTile {
    /// Espace latent compressé : 128 dimensions codées sur 4 bits (64 octets)
    pub latent_kv: [u8; 64],
    /// Résidu binaire de Johnson-Lindenstrauss : 256 bits (32 octets)
    pub residual_bitmap: [u64; 4],
    /// Facteur d'échelle de la quantification de bas rang
    pub scale: f32,
    /// Facteur de correction binaire dynamique calculé analytiquement
    pub dynamic_lambda: f32,
}

pub struct SciRustSlhaEngine;

impl SciRustSlhaEngine {
    /// Calcule le score d'attention asymétrique d'une requête contre une tuile de contexte SLHA v2.
    /// Ce code est conçu pour s'exécuter entièrement dans les registres CPU sans rupture de cache.
    #[target_feature(enable = "avx2,popcnt")]
    pub unsafe fn compute_tile_score(
        q_coarse: *const f32,         // Vecteur de requête Q * W_up (128 dimensions contiguës)
        q_residual_sign: *const u64,  // Signe de la requête packé sur 4 mots de 64 bits
        tile: *const SciRustSlhaTile,
    ) -> f32 {
        // 1. Évaluation de la composante basse-fidélité (Déquantification 4-bit en ligne)
        let mut coarse_accumulator = 0.0f32;
        let latent_ptr = (*tile).latent_kv.as_ptr();
        let scale = (*tile).scale;

        // Boucle déroulée manuellement pour saturer les pipelines superscalaires
        for i in 0..64 {
            let packed_byte = core::ptr::read_volatile(latent_ptr.add(i));

            // Extraction simultanée des deux valeurs de 4 bits (paires et impaires)
            let v1 = (packed_byte & 0x0F) as f32 * scale;
            let v2 = (packed_byte >> 4) as f32 * scale;

            coarse_accumulator += core::ptr::read_volatile(q_coarse.add(i * 2)) * v1;
            coarse_accumulator += core::ptr::read_volatile(q_coarse.add(i * 2 + 1)) * v2;
        }

        // 2. Évaluation de la correction binaire 1-Bit (Bitwise Attention Core)
        // Le compilateur Rust mappe directement ces opérations vers l'instruction matérielle POPCNT
        let mut popcount_accumulator: u32 = 0;
        let tile_residual_ptr = (*tile).residual_bitmap.as_ptr();

        popcount_accumulator += (core::ptr::read_volatile(q_residual_sign.add(0))
            ^ core::ptr::read_volatile(tile_residual_ptr.add(0)))
        .count_ones();
        popcount_accumulator += (core::ptr::read_volatile(q_residual_sign.add(1))
            ^ core::ptr::read_volatile(tile_residual_ptr.add(1)))
        .count_ones();
        popcount_accumulator += (core::ptr::read_volatile(q_residual_sign.add(2))
            ^ core::ptr::read_volatile(tile_residual_ptr.add(2)))
        .count_ones();
        popcount_accumulator += (core::ptr::read_volatile(q_residual_sign.add(3))
            ^ core::ptr::read_volatile(tile_residual_ptr.add(3)))
        .count_ones();

        // Résolution de la distance de Hamming inversée : d_s - (2 * popcount)
        let residual_score = 256.0f32 - (2.0f32 * popcount_accumulator as f32);

        // 3. Fusion linéaire finale avec le lambda co-conscient de la tuile
        coarse_accumulator + ((*tile).dynamic_lambda * residual_score)
    }
}
```

### 5.1 Limitations connues de l'implémentation de référence

Le listing ci-dessus établit la **sémantique** du score, mais plusieurs éléments contredisent les objectifs de performance affichés et seront corrigés lors de la phase d'implémentation (crate buildable + tests + bench) :

- **`read_volatile` dans la boucle chaude.** Les lectures volatiles interdisent au compilateur (LLVM) de vectoriser, réordonner ou coalescer les accès. Elles **annulent** donc l'auto-vectorisation et la saturation superscalaire visées. Pour de la mémoire normale (≠ MMIO), il faut des lectures ordinaires — et idéalement du SIMD explicite. En conséquence, le commentaire « Boucle déroulée manuellement pour saturer les pipelines superscalaires » est **aspirationnel** : la boucle n'est en l'état ni déroulée à la main, ni vectorisée.
- **`#[target_feature(enable = "avx2,…")]` sans aucun intrinsèque AVX2.** Le corps est intégralement scalaire ; seul `popcnt` (via `count_ones()`) est réellement exploité. La déclaration `avx2` est donc trompeuse tant qu'aucun chemin SIMD n'existe.
- **`use std::arch::x86_64::*;` est inutilisé** (avertissement compilateur) tant qu'aucun intrinsèque SIMD n'est appelé.
- **Déquantification INT4 non signée, sans zero-point.** `(octet & 0x0F) as f32 * scale` ne produit que des valeurs dans `[0, 15]·scale`, donc **toujours ≥ 0**. Or des clés/valeurs réelles sont signées : une quantification symétrique (p. ex. `(nibble − 8)·scale`) ou un zero-point explicite est nécessaire pour représenter la base bas-rang sans biais. *(Changement de sémantique — traité séparément de cette révision purement documentaire.)*
- **Pas encore un crate.** Le fichier `scirust/src/attention/slha_v2.rs` est orphelin (ni `Cargo.toml`, ni `lib.rs`, ni déclaration `mod`) : il ne compile pas tel quel et n'est donc pas testé. La correspondance code ↔ équation (2.3) reste à prouver par des tests unitaires.

> Ces points relèvent de la phase « crate buildable + tests / kernel optimisé », distincte de la présente mise en cohérence documentaire.

---

## 6. Protocole d'Évaluation et Validation Empirique

Pour valider les gains de performance de SLHA v2 par rapport aux structures de graphes et d'attention conventionnelles, trois métriques matérielles strictes doivent être mesurées lors des prochains tests de charge sous Debian 13.

**Les valeurs ci-dessous (≥ 85 %, 3,5×–5×, ΔP < 0,04) sont des cibles de conception / hypothèses à valider, et non des résultats mesurés.** Aucune mesure n'a encore été réalisée ; les §6.1–6.3 décrivent le protocole destiné à les confirmer ou les infirmer.

### 6.1 Taux de Cache Misses L1/L2/L3

- **Objectif :** Démontrer que le tuilage statique élimine le phénomène d'attente CPU.
- **Méthodologie :** Instrumentation du binaire via les compteurs de performance matériels du processeur (`perf stat -e L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses`).
- **Cible attendue :** Une baisse de ≥ 85% des lignes de cache invalidées lors du balayage d'un contexte de 32 000 jetons.

### 6.2 Débit d'Inférence sous Stress Temporel

- **Objectif :** Quantifier le gain en jetons par seconde lors des phases d'écriture intensive des agents de codage.
- **Méthodologie :** Comparaison de la latence du premier jeton (Time to First Token) et du débit continu face à un cache KV FP16 non compressé.
- **Cible attendue (hypothèse) :** Accélération d'un facteur 3,5× à 5× sur les architectures cibles sans accélération matérielle externe dédiée. Cette cible **suppose une vectorisation effective** du cœur binaire (cf. §5.1), qui reste à implémenter ; le listing scalaire actuel ne l'atteindra pas tel quel.

### 6.3 Dérive de la Perplexité en Mode Dégradé (Soft-Paging Validation)

- **Objectif :** Valider que le passage du mode HOT au mode WARM (perte du résidu 1-bit) n'altère pas la capacité sémantique de l'agent à comprendre la structure globale d'un code.
- **Méthodologie :** Mesure de la perplexité du modèle sur les suites de tests CCOS (`benchmark --cycles 10000`) en basculant sélectivement les nœuds amonts en mode WARM.
- **Cible attendue (hypothèse) :** Dérive de la perplexité inférieure à ΔP < 0,04, ce qui *confirmerait* — une fois réellement mesuré — l'efficacité de la distribution sémantique de bas rang.

---

## 7. Conclusion

SLHA v2 **propose** qu'en mariant la rigueur d'un système d'exploitation gérant sa mémoire au bit près (CCOS) avec des abstractions mathématiques appliquées aux limites physiques du silicium (SciRust), l'inférence locale puisse changer de paradigme : le cache KV cesserait d'être un fardeau monolithique pour devenir une structure fluide, résiliente et consciente de l'architecture qui l'héberge.

**Cette spécification (v1) reste à valider.** Les cibles de performance du §6 sont des hypothèses non encore mesurées, et l'implémentation de référence du §5 comporte les limitations recensées au §5.1. Les prochaines étapes naturelles sont : (1) un crate buildable accompagné de tests prouvant la correspondance code ↔ équation (2.3), (2) un banc de mesure reproductible, et (3) la levée des limitations du §5.1 (vectorisation réelle, zero-point INT4, layout de tuile sans gaspillage).

---

*Spécification de Recherche Fondamentale et d'Ingénierie Système — Édition 2026 — Forge CHECKUPAUTO*
