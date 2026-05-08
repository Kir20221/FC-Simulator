# FC-Simulator — module ML

> Fichier de référence Claude. Générateur de vérité, architecture du modèle, pipeline d'entraînement, inférence, structure des fichiers.
> Pour la vision projet et le vocabulaire : voir `projet.md`.
> Pour le schéma DB et le format parquet : voir `donnees.md`.

## Rôle du ML — décisions structurantes

Trois rôles possibles avaient été identifiés en début de projet :

1. **Substituer la simulation** (ML répond directement à partir des paramètres) — **écarté**, contredit la philosophie d'émergence.
2. **Accélérer la simulation** (ML prédit les événements pour chaque entité) — **retenu pour le prototype**.
3. **Analyse exploratoire a posteriori** (ML analyse les runs pour comprendre régimes paramétriques, solutions de Fermi…) — **validé, reporté après le rôle 2**.

Le rôle 2 est le périmètre de ce qu'on appelle « le Bloc B ». Le « Bloc A » désigne le générateur de vérité qui produit le corpus d'entraînement.

## Choix d'architecture ML : Transformer Hawkes Process

### Pourquoi un TPP plutôt qu'un MLP, qu'une survie, ou autre

Le moteur ML doit produire, pour chaque planète, **une chronologie d'événements** :
- Soit aucun événement.
- Soit un ou plusieurs événements typés, datés par un timecode.
- À terme, ces événements peuvent dépendre les uns des autres (un contact présuppose une civilisation, qui présuppose la vie).

Le bon cadre statistique pour ça est le **processus ponctuel temporel marqué** (TPP). Mathématiquement, on apprend une fonction d'intensité conditionnelle λₖ*(t | history, features) qui donne le taux instantané d'occurrence du type k à l'instant t, sachant l'historique des événements antérieurs et les features statiques de l'entité.

Alternatives écartées et raisons :
- **MLP one-shot** (label = vie oui/non + timecode) : ne capture pas la stochasticité du générateur, ne se généralise pas aux chaînes d'événements interdépendants, demande de tout refaire dès qu'on ajoute un type.
- **Survie / DeepHit** : excellent pour les événements indépendants conditionnés sur features statiques, mais limite à plusieurs événements interdépendants. Bon point d'entrée mais plafond bas pour la cible long terme.
- **Tick-par-tick + MLP simple** (Bernoulli par pas de temps) : équivalent en expressivité au cas particulier d'un TPP avec intensité constante par morceaux, mais pas plus simple en pratique. Forme dégradée.
- **EasyTPP / bibliothèques TPP** : conçues pour des séquences longues sans features statiques. N'acceptent pas le conditionnement sur features étoile/planète qui est central ici.

Le TPP est le **bon objet sans approximation**, à l'échelle de complexité de la cible long terme. Coût initial supérieur à la survie, mais aucune refonte à prévoir quand on ajoutera des types d'événements et des dépendances historiques.

### Architecture concrète : Transformer Hawkes Process

Suivant Zuo et al. 2020, adapté pour conditionnement sur features statiques :

1. **Encodeur de features statiques** : MLP qui combine features continues normalisées et embeddings catégoriels appris (4 dim chacun) → vecteur de contexte `c` de dim `d_model`.
2. **Encodeur de séquence** : pour chaque événement observé, embedding du type + encodage temporel sinusoïdal du timecode + injection du contexte `c` à chaque position. Puis `nn.TransformerEncoder` avec masque causal.
3. **Tête d'intensité** : pour chaque type k, on calcule λₖ(t | history, features) via projection linéaire + softplus à partir de l'état caché du Transformer + encodage du temps interrogé + contexte. K intensités positives, peu importe K.
4. **Loss** : log-vraisemblance TPP standard.
   ```
   log L = sum_i log λ_{k_i}*(t_i)  -  ∫_0^T_obs Λ*(s) ds
   ```
   où Λ*(s) = sum_k λ_k*(s). Le terme intégral est approximé par Monte Carlo : 32 points uniformes sur [0, T_obs].

### Détails techniques notables

- **Normalisation temporelle interne** : tous les timecodes sont divisés par `t_ref = 1e5` avant d'entrer dans le modèle. Sans cette normalisation, T_obs allant jusqu'à ~5e5 unités fait exploser le terme intégral en début d'entraînement. L'intensité retournée par le modèle est dans l'échelle des timecodes bruts (la division par `t_ref` est appliquée en sortie de softplus), donc tous les consommateurs sont indépendants de cette normalisation interne.
- **Gestion des séquences vides** : un sample sans aucun événement produit un masque de padding tout-à-True, ce qui fait diverger le softmax du Transformer en NaN. Corrigé par `torch.nan_to_num` après le Transformer (ces NaN sont de toute façon ignorés par `seq_mask` dans la loss).
- **Décalage causal pour les intensités aux événements observés** : à la position `i`, l'historique pertinent est `[0..i-1]`, donc on utilise `h[:, :-1]` décalé. Pour la position 0, état nul.
- **Recherche du dernier événement antérieur** dans `intensities_at_query` : implémentée par comptage vectorisé `(t_event < t_query).sum(dim=1) - 1`, sans boucle Python.
- **Inférence par thinning d'Ogata** : algorithme standard pour échantillonner à partir d'un TPP. `lambda_max` adaptatif recalibré à chaque acceptation (safety_factor=1.5).

## Bloc A — générateur de vérité

### Rôle

Le générateur de vérité est le « professeur » du modèle ML. C'est un module Python qui implémente les règles consensuelles (zone habitable, distributions stellaires, modèle multiplicatif d'apparition de la vie) et produit un corpus de planètes étiquetées avec leurs séquences d'événements.

Pas de données réelles disponibles à grande échelle pour entraîner le ML directement. Le générateur joue ce rôle, et le ML imite ensuite son comportement, pour permettre une accélération de l'inférence de plusieurs ordres de grandeur — nécessaire pour scaler à 400 milliards de systèmes.

### Périmètre actuel

**Deux types d'événements** implémentés dans `life_model.EVENT_TYPES` :
- `life_apparition` : modèle multiplicatif (HZ × composition × masse × étoile), tirage Bernoulli, timecode tiré uniformément dans la fenêtre [AGE_MIN_GA, min(durée_vie_étoile, T_sim)].
- `star_main_sequence_end` : déterministe à `t = durée_vie_étoile`, émis seulement si l'étoile meurt avant T_sim.

Pour ajouter un type d'événement, il suffit d'ajouter son libellé à `EVENT_TYPES` et une fonction `_tirer_event_xxx` appelée depuis `evaluer_planete`. Le Bloc B découvrira automatiquement le nouveau type (paramètre `nb_types_evt` déduit du dataset).

### Calibration scientifique

Niveau de fidélité visé : **ordres de grandeur plausibles, pas rigueur académique**.

**Paramètres astrophysiques exposés** (features ML, balayés par système lors de la génération du dataset) :

| Paramètre | Plage de balayage | Rôle |
|---|---|---|
| Masse stellaire moyenne | 0.1 à 2.0 M☉ (log-uniforme) | Centre de la log-normale d'où sont tirées les masses stellaires du système |
| Indice tellurique | 0.0 à 1.0 (uniforme) | Module les seuils de masse définissant la composition planétaire |
| Nombre moyen de planètes par système | 0.5 à 5.0 (uniforme) | Lambda de la Poisson tronquée [1, 8] |
| Durée totale de simulation | 1.0 à 50.0 Ga (uniforme) | Fenêtre temporelle des événements |

Ces 4 paramètres sont **tirés par système** (cohérence physique) lors de la génération du dataset. Ces mêmes plages serviront de bornes min/max aux sliders de l'IHM utilisateur.

**Population stellaire** : six types spectraux (M, K, G, F, A, B), séquence principale uniquement. Pour chaque système, masse stellaire tirée dans une log-normale centrée sur la moyenne paramétrée (σ = 0.4 en log10). Type spectral déduit de la masse, autres propriétés (T, L, durée de vie) tirées dans les plages standard du type. Pas d'âge stellaire à T=0 : tous les systèmes naissent au début de la simulation (l'âge variable des étoiles deviendra un événement dans une version future).

**Distributions des features planétaires** : théoriques plutôt qu'observationnelles, pour ne pas hériter des biais de détection des télescopes actuels. Distance orbitale et masse en log-uniforme. Rayon déduit de la masse via une relation par paliers inspirée de Chen & Kipping 2017. Composition (tellurique / glacée / gazeuse) déterminée par seuils de masse, modulés par l'indice tellurique du système.

**Zone habitable** : bornes Kopparapu 2013, polynôme en fonction de la température effective de l'étoile (limites « moist greenhouse » côté chaud et « maximum greenhouse » côté froid).

**Modèle d'apparition de la vie** : produit multiplicatif de quatre facteurs indépendants dans [0, 1] :
- *facteur HZ* : 1 dans la zone habitable, décroissance gaussienne en log(distance) en dehors.
- *facteur composition* : 1 si tellurique, faible si glacée, nul si gazeuse.
- *facteur masse* : gaussienne en log(M) centrée sur la Terre.
- *facteur étoile* : pénalité par type spectral (G favorable, M et A pénalisés, B quasi-nul).

Ces facteurs sont **figés** : ce sont des lois consensuelles, pas des paramètres exposés.

**Tirage du timecode d'apparition** : uniformément dans la fenêtre [AGE_MIN_GA = 0.5 Ga, min(durée_vie_étoile, T_sim)]. AGE_MIN_GA reste une constante figée. Si la durée de vie de l'étoile ou T_sim est inférieure à AGE_MIN_GA, la vie ne peut pas apparaître.

**Constantes temporelles** : unité de timecode = 100 000 ans (constante figée). Durée totale de simulation : paramètre exposé.

Toutes ces constantes vivent en tête des modules concernés pour faciliter les ajustements.

### Stats produites par le générateur

Snapshot pris au moment de la génération, stocké dans la colonne `stats` JSONB de la table `dataset` :

```json
{
  "nb_systemes": 100000,
  "nb_planetes": 295140,
  "nb_events_total": 115935,
  "nb_events_par_type": {
    "life_apparition": 2987,
    "star_main_sequence_end": 112948
  },
  "chaines_observees": {
    "": 180531,
    "star_main_sequence_end": 111622,
    "life_apparition": 1661,
    "life_apparition -> star_main_sequence_end": 1326
  },
  "nb_telluriques": 96281,
  "nb_glacees": 42326,
  "nb_gazeuses": 156533,
  "repartition_types_spectraux": {"M": 147934, "K": 49181, ...}
}
```

Le champ `chaines_observees` est la **distribution des chronologies effectivement produites**. Clé = chaîne ordonnée par timecode des libellés de types, séparés par ` -> `. La chaîne vide `""` représente les planètes sans événement. Cette représentation reste lisible quel que soit le nombre de types.

## Bloc B — entraînement TPP

### Pipeline

1. **Lecture** des deux parquets (planets + events) du dataset référencé.
2. **Calcul des stats de normalisation** sur le set complet (z-score des features continues, log10 préalable pour celles couvrant plusieurs ordres de grandeur).
3. **Split train/val** par permutation aléatoire (seed configurable, val_fraction par défaut 0.1).
4. **Boucle d'entraînement** : optimizer AdamW (lr=1e-3, weight_decay=1e-5), gradient clipping max_norm=5.0, log par epoch.
5. **Sauvegarde** du `.pth` complet : state_dict du modèle + TPPConfig + FeatureStats + EVENT_TYPES + TrainingConfig + métriques par epoch. Tout ce qui est nécessaire à l'inférence est dans le `.pth`, indépendamment du parquet.

### Features ML

Référentiel défini dans `training/features.py` :

- **Continues directes (z-score)** : `star_temp_K`.
- **Continues log10 puis z-score** : `star_mass_solar`, `star_luminosity_solar`, `star_lifetime_Ga`, `planet_distance_UA`, `planet_mass_terre`, `planet_radius_terre`.
- **Catégorielles (embeddings appris 4 dim)** : `star_type` (6 valeurs), `planet_composition` (3 valeurs).

Les 4 paramètres astrophysiques (`param_*`) ne sont **pas** des features ML — décision validée « pas de fuite d'information ».

### Hyperparamètres par défaut (TrainingConfig)

| Champ | Défaut | Rôle |
|---|---|---|
| `batch_size` | 256 | |
| `nb_epochs` | 20 | |
| `learning_rate` | 1e-3 | |
| `val_fraction` | 0.1 | |
| `seed_split` | 42 | |
| `nb_mc_samples` | 32 | Points Monte Carlo pour le terme intégral |
| `weight_decay` | 1e-5 | |

### Hyperparamètres architecturaux par défaut (TPPConfig)

| Champ | Défaut | Rôle |
|---|---|---|
| `d_model` | 64 | Dimension cachée du Transformer |
| `n_heads` | 4 | Têtes d'attention |
| `n_layers` | 2 | Nombre de couches Transformer |
| `dim_feedforward` | 128 | MLP interne |
| `dropout` | 0.1 | |
| `dim_emb_cat` | 4 | Dim embedding par feature catégorielle |
| `time_freqs` | 16 | Nb de fréquences sinusoïdales pour l'encodage temporel |
| `t_ref` | 1e5 | Échelle de normalisation temporelle interne |

### Métriques observées

Loggées par epoch :
- `train_loss` : -log L moyenné sur le batch (par planète, pas par événement).
- `val_loss` : idem sur le set de validation.
- `val_log_L_par_event` : log-vraisemblance par événement, plus interprétable mais sans valeur absolue (dépend de l'échelle des temps et du nombre de types).

Validation forte (non encore implémentée) : comparaison **chaînes générées par le modèle entraîné** vs **chaînes observées dans le dataset**. C'est la métrique qui permet de juger si le modèle reproduit fidèlement la dynamique du Bloc A. Demande d'utiliser le module d'inférence (thinning) sur tout ou partie du dataset.

### Résultats observés

Validation technique du pipeline complet sur :
- 1000 systèmes / 3 epochs : ~2 secondes.
- 100 000 systèmes (~300 000 planètes, 115 000 événements) / 20 epochs : ~13 minutes sur CPU.

Sur 100k/20 epochs : train_loss 4.19 → 3.39, val_loss 3.85 → 3.35. Pas de surapprentissage (train ≈ val tout au long), légère stagnation à partir de l'epoch 12.

## Inférence — thinning d'Ogata

Implémenté dans `training/inference.py`. Pour une planète donnée :

1. Construire le `PlanetSample` avec features statiques + historique éventuel + T_obs.
2. À chaque pas, tirer un intervalle exponentiel de taux `lambda_max`.
3. Évaluer Λ_total(t_candidate) via le modèle.
4. Accepter avec probabilité `Λ_total / lambda_max` ; si accepté, tirer le type selon les `λ_k / Λ_total`.
5. Recalibrer `lambda_max` périodiquement (safety_factor=1.5).
6. Boucler jusqu'à `t >= T_obs` ou `max_events` atteint.

L'inférence permet de générer des séquences depuis un modèle entraîné, soit sur le dataset d'entraînement (validation), soit dans la chaîne de simulation utilisateur (cible long terme — pas encore intégré, voir `projet.md`).

## Structure des fichiers ML

```
server/app/ml/
├── api.py                          # endpoints /training/ (datasets + models)
├── cli.py                          # CLI symétrique (datasets + models)
├── service.py                      # service partagé pour les datasets
├── dataset.py                      # Active Record Dataset
├── generator/                      # Bloc A — générateur de vérité
│   ├── distributions.py            # Tirages features étoile + planète + paramètres astrophysiques
│   ├── habitability.py             # Zone habitable Kopparapu + facteurs de vie
│   ├── life_model.py               # EVENT_TYPES, evaluer_planete, tirages des événements
│   └── generate.py                 # DataFrames, parquet, stats, DatasetGenerationRequest
└── training/                       # Bloc B — entraînement et inférence TPP
    ├── features.py                 # Référentiel features ML, FeatureStats, normalisation
    ├── data.py                     # PlanetEventsDataset, Batch, collate_fn
    ├── model.py                    # TPPConfig, TimeEncoding, StaticFeaturesEncoder, TransformerHawkes
    ├── loss.py                     # tpp_negative_log_likelihood
    ├── inference.py                # Thinning Ogata, sample_sequence, sample_planets
    ├── trainer.py                  # TrainingConfig, train(), load_model()
    ├── model_record.py             # Active Record Model
    └── service_models.py           # service partagé pour les modèles
```

## État d'avancement et étapes suivantes

Livré et fonctionnel :
- Bloc A multi-événements (vie + fin SP) calibré.
- Bloc B TPP opérationnel, training + inférence + intégration HTTP/CLI.
- Validé techniquement à 100k systèmes.

Pas encore livré :
- **Validation par comparaison chaînes générées vs observées** : métrique de confiance forte sur la qualité du modèle entraîné. Implique d'utiliser le module thinning sur le dataset complet ou un échantillon.
- **Endpoint d'inférence** : `POST /api/v1/training/models/{nom}/sample` qui prend des features et retourne une séquence générée. Pas encore exposé en HTTP/CLI.
- **Intégration dans la chaîne de simulation utilisateur** : préalable côté DB (enrichir `etoile` et `planete` avec features physiques, ajouter les 4 paramètres astrophysiques à `Scenario`).
- **Améliorations training** : early stopping, scheduler de learning rate, sauvegarde du best-val checkpoint plutôt que du dernier.

## Tensions et arbitrages assumés

- **Stochasticité du générateur vs déterminisme du modèle** : le Bloc A est stochastique (Bernoulli + tirage uniforme du timecode), le modèle apprend des distributions sous-jacentes. À l'inférence, la stochasticité est rétablie par le thinning. Cohérent.
- **Pas de fuite des paramètres astrophysiques** : ils ont servi à générer les features physiques, donc les redonner au modèle serait une fuite. Conséquence : le modèle apprend une distribution marginalisée sur la plage de T_sim du dataset. À garder en tête mais accepté pour le prototype.
- **Un seul timecode dans le Bloc A** vs distribution apprise par le modèle : la fenêtre uniforme du Bloc A est captée comme une distribution implicite par le modèle, qui peut générer des timecodes différents pour des planètes similaires. Pas exact mais acceptable pour le prototype.
- **Rareté des événements `life_apparition`** : ~1% des planètes sur les paramètres balayés. Le modèle a peu de signal positif. Plus de données ou un dataset équilibré sont des leviers connus pour la suite.
