# FC-Simulator — modèle de données

> Fichier de référence Claude. Schéma DB, structure des fichiers parquet, règles invariantes, contraintes de stockage.
> Pour la vision projet et le vocabulaire : voir `projet.md`.
> Pour le module ML (générateur, training, inférence) : voir `ml.md`.

## Principes directeurs

- Modélisation conçue pour le prototype (1 million de systèmes, exécution locale) mais scalable jusqu'à 100-400 milliards sans refonte des relations.
- Séparation stricte entre **entités** (univers généré, immuable) et **événements** (journal produit par la simulation).
- Évolutivité par ajout (colonnes, tables) sans dogme. Pas d'anticipation prématurée.
- Toutes les relations en 1-N simples, compatibles avec partitionnement et sharding ultérieurs.

## Quatre zones fonctionnelles

### 1. Zone scénario

Un **scénario** représente un jeu de paramètres saisis par l'utilisateur, associé en 1-1 aux entités générées.

Une **simulation** est une exécution du moteur sur les entités d'un scénario. Un scénario peut donner lieu à N simulations.

### 2. Zone entités

Hiérarchie : galaxie → systèmes solaires → étoiles + planètes. Générées au lancement du scénario, puis figées. Aucune propriété des entités n'est modifiée par les événements.

### 3. Zone événements

Journal append-only produit par chaque simulation. Chaque événement est rattaché à une et une seule entité. Types extensibles via une table de référence.

### 4. Zone ML

Métadonnées des datasets (corpus d'entraînement) et des modèles entraînés. Indépendante des trois autres zones (aucun rapport conceptuel entre datasets/modèles ML et scénarios utilisateur). Les fichiers volumineux (parquet, .pth) vivent sur disque dans un volume Docker, pas en DB.

## Schéma — entités

### Scenario

| Champ | Description |
|---|---|
| id | Identifiant unique |
| nom | Libellé saisi par l'utilisateur |
| date_creation | Horodatage |
| statut_entites | en_attente / en_cours / termine / echec |
| nb_systemes | Nombre de systèmes solaires à générer |
| masse_stellaire_moyenne | Paramètre astrophysique (M☉) — *à ajouter* |
| indice_tellurique | Paramètre astrophysique (0 à 1) — *à ajouter* |
| planetes_par_systeme_moyen | Paramètre astrophysique (lambda Poisson) — *à ajouter* |
| duree_simulation_Ga | Paramètre astrophysique (Ga) — *à ajouter* |

Note : les variables de Drake (r_star, fp, ne, fl, fi, fc, l_drake) ne sont **pas** des paramètres d'entrée. Elles **émergent** de la simulation.

Les 4 paramètres astrophysiques au-dessus existent déjà côté générateur de vérité (`ParametresAstrophysiques` dans `ml/generator/distributions.py`) mais ne sont pas encore en colonnes de la table `scenario`. Cet ajout est un préalable à l'intégration du modèle entraîné dans la chaîne de simulation utilisateur.

### Simulation

| Champ | Description |
|---|---|
| id | Identifiant unique |
| scenario_id | FK vers Scenario |
| date_lancement | Horodatage |
| statut | en_attente / en_cours / terminee / echec |
| progression | Avancement (0.0 à 1.0) |
| seed | Graine aléatoire |

### Galaxie

| Champ | Description |
|---|---|
| id | Identifiant unique |
| scenario_id | FK vers Scenario (1-1 dans le prototype) |

### SystemeSolaire

| Champ | Description |
|---|---|
| id | Identifiant unique |
| galaxie_id | FK vers Galaxie |
| position_x, position_y, position_z | Position 3D dans la galaxie |

### Etoile

| Champ | Description |
|---|---|
| id | Identifiant unique |
| systeme_id | FK vers SystemeSolaire |
| (propriétés physiques) | À enrichir : type spectral, T effective, masse, luminosité, durée de vie. Préalable à l'intégration du modèle ML dans la chaîne. |

Le schéma autorise déjà N étoiles par système (relation 1-N) pour les versions futures (binaires, triples). Une seule étoile par système dans le prototype.

### Planete

| Champ | Description |
|---|---|
| id | Identifiant unique |
| systeme_id | FK vers SystemeSolaire |
| etoile_id | FK vers Etoile |
| (propriétés physiques) | À enrichir : distance orbitale, masse, rayon, composition. Préalable à l'intégration du modèle ML. |

### TypeEvenement

Table de référence extensible. Référentiel des types d'événements observables dans une simulation. Le moteur stochastique actuel et le futur moteur ML produisent des événements typés selon cette table.

| Champ | Description |
|---|---|
| id | Identifiant unique |
| libelle | Nom du type (ex: `life_apparition`, `star_main_sequence_end`, `emergence_civilisation`, …) |

### Evenement

| Champ | Description |
|---|---|
| id | Identifiant unique |
| simulation_id | FK vers Simulation |
| timecode | Position temporelle (entier, unité 100 000 ans) |
| type_evenement_id | FK vers TypeEvenement |
| entite_id | Identifiant de l'entité concernée |
| entite_type | Type de cette entité (galaxie / systeme / etoile / planete) |
| payload | Charge utile JSONB, enrichie au cas par cas |

La référence polymorphe (`entite_id` + `entite_type`) a été choisie pour sa simplicité. Pas de FK SQL stricte vers les entités, contrepartie acceptée à ce stade.

## Schéma — zone ML

### Dataset

Métadonnées des datasets d'entraînement. Les fichiers parquet correspondants vivent dans le volume Docker `/srv/data/training/`.

| Champ | Description |
|---|---|
| nom | Identifiant unique du dataset (clé primaire) |
| date_creation | Horodatage |
| nb_systemes | Nombre de systèmes solaires utilisés pour la génération |
| seed | Graine aléatoire |
| chemin_planets | Chemin du parquet « planètes » dans le volume Docker |
| chemin_events | Chemin du parquet « événements » dans le volume Docker |
| taille_octets | Taille totale (planets + events) |
| stats | Snapshot statistique JSONB calculé au moment de la génération (cf. format dans `ml.md`) |

Les stats sont un snapshot figé : elles ne sont pas recalculées à la lecture, elles reflètent ce qu'a produit la génération.

### Model

Métadonnées des modèles entraînés. Le fichier `.pth` vit dans le même volume Docker que les datasets.

| Champ | Description |
|---|---|
| nom | Identifiant unique du modèle (clé primaire) |
| date_creation | Horodatage |
| dataset_nom | Nom du dataset utilisé pour l'entraînement |
| chemin_fichier | Chemin du `.pth` dans le volume Docker |
| taille_octets | Taille du fichier |
| duree_entrainement_s | Durée totale de l'entraînement en secondes |
| training_config | JSONB — hyperparamètres d'entraînement (TrainingConfig.__dict__) |
| tpp_config | JSONB — configuration architecturale du modèle (TPPConfig.__dict__) |
| metrics_par_epoch | JSONB — liste de dicts (epoch, train_loss, val_loss, val_log_L_par_event) |
| final_train_loss | Loss train de la dernière epoch |
| final_val_loss | Loss val de la dernière epoch |

## Format des fichiers parquet

Un dataset est composé de **deux parquets** :

- `<nom>_planets.parquet` — une ligne par planète
- `<nom>_events.parquet` — une ligne par événement (format long)

Joints par `planet_id`. Format long pour les événements : agnostique au nombre de types et à la structure des chaînes.

### Schéma planets.parquet

| Colonne | Description |
|---|---|
| `planet_id` | Identifiant unique (clé de jointure avec events) |
| `system_id` | Système d'appartenance |
| `param_masse_stellaire_moyenne` | Paramètre astrophysique du système (inspection seulement, *pas* feature ML) |
| `param_indice_tellurique` | idem |
| `param_planetes_par_systeme_moyen` | idem |
| `param_duree_simulation_Ga` | idem |
| `star_type` | M, K, G, F, A, B (feature ML catégorielle) |
| `star_temp_K` | Température effective (feature ML continue) |
| `star_mass_solar` | Masse stellaire (feature ML continue, log10 avant z-score) |
| `star_luminosity_solar` | Luminosité (feature ML continue, log10 avant z-score) |
| `star_lifetime_Ga` | Durée de vie séquence principale (feature ML continue, log10) |
| `planet_distance_UA` | Distance orbitale (feature ML continue, log10) |
| `planet_mass_terre` | Masse planétaire (feature ML continue, log10) |
| `planet_radius_terre` | Rayon planétaire (feature ML continue, log10) |
| `planet_composition` | tellurique / glacee / gazeuse (feature ML catégorielle) |
| `T_obs` | Horizon d'observation en timecode (= duree_simulation_Ga convertie) |
| `life_probability` | Diagnostic : produit multiplicatif des facteurs de vie |
| `f_HZ`, `f_composition`, `f_masse`, `f_etoile` | Diagnostic : facteurs individuels |

Note importante : les 4 colonnes `param_*` sont stockées dans le parquet pour inspection mais ne sont pas données au modèle ML en entrée — décision « pas de fuite d'information », puisqu'elles ont déjà servi à générer les features physiques.

### Schéma events.parquet

| Colonne | Description |
|---|---|
| `planet_id` | FK vers planets.parquet |
| `event_type_id` | Index entier dans le référentiel `EVENT_TYPES` |
| `event_type_label` | Libellé du type (redondant avec id, utile pour inspection) |
| `event_time` | Timecode de l'événement (entier, unité 100 000 ans) |

Le format long permet d'avoir 0, 1, 2 ou N événements par planète sans changer le schéma. La chaîne effective (vie → fin SP) émerge naturellement à partir du tri par `event_time`.

## Diagramme des relations

```
Scenario 1───N Simulation 1───N Evenement N───1 TypeEvenement
   │                                  │
   │                                  │ référence polymorphe
   ▼ 1-1                              │ (entite_id + entite_type)
Galaxie 1───N SystemeSolaire ─────────┤
                  │                   │
                  ├─ 1───N Etoile ────┤
                  │      │            │
                  │      │ 1          │
                  │      │            │
                  └─ 1───N Planete ───┘
                         │ N
                         └─ orbite autour d'une Etoile

Zone ML (indépendante)
Dataset (1)──── (1) parquet planets + (1) parquet events
Model   (1)──── (1) fichier .pth, références dataset_nom
```

## Règles invariantes

1. Un scénario possède exactement un ensemble d'entités (1-1 conceptuel via la galaxie unique).
2. Les entités sont immuables après génération.
3. Chaque événement appartient à une seule simulation et concerne une seule entité.
4. Plusieurs simulations d'un même scénario partagent les mêmes entités mais ont chacune leur propre journal.
5. État de l'univers à T = état initial des entités + rejeu des événements antérieurs à T.
6. Un dataset est identifié par son nom. Suppression d'un dataset = suppression conjointe des deux parquets et de la ligne en DB.
7. Un modèle est identifié par son nom. Suppression d'un modèle = suppression du `.pth` et de la ligne en DB.
8. Les événements d'une planète sont triés par `event_time`. Aucune contrainte d'ordre n'est imposée par le schéma : la cohérence physique (ex: vie avant fin SP) est garantie par le générateur, pas par le format de stockage.

## Évolutions hors prototype, déjà compatibles

- Plusieurs étoiles par système (binaires, triples) : relation déjà en 1-N.
- Nouveaux objets dans les entités (lunes, astéroïdes, comètes…) : ajout de tables avec FK. `entite_type` extensible.
- Nouveaux types d'événements : ajout de lignes dans TypeEvenement et dans `EVENT_TYPES` côté générateur. Aucun changement de schéma. Le format long du parquet events absorbe naturellement.
- Nouvelles chaînes d'événements branchantes (planète A : E1→E2→E4 ; planète B : E1→E3→E4) : aucun changement de schéma. Format long agnostique.
- Payloads par type d'événement : ajout d'une colonne JSONB dans la table `Evenement` (déjà présente) et dans `events.parquet` (à ajouter au moment du besoin).
- Nouveaux paramètres de scénario : ajout de colonnes dans Scenario.
- Évolutions de propriétés physiques sur étoile et planète : ajout de colonnes (préalable à l'intégration ML dans la chaîne de simulation).
- Nouveaux types de ressources ML (métriques d'évaluation, suites d'expériences, comparaisons modèle/dataset) : ajout de tables sur le pattern de `dataset` et `model` (métadonnées en DB, fichiers binaires sur volume Docker).

## Points d'implémentation différés

- Choix d'identifiants (BIGINT, UUID, composites).
- Stratégie d'indexation, en particulier index spatial 3D.
- Partitionnement (par scenario_id, simulation_id, timecode).
- Format de stockage du payload des événements (colonnes typées vs JSONB) — à arbitrer cas par cas selon les besoins réels.
- Versionnage du schéma DB (migrations) — à introduire à la version alpha.
- Validation du format du nom de dataset/modèle (regex Pydantic).

## Style applicatif

- **psycopg 3** pour l'accès direct à Postgres (pas d'ORM).
- **Pydantic v2** pour la validation et la modélisation des objets métier.
- **FastAPI** pour les endpoints, génération automatique de la doc OpenAPI sur `/docs`.
- Modèles métier riches type **Active Record** : `scenario.create(conn)`, `dataset.create(conn)`, `model.create(conn)` — pas de couche repository séparée.
- Connexions ouvertes par l'appelant via `with get_conn() as conn:`, partagées entre les opérations qui doivent être atomiques.
- Fonctionnalités d'administration exposées par deux transports symétriques (HTTP et CLI) qui partagent une même couche service.
