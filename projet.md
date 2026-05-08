# FC-Simulator — projet

> Fichier de référence Claude. Vision projet, philosophie, vocabulaire, choix transverses non-techniques.
> Pour le schéma de données et la structure des fichiers : voir `donnees.md`.
> Pour le module ML (générateur de vérité, training, inférence) : voir `ml.md`.

## Philosophie

L'utilisateur règle des **conditions initiales** astrophysiques (substrat physique). La simulation **fait émerger** les valeurs des variables Drake (ne, fl, fi, fc, L) ainsi que des solutions au paradoxe de Fermi. Ces variables résultantes ne doivent jamais être saisies en entrée — sinon le projet perd sa raison d'être.

Les **lois d'apparition de la vie** (zone habitable, facteur composition, facteur masse, etc.) sont en revanche **figées** et reflètent le consensus scientifique actuel. Les exposer reviendrait à laisser l'utilisateur décider à l'avance « la vie est facile » ou « la vie est difficile » et retrouver mécaniquement ce qu'il a posé.

Le projet est avant tout un **projet technique d'apprentissage et d'exploration**. Les ambitions scientifiques affichées (Drake, Fermi, apparition de la vie) sont les sujets qui rendent l'exercice intéressant, pas un objectif que ce prototype prétendrait atteindre. La calibration scientifique suit l'état de l'art consensuel ; les arbitrages de modélisation et les approximations sont assumés et documentés.

## Vision

Simulateur d'**événements à l'échelle galactique**, du substrat astrophysique aux civilisations. L'utilisateur paramètre un scénario, lance un run, puis explore le résultat via 3 niveaux de zoom Unity : galaxie, système solaire, planète.

Le centre du projet est la **chronologie des événements** qui se produisent dans la galaxie : apparition de la vie, fin de séquence principale d'une étoile, et à terme civilisations, contacts, expansions, extinctions. Drake et l'apparition de la vie ne sont qu'un cas particulier de ce cadre événementiel.

## Échelle

- **Cible finale** : 100 à 400 milliards de systèmes solaires par run.
- **Cible prototype** : 1 million de systèmes — exécutable localement.
- L'architecture est conçue pour la cible finale dès le prototype.
- L'exécution réelle à 400 milliards nécessitera ultérieurement un environnement adapté (cluster, cloud), hors prototype.

## Vocabulaire

- **Scénario** : jeu de paramètres saisi par l'utilisateur.
- **Entités** : ensemble du contenu généré du scénario (galaxie + systèmes + étoiles + planètes). Substrat immuable une fois généré.
- **Simulation** : exécution du moteur sur les entités d'un scénario, qui produit un journal d'événements.
- **Run** : terme courant pour désigner une exécution complète (scénario + entités + une simulation).
- **Paramètres du scénario** : variables que l'utilisateur saisit. Les variables de Drake n'en font *pas* partie ; elles émergent.
- **Événement** : occurrence ponctuelle attachée à une entité (typiquement une planète ou une étoile), datée par un timecode. L'unité de timecode est figée à 100 000 ans.
- **Type d'événement** : libellé extensible référencé par TypeEvenement. Aujourd'hui implémentés : `life_apparition`, `star_main_sequence_end`. Futurs envisagés : émergence civilisation, contact, expansion, extinction, impact cométaire, supernova.
- **Dataset (d'entraînement)** : corpus de planètes étiquetées avec leurs séquences d'événements, produit par le générateur de vérité, utilisé pour entraîner le ML. Indépendant des scénarios utilisateur.
- **Modèle (entraîné)** : fichier `.pth` PyTorch produit par l'entraînement, accompagné de ses métadonnées en DB.
- **Générateur de vérité** : module Python qui implémente les règles consensuelles (zone habitable, distributions stellaires, modèle d'apparition de la vie) et produit les datasets d'entraînement. C'est le « professeur » du modèle ML.

## Modèle conceptuel

```
Galaxie
└── Système solaire
    ├── Étoile(s) [1 ou plusieurs : simple, binaire, triple+]
    ├── Planète(s)
    │   └── Civilisation(s) [rattachée à une planète, à venir]
    └── Autres objets [futurs : astéroïdes, comètes, lunes...]
```

- Une civilisation appartient à une planète, pas à une étoile.
- Plusieurs civilisations peuvent coexister dans un même système (interactions futures : expansion, contact, conflit).
- Le système solaire détient l'environnement qui détermine les conditions d'apparition de la vie.

## Modèle temporel — événementiel

Pas de snapshots réguliers. L'état du monde est reconstitué en rejouant un journal d'événements ordonnés (timecode). Volume estimé ~4 ordres de grandeur de moins que le modèle snapshots. Pattern proche de l'event sourcing.

Le moteur de simulation est une **machine à événements** : pour chaque entité, il détermine quels événements surviennent et à quel timecode. À terme, ces événements peuvent dépendre les uns des autres (un contact suppose une civilisation, qui suppose la vie).

## IHM Unity

Trois niveaux de zoom :
- **Galaxie** : densités + événements signalés (pas affichage individuel des milliards). Chronologie défilante, pause/lecture, clic sur événement pour zoomer.
- **Système solaire** : étoiles, planètes, environnement.
- **Planète** : propriétés physiques, civilisation éventuelle, histoire.

Deux fonctionnalités asynchrones :
1. Lancer un calcul de scénario (peut prendre du temps).
2. Visualiser/rejouer un run déjà calculé.

Pas de runs concurrents nécessaires dans le prototype.

## Choix techniques transverses

### Patterns big data sans outils big data

- Gratuit, peu importe la solution tant qu'elle scale.
- Local et écosystème connu pour le prototype (Python + Postgres standard).
- Adoption de **patterns** big data (streaming, partitionnement, vues agrégées) sans adopter les **outils** big data.
- Bascule vers des outils plus avancés au moment du besoin réel.

### Pas d'ORM

Accès direct à Postgres via psycopg 3. Pydantic v2 pour la validation et la modélisation des objets métier. Style **Active Record** : `scenario.create(conn)` plutôt que `scenario_repo.create(conn, scenario)`. Décision assumée pour le prototype, refactorable.

### Pattern command/controller

Toutes les fonctionnalités d'administration (datasets, entraînements) sont exposées par **deux transports symétriques** :
- HTTP : endpoints FastAPI sous `/api/v1/training/`.
- CLI : commandes invoquées via `docker compose exec api python -m ml.cli ...`.

Les deux transports passent par une même couche service partagée. À chaque endpoint correspond une commande CLI symétrique. Le cœur métier ne dépend ni de FastAPI ni d'argparse.

### Endpoints synchrones

Les endpoints sont synchrones à ce stade. La bascule async (BackgroundTasks + polling) est prévue mais non encore implémentée. À mettre quand un besoin concret apparaît (run long, polling Unity).

## Architecture serveur

| Composant | Choix |
|---|---|
| Stack serveur | Python 3.12 + FastAPI |
| DB | PostgreSQL 16 |
| Format modèle ML | `.pth` natif PyTorch |
| Containerisation | Docker Compose : `db`, `api` (FastAPI + moteur + module ML), `pgadmin` |
| Run = entité centrale | Persisté en DB avec statut, progression, paramètres, métadonnées |

Unity travaille de façon découplée via API HTTP/JSON, préfixe `/api/v1/`, OpenAPI.

Le module ML fait partie intégrante du service `api`. Les fichiers volumineux qu'il produit (datasets parquet, modèles entraînés `.pth`) vivent dans un volume Docker `data/` partagé entre l'hôte et le conteneur. Les métadonnées de ces fichiers sont en DB.

## Endpoints actuels

```
POST   /api/v1/scenarios                         crée un scénario
GET    /api/v1/scenarios/{id}                    état d'un scénario + simulations
POST   /api/v1/scenarios/{id}/entites            génère les entités du scénario
POST   /api/v1/scenarios/{id}/simulations        crée et exécute une simulation
GET    /api/v1/simulations/{id}/evenements       journal d'une simulation
POST   /api/v1/scenarios/full                    pipeline complet (endpoint de confort dev)

POST   /api/v1/training/datasets                 génère un dataset (planets + events)
GET    /api/v1/training/datasets                 liste les datasets
GET    /api/v1/training/datasets/{nom}           métadonnées d'un dataset
DELETE /api/v1/training/datasets/{nom}           supprime un dataset (DB + fichiers)

POST   /api/v1/training/models                   lance un entraînement
GET    /api/v1/training/models                   liste les modèles
GET    /api/v1/training/models/{nom}             métadonnées d'un modèle
DELETE /api/v1/training/models/{nom}             supprime un modèle (DB + .pth)
```

## Commandes CLI symétriques

```
python -m ml.cli datasets create --nom <n> --systemes <N> [--seed <s>]
python -m ml.cli datasets list
python -m ml.cli datasets show <nom>
python -m ml.cli datasets delete <nom>

python -m ml.cli models create --nom <m> --dataset <d> [--nb-epochs <N>] [...]
python -m ml.cli models list
python -m ml.cli models show <nom>
python -m ml.cli models delete <nom>
```

## État d'avancement

Livré et fonctionnel :
- Stack Docker complète, schéma DB stable.
- Génération du substrat galactique paramétrée et calibrée.
- Bloc A (générateur de vérité) en version multi-événements (vie + fin SP).
- Bloc B (entraînement TPP — Transformer Hawkes Process) opérationnel.
- Pipeline complet datasets + models, intégré HTTP/CLI symétrique.
- Validé techniquement sur 100 000 systèmes (~300 000 planètes), 20 epochs, ~13 minutes.

Pas dans le prototype :
- Intégration du modèle entraîné dans la chaîne de simulation utilisateur (préalable : enrichir le schéma `etoile` et `planete` avec les features physiques, et ajouter les 4 paramètres astrophysiques à `Scenario`).
- IHM Unity.
- Scaling au-delà de quelques millions de systèmes.
- Validation par comparaison « chaînes générées vs chaînes observées » (option pour la suite).

## Écarts assumés et points en attente

- **Async** : non implémenté. À mettre quand un besoin concret apparaît.
- **Versionnage du schéma DB** : pas en place tant qu'on est en phase d'exploration. À mettre en version alpha.
- **Repositories séparés** : non. Active Record assumé.
- **Tests unitaires** : absents. Code testé via les endpoints en bout-à-bout.
- **Validation du nom des datasets/modèles** : aucune contrainte de format actuellement. Regex Pydantic à introduire au besoin.
- **Choix d'identifiants** (BIGINT vs UUID), partitionnement, indexation spatiale 3D : différés au passage à l'échelle.
- **Early stopping / scheduler de learning rate / sauvegarde du best-val checkpoint** : pas en place côté training, à introduire si on veut affiner.
