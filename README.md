# FC-Simulator

> Simulateur d'événements à l'échelle de notre voie lactée, des éléments astrophysiques (astres, systèmes solaiures, etc.) aux civilisations et leur dynamique.

Sommes-nous seuls dans l'univers ?
FC-Simulator simule des **chronologies d'événements** qui peuvent se produire dans une galaxie — apparition de la vie, fin de vie d'une étoile, et à terme émergence de civilisations, expansions, premiers contacts, extinctions. Les variables de Drake (taux d'apparition de la vie, durée de vie des civilisations, etc.) **émergent** du jeu d'événements simulé.

C'est avant tout un projet technique d'apprentissage et d'exploration. Les ambitions scientifiques affichées — Drake, Fermi, l'apparition de la vie — sont les sujets qui rendent l'exercice intéressant, pas un objectif que ce prototype prétendrait atteindre, évidemment ! Mais La calibration scientifique (zones habitables, distributions stellaires, relations masse-rayon) tente de suivre l'état de l'art consensuel. Les arbitrages de modélisation et les approximations sont assumés et documentés.

## Philosophie du simulateur

Trois étapes successives, chacune avec son outillage propre :

1. **Construction de la galxie** — l'utilisateur fixe quelques paramètres astrophysiques (masse stellaire moyenne, indice tellurique, nombre moyen de planètes par système, durée de simulation). Le générateur produit alors une galaxie : étoiles avec leurs propriétés physiques, planètes avec leurs caractéristiques, zones habitables calculées par les polynômes de Kopparapu, etc.

2. **Survenue dynamique d'événements** — sur ce terrain, un moteur produit un journal d'événements ordonnés dans le temps (timecode). Chaque événement est attaché à une entité (étoile, planète) et appartient à un type extensible. Le journal est append-only : l'état de la galaxie à un instant T se reconstitue en rejouant les événements antérieurs à T.

3. **Exploration** — l'IHM Unity (à venir) permet d'explorer le résultat à trois échelles : galaxie entière (densités, signalement d'événements), système solaire, planète individuelle.

L'architecture est conçue pour la cible 100 à 400 milliards de systèmes (consensus actuel); le prototype actuel tourne en local sur 1 million.

## Périmètre du prototype

Ce qui est livré aujourd'hui :

- Stack Docker complète (Postgres + FastAPI), schéma DB stable.
- Génération du substrat galactique paramétrée, calibrée scientifiquement.
- Générateur de vérité terrain : produit des datasets parquet de planètes étiquetées avec leurs séquences d'événements.
- Modèle ML d'apprentissage de la dynamique événementielle (Transformer Hawkes Process).
- Pipeline d'entraînement complet, intégré aux mêmes patterns API/CLI que le reste.
- Deux types d'événements implémentés : **apparition de la vie** sur une planète, **fin de séquence principale** d'une étoile.

Ce qui n'est pas dans le prototype :

- Intégration du modèle entraîné dans la chaîne de simulation (préalable : enrichir le schéma des entités avec les features physiques).
- IHM Unity.
- Scaling au-delà de quelques millions de systèmes (nécessite un environnement adapté : cluster, cloud).

> **Pistes d'évolution.** L'extension naturelle est l'ajout de types d'événements (impacts cométaires majeurs, supernovae, émergence de civilisations, contacts inter-civilisations, expansions) et de leurs interdépendances temporelles. Le format de stockage et l'architecture ML sont conçus pour absorber ces extensions sans refonte. À plus long terme, l'analyse a posteriori de runs simulés ouvre la porte à l'exploration des régimes paramétriques qui produisent telles ou telles solutions au paradoxe de Fermi (vide, contact, extinction).

## Choix techniques

### Modèle événementiel

L'état de la galaxie se reconstitue par rejeu d'un journal d'événements ordonnés.

### Apprentissage par processus ponctuel temporel (TPP)

Le moteur ML qui prédit les événements est un **Transformer Hawkes Process**, conditionné sur les features physiques de l'étoile et de la planète. Pour chaque planète, le modèle apprend une fonction d'intensité λₖ(t) par type d'événement, qui donne le taux instantané d'occurrence sachant l'historique des événements antérieurs.

Ce choix d'architecture vient de la cible long terme : à mesure que des types d'événements seront ajoutés, leurs interdépendances temporelles deviendront essentielles (un contact ne peut survenir qu'après l'émergence d'une civilisation, qui ne peut survenir qu'après l'apparition de la vie). Les TPP sont le cadre statistique naturel pour ce genre de chaîne d'événements interdépendants.

L'entraînement repose sur un **générateur de vérité** : un module Python qui implémente les règles (zone habitable, statistiques exoplanètes, relations masse-rayon, modèle multiplicatif d'apparition de la vie) et produit un corpus de planètes étiquetées avec leurs séquences d'événements. Le ML apprend ensuite sur ce corpus, pour pouvoir accélérer l'inférence à 400 milliards de systèmes.

### Patterns big data sans outils big data

Streaming, partitionnement, vues agrégées : oui. Spark, Kafka, Flink : pas tant qu'un besoin réel n'apparaît. Python + Postgres standard pour le prototype, bascule vers des outils plus avancés quand la pression réelle l'imposera.

## Architecture

```
       ┌──────────┐         ┌────────────┐         ┌──────────┐
       │  Unity   │ ◄─────► │  FastAPI   │ ◄─────► │ Postgres │
       │  (IHM)   │  HTTP   │  + moteur  │   SQL   │   16     │
       └──────────┘         │  + module  │         └──────────┘
                            │     ML     │
                            └─────┬──────┘
                                  │
                                  ▼
                            ┌──────────┐
                            │  data/   │
                            │ (parquet │
                            │  + .pth) │
                            └──────────┘
```

Tout est containerisé. Le service `api` héberge à la fois l'API métier et le module ML (générateur de vérité, entraînement, inférence). Les datasets parquet et les modèles entraînés `.pth` vivent dans un volume Docker partagé ; les métadonnées sont en DB.

## Démarrage rapide

Prérequis : Docker, Docker Compose.

```bash
git clone <repo>
cd fc-simulator
docker compose up --build
```

Trois services démarrent :

- **API** : http://localhost:8000 (OpenAPI sur `/docs`)
- **pgAdmin** : http://localhost:5050 (`admin@drake.fr` / `admin`)
- **Postgres** : exposé sur le port 5432


## API

Absolument toutes les commandes pour piloter la simulation ont leur api. Le lancement par terminal est évidemment possible, mais le protoype est pensé pour une utilisation future entièrement via IHM Unity (même la création d'un nouveau dataset d'entrainement paramétré).

Préfixe commun : `/api/v1/`. Documentation interactive complète sur `/docs`.

**Scénarios et simulations**

| Méthode | URL                                  | Description                                 |
|---------|--------------------------------------|---------------------------------------------|
| GET     | `/health`                            | Vie du service + DB joignable               |
| POST    | `/scenarios`                         | Crée un scénario                            |
| GET     | `/scenarios/{id}`                    | État d'un scénario + ses simulations        |
| POST    | `/scenarios/{id}/entites`            | Génère les entités du scénario              |
| POST    | `/scenarios/{id}/simulations`        | Crée et exécute une simulation              |
| GET     | `/simulations/{id}/evenements`       | Journal d'une simulation                    |
| POST    | `/scenarios/full`                    | Pipeline complet (endpoint de confort dev)  |

**Module ML — datasets d'entraînement**

| Méthode | URL                                  | Description                                 |
|---------|--------------------------------------|---------------------------------------------|
| POST    | `/training/datasets`                 | Génère un dataset (planètes + événements)   |
| GET     | `/training/datasets`                 | Liste les datasets                          |
| GET     | `/training/datasets/{nom}`           | Métadonnées d'un dataset                    |
| DELETE  | `/training/datasets/{nom}`           | Supprime un dataset (DB + fichiers)         |

**Module ML — modèles entraînés**

| Méthode | URL                                  | Description                                 |
|---------|--------------------------------------|---------------------------------------------|
| POST    | `/training/models`                   | Lance un entraînement                       |
| GET     | `/training/models`                   | Liste les modèles                           |
| GET     | `/training/models/{nom}`             | Métadonnées d'un modèle                     |
| DELETE  | `/training/models/{nom}`             | Supprime un modèle (DB + .pth)              |

À chaque endpoint du module ML correspond une commande CLI symétrique, exécutable via `docker compose exec api python -m ml.cli ...`.

## Stack technique

- **Python 3.12** — FastAPI, Pydantic v2, psycopg 3
- **NumPy / Pandas / PyArrow** — générateur de vérité, datasets parquet
- **PyTorch** — Transformer Hawkes Process (modèle TPP)
- **PostgreSQL 16** — entités, événements, métadonnées
- **Docker Compose** — orchestration locale
- **Unity** (à venir) — IHM 3 niveaux de zoom

## Structure

```
fc-simulator/
├── docker-compose.yml
├── data/training/         # datasets parquet et modèles .pth (volume Docker)
└── server/
    ├── sql/init.sql       # schéma DB initial
    └── app/
        ├── main.py        # endpoints FastAPI
        ├── db/            # accès Postgres
        ├── simulation/    # substrat + moteur d'événements
        └── ml/            # module ML
            ├── api.py     # endpoints /training/ (datasets + models)
            ├── cli.py     # CLI symétrique
            ├── service.py
            ├── dataset.py
            ├── generator/ # générateur de vérité scientifique
            └── training/  # modèle TPP, entraînement, inférence
```
