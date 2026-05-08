# FC-Simulator

> Simulateur d'événements à l'échelle de notre voie lactée, des éléments astrophysiques (astres, systèmes solaiures, etc.) aux civilisations et leur dynamique.

Sommes-nous seuls dans l'univers ?
FC-Simulator simule des **chronologies d'événements** qui peuvent se produire dans une galaxie (apparition de la vie, fin de vie d'une étoile, premiers contacts, extinctions, etc.). Les variables de Drake (taux d'apparition de la vie, durée de vie des civilisations, etc.) et résolutions du paradoxe de Fermi (pourquoi les extra terrestres e sont pas déjà là ?) **émergent** du jeu d'événements simulé.

Ceci est d'abord un projet technique (Data Science et Machine Learning) d'apprentissage et d'exploration. Les ambitions scientifiques affichées (Drake, Fermi, l'apparition de la vie) rendent l'exercice intéressant, mais ne sont pas un objectif final que ce prototype prétendrait atteindre, évidemment ! Cela-dit, la calibration scientifique (zones habitables, distributions stellaires, relations masse-rayon, évènements interdépendants) tente de suivre au mieux l'état de l'art scientifiaue actuel. Notamment, le projet laisse un paramétrage important à l'utilisateur sur toute la chains du pipeline (de la création du dataset source, l'entrainement et la simulation elle-même).

## Philosophie du simulateur

1. **Construction de la galxie** — l'utilisateur fixe les paramètres astrophysiques (masse stellaire moyenne, indice tellurique, nombre moyen de planètes par système, durée de simulation). Le générateur produit alors une galaxie : étoiles avec leurs propriétés physiques, planètes avec leurs caractéristiques, zones habitables calculées par les polynômes de Kopparapu, etc.

2. **Survenue dynamique des événements** — le moteur produit un journal d'événements ordonnés dans le temps (timecode). Chaque événement est attaché à une entité (galaxie, étoile, planète, etc.). Le journal est en append : l'état de la galaxie à un instant T se reconstitue en rejouant les événements jusqu'à T.

3. **Exploration** — l'IHM Unity permet d'explorer la simulation à trois échelles : galaxie entière (densités, signalement d'événements), système solaire, planète individuelle, tout au long des timecodes.

L'architecture est conçue pour la cible 100 à 400 milliards de systèmes solaires (consensus actuel du nombre de systèmes dans noter galaxie).

## Périmètre du prototype

- Stack Docker complète (Postgres + FastAPI), schéma DB stable
- Génération des éléments galactique paramétrés, calibrés scientifiquement
- Générateur de vérité terrain : produit des datasets parquet de planètes étiquetées avec leurs séquences d'événements
- Modèle ML d'apprentissage de la suite des évènements
- Pipeline d'entraînement complet
- Deux types d'événements implémentés pour l'instant : **apparition de la vie** sur une planète, **fin de séquence principale** d'une étoile.

Ce qui n'est pas dans le prototype :

- Intégration du modèle entraîné dans la chaîne de simulation (préalable : enrichir le schéma des entités avec les features physiques).
- IHM Unity.


> **évolutions futures**
- ajout de types d'événements (impacts cométaires majeurs, supernovae, émergence de civilisations, contacts inter-civilisations, expansions) et de leurs interdépendances temporelles. Le format de stockage et l'architecture ML sont conçus pour absorber ces extensions sans refonte
- analyse a posteriori de runs simulés pour produire des solutions au paradoxe de Fermi (vide, contact, extinction)
- Scaling cluster et cloud

## Choix techniques

### Modèle événementiel

L'état de la galaxie se reconstitue par rejeu d'un journal d'événements ordonnés.

### Apprentissage par processus ponctuel temporel (TPP)

Le moteur ML qui prédit les événements est un **Transformer Hawkes Process** (parfait pour prédire une chaîne d'événements interdépendants), conditionné sur les features physiques (étoiles, planetes) et leurs carctéristiques (masse, orbite, etc.). Pour chaque planète, le modèle apprend une fonction d'intensité par type d'événement selon l'historique des événements antérieurs (un contact ne peut survenir qu'après l'émergence d'une civilisation, qui ne peut survenir qu'après l'apparition de la vie).

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

Absolument toutes les commandes pour piloter la simulation ont leur api. Le protoype est pensé pour une utilisation future entièrement via IHM Unity (même la création d'un nouveau dataset d'entrainement paramétré) ou l'entrainement lui-même.

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
