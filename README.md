# FC-Simulator

> Simulateur d'évolution de civilisations à l'échelle galactique, fondé sur une équation de Drake étendue.

Sommes-nous seuls dans l'univers ? La question reste ouverte, et la formulation la plus connue — l'équation de Drake — propose un cadre pour estimer le nombre de civilisations détectables, mais elle reste statique : un produit de probabilités, pas une dynamique.

FC-Simulator est un simulateur qui transforme cette équation en monde vivant. À partir de paramètres saisis par l'utilisateur, il génère un substrat galactique (étoiles, planètes, zones habitables), puis simule son évolution dans le temps : émergence de la vie, des civilisations, expansions, contacts, extinctions. Le résultat s'explore à trois échelles dans une IHM Unity : la galaxie entière, un système solaire, une planète.

## État d'avancement

| Bloc                                    | Statut         |
|-----------------------------------------|----------------|
| Stack Docker (Postgres + FastAPI)       | ✓ opérationnel |
| Modèle de données (scénarios, entités, événements) | ✓ opérationnel |
| Génération du substrat galactique       | ✓ opérationnel |
| Moteur stochastique d'événements        | ✓ minimal      |
| Générateur de vérité scientifique (ML)  | ✓ opérationnel |
| Entraînement du modèle ML (PyTorch)     | ⌛ en cours     |
| IHM Unity 3 niveaux de zoom             | ⌛ à venir      |
| Scaling 100-400 milliards de systèmes   | ⌛ étape future |

Projet à fonction d'apprentissage et d'exploration. Le prototype tourne en local (1 million de systèmes). L'architecture est conçue pour une cible 100-400 milliards à terme.

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
                            │ (parquet)│
                            └──────────┘
```

Tout est containerisé. Le service `api` héberge à la fois l'API métier et le module ML (générateur de vérité, entraînement). Les datasets d'entraînement vivent dans un volume Docker partagé ; les métadonnées sont en DB.

## Démarrage rapide

Prérequis : Docker, Docker Compose.

```bash
git clone <repo>
cd drake
docker compose up --build
```

Trois services démarrent :

- **API** : http://localhost:8000 (OpenAPI sur `/docs`)
- **pgAdmin** : http://localhost:5050 (`admin@drake.fr` / `admin`)
- **Postgres** : exposé sur le port 5432

## Un exemple concret

Génération d'un dataset d'entraînement de 1000 systèmes solaires (le générateur scientifique calibre les distributions stellaires, calcule les zones habitables selon Kopparapu 2013, et étiquette chaque planète) :

```bash
curl -X POST http://localhost:8000/api/v1/training/datasets \
     -H 'Content-Type: application/json' \
     -d '{"nom": "demo", "nb_systemes": 1000, "seed": 42}'
```

Réponse (extrait) :

```json
{
  "nom": "demo",
  "nb_systemes": 1000,
  "stats": {
    "nb_planetes": 2028,
    "nb_telluriques": 665,
    "nb_vie": 27,
    "taux_vie": 0.013,
    "vie_par_type_spectral": {
      "G": {"vie": 4, "total": 151, "taux": 0.026},
      "K": {"vie": 7, "total": 215, "taux": 0.033},
      "M": {"vie": 16, "total": 1572, "taux": 0.010}
    }
  }
}
```

Hiérarchie attendue confirmée : G (étoiles type Soleil) > K > M, conforme au consensus astrobiologique.

## API

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
| POST    | `/training/datasets`                 | Génère un dataset d'entraînement            |
| GET     | `/training/datasets`                 | Liste les datasets                          |
| GET     | `/training/datasets/{nom}`           | Métadonnées d'un dataset                    |
| DELETE  | `/training/datasets/{nom}`           | Supprime un dataset (DB + fichier)          |

À chaque endpoint du module ML correspond une commande CLI symétrique, exécutable via `docker compose exec api python -m ml.cli ...`.

## Stack technique

- **Python 3.12** — FastAPI, Pydantic v2, psycopg 3
- **NumPy / Pandas / PyArrow** — générateur de vérité, datasets parquet
- **PyTorch** (à venir) — modèle MLP
- **PostgreSQL 16** — entités, événements, métadonnées
- **Docker Compose** — orchestration locale
- **Unity** (à venir) — IHM 3 niveaux de zoom

Pas d'ORM (SQL direct via psycopg). Pas d'outils big data : adoption des **patterns** big data (streaming, partitionnement) sans les **outils**, jusqu'au moment où le besoin réel apparaîtra.

## Structure

```
drake/
├── docker-compose.yml
├── data/training/         # datasets parquet (volume Docker)
└── server/
    ├── sql/init.sql       # schéma DB initial
    └── app/
        ├── main.py        # endpoints FastAPI
        ├── db/            # accès Postgres
        ├── simulation/    # substrat + moteur d'événements
        └── ml/            # module ML
            ├── api.py     # endpoints /training/
            ├── cli.py     # CLI symétrique
            ├── service.py
            └── generator/ # générateur de vérité scientifique
```
