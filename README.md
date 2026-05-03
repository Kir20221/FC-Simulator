# Projet Drake — service simulation

Stack Docker minimale : PostgreSQL + FastAPI + pgAdmin dans un seul `docker-compose`.
Cette première itération a pour seul objectif de **valider la chaîne** :
FastAPI ↔ Postgres, intégrité relationnelle du modèle, écriture/relecture
ordonnée du journal d'événements.

## Services

| Service   | Port hôte | URL                       | Rôle                                  |
|-----------|-----------|---------------------------|---------------------------------------|
| `api`     | 8000      | http://localhost:8000     | API FastAPI + moteur de simulation    |
| `api`     | 8000      | http://localhost:8000/docs| API FastAPI tous les     endpoints    |
| `db`      | 5432      | (psql / clients externes) | PostgreSQL 16                         |
| `pgadmin` | 5050      | http://localhost:5050     | Client web Postgres                   |

Identifiants pgAdmin (interface web) : `admin@drake.fr` / `admin`.
Le serveur **drake-db** est déjà déclaré dans pgAdmin ; au premier clic,
saisir le mot de passe Postgres (`drake`) et cocher *Save Password*.

## Démarrage

```bash
docker compose up --build
```

Au premier démarrage, le volume Postgres est vide : le script
`server/sql/init.sql` est exécuté automatiquement et crée le schéma.

API disponible sur `http://localhost:8000` ; OpenAPI sur `/docs`.

## Les logs

```bash
docker compose logs --tail 100 api
```

## Reset complet

À ce stade, le schéma n'est pas versionné. Pour repartir de zéro :

```bash
docker compose down -v
docker compose up --build
```

## Vérification de bout en bout

```bash
# 1. Vie du service + DB joignable
curl http://localhost:8000/api/v1/health

# 2. Run de validation : crée scénario + substrat (10 systèmes) + simulation + événements
curl -X POST http://localhost:8000/api/v1/run-validation \
     -H 'Content-Type: application/json' \
     -d '{"nom": "test1", "nombre_systemes": 10}'

# Réponse exemple :
# {"scenario_id":1,"galaxie_id":1,"simulation_id":1,"seed":...,"nb_evenements":N}

# 3. État du scénario
curl http://localhost:8000/api/v1/scenarios/1

# 4. Relecture ordonnée du journal
curl http://localhost:8000/api/v1/simulations/1/evenements
```

## Structure

```
docker-compose.yml
pgadmin/
└── servers.json              # serveur drake-db pré-déclaré dans pgAdmin
server/
├── Dockerfile
├── requirements.txt
├── sql/
│   └── init.sql              # schéma DB, exécuté à la 1ère init du volume
└── app/
    ├── main.py               # FastAPI : endpoints
    ├── api/                  # (à venir : routeurs séparés)
    ├── db/
    │   └── connection.py     # pool psycopg
    └── simulation/
        ├── substrat.py       # génération galaxie/systèmes/étoiles/planètes
        └── moteur.py         # production d'événements (minimal)
```
