# FC-Simulator

> Simulateur d'événements à l'échelle de notre voie lactée, des éléments astrophysiques (astres, systèmes solaires, etc.) aux civilisations et leur dynamique.

Sommes-nous seuls dans l'univers ?
FC-Simulator simule des **chronologies d'événements** qui peuvent se produire dans notre galaxie (apparition des systèmes solaires, de la vie, fin de vie d'une étoile, premiers contacts, extinctions, etc.). Fonctionnellement, le projet a été guidé par la volonté de proposer :
- l'émergence de valeurs pour les variables de [l'équation de Drake](https://fr.wikipedia.org/wiki/%C3%89quation_de_Drake) => estimation du nombre de civilisations dans notre galaxie
- quelques résolutions du [paradoxe de Fermi](https://fr.wikipedia.org/wiki/Paradoxe_de_Fermi) => pourquoi les extra terrestres ne sont pas déjà là ?

Mais restons humbles, ce projet est d'abord un projet d'exploration technique (Machine Learning, représentation visuelle dynamique) porté par 3 objectifs :
- **exploration de technologies** (voir stack technique plus bas)
- implémentation d'une simulation totalement **paramétrée de bout en bout** du pipeline par l'utilisateur
- IHM dynamique sur **Unity** pour une immersion visuelle et ludique des prédictions

Ainsi, les ambitions scientifiques affichées (Drake, Fermi) rendent l'exercice intéressant, mais ne sont pas un objectif final que ce prototype prétendrait atteindre, évidemment ! Cela-dit, la calibration scientifique (zones habitables, distributions stellaires, relations masse-rayon, évènements interdépendants) tente de suivre au mieux l'état de l'art scientifique actuel, comme le **suivi d'une spécification client**.

## vue prototypale de la galaxie et les prédictions ML
Les cubes représentent les densités stellaires. Les apparitions colorées repréentent les événements stellaires (2 couleurs car 2 types d'événements implémentés à ce stade)


https://github.com/user-attachments/assets/35e6e836-6705-47fc-971d-ec84d94d0e2b


## Philosophie du simulateur

1. **Construction de la galaxie** — l'utilisateur fixe les paramètres astrophysiques. Le générateur produit alors une galaxie : étoiles avec leurs propriétés physiques, planètes avec leurs caractéristiques, zones habitables calculées, etc.

2. **Survenue dynamique des événements** — le moteur produit un journal d'événements ordonnés dans le temps (timecode). Le journal est en append : l'état de la galaxie à un instant T se reconstitue en rejouant les événements jusqu'à T.

3. **Exploration** — l'IHM Unity permet d'explorer la simulation à trois échelles : galaxie entière (densités, signalement d'événements), système solaire, planète individuelle, tout au long des timecodes.

L'architecture est conçue pour la cible 100 à 400 milliards de systèmes solaires (consensus actuel du nombre de systèmes dans notre galaxie), selon les paramètres choisis par l'utilisateur.

## Périmètre du prototype

- Générateur du dataset (éléments galactiques et événements)
- Modèle ML d'apprentissage de la suite des évènements
- Pipeline d'entraînement complet
- Chaîne complète scénario → entités → simulation par inférence ML → vues zoom galaxie pour Unity
- Deux types d'événements implémentés pour l'instant : **apparition de la vie** sur une planète, **fin de séquence principale** d'une étoile
- Côté Unity : consommation des API
- Implémentation prototypale de la vue galactique et des evénements (vues systemes solaires et planétaires hors prototype)

> **évolutions futures**
- ajout de types d'événements (impacts cométaires majeurs, supernovae, émergence de civilisations, contacts inter-civilisations, expansions) et de leurs interdépendances temporelles. Le format de stockage et l'architecture ML sont conçus pour absorber ces extensions sans refonte
- analyse a posteriori de runs simulés pour produire des solutions au paradoxe de Fermi (vide, contact, extinction)
- Scaling cluster et cloud

## Choix techniques

### Modèle événementiel

L'état de la galaxie se reconstitue par rejeu du journal des événements.
Côté IHM, la simulation Unity est conçue pour rejouer cette chronologie de façon dynamique, avec contrôle du timecode (lecture, pause, navigation), et 3 niveaux de zoom conçus comme des interfaces d'exploration. L'utilisateur ne voit pas une galaxie figée, il assiste à son évolution. On peut explorer les résultats avec un regard technique ML ou causalitées astrophysiques. 

### Apprentissage par processus ponctuel temporel (TPP)

Le moteur ML qui prédit les événements est un **Transformer Hawkes Process** (parfait pour prédire une chaîne d'événements interdépendants), conditionné sur les features physiques (étoiles, planetes) et leurs caractéristiques (masse, orbite, etc.). Pour chaque planète, le modèle apprend une fonction d'intensité par type d'événement selon l'historique des événements antérieurs (un contact ne peut survenir qu'après l'émergence d'une civilisation, qui ne peut survenir qu'après l'apparition de la vie).

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

> **Tout passe par API.** Aucune ligne de code à écrire pour piloter le simulateur, ni pour générer un dataset, ni pour entraîner un modèle, ni pour lancer une simulation. Les exemples ci-dessous utilisent `curl` pour leur concision et leur copiabilité, mais les mêmes appels sont disponibles dans l'interface graphique générée automatiquement sur http://localhost:8000/docs (Swagger UI). Au choix.

Prérequis : Docker, Docker Compose.

### 1. Lancer la stack

```bash
git clone <repo>
cd fc-simulator
docker compose up --build
```

Trois services démarrent :

- **API** : http://localhost:8000 (Swagger UI sur `/docs`)
- **pgAdmin** : http://localhost:5050 (`admin@drake.fr` / `admin`)
- **Postgres** : exposé sur le port 5432

### 2. Générer un mini-dataset d'entraînement

```bash
curl -X POST "http://localhost:8000/api/v1/training/datasets" \
     -H "Content-Type: application/json" \
     -d '{"nom": "ds_demo", "nb_systemes": 1000, "seed": 42}'
```

Quelques secondes. Le dataset est composé de deux fichiers parquet (planètes + événements) dans `data/training/`.

### 3. Entraîner un mini-modèle dessus

```bash
curl -X POST "http://localhost:8000/api/v1/training/models" \
     -H "Content-Type: application/json" \
     -d '{"nom": "m_demo", "dataset_nom": "ds_demo", "nb_epochs": 3}'
```

Quelques secondes également. Le `.pth` est sauvegardé dans `data/training/`. **Ce modèle est volontairement sous-entraîné** : il sert juste à valider le pipeline. Pour des résultats cohérents, voir l'encart « Séquence réaliste » plus bas.

### 4. Lancer un scénario en utilisant le modèle

```bash
curl -X POST "http://localhost:8000/api/v1/scenarios/full?model_nom=m_demo" \
     -H "Content-Type: application/json" \
     -d '{
       "nom": "demo",
       "nb_systemes": 100,
       "parametres": {
         "masse_stellaire_moyenne": 1.0,
         "indice_tellurique": 0.5,
         "planetes_par_systeme_moyen": 2.0,
         "duree_simulation_Ga": 10.0
       }
     }'
```

L'endpoint `full` enchaîne en une fois : création du scénario, génération des entités (substrat physique de la galaxie), et exécution de la simulation par inférence du modèle ML. La réponse contient le `simulation_id`.

### 5. Récupérer les vues zoom galaxie

Avec le `simulation_id` et le `scenario_id` renvoyés à l'étape précédente :

```bash
curl "http://localhost:8000/api/v1/scenarios/<scenario_id>/galaxy/density"
curl "http://localhost:8000/api/v1/simulations/<simulation_id>/galaxy/events"
```

C'est ce que consommera l'IHM Unity au niveau de zoom galaxie : une grille 3D creuse de densité des systèmes, et la liste des événements groupés par système (uniquement les systèmes où la vie est apparue) avec leur position 3D.

> **Séquence réaliste**
>
> Le pipeline ci-dessus passe en quelques minutes mais le modèle est trop peu entraîné pour produire des résultats fidèles aux lois consensuelles encodées par le générateur de vérité. Pour des résultats cohérents (proportions d'événements alignées avec celles du dataset), monter en charge :
> - Étape 2 : `"nb_systemes": 100000` au lieu de 1000.
> - Étape 3 : `"nb_epochs": 20` au lieu de 3.
> - Étape 4 : `"nb_systemes": 1000` ou plus.
>
> Compter environ 15 minutes sur CPU pour l'entraînement à 100k systèmes / 20 epochs, puis quelques dizaines de secondes pour une simulation à 1000 systèmes / 10 Ga.

L'ensemble des endpoints est documenté de façon interactive sur http://localhost:8000/docs. Chaque endpoint a son équivalent CLI symétrique (cf. section **API** ci-dessous).


## API

Absolument toutes les commandes pour piloter la simulation ont leur api. Le protoype est pensé pour une utilisation future entièrement via IHM Unity (même la création d'un nouveau dataset d'entrainement paramétré) ou l'entrainement lui-même.

Préfixe commun : `/api/v1/`. Documentation interactive complète sur `/docs`.

**Scénarios et simulations**

| Méthode | URL                                       | Description                                              |
|---------|-------------------------------------------|----------------------------------------------------------|
| POST    | `/scenarios`                              | Crée un scénario                                         |
| GET     | `/scenarios`                              | Liste les scénarios                                      |
| GET     | `/scenarios/{id}`                         | État d'un scénario + ses simulations                     |
| DELETE  | `/scenarios/{id}`                         | Supprime un scénario (cascade entités + simulations)     |
| POST    | `/scenarios/{id}/entites`                 | Génère les entités du scénario                           |
| POST    | `/scenarios/{id}/simulations`             | Crée et exécute une simulation (`model_nom` requis)      |
| POST    | `/scenarios/full?model_nom=<m>`           | Pipeline complet (endpoint de confort dev)               |

**Zoom galaxie**

| Méthode | URL                                       | Description                                              |
|---------|-------------------------------------------|----------------------------------------------------------|
| GET     | `/scenarios/{id}/galaxy/density`          | Grille 3D régulière creuse de densité des systèmes       |
| GET     | `/simulations/{id}/galaxy/events`         | Événements zoom galaxie + compteur incrémental           |

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

À chaque endpoint correspond une commande CLI symétrique : `docker compose exec api python -m ml.cli ...` pour la partie ML (datasets, models), et `docker compose exec api python -m app.simulation.cli ...` pour les scénarios, simulations et zoom galaxie.

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
    ├── app/
    │   ├── main.py        # endpoints FastAPI
    │   ├── scenario.py    # Active Record Scenario + requests
    │   ├── db/            # accès Postgres
    │   └── simulation/    # générateur d'entités + moteur ML + vues zoom galaxie + CLI
    └── ml/                # module ML
        ├── api.py         # endpoints /training/ (datasets + models)
        ├── cli.py         # CLI symétrique
        ├── service.py
        ├── dataset.py
        ├── generator/     # générateur de vérité scientifique (Bloc A)
        └── training/      # modèle TPP, entraînement, inférence (Bloc B)
```
