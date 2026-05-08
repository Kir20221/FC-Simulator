"""
Endpoints FastAPI pour la gestion des datasets et des modèles entraînés.

Monté sous /api/v1/training par app/main.py.

Symétrique de ml/cli.py : à chaque endpoint correspond une commande CLI.
"""

from fastapi import APIRouter, HTTPException, status

from app.db.connection import get_conn
from .dataset import Dataset
from .generator.generate import DatasetGenerationRequest
from .service import (
    DatasetExisteDejaError,
    DatasetIntrouvableError,
    creer_dataset,
    lire_dataset,
    lister_datasets,
    supprimer_dataset,
)
from .training.model_record import Model
from .training.service_models import (
    ModelExisteDejaError,
    ModelIntrouvableError,
    ModelTrainingRequest,
    creer_modele,
    lire_modele,
    lister_modeles,
    supprimer_modele,
)


router = APIRouter(tags=["training"])


# ============================================================================
# DATASETS
# ============================================================================

@router.post("/datasets", response_model=Dataset, status_code=status.HTTP_201_CREATED)
def post_dataset(request: DatasetGenerationRequest) -> Dataset:
    """
    Génère un dataset d'entraînement (synchrone pour l'instant).

    Le fichier parquet est écrit dans /srv/data/training/, ses métadonnées
    sont persistées en DB.
    """
    try:
        with get_conn() as conn:
            return creer_dataset(conn, request)
    except DatasetExisteDejaError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/datasets", response_model=list[Dataset])
def get_datasets() -> list[Dataset]:
    with get_conn() as conn:
        return lister_datasets(conn)


@router.get("/datasets/{nom}", response_model=Dataset)
def get_dataset(nom: str) -> Dataset:
    try:
        with get_conn() as conn:
            return lire_dataset(conn, nom)
    except DatasetIntrouvableError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/datasets/{nom}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(nom: str) -> None:
    try:
        with get_conn() as conn:
            supprimer_dataset(conn, nom)
    except DatasetIntrouvableError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ============================================================================
# MODELS
# ============================================================================

@router.post("/models", response_model=Model, status_code=status.HTTP_201_CREATED)
def post_model(request: ModelTrainingRequest) -> Model:
    """
    Lance un entraînement (synchrone pour l'instant).

    Le fichier .pth est écrit dans /srv/data/training/, ses métadonnées
    et les résultats d'entraînement sont persistés en DB.
    """
    try:
        with get_conn() as conn:
            return creer_modele(conn, request)
    except ModelExisteDejaError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatasetIntrouvableError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/models", response_model=list[Model])
def get_models() -> list[Model]:
    with get_conn() as conn:
        return lister_modeles(conn)


@router.get("/models/{nom}", response_model=Model)
def get_model(nom: str) -> Model:
    try:
        with get_conn() as conn:
            return lire_modele(conn, nom)
    except ModelIntrouvableError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/models/{nom}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(nom: str) -> None:
    try:
        with get_conn() as conn:
            supprimer_modele(conn, nom)
    except ModelIntrouvableError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
