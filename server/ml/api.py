"""
Endpoints FastAPI pour la gestion des datasets d'entraînement.

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


router = APIRouter(tags=["training"])


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
