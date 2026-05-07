"""
Couche service pour les datasets d'entraînement.

Refonte multi-événements : un dataset = deux parquets (planets + events).
La suppression supprime les deux fichiers + la ligne DB.
"""

from datetime import datetime, timezone

from psycopg import Connection

from .dataset import Dataset
from .generator.generate import (
    DatasetGenerationRequest,
    chemins_parquet,
    generer_dataset,
    supprimer_parquets,
)


class DatasetExisteDejaError(Exception):
    pass


class DatasetIntrouvableError(Exception):
    pass


def creer_dataset(conn: Connection, request: DatasetGenerationRequest) -> Dataset:
    """
    Génère les parquets, calcule les stats, persiste les métadonnées en DB.
    """
    if Dataset.get(conn, request.nom) is not None:
        raise DatasetExisteDejaError(f"Le dataset '{request.nom}' existe déjà.")

    resultat = generer_dataset(request)

    dataset = Dataset(
        nom=resultat.nom,
        date_creation=datetime.now(timezone.utc),
        nb_systemes=request.nb_systemes,
        seed=request.seed,
        chemin_planets=str(resultat.chemin_planets),
        chemin_events=str(resultat.chemin_events),
        taille_octets=resultat.taille_octets,
        stats=resultat.stats,
    )
    dataset.create(conn)
    conn.commit()
    return dataset


def lister_datasets(conn: Connection) -> list[Dataset]:
    return Dataset.list_all(conn)


def lire_dataset(conn: Connection, nom: str) -> Dataset:
    dataset = Dataset.get(conn, nom)
    if dataset is None:
        raise DatasetIntrouvableError(f"Le dataset '{nom}' est introuvable.")
    return dataset


def supprimer_dataset(conn: Connection, nom: str) -> None:
    """Supprime le dataset en DB et les fichiers parquet associés."""
    dataset = Dataset.get(conn, nom)
    if dataset is None:
        raise DatasetIntrouvableError(f"Le dataset '{nom}' est introuvable.")

    chemin_p, chemin_e = chemins_parquet(nom)
    supprimer_parquets(chemin_p, chemin_e)
    Dataset.delete(conn, nom)
    conn.commit()
