"""
Couche service pour les datasets d'entraînement.

Orchestration : génère le parquet (cœur métier) + persiste les
métadonnées en DB. Appelée indifféremment depuis ml/api.py (HTTP) ou
ml/cli.py (CLI).

L'appelant fournit la connexion DB pour conserver le contrôle de la
transaction, conformément au style décidé dans modélisation v2.
"""

from datetime import datetime, timezone
from typing import Optional

from psycopg import Connection

from .dataset import Dataset
from .generator.generate import (
    DatasetGenerationRequest,
    chemin_parquet,
    generer_dataset,
    supprimer_parquet,
)


class DatasetExisteDejaError(Exception):
    pass


class DatasetIntrouvableError(Exception):
    pass


def creer_dataset(conn: Connection, request: DatasetGenerationRequest) -> Dataset:
    """
    Génère le parquet, calcule les stats, persiste les métadonnées en DB.
    Échoue si un dataset du même nom existe déjà (en DB).
    """
    if Dataset.get(conn, request.nom) is not None:
        raise DatasetExisteDejaError(f"Le dataset '{request.nom}' existe déjà.")

    resultat = generer_dataset(request)

    dataset = Dataset(
        nom=resultat.nom,
        date_creation=datetime.now(timezone.utc),
        nb_systemes=request.nb_systemes,
        seed=request.seed,
        chemin_fichier=str(resultat.chemin),
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
    """Supprime le dataset en DB et le fichier parquet associé."""
    dataset = Dataset.get(conn, nom)
    if dataset is None:
        raise DatasetIntrouvableError(f"Le dataset '{nom}' est introuvable.")

    supprimer_parquet(chemin_parquet(nom))
    Dataset.delete(conn, nom)
    conn.commit()
