"""
Métadonnées des datasets d'entraînement.

Style Active Record cohérent avec le reste du projet (cf. modélisation v2) :
les méthodes de persistance sont sur le modèle lui-même, pas dans un
repository séparé.

Les fichiers parquet vivent sur disque (volume Docker), seules les
métadonnées sont en DB.
"""

from datetime import datetime
from typing import Optional

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from .generator.generate import DatasetStats


class Dataset(BaseModel):
    nom: str
    date_creation: datetime
    nb_systemes: int
    seed: int
    chemin_fichier: str
    taille_octets: int
    stats: DatasetStats

    # ----------------------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------------------

    def create(self, conn: Connection) -> None:
        """Insère le dataset en DB. Échoue si le nom existe déjà."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset (
                    nom, date_creation, nb_systemes, seed,
                    chemin_fichier, taille_octets, stats
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    self.nom,
                    self.date_creation,
                    self.nb_systemes,
                    self.seed,
                    self.chemin_fichier,
                    self.taille_octets,
                    Jsonb(self.stats.model_dump()),
                ),
            )

    @classmethod
    def get(cls, conn: Connection, nom: str) -> Optional["Dataset"]:
        """Charge un dataset par son nom, ou None s'il n'existe pas."""
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM dataset WHERE nom = %s", (nom,))
            row = cur.fetchone()
            if row is None:
                return None
            return cls(**row)

    @classmethod
    def list_all(cls, conn: Connection) -> list["Dataset"]:
        """Liste tous les datasets, du plus récent au plus ancien."""
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM dataset ORDER BY date_creation DESC")
            return [cls(**row) for row in cur.fetchall()]

    @classmethod
    def delete(cls, conn: Connection, nom: str) -> bool:
        """Supprime un dataset de la DB. Retourne True si supprimé."""
        with conn.cursor() as cur:
            cur.execute("DELETE FROM dataset WHERE nom = %s", (nom,))
            return cur.rowcount > 0
