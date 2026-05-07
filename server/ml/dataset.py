"""
Métadonnées des datasets d'entraînement.

Refonte multi-événements : un dataset est désormais composé de deux
fichiers parquet (planets + events). Les deux chemins sont persistés
en DB.

Style Active Record cohérent avec le reste du projet.
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
    chemin_planets: str
    chemin_events: str
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
                    chemin_planets, chemin_events, taille_octets, stats
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    self.nom,
                    self.date_creation,
                    self.nb_systemes,
                    self.seed,
                    self.chemin_planets,
                    self.chemin_events,
                    self.taille_octets,
                    Jsonb(self.stats.model_dump()),
                ),
            )

    @classmethod
    def get(cls, conn: Connection, nom: str) -> Optional["Dataset"]:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM dataset WHERE nom = %s", (nom,))
            row = cur.fetchone()
            if row is None:
                return None
            return cls(**row)

    @classmethod
    def list_all(cls, conn: Connection) -> list["Dataset"]:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM dataset ORDER BY date_creation DESC")
            return [cls(**row) for row in cur.fetchall()]

    @classmethod
    def delete(cls, conn: Connection, nom: str) -> bool:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM dataset WHERE nom = %s", (nom,))
            return cur.rowcount > 0
