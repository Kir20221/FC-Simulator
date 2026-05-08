"""
Métadonnées des modèles entraînés.

Style Active Record cohérent avec Dataset : les méthodes de persistance
sont sur la classe elle-même.

Le fichier .pth vit dans le volume Docker, seules les métadonnées sont
en DB.
"""

from datetime import datetime
from typing import Any, Optional

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel


class Model(BaseModel):
    nom: str
    date_creation: datetime
    dataset_nom: str
    chemin_fichier: str
    taille_octets: int
    duree_entrainement_s: float
    training_config: dict[str, Any]
    tpp_config: dict[str, Any]
    metrics_par_epoch: list[dict[str, Any]]
    final_train_loss: float
    final_val_loss: float

    # ----------------------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------------------

    def create(self, conn: Connection) -> None:
        """Insère le modèle en DB. Échoue si le nom existe déjà."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model (
                    nom, date_creation, dataset_nom, chemin_fichier,
                    taille_octets, duree_entrainement_s,
                    training_config, tpp_config, metrics_par_epoch,
                    final_train_loss, final_val_loss
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    self.nom,
                    self.date_creation,
                    self.dataset_nom,
                    self.chemin_fichier,
                    self.taille_octets,
                    self.duree_entrainement_s,
                    Jsonb(self.training_config),
                    Jsonb(self.tpp_config),
                    Jsonb(self.metrics_par_epoch),
                    self.final_train_loss,
                    self.final_val_loss,
                ),
            )

    @classmethod
    def get(cls, conn: Connection, nom: str) -> Optional["Model"]:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM model WHERE nom = %s", (nom,))
            row = cur.fetchone()
            if row is None:
                return None
            return cls(**row)

    @classmethod
    def list_all(cls, conn: Connection) -> list["Model"]:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM model ORDER BY date_creation DESC")
            return [cls(**row) for row in cur.fetchall()]

    @classmethod
    def delete(cls, conn: Connection, nom: str) -> bool:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM model WHERE nom = %s", (nom,))
            return cur.rowcount > 0
