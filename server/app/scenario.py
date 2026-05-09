# scenario.py
from __future__ import annotations

import psycopg
from pydantic import BaseModel, Field


class ParametresAstrophysiques(BaseModel):
    """Paramètres astrophysiques du scénario.

    Alignés sur ml/generator/distributions.py:ParametresAstrophysiques.
    Les variables de Drake (r_star, fp, ne, fl, fi, fc, l_drake) ne sont PAS
    des paramètres d'entrée : elles émergent de la simulation.
    """

    masse_stellaire_moyenne: float = Field(
        default=1.0,
        gt=0,
        description="Masse stellaire moyenne en M☉ (centre de la log-normale).",
    )
    indice_tellurique: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Indice tellurique [0, 1] (module les seuils de composition).",
    )
    planetes_par_systeme_moyen: float = Field(
        default=2.0,
        gt=0,
        description="Lambda de la Poisson tronquée [1, 8].",
    )
    duree_simulation_Ga: float = Field(
        default=10.0,
        gt=0,
        description="Durée totale de simulation en Ga.",
    )


class Scenario(BaseModel):
    id:          int | None = None
    nom:         str
    parametres:  ParametresAstrophysiques
    nb_systemes: int = Field(ge=1)

    def create(self, conn: psycopg.Connection) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scenario "
                "(nom, nb_systemes, masse_stellaire_moyenne, indice_tellurique, "
                " planetes_par_systeme_moyen, duree_simulation_Ga) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    self.nom,
                    self.nb_systemes,
                    self.parametres.masse_stellaire_moyenne,
                    self.parametres.indice_tellurique,
                    self.parametres.planetes_par_systeme_moyen,
                    self.parametres.duree_simulation_Ga,
                ),
            )
            self.id = cur.fetchone()[0]
            return self.id


class ScenarioRequest(BaseModel):
    nom:         str
    nb_systemes: int = Field(default=10, ge=1, le=10_000_000)
    parametres:  ParametresAstrophysiques = Field(default_factory=ParametresAstrophysiques)

    def to_scenario(self) -> Scenario:
        return Scenario(
            nom=self.nom,
            parametres=self.parametres,
            nb_systemes=self.nb_systemes,
        )


class SimulationRequest(BaseModel):
    """Paramètres d'exécution d'une simulation.

    Le modèle ML utilisé est obligatoire : il définit la dynamique
    d'occurrence des événements via inférence TPP. Le seed est optionnel,
    tiré au hasard si non fourni.
    """
    model_nom: str = Field(..., description="Nom du modèle entraîné à utiliser pour l'inférence.")
    seed:      int | None = None
