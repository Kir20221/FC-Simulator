# scenario.py
from __future__ import annotations

import psycopg
from pydantic import BaseModel, Field


class ParametresDrake(BaseModel):
    r_star:  float = Field(default=1.5,      ge=0)
    fp:      float = Field(default=0.5,      ge=0, le=1)
    ne:      float = Field(default=0.2,      ge=0)
    fl:      float = Field(default=0.1,      ge=0, le=1)
    fi:      float = Field(default=0.1,      ge=0, le=1)
    fc:      float = Field(default=0.1,      ge=0, le=1)
    l_drake: float = Field(default=10_000.0, ge=0)


class Scenario(BaseModel):
    id:              int | None = None
    nom:             str
    parametres:      ParametresDrake
    nombre_systemes: int = Field(ge=1)

    def create(self, conn: psycopg.Connection) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scenario "
                "(nom, r_star, fp, ne, fl, fi, fc, l_drake, nombre_systemes) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    self.nom,
                    self.parametres.r_star,
                    self.parametres.fp,
                    self.parametres.ne,
                    self.parametres.fl,
                    self.parametres.fi,
                    self.parametres.fc,
                    self.parametres.l_drake,
                    self.nombre_systemes,
                ),
            )
            self.id = cur.fetchone()[0]
            return self.id


class ScenarioRequest(BaseModel):
    nom:             str
    nombre_systemes: int = Field(default=10, ge=1, le=10_000)
    parametres:      ParametresDrake = Field(default_factory=ParametresDrake)

    def to_scenario(self) -> Scenario:
        return Scenario(
            nom=self.nom,
            parametres=self.parametres,
            nombre_systemes=self.nombre_systemes,
        )


class SimulationRequest(BaseModel):
    seed: int | None = None