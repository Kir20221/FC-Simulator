"""Application FastAPI — point d'entrée du service.

Endpoints fournis dans cette première itération :
- GET  /api/v1/health           → contrôle de vie (DB joignable)
- POST /api/v1/run-validation   → enchaîne scenario + substrat + simulation + événements
                                  pour valider la chaîne complète. Synchrone : pas de
                                  BackgroundTasks à ce stade.
- GET  /api/v1/scenarios/{id}   → retourne l'état d'un scénario, ses simulations
                                  et le nombre d'événements produits
- GET  /api/v1/simulations/{id}/evenements → relecture ordonnée du journal

Les endpoints "officiels" du contrat (création de scénario, polling de run, etc.)
seront ajoutés ultérieurement.
"""
from __future__ import annotations

import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db.connection import pool, get_conn
from app.simulation.moteur import executer_simulation
from app.simulation.substrat import generer_substrat


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pool.open()
    pool.wait()
    yield
    pool.close()


app = FastAPI(title="Drake — service simulation", version="0.0.1", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #


class ParametresDrake(BaseModel):
    r_star:  float = 1.5
    fp:      float = 0.5
    ne:      float = 0.2
    fl:      float = 0.1
    fi:      float = 0.1
    fc:      float = 0.1
    l_drake: float = 10_000.0


class DemandeValidation(BaseModel):
    nom:             str = "validation"
    nombre_systemes: int = Field(default=10, ge=1, le=10_000)
    drake:           ParametresDrake = ParametresDrake()
    seed:            int | None = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/v1/health")
def health():
    """Vérifie la connexion à la base."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
    return {"status": "ok"}


@app.post("/api/v1/run-validation")
def run_validation(demande: DemandeValidation):
    """Crée un scénario, génère son substrat, lance une simulation, écrit des événements.

    Synchrone à dessein : le but est de valider la mécanique, pas d'exercer la
    chaîne asynchrone (qui sera mise en place ensuite).
    """
    seed = demande.seed if demande.seed is not None else random.randint(0, 2**63 - 1)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scenario "
                "(nom, r_star, fp, ne, fl, fi, fc, l_drake, nombre_systemes) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    demande.nom,
                    demande.drake.r_star,
                    demande.drake.fp,
                    demande.drake.ne,
                    demande.drake.fl,
                    demande.drake.fi,
                    demande.drake.fc,
                    demande.drake.l_drake,
                    demande.nombre_systemes,
                ),
            )
            scenario_id = cur.fetchone()[0]

        galaxie_id = generer_substrat(conn, scenario_id, seed)

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO simulation (scenario_id, seed) VALUES (%s, %s) RETURNING id",
                (scenario_id, seed),
            )
            simulation_id = cur.fetchone()[0]

        nb_evenements = executer_simulation(conn, simulation_id)

    return {
        "scenario_id":   scenario_id,
        "galaxie_id":    galaxie_id,
        "simulation_id": simulation_id,
        "seed":          seed,
        "nb_evenements": nb_evenements,
    }


@app.get("/api/v1/scenarios/{scenario_id}")
def get_scenario(scenario_id: int):
    """État synthétique d'un scénario : statut substrat + simulations associées."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, nom, statut_substrat, nombre_systemes "
            "FROM scenario WHERE id = %s",
            (scenario_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Scénario inconnu")
        scenario = {
            "id":              row[0],
            "nom":             row[1],
            "statut_substrat": row[2],
            "nombre_systemes": row[3],
        }

        cur.execute(
            "SELECT s.id, s.statut, s.progression, "
            "       (SELECT COUNT(*) FROM evenement e WHERE e.simulation_id = s.id) "
            "FROM simulation s WHERE s.scenario_id = %s ORDER BY s.id",
            (scenario_id,),
        )
        scenario["simulations"] = [
            {"id": r[0], "statut": r[1], "progression": r[2], "nb_evenements": r[3]}
            for r in cur.fetchall()
        ]

    return scenario


@app.get("/api/v1/simulations/{simulation_id}/evenements")
def get_evenements(simulation_id: int, limite: int = 100):
    """Relecture du journal d'une simulation, ordonnée par timecode croissant."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM simulation WHERE id = %s", (simulation_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Simulation inconnue")

        cur.execute(
            "SELECT e.id, e.timecode, t.libelle, e.substrat_type, e.substrat_id, e.payload "
            "FROM evenement e "
            "JOIN type_evenement t ON t.id = e.type_evenement_id "
            "WHERE e.simulation_id = %s "
            "ORDER BY e.timecode ASC, e.id ASC "
            "LIMIT %s",
            (simulation_id, limite),
        )
        evenements = [
            {
                "id":            r[0],
                "timecode":      r[1],
                "type":          r[2],
                "substrat_type": r[3],
                "substrat_id":   r[4],
                "payload":       r[5],
            }
            for r in cur.fetchall()
        ]

    return {"simulation_id": simulation_id, "evenements": evenements}
