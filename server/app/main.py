# main.py
from __future__ import annotations
from ml.api import router as ml_router

import random

from fastapi import FastAPI, HTTPException

from app.db.connection import get_conn
from app.scenario import ScenarioRequest, SimulationRequest
from app.simulation.entites import generer_entites
from app.simulation.moteur import executer_simulation

app = FastAPI(title="FC-Simulator — service simulation", version="0.0.1")
app.include_router(ml_router, prefix="/api/v1/training")

@app.post("/api/v1/scenarios")
def create_scenario(req: ScenarioRequest):
    with get_conn() as conn:
        scenario = req.to_scenario()
        scenario.create(conn)
    return {"scenario_id": scenario.id}


@app.post("/api/v1/scenarios/full")
def create_scenario_full(req: ScenarioRequest):

    result_scenario = create_scenario(req)
    scenario_id = result_scenario["scenario_id"]
    result_entites = create_entites(scenario_id)
    result_simulation = create_simulation(scenario_id, SimulationRequest())

    return {
        "scenario_id":   scenario_id,
        "galaxie_id":    result_entites["galaxie_id"],
        "simulation_id": result_simulation["simulation_id"],
        "nb_evenements": result_simulation["nb_evenements"],
    }


@app.get("/api/v1/scenarios/{scenario_id}")
def get_scenario(scenario_id: int):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, nom, statut_entites, nb_systemes "
            "FROM scenario WHERE id = %s",
            (scenario_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Scénario inconnu")
        scenario = {
            "id":             row[0],
            "nom":            row[1],
            "statut_entites": row[2],
            "nb_systemes":    row[3],
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


@app.post("/api/v1/scenarios/{scenario_id}/entites")
def create_entites(scenario_id: int, seed: int | None = None):
    seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    with get_conn() as conn:
        galaxie_id = generer_entites(conn, scenario_id, seed)
    return {"scenario_id": scenario_id, "galaxie_id": galaxie_id, "seed": seed}


@app.post("/api/v1/scenarios/{scenario_id}/simulations")
def create_simulation(scenario_id: int, req: SimulationRequest):
    seed = req.seed if req.seed is not None else random.randint(0, 2**63 - 1)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO simulation (scenario_id, seed) VALUES (%s, %s) RETURNING id",
                (scenario_id, seed),
            )
            simulation_id = cur.fetchone()[0]

        nb_evenements = executer_simulation(conn, simulation_id)

    return {
        "scenario_id":   scenario_id,
        "simulation_id": simulation_id,
        "seed":          seed,
        "nb_evenements": nb_evenements,
    }


@app.get("/api/v1/simulations/{simulation_id}/evenements")
def get_evenements(simulation_id: int, limite: int = 100):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM simulation WHERE id = %s", (simulation_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Simulation inconnue")

        cur.execute(
            "SELECT e.id, e.timecode, t.libelle, e.entite_type, e.entite_id, e.payload "
            "FROM evenement e "
            "JOIN type_evenement t ON t.id = e.type_evenement_id "
            "WHERE e.simulation_id = %s "
            "ORDER BY e.timecode ASC, e.id ASC "
            "LIMIT %s",
            (simulation_id, limite),
        )
        evenements = [
            {
                "id":          r[0],
                "timecode":    r[1],
                "type":        r[2],
                "entite_type": r[3],
                "entite_id":   r[4],
                "payload":     r[5],
            }
            for r in cur.fetchall()
        ]

    return {"simulation_id": simulation_id, "evenements": evenements}
