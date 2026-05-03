# main.py
from __future__ import annotations
import random
from fastapi import FastAPI, HTTPException

from app.db.connection import get_conn
from app.scenario import Scenario, ScenarioRequest, SimulationRequest
from app.simulation.moteur import executer_simulation
from app.simulation.substrat import generer_substrat

app = FastAPI(title="FCSimulation", version="0.0.1")

# sert à vérifier que la connexion à la base est opérationnelle au démarrage du service
@app.get("/api/v1/health")
def health():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
    return {"status": "ok"}

@app.post("/api/v1/scenarios")
def create_scenario(req: ScenarioRequest):
    with get_conn() as conn:
        scenario = req.to_scenario()
        scenario.create(conn)
    return {"scenario_id": scenario.id}


@app.get("/api/v1/scenarios/{scenario_id}")
def get_scenario(scenario_id: int):
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


@app.post("/api/v1/scenarios/{scenario_id}/substrat")
def create_substrat(scenario_id: int, seed: int | None = None):
    seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    with get_conn() as conn:
        galaxie_id = generer_substrat(conn, scenario_id, seed)
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