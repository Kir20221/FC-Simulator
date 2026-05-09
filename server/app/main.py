# main.py
from __future__ import annotations
from ml.api import router as ml_router

from fastapi import FastAPI, HTTPException, Query

from app.db.connection import get_conn
from app.scenario import ScenarioRequest, SimulationRequest
from app.simulation import service
from app.simulation.galaxy_view import (
    calculer_densite,
    calculer_events_zoom_galaxie,
)

app = FastAPI(title="FC-Simulator — service simulation", version="0.0.1")
app.include_router(ml_router, prefix="/api/v1/training")


# ============================================================================
# Scénarios
# ============================================================================

@app.post("/api/v1/scenarios")
def create_scenario(req: ScenarioRequest):
    with get_conn() as conn:
        return service.creer_scenario(conn, req)


@app.get("/api/v1/scenarios")
def list_scenarios():
    with get_conn() as conn:
        return service.lister_scenarios(conn)


@app.get("/api/v1/scenarios/{scenario_id}")
def get_scenario(scenario_id: int):
    with get_conn() as conn:
        try:
            return service.lire_scenario(conn, scenario_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/v1/scenarios/{scenario_id}")
def delete_scenario(scenario_id: int):
    with get_conn() as conn:
        try:
            return service.supprimer_scenario(conn, scenario_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/v1/scenarios/{scenario_id}/entites")
def create_entites(scenario_id: int, seed: int | None = None):
    with get_conn() as conn:
        return service.generer_entites_scenario(conn, scenario_id, seed)


@app.post("/api/v1/scenarios/{scenario_id}/simulations")
def create_simulation(scenario_id: int, req: SimulationRequest):
    with get_conn() as conn:
        return service.creer_simulation(conn, scenario_id, req)


@app.post("/api/v1/scenarios/full")
def create_scenario_full(
    req: ScenarioRequest,
    model_nom: str = Query(..., description="Modèle ML à utiliser pour la simulation."),
):
    with get_conn() as conn:
        return service.creer_scenario_full(conn, req, model_nom)


# ============================================================================
# Zoom galaxie
# ============================================================================

@app.get("/api/v1/scenarios/{scenario_id}/galaxy/density")
def get_galaxy_density(
    scenario_id: int,
    nx: int = Query(default=50, ge=1, le=200),
    ny: int = Query(default=50, ge=1, le=200),
    nz: int = Query(default=10, ge=1, le=200),
):
    """Grille 3D régulière creuse des systèmes du scénario."""
    try:
        with get_conn() as conn:
            return calculer_densite(conn, scenario_id, nx, ny, nz)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/v1/simulations/{simulation_id}/galaxy/events")
def get_galaxy_events(
    simulation_id: int,
    n_taux_rafraichissement: int = Query(default=100, ge=1),
):
    """Événements groupés par système (uniquement ceux ayant au moins
    une life_apparition) + compteur incrémental événementiel."""
    try:
        with get_conn() as conn:
            return calculer_events_zoom_galaxie(
                conn, simulation_id, n_taux_rafraichissement,
            )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
