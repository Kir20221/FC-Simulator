# simulation/service.py
"""
Couche service partagée pour les opérations scenario / simulation.

Logique pure (pas de FastAPI, pas d'argparse, pas de print). Réutilisée
par les endpoints HTTP (app/main.py) et par la CLI (app/simulation/cli.py).

Convention : la connexion est ouverte par l'appelant, le service
n'ouvre pas de connexion. Permet à l'appelant de composer plusieurs
opérations dans la même transaction si besoin.
"""

from __future__ import annotations

import random

import psycopg

from app.scenario import ScenarioRequest, SimulationRequest
from app.simulation.entites import generer_entites
from app.simulation.moteur import executer_simulation


# ============================================================================
# SCÉNARIOS
# ============================================================================

def creer_scenario(conn: psycopg.Connection, req: ScenarioRequest) -> dict:
    """Crée un scénario en DB. Retourne {scenario_id}."""
    scenario = req.to_scenario()
    scenario.create(conn)
    return {"scenario_id": scenario.id}


def lister_scenarios(conn: psycopg.Connection) -> dict:
    """Liste les scénarios, plus récent en premier."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, nom, date_creation, statut_entites, nb_systemes, "
            "       masse_stellaire_moyenne, indice_tellurique, "
            "       planetes_par_systeme_moyen, duree_simulation_Ga "
            "FROM scenario "
            "ORDER BY date_creation DESC, id DESC",
        )
        scenarios = [_serialiser_scenario_ligne(r) for r in cur.fetchall()]
    return {"scenarios": scenarios}


def lire_scenario(conn: psycopg.Connection, scenario_id: int) -> dict:
    """État détaillé d'un scénario + ses simulations."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, nom, date_creation, statut_entites, nb_systemes, "
            "       masse_stellaire_moyenne, indice_tellurique, "
            "       planetes_par_systeme_moyen, duree_simulation_Ga "
            "FROM scenario WHERE id = %s",
            (scenario_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"Scénario {scenario_id} inconnu")
        scenario = _serialiser_scenario_ligne(row)

        cur.execute(
            "SELECT s.id, s.statut, s.progression, s.model_nom, "
            "       (SELECT COUNT(*) FROM evenement e WHERE e.simulation_id = s.id) "
            "FROM simulation s WHERE s.scenario_id = %s ORDER BY s.id",
            (scenario_id,),
        )
        scenario["simulations"] = [
            {
                "id": r[0], "statut": r[1], "progression": r[2],
                "model_nom": r[3], "nb_evenements": r[4],
            }
            for r in cur.fetchall()
        ]
    return scenario


def supprimer_scenario(conn: psycopg.Connection, scenario_id: int) -> dict:
    """Supprime un scénario et tout ce qui en dépend (cascade DB)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM scenario WHERE id = %s", (scenario_id,))
        if cur.rowcount == 0:
            raise LookupError(f"Scénario {scenario_id} inconnu")
    return {"scenario_id": scenario_id, "supprime": True}


def _serialiser_scenario_ligne(row: tuple) -> dict:
    """Sérialise une ligne SELECT du scénario (9 colonnes attendues)."""
    return {
        "id":             row[0],
        "nom":            row[1],
        "date_creation":  row[2].isoformat() if row[2] is not None else None,
        "statut_entites": row[3],
        "nb_systemes":    row[4],
        "parametres": {
            "masse_stellaire_moyenne":    float(row[5]),
            "indice_tellurique":          float(row[6]),
            "planetes_par_systeme_moyen": float(row[7]),
            "duree_simulation_Ga":        float(row[8]),
        },
    }


# ============================================================================
# ENTITÉS
# ============================================================================

def generer_entites_scenario(
    conn: psycopg.Connection,
    scenario_id: int,
    seed: int | None = None,
) -> dict:
    """Génère les entités d'un scénario. Seed tiré au hasard si non fourni."""
    seed_effectif = seed if seed is not None else random.randint(0, 2**63 - 1)
    galaxie_id = generer_entites(conn, scenario_id, seed_effectif)
    return {"scenario_id": scenario_id, "galaxie_id": galaxie_id, "seed": seed_effectif}


# ============================================================================
# SIMULATIONS
# ============================================================================

def creer_simulation(
    conn: psycopg.Connection,
    scenario_id: int,
    req: SimulationRequest,
) -> dict:
    """Crée une simulation et l'exécute. Modèle ML obligatoire."""
    seed_effectif = req.seed if req.seed is not None else random.randint(0, 2**63 - 1)

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO simulation (scenario_id, seed, model_nom) "
            "VALUES (%s, %s, %s) RETURNING id",
            (scenario_id, seed_effectif, req.model_nom),
        )
        simulation_id = cur.fetchone()[0]

    nb_evenements = executer_simulation(conn, simulation_id)

    return {
        "scenario_id":   scenario_id,
        "simulation_id": simulation_id,
        "seed":          seed_effectif,
        "model_nom":     req.model_nom,
        "nb_evenements": nb_evenements,
    }


# ============================================================================
# PIPELINE COMPLET
# ============================================================================

def creer_scenario_full(
    conn: psycopg.Connection,
    req: ScenarioRequest,
    model_nom: str,
) -> dict:
    """Pipeline complet : crée scénario, génère entités, lance simulation
    avec le modèle ML fourni."""
    res_scenario = creer_scenario(conn, req)
    scenario_id = res_scenario["scenario_id"]
    res_entites = generer_entites_scenario(conn, scenario_id)
    res_simulation = creer_simulation(
        conn, scenario_id, SimulationRequest(model_nom=model_nom),
    )
    return {
        "scenario_id":   scenario_id,
        "galaxie_id":    res_entites["galaxie_id"],
        "simulation_id": res_simulation["simulation_id"],
        "model_nom":     model_nom,
        "nb_evenements": res_simulation["nb_evenements"],
    }
