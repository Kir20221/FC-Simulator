# simulation/entites.py
from __future__ import annotations

import random

import psycopg


def generer_entites(conn: psycopg.Connection, scenario_id: int, seed: int) -> int:
    """Génère la galaxie et toutes ses entités (systèmes, étoiles, planètes).
    Retourne l'id de la galaxie créée."""
    rng = random.Random(seed)

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE scenario SET statut_entites = 'en_cours' WHERE id = %s",
            (scenario_id,),
        )

        cur.execute("SELECT nombre_systemes FROM scenario WHERE id = %s", (scenario_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Scénario {scenario_id} introuvable")
        nombre_systemes = row[0]

        cur.execute(
            "INSERT INTO galaxie (scenario_id) VALUES (%s) RETURNING id",
            (scenario_id,),
        )
        galaxie_id = cur.fetchone()[0]

        for _ in range(nombre_systemes):
            x = rng.uniform(-50_000.0, 50_000.0)
            y = rng.uniform(-50_000.0, 50_000.0)
            z = rng.uniform(-1_000.0, 1_000.0)

            cur.execute(
                "INSERT INTO systeme_solaire (galaxie_id, position_x, position_y, position_z) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (galaxie_id, x, y, z),
            )
            systeme_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO etoile (systeme_id) VALUES (%s) RETURNING id",
                (systeme_id,),
            )
            etoile_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO planete (systeme_id, etoile_id) VALUES (%s, %s)",
                (systeme_id, etoile_id),
            )

        cur.execute(
            "UPDATE scenario SET statut_entites = 'termine' WHERE id = %s",
            (scenario_id,),
        )

    return galaxie_id