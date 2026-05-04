# simulation/moteur.py
from __future__ import annotations

import random

import psycopg


def executer_simulation(conn: psycopg.Connection, simulation_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'en_cours', progression = 0.0 WHERE id = %s",
            (simulation_id,),
        )

        cur.execute(
            "SELECT s.seed, g.id "
            "FROM simulation s "
            "JOIN galaxie g ON g.scenario_id = s.scenario_id "
            "WHERE s.id = %s",
            (simulation_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Simulation {simulation_id} introuvable ou entités non générées")
        seed, galaxie_id = row

        rng = random.Random(seed)

        cur.execute(
            "SELECT p.id "
            "FROM planete p "
            "JOIN systeme_solaire ss ON ss.id = p.systeme_id "
            "WHERE ss.galaxie_id = %s",
            (galaxie_id,),
        )
        planetes = [r[0] for r in cur.fetchall()]

        cur.execute("SELECT libelle, id FROM type_evenement")
        types = dict(cur.fetchall())

        nb_evenements = 0
        for planete_id in planetes:
            if rng.random() < 0.5:
                t_vie = rng.randint(1_000_000, 5_000_000)
                cur.execute(
                    "INSERT INTO evenement "
                    "(simulation_id, timecode, type_evenement_id, entite_id, entite_type, payload) "
                    "VALUES (%s, %s, %s, %s, 'planete', %s)",
                    (simulation_id, t_vie, types["emergence_vie"], planete_id, None),
                )
                nb_evenements += 1

                if rng.random() < 0.3:
                    t_civ = t_vie + rng.randint(100_000, 1_000_000)
                    cur.execute(
                        "INSERT INTO evenement "
                        "(simulation_id, timecode, type_evenement_id, entite_id, entite_type, payload) "
                        "VALUES (%s, %s, %s, %s, 'planete', %s)",
                        (simulation_id, t_civ, types["emergence_civilisation"], planete_id, None),
                    )
                    nb_evenements += 1

        cur.execute(
            "UPDATE simulation SET statut = 'terminee', progression = 1.0 WHERE id = %s",
            (simulation_id,),
        )

    return nb_evenements