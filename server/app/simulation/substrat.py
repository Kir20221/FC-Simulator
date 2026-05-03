"""Génération du substrat pour un scénario.

Pour ce premier run de validation, le substrat est volontairement minimal :
- 1 galaxie par scénario
- N systèmes solaires (paramètre du scénario)
- 1 étoile par système
- 1 planète par système, orbitant autour de cette étoile

La structure relationnelle est cependant celle de la cible : chaque entité
existe en table dédiée, prête à accueillir des propriétés physiques et
à supporter plusieurs étoiles/planètes par système dans les versions futures.
"""
from __future__ import annotations

import random

import psycopg


def generer_substrat(conn: psycopg.Connection, scenario_id: int, seed: int) -> int:
    """Génère le substrat d'un scénario et retourne l'id de la galaxie créée.

    Le scénario doit exister. Sa colonne `nombre_systemes` détermine le nombre
    de systèmes générés.
    """
    rng = random.Random(seed)

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE scenario SET statut_substrat = 'en_cours' WHERE id = %s",
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

        # Génération en lot des systèmes, étoiles, planètes.
        # Pour le prototype on procède simplement ; le passage à l'échelle
        # se fera via COPY ou batch inserts au moment du besoin réel.
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
            "UPDATE scenario SET statut_substrat = 'termine' WHERE id = %s",
            (scenario_id,),
        )

    return galaxie_id
