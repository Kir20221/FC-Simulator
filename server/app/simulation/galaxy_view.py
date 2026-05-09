# simulation/galaxy_view.py
"""
Couche service du zoom galaxie.

Deux fonctions principales, pures (pas de FastAPI, pas d'argparse), pour
être réutilisées indistinctement par les endpoints HTTP et la CLI :

- calculer_densite(scenario_id, nx, ny, nz)
    Grille 3D régulière creuse des systèmes du scénario. Bornes spatiales
    déduites des positions effectives. Format creux : seules les cellules
    non-vides sont retournées.

- calculer_events_zoom_galaxie(simulation_id, n_taux_rafraichissement)
    Événements groupés par système, restreints aux systèmes ayant au moins
    une life_apparition. Chaque système porte sa position 3D et tous ses
    événements (étoile + planètes). Plus un compteur incrémental
    événementiel : un checkpoint tous les n événements.
"""

from __future__ import annotations

import psycopg


# ============================================================================
# DENSITÉ
# ============================================================================

def calculer_densite(
    conn: psycopg.Connection,
    scenario_id: int,
    nx: int,
    ny: int,
    nz: int,
) -> dict:
    """
    Calcule la grille de densité 3D des systèmes du scénario.

    La grille est dimensionnée sur la boîte englobante des positions
    effectives. Format creux : seules les cellules non-vides sont
    retournées, ce qui évite d'envoyer des dizaines de milliers de zéros
    pour une galaxie en disque.
    """
    with conn.cursor() as cur:
        # Existence du scénario + galaxie associée.
        cur.execute(
            "SELECT g.id "
            "FROM galaxie g "
            "WHERE g.scenario_id = %s",
            (scenario_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(
                f"Aucune galaxie pour le scénario {scenario_id} "
                "(entités non générées ?)"
            )
        galaxie_id = row[0]

        # Bornes spatiales effectives.
        cur.execute(
            "SELECT MIN(position_x), MAX(position_x), "
            "       MIN(position_y), MAX(position_y), "
            "       MIN(position_z), MAX(position_z), "
            "       COUNT(*) "
            "FROM systeme_solaire WHERE galaxie_id = %s",
            (galaxie_id,),
        )
        x_min, x_max, y_min, y_max, z_min, z_max, total = cur.fetchone()

        if total == 0:
            return _payload_densite_vide(scenario_id, nx, ny, nz)

        # Comptage par cellule via WIDTH_BUCKET. Le bucket = nx + 1 contient
        # les valeurs égales à la borne max ; on le fusionne avec nx.
        cur.execute(
            "SELECT "
            "  LEAST(WIDTH_BUCKET(position_x, %s, %s, %s), %s) - 1 AS ix, "
            "  LEAST(WIDTH_BUCKET(position_y, %s, %s, %s), %s) - 1 AS iy, "
            "  LEAST(WIDTH_BUCKET(position_z, %s, %s, %s), %s) - 1 AS iz, "
            "  COUNT(*) AS n "
            "FROM systeme_solaire "
            "WHERE galaxie_id = %s "
            "GROUP BY ix, iy, iz",
            (
                x_min, x_max, nx, nx,
                y_min, y_max, ny, ny,
                z_min, z_max, nz, nz,
                galaxie_id,
            ),
        )
        cellules = [
            {"ix": int(r[0]), "iy": int(r[1]), "iz": int(r[2]), "count": int(r[3])}
            for r in cur.fetchall()
        ]

    return {
        "scenario_id":    scenario_id,
        "nb_systemes":    int(total),
        "grille": {
            "nx": nx, "ny": ny, "nz": nz,
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
            "z_min": float(z_min), "z_max": float(z_max),
        },
        "cellules":       cellules,
        "nb_cellules_non_vides": len(cellules),
    }


def _payload_densite_vide(scenario_id: int, nx: int, ny: int, nz: int) -> dict:
    return {
        "scenario_id":    scenario_id,
        "nb_systemes":    0,
        "grille": {
            "nx": nx, "ny": ny, "nz": nz,
            "x_min": 0.0, "x_max": 0.0,
            "y_min": 0.0, "y_max": 0.0,
            "z_min": 0.0, "z_max": 0.0,
        },
        "cellules":       [],
        "nb_cellules_non_vides": 0,
    }


# ============================================================================
# ÉVÉNEMENTS ZOOM GALAXIE
# ============================================================================

def calculer_events_zoom_galaxie(
    conn: psycopg.Connection,
    simulation_id: int,
    n_taux_rafraichissement: int,
) -> dict:
    """
    Construit la vue événements du zoom galaxie pour une simulation.

    Restriction : seuls les systèmes ayant au moins une life_apparition
    sont remontés. Pour ces systèmes, on remonte tous les événements
    (étoile + planètes), avec leur position 3D groupée au niveau système.

    Compteur incrémental événementiel : on parcourt l'ensemble des
    événements (toutes simulations confondues du run) ordonnés par
    timecode, on incrémente un dict de comptes par type, et on snapshote
    tous les n événements.
    """
    with conn.cursor() as cur:
        # Existence de la simulation.
        cur.execute("SELECT 1 FROM simulation WHERE id = %s", (simulation_id,))
        if cur.fetchone() is None:
            raise LookupError(f"Simulation {simulation_id} inconnue")

        # Compteur incrémental événementiel : parcours global ordonné.
        cur.execute(
            "SELECT t.libelle, e.timecode "
            "FROM evenement e "
            "JOIN type_evenement t ON t.id = e.type_evenement_id "
            "WHERE e.simulation_id = %s "
            "ORDER BY e.timecode ASC, e.id ASC",
            (simulation_id,),
        )
        compteur_courant: dict[str, int] = {}
        checkpoints: list[dict] = []
        nb_total_events = 0
        for libelle, timecode in cur:
            compteur_courant[libelle] = compteur_courant.get(libelle, 0) + 1
            nb_total_events += 1
            if nb_total_events % n_taux_rafraichissement == 0:
                checkpoints.append({
                    "timecode":    int(timecode),
                    "counts":      dict(compteur_courant),
                })

        # Toujours ajouter un checkpoint final (état total) s'il n'a pas
        # été émis par le pas de rafraîchissement.
        if nb_total_events > 0 and (
            not checkpoints
            or checkpoints[-1]["counts"] != compteur_courant
        ):
            cur.execute(
                "SELECT MAX(e.timecode) "
                "FROM evenement e "
                "WHERE e.simulation_id = %s",
                (simulation_id,),
            )
            timecode_final = cur.fetchone()[0]
            checkpoints.append({
                "timecode":    int(timecode_final),
                "counts":      dict(compteur_courant),
            })

        compteur_final = dict(compteur_courant)

        # Identification des systèmes ayant au moins une life_apparition.
        cur.execute(
            "SELECT DISTINCT p.systeme_id "
            "FROM evenement e "
            "JOIN type_evenement t ON t.id = e.type_evenement_id "
            "JOIN planete p ON p.id = e.entite_id "
            "WHERE e.simulation_id = %s "
            "  AND e.entite_type = 'planete' "
            "  AND t.libelle = 'life_apparition'",
            (simulation_id,),
        )
        systemes_remarquables = [int(r[0]) for r in cur.fetchall()]

        if not systemes_remarquables:
            return {
                "simulation_id":           simulation_id,
                "n_taux_rafraichissement": n_taux_rafraichissement,
                "nb_events_total":         nb_total_events,
                "compteur_final":          compteur_final,
                "compteur_checkpoints":    checkpoints,
                "systemes":                [],
                "nb_systemes_remarquables": 0,
            }

        # Position des systèmes remarquables.
        cur.execute(
            "SELECT id, position_x, position_y, position_z "
            "FROM systeme_solaire "
            "WHERE id = ANY(%s)",
            (systemes_remarquables,),
        )
        positions = {
            int(r[0]): {
                "position_x": float(r[1]),
                "position_y": float(r[2]),
                "position_z": float(r[3]),
            }
            for r in cur.fetchall()
        }

        # Tous les événements de ces systèmes (rattachés à une planète OU
        # à l'étoile du système). On joint via la table planete pour les
        # événements rattachés planète, et via la table etoile pour ceux
        # rattachés étoile, puis on UNION les deux flux.
        cur.execute(
            "WITH systemes_cibles AS (SELECT UNNEST(%s::bigint[]) AS systeme_id) "
            "SELECT * FROM ( "
            "  SELECT p.systeme_id, e.timecode, t.libelle, "
            "         e.entite_type, e.entite_id "
            "  FROM evenement e "
            "  JOIN type_evenement t ON t.id = e.type_evenement_id "
            "  JOIN planete p ON p.id = e.entite_id "
            "  JOIN systemes_cibles sc ON sc.systeme_id = p.systeme_id "
            "  WHERE e.simulation_id = %s AND e.entite_type = 'planete' "
            "  UNION ALL "
            "  SELECT et.systeme_id, e.timecode, t.libelle, "
            "         e.entite_type, e.entite_id "
            "  FROM evenement e "
            "  JOIN type_evenement t ON t.id = e.type_evenement_id "
            "  JOIN etoile et ON et.id = e.entite_id "
            "  JOIN systemes_cibles sc ON sc.systeme_id = et.systeme_id "
            "  WHERE e.simulation_id = %s AND e.entite_type = 'etoile' "
            ") AS evts "
            "ORDER BY systeme_id, timecode ASC",
            (systemes_remarquables, simulation_id, simulation_id),
        )

        evts_par_systeme: dict[int, list[dict]] = {
            sid: [] for sid in systemes_remarquables
        }
        for systeme_id, timecode, libelle, entite_type, entite_id in cur:
            evts_par_systeme[int(systeme_id)].append({
                "timecode":    int(timecode),
                "type":        libelle,
                "entite_type": entite_type,
                "entite_id":   int(entite_id),
            })

    systemes_payload = [
        {
            "system_id":   sid,
            **positions[sid],
            "evenements":  evts_par_systeme[sid],
            "nb_evenements": len(evts_par_systeme[sid]),
        }
        for sid in systemes_remarquables
        if sid in positions  # garde-fou
    ]

    return {
        "simulation_id":           simulation_id,
        "n_taux_rafraichissement": n_taux_rafraichissement,
        "nb_events_total":         nb_total_events,
        "compteur_final":          compteur_final,
        "compteur_checkpoints":    checkpoints,
        "systemes":                systemes_payload,
        "nb_systemes_remarquables": len(systemes_payload),
    }
