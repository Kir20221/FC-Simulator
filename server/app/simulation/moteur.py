"""Moteur de simulation — version minimale pour valider le journal d'événements.

À ce stade, le moteur ne calcule rien de scientifiquement significatif.
Il produit quelques événements bidons rattachés à des planètes du substrat,
afin de valider :
- l'écriture append-only dans la table `evenement`
- le respect des FK (simulation, type_evenement, substrat)
- la relecture ordonnée par timecode
"""
from __future__ import annotations

import random

import psycopg


def executer_simulation(conn: psycopg.Connection, simulation_id: int) -> int:
    """Exécute une simulation minimale et retourne le nombre d'événements écrits."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'en_cours', progression = 0.0 WHERE id = %s",
            (simulation_id,),
        )

        # Récupération du contexte : seed et galaxie associée via le scénario.
        cur.execute(
            "SELECT s.seed, g.id "
            "FROM simulation s "
            "JOIN galaxie g ON g.scenario_id = s.scenario_id "
            "WHERE s.id = %s",
            (simulation_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Simulation {simulation_id} introuvable ou substrat absent")
        seed, galaxie_id = row

        rng = random.Random(seed)

        # Récupération des planètes du substrat.
        cur.execute(
            "SELECT p.id "
            "FROM planete p "
            "JOIN systeme_solaire ss ON ss.id = p.systeme_id "
            "WHERE ss.galaxie_id = %s",
            (galaxie_id,),
        )
        planetes = [r[0] for r in cur.fetchall()]

        # Cache des ids de types d'événements.
        cur.execute("SELECT libelle, id FROM type_evenement")
        types = dict(cur.fetchall())

        nb_evenements = 0
        for planete_id in planetes:
            # Pour chaque planète, on tire potentiellement quelques événements.
            # Logique factice : suffisant pour valider la chaîne.
            if rng.random() < 0.5:
                t_vie = rng.randint(1_000_000, 5_000_000)
                cur.execute(
                    "INSERT INTO evenement "
                    "(simulation_id, timecode, type_evenement_id, substrat_id, substrat_type, payload) "
                    "VALUES (%s, %s, %s, %s, 'planete', %s)",
                    (simulation_id, t_vie, types["emergence_vie"], planete_id, None),
                )
                nb_evenements += 1

                if rng.random() < 0.3:
                    t_civ = t_vie + rng.randint(100_000, 1_000_000)
                    cur.execute(
                        "INSERT INTO evenement "
                        "(simulation_id, timecode, type_evenement_id, substrat_id, substrat_type, payload) "
                        "VALUES (%s, %s, %s, %s, 'planete', %s)",
                        (simulation_id, t_civ, types["emergence_civilisation"], planete_id, None),
                    )
                    nb_evenements += 1

        cur.execute(
            "UPDATE simulation SET statut = 'terminee', progression = 1.0 WHERE id = %s",
            (simulation_id,),
        )

    return nb_evenements
