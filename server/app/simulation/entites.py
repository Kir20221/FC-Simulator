# simulation/entites.py
"""
Génération des entités d'un scénario.

Phase 1 du pipeline scénario : tire le substrat physique (galaxie, systèmes,
étoiles, planètes) à partir des 4 paramètres astrophysiques fixés du scénario
et persiste tout en DB.

Les événements ne sont PAS produits ici : ils relèvent de la phase 2
(moteur de simulation), qui relit les entités depuis la DB et appelle
evaluer_planete() avec le seed propre à la simulation.

Le code réutilise le tirage du Bloc A (ml.generator) afin que les entités
d'un scénario soient strictement de la même nature que celles d'un dataset.
"""

from __future__ import annotations

import math

import numpy as np
import psycopg

from ml.generator.distributions import ParametresAstrophysiques
from ml.generator.generate import tirer_systeme


# ============================================================================
# CONSTANTES — FORME DE LA GALAXIE
# ============================================================================
# Disque simple, valeurs purement esthétiques (n'influent pas sur la physique).
# Unités arbitraires : Unity rescale à l'affichage.

R_DISK = 15_000.0   # échelle radiale de l'exponentielle (unités arbitraires)
H_DISK = 500.0      # écart-type vertical de la gaussienne


# ============================================================================
# TIRAGE DE LA POSITION 3D D'UN SYSTÈME
# ============================================================================

def _tirer_position(rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Disque galactique simple :
    - r ~ Exponential(R_DISK)
    - theta ~ Uniform(0, 2*pi)
    - z ~ Normal(0, H_DISK)
    """
    r = float(rng.exponential(R_DISK))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    z = float(rng.normal(0.0, H_DISK))
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y, z


# ============================================================================
# GÉNÉRATION COMPLÈTE
# ============================================================================

def generer_entites(conn: psycopg.Connection, scenario_id: int, seed: int) -> int:
    """
    Génère la galaxie et toutes ses entités (systèmes, étoiles, planètes)
    pour un scénario. Retourne l'id de la galaxie créée.

    Les 4 paramètres astrophysiques sont lus depuis la table scenario et
    appliqués uniformément à tous les systèmes du scénario.
    """
    rng = np.random.default_rng(seed)

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE scenario SET statut_entites = 'en_cours' WHERE id = %s",
            (scenario_id,),
        )

        cur.execute(
            "SELECT nb_systemes, masse_stellaire_moyenne, indice_tellurique, "
            "       planetes_par_systeme_moyen, duree_simulation_Ga "
            "FROM scenario WHERE id = %s",
            (scenario_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Scénario {scenario_id} introuvable")

        nb_systemes, masse_stell, indice_tell, planetes_moy, duree_ga = row
        params = ParametresAstrophysiques(
            masse_stellaire_moyenne=float(masse_stell),
            indice_tellurique=float(indice_tell),
            planetes_par_systeme_moyen=float(planetes_moy),
            duree_simulation_Ga=float(duree_ga),
        )

        cur.execute(
            "INSERT INTO galaxie (scenario_id) VALUES (%s) RETURNING id",
            (scenario_id,),
        )
        galaxie_id = cur.fetchone()[0]

        for _ in range(nb_systemes):
            x, y, z = _tirer_position(rng)
            cur.execute(
                "INSERT INTO systeme_solaire "
                "(galaxie_id, position_x, position_y, position_z) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (galaxie_id, x, y, z),
            )
            systeme_id = cur.fetchone()[0]

            systeme = tirer_systeme(rng, params)
            etoile = systeme.etoile

            cur.execute(
                "INSERT INTO etoile "
                "(systeme_id, star_type, star_temp_K, star_mass_solar, "
                " star_luminosity_solar, star_lifetime_Ga) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    systeme_id,
                    etoile.type_spectral,
                    etoile.temperature_K,
                    etoile.masse_solaire,
                    etoile.luminosite_solaire,
                    etoile.duree_vie_Ga,
                ),
            )
            etoile_id = cur.fetchone()[0]

            for planete in systeme.planetes:
                cur.execute(
                    "INSERT INTO planete "
                    "(systeme_id, etoile_id, planet_distance_UA, "
                    " planet_mass_terre, planet_radius_terre, planet_composition) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        systeme_id,
                        etoile_id,
                        planete.distance_UA,
                        planete.masse_terre,
                        planete.rayon_terre,
                        planete.composition,
                    ),
                )

        cur.execute(
            "UPDATE scenario SET statut_entites = 'termine' WHERE id = %s",
            (scenario_id,),
        )

    return galaxie_id
