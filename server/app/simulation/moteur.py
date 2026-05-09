# simulation/moteur.py
"""
Moteur de simulation : wrapper du générateur de vérité (Bloc A).

Phase 2 du pipeline scénario : relit les entités depuis la DB, appelle
evaluer_planete() planète par planète avec le seed propre à la simulation,
persiste les événements dans la table evenement.

Règles de rattachement des événements :
- life_apparition         → entite_type='planete', entite_id=planete.id
- star_main_sequence_end  → entite_type='etoile',  entite_id=etoile.id
  (dédupliqué au niveau système : émis une seule fois par étoile, même
  si plusieurs planètes orbitent autour, puisque le générateur le produit
  pour chaque planète indépendamment.)

Le moteur itère par système (et non par planète), ce qui simplifie la
déduplication des événements stellaires et permet un traitement en
streaming pour les grandes galaxies.
"""

from __future__ import annotations

import numpy as np
import psycopg

from ml.generator.distributions import (
    Etoile,
    ParametresAstrophysiques,
    Planete,
)
from ml.generator.life_model import EVENT_TYPES, evaluer_planete


# Curseur serveur : streaming sans charger toute la galaxie en RAM.
CURSOR_NAME = "moteur_systemes"


def executer_simulation(conn: psycopg.Connection, simulation_id: int) -> int:
    """
    Exécute la simulation : produit les événements et les persiste.
    Retourne le nombre total d'événements insérés.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'en_cours', progression = 0.0 WHERE id = %s",
            (simulation_id,),
        )

        # Récup seed + galaxie + paramètres astrophysiques du scénario
        cur.execute(
            "SELECT s.seed, g.id, "
            "       sc.masse_stellaire_moyenne, sc.indice_tellurique, "
            "       sc.planetes_par_systeme_moyen, sc.duree_simulation_Ga "
            "FROM simulation s "
            "JOIN scenario sc ON sc.id = s.scenario_id "
            "JOIN galaxie  g  ON g.scenario_id = sc.id "
            "WHERE s.id = %s",
            (simulation_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(
                f"Simulation {simulation_id} introuvable ou entités non générées"
            )
        seed, galaxie_id, masse_stell, indice_tell, planetes_moy, duree_ga = row

        params = ParametresAstrophysiques(
            masse_stellaire_moyenne=float(masse_stell),
            indice_tellurique=float(indice_tell),
            planetes_par_systeme_moyen=float(planetes_moy),
            duree_simulation_Ga=float(duree_ga),
        )

        rng = np.random.default_rng(seed)

        # Référentiel des types d'événements en DB → id par libellé.
        cur.execute("SELECT libelle, id FROM type_evenement")
        type_id_par_libelle: dict[str, int] = dict(cur.fetchall())
        for libelle in EVENT_TYPES:
            if libelle not in type_id_par_libelle:
                raise RuntimeError(
                    f"Type d'événement '{libelle}' absent de la table type_evenement"
                )

    # Traitement en streaming, système par système.
    nb_evenements = 0
    nb_systemes_traites = 0

    # Curseur serveur pour les étoiles de la galaxie.
    with conn.cursor(name=CURSOR_NAME) as cur_etoiles:
        cur_etoiles.execute(
            "SELECT e.id, e.systeme_id, e.star_type, e.star_temp_K, "
            "       e.star_mass_solar, e.star_luminosity_solar, e.star_lifetime_Ga "
            "FROM etoile e "
            "JOIN systeme_solaire ss ON ss.id = e.systeme_id "
            "WHERE ss.galaxie_id = %s "
            "ORDER BY e.systeme_id",
            (galaxie_id,),
        )

        for row_etoile in cur_etoiles:
            (etoile_id, systeme_id, star_type, star_temp, star_mass,
             star_lum, star_lifetime) = row_etoile

            etoile_dc = Etoile(
                type_spectral=star_type,
                temperature_K=float(star_temp),
                masse_solaire=float(star_mass),
                luminosite_solaire=float(star_lum),
                duree_vie_Ga=float(star_lifetime),
            )

            # Lecture des planètes de cette étoile (curseur classique,
            # le nombre est petit : N planètes par système).
            with conn.cursor() as cur_planetes:
                cur_planetes.execute(
                    "SELECT id, planet_distance_UA, planet_mass_terre, "
                    "       planet_radius_terre, planet_composition "
                    "FROM planete WHERE etoile_id = %s",
                    (etoile_id,),
                )
                planetes_db = cur_planetes.fetchall()

            evt_etoile_emis = False
            evenements_a_inserer: list[tuple] = []

            for (planete_id, p_dist, p_mass, p_radius, p_comp) in planetes_db:
                planete_dc = Planete(
                    distance_UA=float(p_dist),
                    masse_terre=float(p_mass),
                    rayon_terre=float(p_radius),
                    composition=p_comp,
                )

                res = evaluer_planete(etoile_dc, planete_dc, params, rng)

                for evt in res.evenements:
                    libelle = evt.type_libelle
                    type_id = type_id_par_libelle[libelle]

                    if libelle == "life_apparition":
                        evenements_a_inserer.append((
                            simulation_id,
                            evt.timecode,
                            type_id,
                            planete_id,
                            "planete",
                            None,
                        ))
                    elif libelle == "star_main_sequence_end":
                        # Dédupliqué : une seule occurrence par étoile.
                        if not evt_etoile_emis:
                            evenements_a_inserer.append((
                                simulation_id,
                                evt.timecode,
                                type_id,
                                etoile_id,
                                "etoile",
                                None,
                            ))
                            evt_etoile_emis = True
                    else:
                        # Tout futur type d'événement : par défaut rattaché
                        # à la planète. Sera ajusté quand le type sera
                        # implémenté.
                        evenements_a_inserer.append((
                            simulation_id,
                            evt.timecode,
                            type_id,
                            planete_id,
                            "planete",
                            None,
                        ))

            # Insertion groupée des événements de ce système.
            if evenements_a_inserer:
                with conn.cursor() as cur_ins:
                    cur_ins.executemany(
                        "INSERT INTO evenement "
                        "(simulation_id, timecode, type_evenement_id, "
                        " entite_id, entite_type, payload) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        evenements_a_inserer,
                    )
                nb_evenements += len(evenements_a_inserer)

            nb_systemes_traites += 1

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'terminee', progression = 1.0 "
            "WHERE id = %s",
            (simulation_id,),
        )

    return nb_evenements
