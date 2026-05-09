# simulation/moteur.py
"""
Moteur de simulation : utilise un modèle TPP entraîné pour échantillonner
la chronologie des événements de chaque planète d'un scénario.

Phase 2 du pipeline scénario :
1. Charge le modèle entraîné dont le nom est porté par la simulation.
2. Itère sur les systèmes du scénario (en streaming via curseur serveur).
3. Pour chaque planète : construit le PlanetSample avec features
   normalisées via les FeatureStats du modèle, appelle sample_sequence
   (thinning d'Ogata) qui retourne une séquence d'événements typés/datés.
4. Persiste les événements en DB.

Règles de rattachement des événements (inchangées) :
- life_apparition         → entite_type='planete', entite_id=planete.id
- star_main_sequence_end  → entite_type='etoile',  entite_id=etoile.id
  (dédupliqué au niveau système : émis une seule fois par étoile, même
  si plusieurs planètes orbitent autour, puisque le générateur le produit
  pour chaque planète indépendamment.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import torch

from ml.training.data import PlanetSample
from ml.training.features import (
    transformer_categorielles,
    transformer_continues,
)
from ml.training.inference import sample_sequence
from ml.training.trainer import load_model


# Curseur serveur : streaming sans charger toute la galaxie en RAM.
CURSOR_NAME = "moteur_systemes"

# Constante figée du Bloc A : 1 unité de timecode = 100 000 ans.
# Dupliquée ici pour ne pas dépendre du module générateur (le moteur ne
# parle qu'au modèle ML désormais). Source : ml/generator/life_model.py.
TIMECODE_UNITE_ANS = 100_000


def executer_simulation(conn: psycopg.Connection, simulation_id: int) -> int:
    """
    Exécute la simulation par inférence TPP. Retourne le nombre total
    d'événements insérés.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'en_cours', progression = 0.0 WHERE id = %s",
            (simulation_id,),
        )

        # Récup seed + galaxie + paramètres astrophysiques + modèle.
        cur.execute(
            "SELECT s.seed, s.model_nom, g.id, sc.duree_simulation_Ga "
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
        seed, model_nom, galaxie_id, duree_ga = row

        # Chemin du .pth depuis la table model.
        cur.execute(
            "SELECT chemin_fichier FROM model WHERE nom = %s",
            (model_nom,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Modèle '{model_nom}' introuvable en DB")
        chemin_modele = Path(row[0])

        # Référentiel des types d'événements en DB → id par libellé.
        cur.execute("SELECT libelle, id FROM type_evenement")
        type_id_par_libelle: dict[str, int] = dict(cur.fetchall())

    # Chargement du modèle entraîné. Tout est dans le .pth : architecture,
    # poids, stats de normalisation, référentiel d'événements.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_stats, event_types_modele, _tpp_cfg = load_model(
        chemin_modele, device=device,
    )

    # Vérification de cohérence du référentiel des types : ceux du modèle
    # doivent tous être présents en DB. Sinon échec explicite.
    for libelle in event_types_modele:
        if libelle not in type_id_par_libelle:
            raise RuntimeError(
                f"Type d'événement '{libelle}' (présent dans le modèle '{model_nom}') "
                f"absent de la table type_evenement"
            )

    # Seed numpy pour la stochasticité du thinning.
    np.random.seed(seed % (2**32))

    # Horizon d'observation en timecode (mêmes formules que life_model).
    T_obs = int(duree_ga * 1e9 / TIMECODE_UNITE_ANS)

    # Traitement en streaming, système par système.
    nb_evenements = 0

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
                # Construction d'un DataFrame d'une ligne pour réutiliser
                # les transformateurs du module features.
                df_p = pd.DataFrame([{
                    "star_type":             star_type,
                    "star_temp_K":           float(star_temp),
                    "star_mass_solar":       float(star_mass),
                    "star_luminosity_solar": float(star_lum),
                    "star_lifetime_Ga":      float(star_lifetime),
                    "planet_distance_UA":    float(p_dist),
                    "planet_mass_terre":     float(p_mass),
                    "planet_radius_terre":   float(p_radius),
                    "planet_composition":    p_comp,
                }])
                cont = transformer_continues(df_p, feature_stats)[0]
                cat_arrays = transformer_categorielles(df_p, feature_stats)
                cat = {k: int(v[0]) for k, v in cat_arrays.items()}

                sample = PlanetSample(
                    cont=cont,
                    cat=cat,
                    times=np.zeros(0, dtype=np.int64),
                    types=np.zeros(0, dtype=np.int64),
                    T_obs=T_obs,
                )

                seq = sample_sequence(model, sample, device)

                for t, type_id_modele in zip(seq.times, seq.types):
                    libelle = event_types_modele[type_id_modele]
                    type_id_db = type_id_par_libelle[libelle]

                    if libelle == "life_apparition":
                        evenements_a_inserer.append((
                            simulation_id, int(t), type_id_db,
                            planete_id, "planete", None,
                        ))
                    elif libelle == "star_main_sequence_end":
                        if not evt_etoile_emis:
                            evenements_a_inserer.append((
                                simulation_id, int(t), type_id_db,
                                etoile_id, "etoile", None,
                            ))
                            evt_etoile_emis = True
                    else:
                        # Tout futur type d'événement : par défaut rattaché
                        # à la planète. Sera ajusté quand le type sera
                        # implémenté.
                        evenements_a_inserer.append((
                            simulation_id, int(t), type_id_db,
                            planete_id, "planete", None,
                        ))

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

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'terminee', progression = 1.0 "
            "WHERE id = %s",
            (simulation_id,),
        )

    return nb_evenements
