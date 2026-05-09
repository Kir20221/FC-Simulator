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

Règles de rattachement des événements :
- life_apparition         → entite_type='planete', entite_id=planete.id
- star_main_sequence_end  → entite_type='etoile',  entite_id=etoile.id
  (dédupliqué au niveau système : émis une seule fois par étoile, même
  si plusieurs planètes orbitent autour, puisque le générateur le produit
  pour chaque planète indépendamment.)

Cette version est INSTRUMENTÉE : elle imprime des mesures de temps à
chaque étape pour diagnostiquer les performances. Logs visibles via
`docker compose logs -f api`.
"""

from __future__ import annotations

import time
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
TIMECODE_UNITE_ANS = 100_000

# Fréquence des logs de progression (tous les N systèmes traités).
LOG_EVERY_N_SYSTEMES = 10


def _log(msg: str) -> None:
    """Print avec flush forcé pour que les logs apparaissent en temps réel
    dans docker compose logs."""
    print(f"[moteur] {msg}", flush=True)


def executer_simulation(conn: psycopg.Connection, simulation_id: int) -> int:
    """
    Exécute la simulation par inférence TPP. Retourne le nombre total
    d'événements insérés.
    """
    t_start = time.time()
    _log(f"=== Simulation {simulation_id} : démarrage ===")

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
        _log(f"model_nom={model_nom}, galaxie_id={galaxie_id}, "
             f"duree_simulation_Ga={duree_ga}, seed={seed}")

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

    # Chargement du modèle entraîné.
    t_load_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"chargement du modèle '{model_nom}' depuis {chemin_modele} "
         f"sur device={device} ...")
    model, feature_stats, event_types_modele, _tpp_cfg = load_model(
        chemin_modele, device=device,
    )
    t_load = time.time() - t_load_start
    _log(f"modèle chargé en {t_load:.2f}s ; "
         f"event_types={event_types_modele}")

    # Vérification cohérence du référentiel.
    for libelle in event_types_modele:
        if libelle not in type_id_par_libelle:
            raise RuntimeError(
                f"Type d'événement '{libelle}' (présent dans le modèle '{model_nom}') "
                f"absent de la table type_evenement"
            )

    # Seed numpy pour la stochasticité du thinning.
    np.random.seed(seed % (2**32))

    # Horizon d'observation en timecode.
    T_obs = int(duree_ga * 1e9 / TIMECODE_UNITE_ANS)
    _log(f"T_obs={T_obs} timecodes")

    # Compteurs globaux pour le diagnostic.
    nb_evenements = 0
    nb_systemes_traites = 0
    nb_planetes_traitees = 0
    t_inference_total = 0.0
    t_db_total = 0.0
    nb_events_par_libelle: dict[str, int] = {l: 0 for l in event_types_modele}

    t_loop_start = time.time()

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

            t_db = time.time()
            with conn.cursor() as cur_planetes:
                cur_planetes.execute(
                    "SELECT id, planet_distance_UA, planet_mass_terre, "
                    "       planet_radius_terre, planet_composition "
                    "FROM planete WHERE etoile_id = %s",
                    (etoile_id,),
                )
                planetes_db = cur_planetes.fetchall()
            t_db_total += time.time() - t_db

            evt_etoile_emis = False
            evenements_a_inserer: list[tuple] = []

            for (planete_id, p_dist, p_mass, p_radius, p_comp) in planetes_db:
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

                t_inf = time.time()
                seq = sample_sequence(model, sample, device)
                t_inference_total += time.time() - t_inf

                nb_planetes_traitees += 1

                for t, type_id_modele in zip(seq.times, seq.types):
                    libelle = event_types_modele[type_id_modele]
                    nb_events_par_libelle[libelle] = nb_events_par_libelle.get(libelle, 0) + 1
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
                        evenements_a_inserer.append((
                            simulation_id, int(t), type_id_db,
                            planete_id, "planete", None,
                        ))

            if evenements_a_inserer:
                t_db = time.time()
                with conn.cursor() as cur_ins:
                    cur_ins.executemany(
                        "INSERT INTO evenement "
                        "(simulation_id, timecode, type_evenement_id, "
                        " entite_id, entite_type, payload) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        evenements_a_inserer,
                    )
                t_db_total += time.time() - t_db
                nb_evenements += len(evenements_a_inserer)

            nb_systemes_traites += 1

            if nb_systemes_traites % LOG_EVERY_N_SYSTEMES == 0:
                t_ecoule = time.time() - t_loop_start
                t_par_systeme = t_ecoule / nb_systemes_traites
                t_par_planete = (
                    t_inference_total / nb_planetes_traitees
                    if nb_planetes_traitees > 0 else 0.0
                )
                _log(
                    f"... {nb_systemes_traites} systèmes ({nb_planetes_traitees} planètes), "
                    f"{nb_evenements} événements en {t_ecoule:.1f}s "
                    f"({t_par_systeme*1000:.0f} ms/système, "
                    f"{t_par_planete*1000:.1f} ms/planète d'inférence)"
                )

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE simulation SET statut = 'terminee', progression = 1.0 "
            "WHERE id = %s",
            (simulation_id,),
        )

    t_total = time.time() - t_start
    t_loop = time.time() - t_loop_start
    _log(
        f"=== Simulation {simulation_id} : terminée en {t_total:.1f}s "
        f"(load={t_load:.2f}s, boucle={t_loop:.1f}s, "
        f"inférence={t_inference_total:.1f}s, db_intra_boucle={t_db_total:.1f}s) ==="
    )
    _log(
        f"Totaux : {nb_systemes_traites} systèmes, {nb_planetes_traitees} planètes, "
        f"{nb_evenements} événements"
    )
    _log(f"Répartition par type : {nb_events_par_libelle}")

    return nb_evenements
