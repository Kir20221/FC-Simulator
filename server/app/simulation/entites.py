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
# CONSTANTES — MORPHOLOGIE DE LA VOIE LACTÉE
# ============================================================================
# Modèle multi-composantes calibré sur la littérature consensuelle :
#   - Bland-Hawthorn & Gerhard 2016 (ARA&A) pour la revue d'ensemble.
#   - Reid et al. 2019 pour la géométrie des bras (pitch ~12°, 4 bras nommés).
#   - Wegg, Gerhard & Portail 2015 pour la barre (longueur, orientation).
#   - Jurić et al. 2008, Bovy 2017 pour les échelles des disques mince/épais.
#   - Bullock & Johnston 2005 pour le profil du halo stellaire.
#
# Unités : parsecs (pc) dans le code. Unity rescale à l'affichage.
# Le repère est tel que le centre galactique est à l'origine, le plan
# galactique est z = 0.

# --- Fractions de masse stellaire par composante --------------------------
# Le tirage de position commence par choisir la composante d'appartenance
# du système. Valeurs ordres de grandeur, modulables.
FRAC_BRAS         = 0.55
FRAC_DISQUE_MINCE = 0.15  # composante diffuse du disque mince (hors bras)
FRAC_DISQUE_EPAIS = 0.10
FRAC_BARRE        = 0.08
FRAC_BULBE        = 0.10
FRAC_HALO         = 0.02
# Somme = 1.0 (vérifié dans _fractions_composantes()).

# --- Disque mince (échelles utilisées par les bras + composante diffuse) --
ECHELLE_R_DISQUE_MINCE = 2_500.0  # échelle radiale exponentielle (pc)
ECHELLE_Z_DISQUE_MINCE = 300.0    # échelle verticale exponentielle (pc)

# --- Disque épais ---------------------------------------------------------
ECHELLE_R_DISQUE_EPAIS = 3_500.0
ECHELLE_Z_DISQUE_EPAIS = 900.0

# --- Bras spiraux logarithmiques ------------------------------------------
# 4 bras nommés, pitch ~12°. Densités relatives non uniformes :
# Scutum-Centaurus et Perseus sont les bras stellaires dominants ;
# Sagittarius et Norma sont mineurs en population stellaire vieille.
NB_BRAS = 4
PITCH_ANGLE_RAD = math.radians(12.0)
# Phases de référence (angle au rayon de référence) des 4 bras, en radians.
# Valeurs approchées d'après Reid et al. 2019, espacement ~90°.
PHASES_BRAS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
# Densités relatives des 4 bras (somme normalisée en interne).
# Ordre : Norma, Sagittarius, Scutum-Centaurus, Perseus.
POIDS_BRAS = (0.18, 0.22, 0.32, 0.28)
# Dispersion gaussienne perpendiculaire au bras (épaisseur du bras vu de face).
LARGEUR_BRAS = 500.0  # pc
# Rayon de référence pour la spirale logarithmique (rayon où theta = phase).
R_REF_BRAS = 8_000.0
# Bornes radiales d'extension des bras.
R_MIN_BRAS = 3_000.0
R_MAX_BRAS = 18_000.0

# --- Barre centrale (ellipsoïde anisotrope dans le plan) ------------------
# Barre Galactique : ~5 kpc demi-grand axe, orientation ~27° par rapport
# à la ligne Soleil-centre galactique (Wegg, Gerhard & Portail 2015).
DEMI_GRAND_AXE_BARRE = 2_500.0  # pc (sigma le long de l'axe principal)
DEMI_PETIT_AXE_BARRE = 800.0    # pc (sigma perpendiculaire dans le plan)
HAUTEUR_BARRE        = 250.0    # pc (sigma vertical)
ANGLE_BARRE_RAD      = math.radians(27.0)

# --- Bulbe central (sphéroïdal quasi isotrope) ----------------------------
RAYON_BULBE = 700.0  # pc (sigma 3D, légèrement aplati en pratique)
APLATISSEMENT_BULBE = 0.6  # sigma_z / sigma_r

# --- Halo stellaire (profil en loi de puissance, sphéroïdal) --------------
# Densité ∝ r^(-alpha), alpha ~ 2.5 à 3 dans la littérature.
# On tire r selon une loi de puissance tronquée, puis direction isotrope.
HALO_R_MIN     = 1_000.0    # pc
HALO_R_MAX     = 40_000.0   # pc
HALO_ALPHA     = 2.7        # pente du profil de densité
HALO_APLATISSEMENT = 0.7    # facteur d'aplatissement vertical


# ============================================================================
# TIRAGE DE LA POSITION 3D D'UN SYSTÈME
# ============================================================================

def _fractions_composantes() -> np.ndarray:
    """Vecteur des 6 fractions, vérifié à 1.0, dans l'ordre :
    bras, disque mince, disque épais, barre, bulbe, halo."""
    fractions = np.array([
        FRAC_BRAS,
        FRAC_DISQUE_MINCE,
        FRAC_DISQUE_EPAIS,
        FRAC_BARRE,
        FRAC_BULBE,
        FRAC_HALO,
    ])
    total = fractions.sum()
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(
            f"Somme des fractions de composantes = {total}, attendu 1.0"
        )
    return fractions


_FRACTIONS = _fractions_composantes()
_POIDS_BRAS_NORM = np.array(POIDS_BRAS) / sum(POIDS_BRAS)


def _tirer_position_bras(rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Spirale logarithmique à 4 bras (pitch fixe).
    - Choix du bras selon POIDS_BRAS.
    - Tirage de r selon une exponentielle tronquée du disque mince.
    - theta = phase_bras + log(r / R_REF_BRAS) / tan(pitch)
    - Dispersion gaussienne perpendiculaire (largeur du bras).
    - z gaussien d'échelle du disque mince.
    """
    # Choix du bras
    idx_bras = int(rng.choice(NB_BRAS, p=_POIDS_BRAS_NORM))
    phase = PHASES_BRAS[idx_bras]

    # Rayon le long du bras : exponentielle du disque mince tronquée
    r = float(rng.exponential(ECHELLE_R_DISQUE_MINCE))
    # Rejet doux : on borne le rayon, plus simple et statistiquement équivalent
    # pour une exponentielle aux bornes choisies.
    r = max(R_MIN_BRAS, min(r, R_MAX_BRAS))

    # Angle du bras à ce rayon (spirale logarithmique)
    theta_bras = phase + math.log(r / R_REF_BRAS) / math.tan(PITCH_ANGLE_RAD)

    # Dispersion perpendiculaire au bras (dans le plan)
    # En coordonnées polaires locales : on perturbe theta autour du bras
    # avec une amplitude angulaire ~ LARGEUR_BRAS / r (longueur d'arc).
    dtheta = float(rng.normal(0.0, LARGEUR_BRAS / r))
    theta = theta_bras + dtheta

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = float(rng.normal(0.0, ECHELLE_Z_DISQUE_MINCE))
    return x, y, z


def _tirer_position_disque(
    rng: np.random.Generator,
    echelle_r: float,
    echelle_z: float,
) -> tuple[float, float, float]:
    """Disque exponentiel radial + exponentiel vertical symétrique."""
    r = float(rng.exponential(echelle_r))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    # Exponentielle vertical à deux côtés (signe tiré indépendamment).
    z = float(rng.exponential(echelle_z)) * (1.0 if rng.random() < 0.5 else -1.0)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y, z


def _tirer_position_barre(rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Ellipsoïde gaussien anisotrope, orienté à ANGLE_BARRE_RAD dans le plan.
    Tirage dans le repère propre de la barre (u, v, w) puis rotation vers (x, y).
    """
    u = float(rng.normal(0.0, DEMI_GRAND_AXE_BARRE))
    v = float(rng.normal(0.0, DEMI_PETIT_AXE_BARRE))
    w = float(rng.normal(0.0, HAUTEUR_BARRE))
    cos_a = math.cos(ANGLE_BARRE_RAD)
    sin_a = math.sin(ANGLE_BARRE_RAD)
    x = u * cos_a - v * sin_a
    y = u * sin_a + v * cos_a
    return x, y, w


def _tirer_position_bulbe(rng: np.random.Generator) -> tuple[float, float, float]:
    """Sphéroïde gaussien légèrement aplati au centre."""
    x = float(rng.normal(0.0, RAYON_BULBE))
    y = float(rng.normal(0.0, RAYON_BULBE))
    z = float(rng.normal(0.0, RAYON_BULBE * APLATISSEMENT_BULBE))
    return x, y, z


def _tirer_position_halo(rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Halo sphéroïdal : r tiré selon densité ∝ r^(-alpha) tronquée,
    direction isotrope puis aplatissement vertical.

    Pour densité de nombre ∝ r^(-alpha), la PDF du rayon (en 3D, surface 4πr²)
    est ∝ r^(2-alpha). On échantillonne par méthode de la transformée inverse.
    """
    p = 2.0 - HALO_ALPHA  # exposant de la PDF de r
    u = rng.random()
    if abs(p + 1.0) < 1e-6:
        # Cas dégénéré (PDF ∝ 1/r), équivalent log-uniforme
        r = HALO_R_MIN * (HALO_R_MAX / HALO_R_MIN) ** u
    else:
        r_min_p = HALO_R_MIN ** (p + 1.0)
        r_max_p = HALO_R_MAX ** (p + 1.0)
        r = (u * (r_max_p - r_min_p) + r_min_p) ** (1.0 / (p + 1.0))
    # Direction isotrope
    cos_phi = float(rng.uniform(-1.0, 1.0))
    sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    x = r * sin_phi * math.cos(theta)
    y = r * sin_phi * math.sin(theta)
    z = r * cos_phi * HALO_APLATISSEMENT
    return x, y, z


def _tirer_position(rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Tirage de la position 3D d'un système selon le modèle multi-composantes
    de la Voie lactée. La composante d'appartenance est tirée selon les
    fractions de masse stellaire, puis la position est tirée dans cette
    composante.
    """
    composante = int(rng.choice(6, p=_FRACTIONS))
    if composante == 0:
        return _tirer_position_bras(rng)
    elif composante == 1:
        return _tirer_position_disque(
            rng, ECHELLE_R_DISQUE_MINCE, ECHELLE_Z_DISQUE_MINCE
        )
    elif composante == 2:
        return _tirer_position_disque(
            rng, ECHELLE_R_DISQUE_EPAIS, ECHELLE_Z_DISQUE_EPAIS
        )
    elif composante == 3:
        return _tirer_position_barre(rng)
    elif composante == 4:
        return _tirer_position_bulbe(rng)
    else:
        return _tirer_position_halo(rng)


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
