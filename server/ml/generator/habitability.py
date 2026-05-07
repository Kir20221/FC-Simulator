"""
Zone habitable et facteurs de probabilité d'apparition de la vie.

Le produit multiplicatif des facteurs ci-dessous donne P(vie | conditions).
Chaque facteur est dans [0, 1], 1 signifiant « conditions favorables ».

Refonte paramétrique : le facteur d'âge a été supprimé. Tous les systèmes
naissent à T=0 dans la simulation, l'âge stellaire à l'apparition de la vie
est encadré directement par la fenêtre [AGE_MIN_GA, durée_vie_étoile] dans
life_model.py, sans facteur multiplicatif additionnel.

Référence zone habitable : Kopparapu et al. 2013/2014 — bornes
"moist greenhouse" (chaud) et "maximum greenhouse" (froid).
"""

import numpy as np

from .distributions import Etoile, Planete


# ============================================================================
# CONSTANTES DE CALIBRATION — ZONE HABITABLE
# ============================================================================

# Bornes Kopparapu 2013, exprimées en flux stellaire reçu (S/S_terre)
# pour une étoile type Soleil (T_eff = 5780 K).
S_EFF_SUN_CHAUD = 1.0140      # moist greenhouse à T=5780K
S_EFF_SUN_FROID = 0.3438      # maximum greenhouse à T=5780K

# Coefficients du polynôme Kopparapu (a, b, c, d) pour S_eff(T*).
# T* = T_eff - 5780. S_eff(T*) = S_eff_sun + a*T* + b*T*^2 + c*T*^3 + d*T*^4
COEFS_KOPPARAPU_CHAUD = (8.1774e-5, 1.7063e-9, -4.3241e-12, -6.6462e-16)
COEFS_KOPPARAPU_FROID = (5.8942e-5, 1.6558e-9, -3.0045e-12, -5.2983e-16)

# Largeur de la décroissance gaussienne hors HZ (en log10(distance)).
HZ_LARGEUR_LOG = 0.15

# Bornes du domaine de validité Kopparapu : 2600 K à 7200 K.
KOPPARAPU_T_MIN = 2600.0
KOPPARAPU_T_MAX = 7200.0


# ============================================================================
# CONSTANTES DE CALIBRATION — AUTRES FACTEURS
# ============================================================================

# Facteur de masse planétaire : optimum gaussien centré sur 1 M_terre.
MASSE_OPTIMUM_TERRE = 1.0
MASSE_LARGEUR_LOG = 0.5  # en log10(M)

# Facteur composition : valeur par type.
COMPOSITION_PROBA = {
    "tellurique": 1.0,
    "glacee": 0.05,
    "gazeuse": 0.0,
}

# Facteur étoile : pénalité par type spectral.
ETOILE_PROBA = {
    "O": 0.0,
    "B": 0.05,
    "A": 0.2,
    "F": 0.9,
    "G": 1.0,
    "K": 0.95,
    "M": 0.4,
}


# ============================================================================
# ZONE HABITABLE
# ============================================================================

def _flux_seuil(t_eff: float, coefs: tuple, s_eff_sun: float) -> float:
    """Polynôme Kopparapu : flux stellaire seuil à la température T_eff."""
    t = np.clip(t_eff, KOPPARAPU_T_MIN, KOPPARAPU_T_MAX) - 5780.0
    a, b, c, d = coefs
    return s_eff_sun + a * t + b * t**2 + c * t**3 + d * t**4


def bornes_zone_habitable_UA(etoile: Etoile) -> tuple[float, float]:
    """
    Retourne (d_inner, d_outer) en UA, bornes de la zone habitable.
    Calcul : d = sqrt(L / S_eff), avec L en luminosités solaires.
    """
    s_chaud = _flux_seuil(etoile.temperature_K, COEFS_KOPPARAPU_CHAUD, S_EFF_SUN_CHAUD)
    s_froid = _flux_seuil(etoile.temperature_K, COEFS_KOPPARAPU_FROID, S_EFF_SUN_FROID)
    d_inner = float(np.sqrt(etoile.luminosite_solaire / s_chaud))
    d_outer = float(np.sqrt(etoile.luminosite_solaire / s_froid))
    return d_inner, d_outer


def facteur_HZ(etoile: Etoile, planete: Planete) -> float:
    """
    Facteur P_HZ : 1 dans la HZ, décroissance gaussienne en log10(distance)
    de part et d'autre.
    """
    d_in, d_out = bornes_zone_habitable_UA(etoile)
    d = planete.distance_UA

    if d_in <= d <= d_out:
        return 1.0

    if d < d_in:
        ecart_log = np.log10(d_in) - np.log10(d)
    else:
        ecart_log = np.log10(d) - np.log10(d_out)

    return float(np.exp(-(ecart_log / HZ_LARGEUR_LOG) ** 2))


# ============================================================================
# AUTRES FACTEURS
# ============================================================================

def facteur_composition(planete: Planete) -> float:
    return COMPOSITION_PROBA.get(planete.composition, 0.0)


def facteur_masse(planete: Planete) -> float:
    """Gaussienne en log10(M) centrée sur la Terre."""
    ecart_log = np.log10(planete.masse_terre) - np.log10(MASSE_OPTIMUM_TERRE)
    return float(np.exp(-(ecart_log / MASSE_LARGEUR_LOG) ** 2))


def facteur_etoile(etoile: Etoile) -> float:
    return ETOILE_PROBA.get(etoile.type_spectral, 0.0)
