"""
Tirages des features étoile et planète pour le générateur de vérité.

Toutes les constantes de calibration sont regroupées en tête de fichier
pour faciliter les ajustements lors des tests fonctionnels.
"""

from dataclasses import dataclass
import numpy as np


# ============================================================================
# CONSTANTES DE CALIBRATION
# ============================================================================

# Types spectraux et fractions galactiques (séquence principale).
# Source : approximations standard, type Reid & Hawley.
TYPES_SPECTRAUX = ["M", "K", "G", "F", "A", "B"]
_FRACTIONS_BRUTES = np.array([0.76, 0.12, 0.076, 0.03, 0.006, 0.0013])
FRACTIONS_TYPES = (_FRACTIONS_BRUTES / _FRACTIONS_BRUTES.sum()).tolist()

# Propriétés physiques par type spectral (séquence principale).
# Pour chaque type : (T_eff_min, T_eff_max, M_min, M_max, L_min, L_max,
#                     duree_vie_min_Ga, duree_vie_max_Ga)
# Sources : Habets & Heintze 1981, tables stellaires standard.
PROPRIETES_TYPES = {
    "M": (2400, 3700, 0.08, 0.45, 0.0001, 0.08, 50.0, 1000.0),
    "K": (3700, 5200, 0.45, 0.80, 0.08, 0.60, 15.0, 50.0),
    "G": (5200, 6000, 0.80, 1.04, 0.60, 1.50, 8.0, 15.0),
    "F": (6000, 7500, 1.04, 1.40, 1.50, 5.00, 3.0, 8.0),
    "A": (7500, 10000, 1.40, 2.10, 5.00, 25.0, 0.5, 3.0),
    "B": (10000, 30000, 2.10, 16.0, 25.0, 30000.0, 0.01, 0.5),
}

# Âge max possible (borne univers + marge prospective pour le projet Drake).
AGE_MAX_GA = 20.0

# Distribution du nombre de planètes par système : Poisson tronquée.
PLANETES_LAMBDA = 1.6
PLANETES_MIN = 1
PLANETES_MAX = 8

# Distance orbitale : log-uniforme.
DISTANCE_MIN_UA = 0.05
DISTANCE_MAX_UA = 50.0

# Masse planétaire : log-uniforme, en masses terrestres.
MASSE_MIN_TERRE = 0.05
MASSE_MAX_TERRE = 4000.0

# Seuils de composition par masse (en masses terrestres).
# Approximation : tellurique / glacée / gazeuse.
SEUIL_TELLURIQUE_GLACEE = 2.0
SEUIL_GLACEE_GAZEUSE = 10.0


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class Etoile:
    type_spectral: str
    temperature_K: float
    masse_solaire: float
    luminosite_solaire: float
    age_Ga: float
    duree_vie_Ga: float


@dataclass
class Planete:
    distance_UA: float
    masse_terre: float
    rayon_terre: float
    composition: str  # "tellurique" | "glacee" | "gazeuse"


# ============================================================================
# TIRAGES
# ============================================================================

def tirer_type_spectral(rng: np.random.Generator) -> str:
    return rng.choice(TYPES_SPECTRAUX, p=FRACTIONS_TYPES)


def tirer_etoile(rng: np.random.Generator) -> Etoile:
    type_sp = tirer_type_spectral(rng)
    t_min, t_max, m_min, m_max, l_min, l_max, dv_min, dv_max = PROPRIETES_TYPES[type_sp]

    # Tirage log-uniforme pour la luminosité (étendue très large).
    luminosite = float(np.exp(rng.uniform(np.log(l_min), np.log(l_max))))
    # Tirage uniforme pour T et M (étendue plus modeste).
    temperature = float(rng.uniform(t_min, t_max))
    masse = float(rng.uniform(m_min, m_max))
    duree_vie = float(np.exp(rng.uniform(np.log(dv_min), np.log(dv_max))))

    # Âge : uniforme entre 0 et min(durée de vie, AGE_MAX_GA).
    age_max = min(duree_vie, AGE_MAX_GA)
    age = float(rng.uniform(0.0, age_max))

    return Etoile(
        type_spectral=type_sp,
        temperature_K=temperature,
        masse_solaire=masse,
        luminosite_solaire=luminosite,
        age_Ga=age,
        duree_vie_Ga=duree_vie,
    )


def tirer_nombre_planetes(rng: np.random.Generator) -> int:
    """Poisson tronquée [PLANETES_MIN, PLANETES_MAX]."""
    while True:
        n = rng.poisson(PLANETES_LAMBDA)
        if PLANETES_MIN <= n <= PLANETES_MAX:
            return int(n)


def _rayon_depuis_masse(masse_terre: float) -> float:
    """
    Relation masse-rayon par paliers, inspirée Chen & Kipping 2017.
    Approximation grossière, suffisante pour ordres de grandeur.
    """
    if masse_terre < SEUIL_TELLURIQUE_GLACEE:
        # Telluriques : R ∝ M^0.28
        return masse_terre ** 0.28
    elif masse_terre < SEUIL_GLACEE_GAZEUSE * 10:
        # Néptuniens : R ∝ M^0.59 (raccordé au régime tellurique).
        r_ref = SEUIL_TELLURIQUE_GLACEE ** 0.28
        return r_ref * (masse_terre / SEUIL_TELLURIQUE_GLACEE) ** 0.59
    else:
        # Joviens : rayon quasi-constant ~11 R_terre, légère croissance.
        r_ref = SEUIL_TELLURIQUE_GLACEE ** 0.28 * (100.0 / SEUIL_TELLURIQUE_GLACEE) ** 0.59
        return r_ref * (masse_terre / 100.0) ** 0.04


def _composition_depuis_masse(masse_terre: float) -> str:
    if masse_terre < SEUIL_TELLURIQUE_GLACEE:
        return "tellurique"
    elif masse_terre < SEUIL_GLACEE_GAZEUSE:
        return "glacee"
    else:
        return "gazeuse"


def tirer_planete(rng: np.random.Generator) -> Planete:
    distance = float(np.exp(rng.uniform(np.log(DISTANCE_MIN_UA), np.log(DISTANCE_MAX_UA))))
    masse = float(np.exp(rng.uniform(np.log(MASSE_MIN_TERRE), np.log(MASSE_MAX_TERRE))))
    rayon = _rayon_depuis_masse(masse)
    composition = _composition_depuis_masse(masse)

    return Planete(
        distance_UA=distance,
        masse_terre=masse,
        rayon_terre=rayon,
        composition=composition,
    )