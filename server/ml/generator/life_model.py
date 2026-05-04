"""
Modèle d'apparition de la vie : produit multiplicatif des facteurs +
tirage uniforme du timecode dans la fenêtre des conditions propices.

Le timecode est exprimé en unités de TIMECODE_UNITE_ANS années depuis
la formation de l'étoile. La fenêtre propice est [AGE_MIN_GA, durée_vie_etoile]
ramenée en unités de timecode et bornée par AGE_MAX_GA.
"""

from dataclasses import dataclass
import numpy as np

from .distributions import Etoile, Planete
from .habitability import (
    facteur_HZ,
    facteur_composition,
    facteur_masse,
    facteur_etoile,
    facteur_age,
    AGE_MIN_GA,
)


# ============================================================================
# CONSTANTES DE CALIBRATION
# ============================================================================

# Unité de timecode en années. 1 unité = 100 000 ans.
TIMECODE_UNITE_ANS = 100_000

# Durée totale de la simulation, en Ga.
DUREE_TOTALE_GA = 20.0


# ============================================================================
# STRUCTURES
# ============================================================================

@dataclass
class ResultatVie:
    proba_vie: float
    vie_apparait: bool
    timecode_apparition: int | None  # None si vie_apparait == False
    # Détail des facteurs pour debug / inspection.
    f_HZ: float
    f_composition: float
    f_masse: float
    f_etoile: float
    f_age: float


# ============================================================================
# CALCUL
# ============================================================================

def proba_apparition_vie(etoile: Etoile, planete: Planete) -> tuple[float, dict]:
    """Produit multiplicatif des 5 facteurs. Retourne (proba, détail)."""
    f_hz = facteur_HZ(etoile, planete)
    f_comp = facteur_composition(planete)
    f_mass = facteur_masse(planete)
    f_star = facteur_etoile(etoile)
    f_age = facteur_age(etoile)

    proba = f_hz * f_comp * f_mass * f_star * f_age
    detail = {
        "f_HZ": f_hz,
        "f_composition": f_comp,
        "f_masse": f_mass,
        "f_etoile": f_star,
        "f_age": f_age,
    }
    return proba, detail


def _ga_vers_timecode(annees_ga: float) -> int:
    """Convertit une durée en Ga vers le nombre d'unités de timecode."""
    return int(annees_ga * 1e9 / TIMECODE_UNITE_ANS)


def tirer_timecode_apparition(etoile: Etoile, rng: np.random.Generator) -> int:
    """
    Tirage uniforme du timecode d'apparition dans la fenêtre :
    [AGE_MIN_GA, min(durée_vie_étoile, DUREE_TOTALE_GA)].

    Retour en unités de TIMECODE_UNITE_ANS.
    """
    t_min_ga = AGE_MIN_GA
    t_max_ga = min(etoile.duree_vie_Ga, DUREE_TOTALE_GA)

    # Sécurité : si la fenêtre est dégénérée, on retourne le bord inférieur.
    if t_max_ga <= t_min_ga:
        return _ga_vers_timecode(t_min_ga)

    instant_ga = rng.uniform(t_min_ga, t_max_ga)
    return _ga_vers_timecode(instant_ga)


def evaluer_planete(
    etoile: Etoile,
    planete: Planete,
    rng: np.random.Generator,
) -> ResultatVie:
    """
    Évalue une planète : tire P(vie), tire le tirage de Bernoulli,
    et si succès, tire le timecode.
    """
    proba, detail = proba_apparition_vie(etoile, planete)

    vie = bool(rng.random() < proba)
    timecode = tirer_timecode_apparition(etoile, rng) if vie else None

    return ResultatVie(
        proba_vie=proba,
        vie_apparait=vie,
        timecode_apparition=timecode,
        f_HZ=detail["f_HZ"],
        f_composition=detail["f_composition"],
        f_masse=detail["f_masse"],
        f_etoile=detail["f_etoile"],
        f_age=detail["f_age"],
    )