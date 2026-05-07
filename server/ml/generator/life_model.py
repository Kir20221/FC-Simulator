"""
Modèle d'événements par planète.

Refonte multi-événements pour le Bloc B (TPP) :
- Au lieu d'une étiquette unique (vie : oui/non + timecode), on produit
  une liste d'événements pour chaque planète.
- Deux types d'événements pour le prototype :
    * "life_apparition" : modèle multiplicatif inchangé.
    * "star_main_sequence_end" : déterministe, à t = duree_vie_etoile,
      si cette durée tombe dans la fenêtre d'observation [0, T_sim].
- Chaque planète peut produire 0, 1 ou 2 événements, dans l'ordre temporel.

Les types d'événements et leur logique de tirage sont définis ici. Le
Bloc B (modèle ML) restera agnostique au nombre et au libellé des types :
il consommera simplement l'index de type fourni par EVENT_TYPES.
"""

from dataclasses import dataclass
import numpy as np

from .distributions import Etoile, Planete, ParametresAstrophysiques
from .habitability import (
    facteur_HZ,
    facteur_composition,
    facteur_masse,
    facteur_etoile,
)


# ============================================================================
# CONSTANTES DE CALIBRATION
# ============================================================================

# Unité de timecode en années. 1 unité = 100 000 ans.
TIMECODE_UNITE_ANS = 100_000

# Âge minimum requis (en âge stellaire) pour que la vie ait pu apparaître.
AGE_MIN_GA = 0.5


# ============================================================================
# RÉFÉRENTIEL DES TYPES D'ÉVÉNEMENTS
# ============================================================================
# Ordre stable. L'index dans cette liste sert d'identifiant entier dans
# le dataset (colonne event_type_id). Pour ajouter un nouveau type, on
# append à la fin et on implémente sa logique de tirage dans une fonction
# dédiée appelée depuis evaluer_planete().

EVENT_TYPES = [
    "life_apparition",
    "star_main_sequence_end",
]


def event_type_id(libelle: str) -> int:
    return EVENT_TYPES.index(libelle)


# ============================================================================
# STRUCTURE D'UN ÉVÉNEMENT
# ============================================================================

@dataclass
class Evenement:
    type_id: int
    timecode: int  # en unités de TIMECODE_UNITE_ANS

    @property
    def type_libelle(self) -> str:
        return EVENT_TYPES[self.type_id]


@dataclass
class ResultatPlanete:
    """Résultat de l'évaluation d'une planète : la liste de ses événements
    et les facteurs intermédiaires (pour debug / inspection)."""
    evenements: list[Evenement]
    proba_vie: float
    f_HZ: float
    f_composition: float
    f_masse: float
    f_etoile: float
    T_obs: int  # horizon d'observation en timecode (= T_sim convertie)


# ============================================================================
# CONVERSIONS
# ============================================================================

def _ga_vers_timecode(annees_ga: float) -> int:
    return int(annees_ga * 1e9 / TIMECODE_UNITE_ANS)


# ============================================================================
# CALCUL DES FACTEURS DE VIE
# ============================================================================

def proba_apparition_vie(etoile: Etoile, planete: Planete) -> tuple[float, dict]:
    """Produit multiplicatif des 4 facteurs. Retourne (proba, détail)."""
    f_hz = facteur_HZ(etoile, planete)
    f_comp = facteur_composition(planete)
    f_mass = facteur_masse(planete)
    f_star = facteur_etoile(etoile)

    proba = f_hz * f_comp * f_mass * f_star
    detail = {
        "f_HZ": f_hz,
        "f_composition": f_comp,
        "f_masse": f_mass,
        "f_etoile": f_star,
    }
    return proba, detail


# ============================================================================
# TIRAGE DES ÉVÉNEMENTS PAR TYPE
# ============================================================================

def _tirer_event_vie(
    etoile: Etoile,
    proba_vie: float,
    T_sim_Ga: float,
    rng: np.random.Generator,
) -> Evenement | None:
    """
    Tirage de l'événement 'apparition de la vie'.
    Fenêtre : [AGE_MIN_GA, min(duree_vie_etoile, T_sim)].
    Retourne None si pas d'apparition.
    """
    t_max_ga = min(etoile.duree_vie_Ga, T_sim_Ga)

    # Fenêtre dégénérée : pas d'apparition possible.
    if t_max_ga <= AGE_MIN_GA:
        return None

    # Bernoulli sur la proba multiplicative.
    if rng.random() >= proba_vie:
        return None

    instant_ga = rng.uniform(AGE_MIN_GA, t_max_ga)
    return Evenement(
        type_id=event_type_id("life_apparition"),
        timecode=_ga_vers_timecode(instant_ga),
    )


def _tirer_event_fin_sp(
    etoile: Etoile,
    T_sim_Ga: float,
) -> Evenement | None:
    """
    Événement 'fin de séquence principale' : déterministe à t = duree_vie_Ga
    s'il tombe dans la fenêtre d'observation. Sinon pas d'événement émis
    (la simulation se termine avant que l'étoile ne quitte la SP).
    """
    if etoile.duree_vie_Ga >= T_sim_Ga:
        return None

    return Evenement(
        type_id=event_type_id("star_main_sequence_end"),
        timecode=_ga_vers_timecode(etoile.duree_vie_Ga),
    )


# ============================================================================
# ÉVALUATION D'UNE PLANÈTE → LISTE D'ÉVÉNEMENTS
# ============================================================================

def evaluer_planete(
    etoile: Etoile,
    planete: Planete,
    params: ParametresAstrophysiques,
    rng: np.random.Generator,
) -> ResultatPlanete:
    """
    Calcule la séquence d'événements d'une planète sur la fenêtre
    d'observation [0, T_sim]. Les événements sont triés par timecode.
    """
    proba, detail = proba_apparition_vie(etoile, planete)
    T_obs = _ga_vers_timecode(params.duree_simulation_Ga)

    evts: list[Evenement] = []

    # Vie : conditionnée par la durée de vie stellaire et T_sim.
    e_vie = _tirer_event_vie(etoile, proba, params.duree_simulation_Ga, rng)
    if e_vie is not None:
        evts.append(e_vie)

    # Fin SP : déterministe si l'étoile meurt avant T_sim.
    e_fsp = _tirer_event_fin_sp(etoile, params.duree_simulation_Ga)
    if e_fsp is not None:
        evts.append(e_fsp)

    evts.sort(key=lambda e: e.timecode)

    return ResultatPlanete(
        evenements=evts,
        proba_vie=proba,
        f_HZ=detail["f_HZ"],
        f_composition=detail["f_composition"],
        f_masse=detail["f_masse"],
        f_etoile=detail["f_etoile"],
        T_obs=T_obs,
    )
