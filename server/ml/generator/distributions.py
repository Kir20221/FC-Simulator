"""
Tirages des features étoile et planète pour le générateur de vérité.

Refonte paramétrique : les conditions astrophysiques de l'univers sont
exposées via la dataclass ParametresAstrophysiques. Lors de la génération
d'un dataset d'entraînement, ces paramètres sont tirés par système dans
des plages de balayage (constantes du générateur). Lors d'une simulation
utilisateur, ils seront fixés à des valeurs uniques choisies dans l'IHM.

Toutes les constantes de calibration sont regroupées en tête de fichier.
"""

from dataclasses import dataclass
import numpy as np


# ============================================================================
# CONSTANTES DE CALIBRATION — SUBSTRAT STELLAIRE
# ============================================================================

# Types spectraux et plages de masse associées (séquence principale).
# Source : tables stellaires standard, type Habets & Heintze 1981.
TYPES_SPECTRAUX = ["M", "K", "G", "F", "A", "B"]

# Pour chaque type : (T_eff_min, T_eff_max, M_min, M_max, L_min, L_max,
#                     duree_vie_min_Ga, duree_vie_max_Ga)
PROPRIETES_TYPES = {
    "M": (2400, 3700, 0.08, 0.45, 0.0001, 0.08, 50.0, 1000.0),
    "K": (3700, 5200, 0.45, 0.80, 0.08, 0.60, 15.0, 50.0),
    "G": (5200, 6000, 0.80, 1.04, 0.60, 1.50, 8.0, 15.0),
    "F": (6000, 7500, 1.04, 1.40, 1.50, 5.00, 3.0, 8.0),
    "A": (7500, 10000, 1.40, 2.10, 5.00, 25.0, 0.5, 3.0),
    "B": (10000, 30000, 2.10, 16.0, 25.0, 30000.0, 0.01, 0.5),
}

# Bornes physiques globales pour la masse stellaire (M☉).
MASSE_STELLAIRE_MIN = 0.08
MASSE_STELLAIRE_MAX = 16.0

# Dispersion (en log10 M☉) autour de la masse moyenne paramétrée.
# Une dispersion fixe permet de couvrir un éventail de types spectraux
# autour de la moyenne sans laisser l'utilisateur la régler (paramètre
# non exposé dans l'IHM pour le prototype).
MASSE_STELLAIRE_SIGMA_LOG10 = 0.4


# ============================================================================
# CONSTANTES DE CALIBRATION — SUBSTRAT PLANÉTAIRE
# ============================================================================

# Bornes Poisson pour le nombre de planètes par système.
PLANETES_MIN = 1
PLANETES_MAX = 8

# Distance orbitale : log-uniforme.
DISTANCE_MIN_UA = 0.05
DISTANCE_MAX_UA = 50.0

# Masse planétaire : log-uniforme, en masses terrestres.
MASSE_MIN_TERRE = 0.05
MASSE_MAX_TERRE = 4000.0

# Seuils de composition par masse à indice tellurique de référence (0.5).
# Ils sont modulés ensuite par l'indice tellurique pour chaque système.
SEUIL_TELLURIQUE_GLACEE_REF = 2.0
SEUIL_GLACEE_GAZEUSE_REF = 10.0

# Amplitude de modulation des seuils par l'indice tellurique.
# À indice = 1.0, le seuil tellurique est multiplié par AMPLITUDE_TELLURIQUE.
# À indice = 0.0, il est divisé par AMPLITUDE_TELLURIQUE.
# À indice = 0.5, pas de modulation (valeur de référence).
AMPLITUDE_TELLURIQUE = 3.0


# ============================================================================
# CONSTANTES DE CALIBRATION — PLAGES DE BALAYAGE PARAMÉTRIQUE
# ============================================================================
# Plages utilisées par le générateur de vérité pour échantillonner les
# paramètres lors de la génération d'un dataset d'entraînement. L'IHM
# de simulation reportera ces bornes comme min/max des sliders utilisateur.

PLAGE_MASSE_STELLAIRE_MOYENNE = (0.1, 2.0)        # M☉
PLAGE_INDICE_TELLURIQUE = (0.0, 1.0)              # adimensionné
PLAGE_PLANETES_PAR_SYSTEME_MOYEN = (0.5, 5.0)     # adimensionné (lambda Poisson)
PLAGE_DUREE_SIMULATION_GA = (1.0, 50.0)           # Ga


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class ParametresAstrophysiques:
    """
    Conditions astrophysiques d'un univers (un système, ou un scénario
    utilisateur entier). Quatre paramètres exposés à l'utilisateur, qui
    deviennent des features ML dans le dataset d'entraînement.
    """
    masse_stellaire_moyenne: float       # M☉
    indice_tellurique: float             # [0, 1]
    planetes_par_systeme_moyen: float    # lambda Poisson tronquée
    duree_simulation_Ga: float           # Ga


@dataclass
class Etoile:
    type_spectral: str
    temperature_K: float
    masse_solaire: float
    luminosite_solaire: float
    duree_vie_Ga: float


@dataclass
class Planete:
    distance_UA: float
    masse_terre: float
    rayon_terre: float
    composition: str  # "tellurique" | "glacee" | "gazeuse"


# ============================================================================
# TIRAGE DES PARAMÈTRES (BALAYAGE PAR SYSTÈME)
# ============================================================================

def tirer_parametres(rng: np.random.Generator) -> ParametresAstrophysiques:
    """
    Tire un jeu de paramètres astrophysiques pour un système, dans les
    plages de balayage du générateur de vérité.

    Convention :
    - masse stellaire moyenne : log-uniforme (couvre plusieurs ordres de grandeur)
    - autres paramètres : uniforme sur leur plage
    """
    m_min, m_max = PLAGE_MASSE_STELLAIRE_MOYENNE
    masse_moyenne = float(np.exp(rng.uniform(np.log(m_min), np.log(m_max))))

    it_min, it_max = PLAGE_INDICE_TELLURIQUE
    indice_tellurique = float(rng.uniform(it_min, it_max))

    p_min, p_max = PLAGE_PLANETES_PAR_SYSTEME_MOYEN
    planetes_moyen = float(rng.uniform(p_min, p_max))

    d_min, d_max = PLAGE_DUREE_SIMULATION_GA
    duree = float(rng.uniform(d_min, d_max))

    return ParametresAstrophysiques(
        masse_stellaire_moyenne=masse_moyenne,
        indice_tellurique=indice_tellurique,
        planetes_par_systeme_moyen=planetes_moyen,
        duree_simulation_Ga=duree,
    )


# ============================================================================
# TIRAGE DE L'ÉTOILE
# ============================================================================

def _type_spectral_depuis_masse(masse_solaire: float) -> str:
    """Recherche dans PROPRIETES_TYPES le type spectral contenant cette masse."""
    for type_sp in TYPES_SPECTRAUX:
        _, _, m_min, m_max, _, _, _, _ = PROPRIETES_TYPES[type_sp]
        if m_min <= masse_solaire < m_max:
            return type_sp
    # Cas limite : masse au-delà du plus grand type → on retourne le plus massif.
    return TYPES_SPECTRAUX[-1]


def tirer_etoile(rng: np.random.Generator, params: ParametresAstrophysiques) -> Etoile:
    """
    Tire une étoile dont la masse est échantillonnée dans une log-normale
    centrée sur params.masse_stellaire_moyenne, avec dispersion fixe.
    Les autres propriétés (T, L, durée de vie) sont tirées dans les plages
    du type spectral correspondant à la masse.
    """
    log_moyenne = np.log10(params.masse_stellaire_moyenne)
    log_masse = rng.normal(log_moyenne, MASSE_STELLAIRE_SIGMA_LOG10)
    masse = float(np.clip(10 ** log_masse, MASSE_STELLAIRE_MIN, MASSE_STELLAIRE_MAX))

    type_sp = _type_spectral_depuis_masse(masse)
    t_min, t_max, _, _, l_min, l_max, dv_min, dv_max = PROPRIETES_TYPES[type_sp]

    temperature = float(rng.uniform(t_min, t_max))
    luminosite = float(np.exp(rng.uniform(np.log(l_min), np.log(l_max))))
    duree_vie = float(np.exp(rng.uniform(np.log(dv_min), np.log(dv_max))))

    return Etoile(
        type_spectral=type_sp,
        temperature_K=temperature,
        masse_solaire=masse,
        luminosite_solaire=luminosite,
        duree_vie_Ga=duree_vie,
    )


# ============================================================================
# TIRAGE DU NOMBRE DE PLANÈTES
# ============================================================================

def tirer_nombre_planetes(
    rng: np.random.Generator,
    params: ParametresAstrophysiques,
) -> int:
    """Poisson tronquée [PLANETES_MIN, PLANETES_MAX] de moyenne paramétrée."""
    while True:
        n = rng.poisson(params.planetes_par_systeme_moyen)
        if PLANETES_MIN <= n <= PLANETES_MAX:
            return int(n)


# ============================================================================
# TIRAGE DE LA PLANÈTE
# ============================================================================

def _seuils_composition(indice_tellurique: float) -> tuple[float, float]:
    """
    Module les seuils de composition selon l'indice tellurique.
    indice = 0.5 → seuils de référence (cas neutre).
    indice = 1.0 → seuils multipliés par AMPLITUDE_TELLURIQUE (plus de telluriques).
    indice = 0.0 → seuils divisés par AMPLITUDE_TELLURIQUE (moins de telluriques).
    Interpolation log-linéaire entre ces points.
    """
    log_facteur = (indice_tellurique - 0.5) * 2.0 * np.log(AMPLITUDE_TELLURIQUE)
    facteur = float(np.exp(log_facteur))
    return (
        SEUIL_TELLURIQUE_GLACEE_REF * facteur,
        SEUIL_GLACEE_GAZEUSE_REF * facteur,
    )


def _rayon_depuis_masse(masse_terre: float, seuil_tellurique: float) -> float:
    """
    Relation masse-rayon par paliers, inspirée Chen & Kipping 2017.
    Le seuil tellurique est paramétré (dépend de l'indice tellurique).
    """
    if masse_terre < seuil_tellurique:
        return masse_terre ** 0.28
    elif masse_terre < seuil_tellurique * 50:
        # Régime néptunien, raccordé au régime tellurique.
        r_ref = seuil_tellurique ** 0.28
        return r_ref * (masse_terre / seuil_tellurique) ** 0.59
    else:
        # Régime jovien : rayon quasi-constant.
        ref_neptune = seuil_tellurique * 50
        r_ref = seuil_tellurique ** 0.28 * (ref_neptune / seuil_tellurique) ** 0.59
        return r_ref * (masse_terre / ref_neptune) ** 0.04


def _composition_depuis_masse(
    masse_terre: float,
    seuil_tellurique: float,
    seuil_glacee: float,
) -> str:
    if masse_terre < seuil_tellurique:
        return "tellurique"
    elif masse_terre < seuil_glacee:
        return "glacee"
    else:
        return "gazeuse"


def tirer_planete(rng: np.random.Generator, params: ParametresAstrophysiques) -> Planete:
    """
    Tire une planète. Les seuils de composition sont modulés par
    l'indice tellurique du système.
    """
    seuil_tell, seuil_glac = _seuils_composition(params.indice_tellurique)

    distance = float(np.exp(rng.uniform(np.log(DISTANCE_MIN_UA), np.log(DISTANCE_MAX_UA))))
    masse = float(np.exp(rng.uniform(np.log(MASSE_MIN_TERRE), np.log(MASSE_MAX_TERRE))))
    rayon = _rayon_depuis_masse(masse, seuil_tell)
    composition = _composition_depuis_masse(masse, seuil_tell, seuil_glac)

    return Planete(
        distance_UA=distance,
        masse_terre=masse,
        rayon_terre=rayon,
        composition=composition,
    )
