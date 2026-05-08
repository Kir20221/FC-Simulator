"""
Cœur métier du générateur de vérité — version multi-événements.

Refonte pour le Bloc B (TPP) :
- Le dataset n'est plus un parquet unique « une ligne par planète » avec
  étiquette one-shot, mais deux parquets :
    * <nom>_planets.parquet : une ligne par planète, avec features +
      durée d'observation T_obs.
    * <nom>_events.parquet  : une ligne par événement, avec planet_id,
      event_type_id, event_time.
- Format long pour les événements (extensible à N types sans changer le
  schéma).
- Le code reste agnostique au nombre de types : life_model.EVENT_TYPES
  est la seule source de vérité pour le référentiel.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .distributions import (
    tirer_etoile,
    tirer_nombre_planetes,
    tirer_parametres,
    tirer_planete,
)
from .life_model import EVENT_TYPES, evaluer_planete


# ============================================================================
# CONSTANTES
# ============================================================================

RACINE_DATASETS = Path("/srv/data/training")

# Schéma planète : features ML + horizon d'observation.
# Note : les 4 paramètres astrophysiques restent dans le parquet pour
# inspection et statistiques, mais ne seront PAS donnés en entrée au
# modèle ML (cf. discussion « pas de fuite d'information »).
COLONNES_PLANETS = [
    "planet_id",
    "system_id",
    # Paramètres astrophysiques (inspection seulement)
    "param_masse_stellaire_moyenne",
    "param_indice_tellurique",
    "param_planetes_par_systeme_moyen",
    "param_duree_simulation_Ga",
    # Features étoile (entrées ML)
    "star_type",
    "star_temp_K",
    "star_mass_solar",
    "star_luminosity_solar",
    "star_lifetime_Ga",
    # Features planète (entrées ML)
    "planet_distance_UA",
    "planet_mass_terre",
    "planet_radius_terre",
    "planet_composition",
    # Horizon d'observation
    "T_obs",
    # Diagnostic
    "life_probability",
    "f_HZ",
    "f_composition",
    "f_masse",
    "f_etoile",
]

# Schéma événements : un événement par ligne, format long.
COLONNES_EVENTS = [
    "planet_id",
    "event_type_id",
    "event_type_label",  # redondant avec id mais utile pour inspection
    "event_time",        # en unités de timecode (100 000 ans)
]


# ============================================================================
# REQUEST / RESPONSE
# ============================================================================

class DatasetGenerationRequest(BaseModel):
    nom: str = Field(..., description="Identifiant du dataset.")
    nb_systemes: int = Field(..., ge=1)
    seed: int = Field(default=42)


class DatasetStats(BaseModel):
    """Stats globales du dataset, agnostiques au nombre de types."""
    nb_systemes: int
    nb_planetes: int
    nb_events_total: int
    # Comptage par type d'événement (clé = libellé du type)
    nb_events_par_type: dict[str, int]
    # Distribution des chaînes d'événements observées.
    # Clé = chaîne ordonnée par timecode, formée des libellés de types
    # séparés par " -> " (ex: "life_apparition -> star_main_sequence_end").
    # La chaîne vide "" représente les planètes sans événement.
    # Cette représentation reste lisible quel que soit le nombre de types.
    chaines_observees: dict[str, int]
    # Stats compositions / types spectraux (inchangées)
    nb_telluriques: int
    nb_glacees: int
    nb_gazeuses: int
    repartition_types_spectraux: dict[str, int]


# ============================================================================
# GÉNÉRATION (CŒUR PUR — PAS D'I/O)
# ============================================================================

def generer_dataframes(
    nb_systemes: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produit les deux DataFrames en mémoire (planètes + événements).
    """
    rng = np.random.default_rng(seed)

    lignes_planetes: list[dict] = []
    lignes_events: list[dict] = []
    planet_id_counter = 0

    for sys_id in range(nb_systemes):
        params = tirer_parametres(rng)
        etoile = tirer_etoile(rng, params)
        nb_planetes = tirer_nombre_planetes(rng, params)

        for _ in range(nb_planetes):
            planete = tirer_planete(rng, params)
            res = evaluer_planete(etoile, planete, params, rng)
            pid = planet_id_counter
            planet_id_counter += 1

            lignes_planetes.append({
                "planet_id": pid,
                "system_id": sys_id,
                "param_masse_stellaire_moyenne": params.masse_stellaire_moyenne,
                "param_indice_tellurique": params.indice_tellurique,
                "param_planetes_par_systeme_moyen": params.planetes_par_systeme_moyen,
                "param_duree_simulation_Ga": params.duree_simulation_Ga,
                "star_type": etoile.type_spectral,
                "star_temp_K": etoile.temperature_K,
                "star_mass_solar": etoile.masse_solaire,
                "star_luminosity_solar": etoile.luminosite_solaire,
                "star_lifetime_Ga": etoile.duree_vie_Ga,
                "planet_distance_UA": planete.distance_UA,
                "planet_mass_terre": planete.masse_terre,
                "planet_radius_terre": planete.rayon_terre,
                "planet_composition": planete.composition,
                "T_obs": res.T_obs,
                "life_probability": res.proba_vie,
                "f_HZ": res.f_HZ,
                "f_composition": res.f_composition,
                "f_masse": res.f_masse,
                "f_etoile": res.f_etoile,
            })

            for evt in res.evenements:
                lignes_events.append({
                    "planet_id": pid,
                    "event_type_id": evt.type_id,
                    "event_type_label": evt.type_libelle,
                    "event_time": evt.timecode,
                })

    df_planets = pd.DataFrame(lignes_planetes, columns=COLONNES_PLANETS)
    df_events = pd.DataFrame(lignes_events, columns=COLONNES_EVENTS)
    return df_planets, df_events


def calculer_stats(df_planets: pd.DataFrame, df_events: pd.DataFrame) -> DatasetStats:
    """Statistiques agrégées."""
    nb_planetes = len(df_planets)

    # Comptage par type d'événement (toutes les clés présentes,
    # même si un type a 0 occurrence — utile pour inspecter le dataset)
    nb_par_type = {label: 0 for label in EVENT_TYPES}
    if len(df_events) > 0:
        counts = df_events["event_type_label"].value_counts().to_dict()
        for k, v in counts.items():
            nb_par_type[str(k)] = int(v)

    # Distribution des chaînes d'événements observées (par planète).
    # On agrège les événements par planet_id, ordonnés par event_time, puis
    # on forme la chaîne lisible "type1 -> type2 -> ...". Les planètes sans
    # événement sont représentées par la chaîne vide "".
    chaines_planetes: dict[int, str] = {pid: "" for pid in df_planets["planet_id"]}
    if len(df_events) > 0:
        events_tries = df_events.sort_values(["planet_id", "event_time"])
        for pid, grp in events_tries.groupby("planet_id"):
            chaines_planetes[int(pid)] = " -> ".join(grp["event_type_label"].tolist())
    chaines = pd.Series(list(chaines_planetes.values())).value_counts().to_dict()
    chaines_observees = {str(k): int(v) for k, v in chaines.items()}

    return DatasetStats(
        nb_systemes=int(df_planets["system_id"].nunique()),
        nb_planetes=nb_planetes,
        nb_events_total=int(len(df_events)),
        nb_events_par_type=nb_par_type,
        chaines_observees=chaines_observees,
        nb_telluriques=int((df_planets["planet_composition"] == "tellurique").sum()),
        nb_glacees=int((df_planets["planet_composition"] == "glacee").sum()),
        nb_gazeuses=int((df_planets["planet_composition"] == "gazeuse").sum()),
        repartition_types_spectraux={
            str(k): int(v) for k, v in df_planets["star_type"].value_counts().items()
        },
    )


# ============================================================================
# I/O DISQUE
# ============================================================================

def chemins_parquet(nom: str) -> tuple[Path, Path]:
    """Retourne (chemin_planets, chemin_events)."""
    return (
        RACINE_DATASETS / f"{nom}_planets.parquet",
        RACINE_DATASETS / f"{nom}_events.parquet",
    )


def ecrire_parquets(
    df_planets: pd.DataFrame,
    df_events: pd.DataFrame,
    chemin_planets: Path,
    chemin_events: Path,
) -> None:
    chemin_planets.parent.mkdir(parents=True, exist_ok=True)
    df_planets.to_parquet(chemin_planets, engine="pyarrow", compression="snappy", index=False)
    df_events.to_parquet(chemin_events, engine="pyarrow", compression="snappy", index=False)


def lire_parquets(chemin_planets: Path, chemin_events: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(chemin_planets, engine="pyarrow"),
        pd.read_parquet(chemin_events, engine="pyarrow"),
    )


def supprimer_parquets(chemin_planets: Path, chemin_events: Path) -> None:
    if chemin_planets.exists():
        chemin_planets.unlink()
    if chemin_events.exists():
        chemin_events.unlink()


# ============================================================================
# RÉSULTAT D'UNE GÉNÉRATION
# ============================================================================

@dataclass
class ResultatGeneration:
    nom: str
    chemin_planets: Path
    chemin_events: Path
    taille_octets: int  # somme des deux fichiers
    stats: DatasetStats
    request: DatasetGenerationRequest


def generer_dataset(request: DatasetGenerationRequest) -> ResultatGeneration:
    """Génère, écrit, calcule les stats."""
    df_planets, df_events = generer_dataframes(request.nb_systemes, request.seed)
    chemin_planets, chemin_events = chemins_parquet(request.nom)
    ecrire_parquets(df_planets, df_events, chemin_planets, chemin_events)
    stats = calculer_stats(df_planets, df_events)

    taille_totale = chemin_planets.stat().st_size + chemin_events.stat().st_size

    return ResultatGeneration(
        nom=request.nom,
        chemin_planets=chemin_planets,
        chemin_events=chemin_events,
        taille_octets=taille_totale,
        stats=stats,
        request=request,
    )
