"""
Cœur métier du générateur de vérité — sans dépendance au transport
(ni FastAPI, ni argparse). Appelé indifféremment depuis ml/api.py (HTTP)
ou ml/cli.py (CLI).

Convention : tout résultat retourné est sérialisable JSON pour faciliter
l'usage HTTP. Les fonctions ne font pas d'I/O au-delà de l'écriture du
parquet et de la lecture pour les stats.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .distributions import (
    tirer_etoile,
    tirer_nombre_planetes,
    tirer_planete,
)
from .life_model import evaluer_planete


# ============================================================================
# CONSTANTES
# ============================================================================

# Racine où vivent les datasets. Le path est relatif au conteneur Docker,
# qui monte ./data depuis l'hôte.
RACINE_DATASETS = Path("/srv/data/training")

# Champs du parquet de sortie. Ordre stable, types explicites.
COLONNES_PARQUET = [
    "system_id",
    "star_type",
    "star_temp_K",
    "star_mass_solar",
    "star_luminosity_solar",
    "star_age_Ga",
    "star_lifetime_Ga",
    "planet_distance_UA",
    "planet_mass_terre",
    "planet_radius_terre",
    "planet_composition",
    "life_probability",
    "life_appears",
    "life_timecode",
    "f_HZ",
    "f_composition",
    "f_masse",
    "f_etoile",
    "f_age",
]


# ============================================================================
# REQUEST / RESPONSE — SCHÉMAS PARTAGÉS CLI ET HTTP
# ============================================================================

class DatasetGenerationRequest(BaseModel):
    """Paramètres de génération d'un dataset. Validés côté HTTP par FastAPI,
    instanciés côté CLI à partir des arguments argparse."""
    nom: str = Field(..., description="Identifiant du dataset (ex: 'v1').")
    nb_systemes: int = Field(..., ge=1, description="Nombre de systèmes solaires à générer.")
    seed: int = Field(default=42, description="Graine aléatoire.")


class DatasetStats(BaseModel):
    """Résumé statistique d'un dataset. Calculé après génération ou à la
    demande sur un dataset existant."""
    nb_systemes: int
    nb_planetes: int
    nb_telluriques: int
    nb_glacees: int
    nb_gazeuses: int
    nb_vie: int
    taux_vie: float
    proba_vie_moyenne: float
    proba_vie_max: float
    repartition_types_spectraux: dict[str, int]
    vie_par_type_spectral: dict[str, dict]


# ============================================================================
# GÉNÉRATION (COEUR PUR — PAS D'I/O)
# ============================================================================

def generer_dataframe(nb_systemes: int, seed: int) -> pd.DataFrame:
    """Produit le DataFrame en mémoire. Pas d'écriture disque ici."""
    rng = np.random.default_rng(seed)
    lignes = []

    for sys_id in range(nb_systemes):
        etoile = tirer_etoile(rng)
        nb_planetes = tirer_nombre_planetes(rng)

        for _ in range(nb_planetes):
            planete = tirer_planete(rng)
            res = evaluer_planete(etoile, planete, rng)

            lignes.append({
                "system_id": sys_id,
                "star_type": etoile.type_spectral,
                "star_temp_K": etoile.temperature_K,
                "star_mass_solar": etoile.masse_solaire,
                "star_luminosity_solar": etoile.luminosite_solaire,
                "star_age_Ga": etoile.age_Ga,
                "star_lifetime_Ga": etoile.duree_vie_Ga,
                "planet_distance_UA": planete.distance_UA,
                "planet_mass_terre": planete.masse_terre,
                "planet_radius_terre": planete.rayon_terre,
                "planet_composition": planete.composition,
                "life_probability": res.proba_vie,
                "life_appears": res.vie_apparait,
                "life_timecode": res.timecode_apparition if res.timecode_apparition is not None else -1,
                "f_HZ": res.f_HZ,
                "f_composition": res.f_composition,
                "f_masse": res.f_masse,
                "f_etoile": res.f_etoile,
                "f_age": res.f_age,
            })

    return pd.DataFrame(lignes, columns=COLONNES_PARQUET)


def calculer_stats(df: pd.DataFrame) -> DatasetStats:
    """Résumé statistique d'un DataFrame de planètes."""
    nb_planetes = len(df)
    nb_vie = int(df["life_appears"].sum())

    vie_par_type_df = df.groupby("star_type")["life_appears"].agg(["sum", "count"])
    vie_par_type_df["taux"] = vie_par_type_df["sum"] / vie_par_type_df["count"]
    vie_par_type = {
        str(idx): {
            "vie": int(row["sum"]),
            "total": int(row["count"]),
            "taux": float(row["taux"]),
        }
        for idx, row in vie_par_type_df.iterrows()
    }

    return DatasetStats(
        nb_systemes=int(df["system_id"].nunique()),
        nb_planetes=nb_planetes,
        nb_telluriques=int((df["planet_composition"] == "tellurique").sum()),
        nb_glacees=int((df["planet_composition"] == "glacee").sum()),
        nb_gazeuses=int((df["planet_composition"] == "gazeuse").sum()),
        nb_vie=nb_vie,
        taux_vie=float(nb_vie / nb_planetes) if nb_planetes else 0.0,
        proba_vie_moyenne=float(df["life_probability"].mean()),
        proba_vie_max=float(df["life_probability"].max()),
        repartition_types_spectraux={
            str(k): int(v) for k, v in df["star_type"].value_counts().items()
        },
        vie_par_type_spectral=vie_par_type,
    )


# ============================================================================
# I/O DISQUE
# ============================================================================

def chemin_parquet(nom: str) -> Path:
    return RACINE_DATASETS / f"{nom}_planetes.parquet"


def ecrire_parquet(df: pd.DataFrame, chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chemin, engine="pyarrow", compression="snappy", index=False)


def lire_parquet(chemin: Path) -> pd.DataFrame:
    return pd.read_parquet(chemin, engine="pyarrow")


def supprimer_parquet(chemin: Path) -> None:
    if chemin.exists():
        chemin.unlink()


# ============================================================================
# RÉSULTAT D'UNE GÉNÉRATION (sans persistance DB — celle-ci est ajoutée
# par la couche appelante)
# ============================================================================

@dataclass
class ResultatGeneration:
    nom: str
    chemin: Path
    taille_octets: int
    stats: DatasetStats
    request: DatasetGenerationRequest


def generer_dataset(request: DatasetGenerationRequest) -> ResultatGeneration:
    """
    Génère un dataset, l'écrit sur disque, calcule ses stats. Retourne tout
    ce qu'il faut à la couche appelante pour persister les métadonnées en DB.
    """
    df = generer_dataframe(request.nb_systemes, request.seed)
    chemin = chemin_parquet(request.nom)
    ecrire_parquet(df, chemin)
    stats = calculer_stats(df)

    return ResultatGeneration(
        nom=request.nom,
        chemin=chemin,
        taille_octets=chemin.stat().st_size,
        stats=stats,
        request=request,
    )