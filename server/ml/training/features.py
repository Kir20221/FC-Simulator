"""
Préparation des features pour le Bloc B (TPP).

Responsabilités :
- Définir quelles colonnes du parquet planets sont des features ML.
- Encoder les catégorielles (type spectral, composition) en index entiers
  (les embeddings sont appris dans le modèle, ce module ne fait que la
  conversion str -> int).
- Normaliser les continues (log10 pour celles couvrant plusieurs ordres
  de grandeur, puis z-score).
- Sérialiser les statistiques de normalisation pour l'inférence (le
  modèle .pth doit pouvoir transformer des features brutes au moment de
  l'utilisation, sans recharger le dataset d'entraînement).

Choix explicites :
- Les 4 paramètres astrophysiques (param_*) ne sont PAS des features
  ML — décision validée plus tôt (« pas de fuite d'information »).
- T_obs n'est pas une feature en entrée du modèle ; il est utilisé par
  la loss (borne d'intégration) et par l'inférence (horizon d'échantillonnage).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================================
# RÉFÉRENTIEL DES FEATURES
# ============================================================================

# Continues : standardisées z-score directement.
FEATURES_CONTINUES = [
    "star_temp_K",
]

# Continues couvrant plusieurs ordres de grandeur : log10 puis z-score.
FEATURES_CONTINUES_LOG = [
    "star_mass_solar",
    "star_luminosity_solar",
    "star_lifetime_Ga",
    "planet_distance_UA",
    "planet_mass_terre",
    "planet_radius_terre",
]

# Catégorielles : converties en index entier, embeddings appris dans le modèle.
# L'ordre dans la liste de catégories définit la correspondance label -> index.
FEATURES_CATEGORIELLES: dict[str, list[str]] = {
    "star_type": ["M", "K", "G", "F", "A", "B"],
    "planet_composition": ["tellurique", "glacee", "gazeuse"],
}


# ============================================================================
# STATISTIQUES DE NORMALISATION
# ============================================================================

@dataclass
class FeatureStats:
    """
    Statistiques de normalisation calculées sur le set d'entraînement.
    Sauvegardées avec le modèle pour pouvoir appliquer la même transformation
    à l'inférence.
    """
    # Pour chaque feature continue (log ou pas) : (mean, std) après éventuel log10.
    continues_mean_std: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Pour chaque feature catégorielle : la liste ordonnée des labels possibles.
    # Permet de retrouver l'index à partir du label.
    categorielles_labels: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "continues_mean_std": self.continues_mean_std,
            "categorielles_labels": self.categorielles_labels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(
            continues_mean_std={k: tuple(v) for k, v in d["continues_mean_std"].items()},
            categorielles_labels=d["categorielles_labels"],
        )


def calculer_stats(df_planets: pd.DataFrame) -> FeatureStats:
    """Calcule les statistiques de normalisation sur le set d'entraînement."""
    stats = FeatureStats()

    for col in FEATURES_CONTINUES:
        x = df_planets[col].to_numpy(dtype=np.float64)
        stats.continues_mean_std[col] = (float(x.mean()), float(x.std() + 1e-12))

    for col in FEATURES_CONTINUES_LOG:
        x = np.log10(df_planets[col].to_numpy(dtype=np.float64) + 1e-30)
        stats.continues_mean_std[col] = (float(x.mean()), float(x.std() + 1e-12))

    stats.categorielles_labels = {k: list(v) for k, v in FEATURES_CATEGORIELLES.items()}
    return stats


# ============================================================================
# APPLICATION DES STATS — TRANSFORMATIONS
# ============================================================================

def transformer_continues(df_planets: pd.DataFrame, stats: FeatureStats) -> np.ndarray:
    """
    Retourne un tenseur numpy (N, F_cont) où F_cont = len(continues) + len(continues_log).
    Ordre des colonnes : FEATURES_CONTINUES puis FEATURES_CONTINUES_LOG.
    """
    cols: list[np.ndarray] = []
    for col in FEATURES_CONTINUES:
        x = df_planets[col].to_numpy(dtype=np.float64)
        m, s = stats.continues_mean_std[col]
        cols.append((x - m) / s)
    for col in FEATURES_CONTINUES_LOG:
        x = np.log10(df_planets[col].to_numpy(dtype=np.float64) + 1e-30)
        m, s = stats.continues_mean_std[col]
        cols.append((x - m) / s)
    return np.stack(cols, axis=1).astype(np.float32)


def transformer_categorielles(
    df_planets: pd.DataFrame,
    stats: FeatureStats,
) -> dict[str, np.ndarray]:
    """
    Retourne un dict {nom_feature: array d'index entiers (N,)}.
    Les labels inconnus sont remappés sur 0 (avec warning silencieux ; en
    pratique le set d'entraînement couvre tous les labels possibles
    puisqu'ils sont définis par construction du Bloc A).
    """
    result: dict[str, np.ndarray] = {}
    for col, labels in stats.categorielles_labels.items():
        mapping = {lab: i for i, lab in enumerate(labels)}
        idx = df_planets[col].map(mapping).fillna(0).to_numpy(dtype=np.int64)
        result[col] = idx
    return result


# ============================================================================
# DIMENSIONS — UTILES POUR INSTANCIER LE MODÈLE
# ============================================================================

def dim_continues() -> int:
    return len(FEATURES_CONTINUES) + len(FEATURES_CONTINUES_LOG)


def dim_categorielles() -> dict[str, int]:
    """{nom_feature: nb_categories}."""
    return {k: len(v) for k, v in FEATURES_CATEGORIELLES.items()}
