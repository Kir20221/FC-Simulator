"""
Dataset PyTorch et data loader pour le Bloc B (TPP).

Pour chaque planète, on fournit au modèle :
- les features statiques (continues normalisées + catégorielles en index),
- la séquence d'événements (timecodes + types), pouvant être vide,
- l'horizon d'observation T_obs.

Le collate_fn pad les séquences à la longueur max du batch avec un masque.
Les types catégoriels d'événements sont indexés par EVENT_TYPES (référentiel
défini dans le Bloc A, life_model.EVENT_TYPES).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    FeatureStats,
    transformer_categorielles,
    transformer_continues,
)


# ============================================================================
# STRUCTURE D'UN ÉLÉMENT DU DATASET
# ============================================================================

@dataclass
class PlanetSample:
    """Un sample = une planète."""
    cont: np.ndarray          # (F_cont,) float32
    cat: dict[str, int]       # {nom_feature: index}
    times: np.ndarray         # (L,) int64, possiblement vide
    types: np.ndarray         # (L,) int64, possiblement vide
    T_obs: int                # horizon en timecode


# ============================================================================
# DATASET
# ============================================================================

class PlanetEventsDataset(Dataset):
    """
    Lit les deux parquets (planets + events) et joint en mémoire la séquence
    d'événements à chaque planète. Pour 100k systèmes (~300k planètes), cela
    tient largement en RAM ; pour des datasets bien plus gros, on passera à
    une lecture par chunks (hors prototype).
    """

    def __init__(
        self,
        chemin_planets: Path | str,
        chemin_events: Path | str,
        stats: FeatureStats,
    ):
        df_planets = pd.read_parquet(chemin_planets, engine="pyarrow")
        df_events = pd.read_parquet(chemin_events, engine="pyarrow")

        # Pré-calcul des features statiques sous forme tensorielle.
        cont = transformer_continues(df_planets, stats)        # (N, F_cont)
        cat = transformer_categorielles(df_planets, stats)     # dict de (N,)

        self._planet_ids = df_planets["planet_id"].to_numpy(dtype=np.int64)
        self._cont = cont
        self._cat = cat
        self._cat_keys = list(cat.keys())
        self._T_obs = df_planets["T_obs"].to_numpy(dtype=np.int64)

        # Indexation des événements par planet_id, triés par event_time.
        # On construit un dict {planet_id: (times[], types[])}.
        self._sequences: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if len(df_events) > 0:
            df_events = df_events.sort_values(["planet_id", "event_time"])
            for pid, grp in df_events.groupby("planet_id"):
                self._sequences[int(pid)] = (
                    grp["event_time"].to_numpy(dtype=np.int64),
                    grp["event_type_id"].to_numpy(dtype=np.int64),
                )

    def __len__(self) -> int:
        return len(self._planet_ids)

    def __getitem__(self, idx: int) -> PlanetSample:
        pid = int(self._planet_ids[idx])
        times, types = self._sequences.get(pid, (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        ))
        return PlanetSample(
            cont=self._cont[idx],
            cat={k: int(self._cat[k][idx]) for k in self._cat_keys},
            times=times,
            types=types,
            T_obs=int(self._T_obs[idx]),
        )


# ============================================================================
# COLLATE — PADDING DES SÉQUENCES
# ============================================================================

@dataclass
class Batch:
    """Batch tensoriel prêt pour le modèle."""
    cont: torch.Tensor                       # (B, F_cont) float32
    cat: dict[str, torch.Tensor]             # {nom: (B,) int64}
    times: torch.Tensor                      # (B, L_max) int64, padding = 0
    types: torch.Tensor                      # (B, L_max) int64, padding = 0
    seq_mask: torch.Tensor                   # (B, L_max) bool, True aux positions valides
    seq_lengths: torch.Tensor                # (B,) int64
    T_obs: torch.Tensor                      # (B,) int64

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            cont=self.cont.to(device),
            cat={k: v.to(device) for k, v in self.cat.items()},
            times=self.times.to(device),
            types=self.types.to(device),
            seq_mask=self.seq_mask.to(device),
            seq_lengths=self.seq_lengths.to(device),
            T_obs=self.T_obs.to(device),
        )


def collate_fn(samples: list[PlanetSample]) -> Batch:
    B = len(samples)
    L_max = max((len(s.times) for s in samples), default=0)
    # Cas pathologique : si tout le batch est sans événement, L_max = 0 ;
    # on force L_max >= 1 pour éviter des tenseurs de dim 0 dans le modèle.
    L_max = max(L_max, 1)

    cont = torch.from_numpy(np.stack([s.cont for s in samples], axis=0))
    cat_keys = list(samples[0].cat.keys())
    cat = {
        k: torch.tensor([s.cat[k] for s in samples], dtype=torch.long)
        for k in cat_keys
    }

    times = torch.zeros((B, L_max), dtype=torch.long)
    types = torch.zeros((B, L_max), dtype=torch.long)
    seq_mask = torch.zeros((B, L_max), dtype=torch.bool)
    seq_lengths = torch.zeros(B, dtype=torch.long)
    T_obs = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(samples):
        L = len(s.times)
        if L > 0:
            times[i, :L] = torch.from_numpy(s.times.copy())
            types[i, :L] = torch.from_numpy(s.types.copy())
            seq_mask[i, :L] = True
        seq_lengths[i] = L
        T_obs[i] = s.T_obs

    return Batch(
        cont=cont,
        cat=cat,
        times=times,
        types=types,
        seq_mask=seq_mask,
        seq_lengths=seq_lengths,
        T_obs=T_obs,
    )
