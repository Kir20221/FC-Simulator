"""
Boucle d'entraînement et sauvegarde du modèle.

Le .pth contient :
- state_dict du modèle,
- config TPP (TPPConfig) pour reconstruire l'architecture,
- statistiques de normalisation des features (FeatureStats) pour
  appliquer la même transformation à l'inférence,
- référentiel des types d'événements (EVENT_TYPES) pour interpréter
  les sorties du modèle.

Tout ce qui est nécessaire à l'inférence est dans le .pth, indépendamment
du parquet d'entraînement.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .data import PlanetEventsDataset, collate_fn
from .features import (
    FeatureStats,
    calculer_stats,
    dim_categorielles,
    dim_continues,
)
from .loss import tpp_negative_log_likelihood
from .model import TPPConfig, TransformerHawkes


# ============================================================================
# CONFIGURATION D'ENTRAÎNEMENT
# ============================================================================

@dataclass
class TrainingConfig:
    # Données
    val_fraction: float = 0.1
    seed_split: int = 42
    # Optimisation
    batch_size: int = 256
    nb_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    nb_mc_samples: int = 32
    # Hyperparams modèle (transmis à TPPConfig)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    dim_emb_cat: int = 4
    time_freqs: int = 16


@dataclass
class TrainingResult:
    chemin_modele: Path
    config: TrainingConfig
    nb_epochs_executees: int
    metrics_par_epoch: list[dict]    # un dict par epoch : train + val
    duree_totale_s: float
    final_train_loss: float
    final_val_loss: float


# ============================================================================
# BUILDER : INSTANCIATION DU MODÈLE
# ============================================================================

def build_model(
    feature_stats: FeatureStats,
    nb_types_evt: int,
    train_cfg: TrainingConfig,
) -> tuple[TransformerHawkes, TPPConfig]:
    tpp_cfg = TPPConfig(
        dim_continues=dim_continues(),
        dim_categorielles=dim_categorielles(),
        nb_types_evt=nb_types_evt,
        dim_emb_cat=train_cfg.dim_emb_cat,
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers,
        dim_feedforward=train_cfg.dim_feedforward,
        dropout=train_cfg.dropout,
        time_freqs=train_cfg.time_freqs,
    )
    return TransformerHawkes(tpp_cfg), tpp_cfg


# ============================================================================
# BOUCLE D'ENTRAÎNEMENT
# ============================================================================

def train(
    chemin_planets: Path,
    chemin_events: Path,
    event_types: list[str],
    chemin_modele_out: Path,
    train_cfg: TrainingConfig,
    device: torch.device | None = None,
) -> TrainingResult:
    """
    Entraîne un Transformer Hawkes sur le dataset (planets + events) du Bloc A.
    Sauvegarde le .pth en sortie. Retourne les métriques observées.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()

    # ----------------------------------------------------------------
    # 1. Statistiques sur le set complet (avant split)
    # ----------------------------------------------------------------
    df_planets = pd.read_parquet(chemin_planets, engine="pyarrow")
    feature_stats = calculer_stats(df_planets)

    full_dataset = PlanetEventsDataset(chemin_planets, chemin_events, feature_stats)

    # ----------------------------------------------------------------
    # 2. Split train / val
    # ----------------------------------------------------------------
    n = len(full_dataset)
    rng = np.random.default_rng(train_cfg.seed_split)
    perm = rng.permutation(n)
    n_val = int(n * train_cfg.val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_set = Subset(full_dataset, train_idx.tolist())
    val_set = Subset(full_dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ----------------------------------------------------------------
    # 3. Modèle + optimizer
    # ----------------------------------------------------------------
    model, tpp_cfg = build_model(
        feature_stats=feature_stats,
        nb_types_evt=len(event_types),
        train_cfg=train_cfg,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ----------------------------------------------------------------
    # 4. Boucle
    # ----------------------------------------------------------------
    metrics_par_epoch: list[dict] = []
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for epoch in range(train_cfg.nb_epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _ = tpp_negative_log_likelihood(
                model, batch, nb_mc_samples=train_cfg.nb_mc_samples
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # Val
        model.eval()
        val_losses = []
        val_log_L_evt = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss, m = tpp_negative_log_likelihood(
                    model, batch, nb_mc_samples=train_cfg.nb_mc_samples
                )
                val_losses.append(loss.item())
                val_log_L_evt.append(m["log_L_par_event"])
        val_loss = float(np.mean(val_losses))
        val_log_L_per_evt = float(np.mean(val_log_L_evt))

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_log_L_par_event": val_log_L_per_evt,
        }
        metrics_par_epoch.append(epoch_metrics)
        last_train_loss = train_loss
        last_val_loss = val_loss

        print(
            f"epoch {epoch+1:3d}/{train_cfg.nb_epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_logL/event {val_log_L_per_evt:.4f}"
        )

    duree = time.time() - t0

    # ----------------------------------------------------------------
    # 5. Sauvegarde
    # ----------------------------------------------------------------
    chemin_modele_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tpp_config": tpp_cfg.__dict__,
            "feature_stats": feature_stats.to_dict(),
            "event_types": event_types,
            "training_config": train_cfg.__dict__,
            "metrics_par_epoch": metrics_par_epoch,
        },
        chemin_modele_out,
    )

    return TrainingResult(
        chemin_modele=chemin_modele_out,
        config=train_cfg,
        nb_epochs_executees=train_cfg.nb_epochs,
        metrics_par_epoch=metrics_par_epoch,
        duree_totale_s=duree,
        final_train_loss=last_train_loss,
        final_val_loss=last_val_loss,
    )


# ============================================================================
# CHARGEMENT D'UN MODÈLE ENTRAÎNÉ
# ============================================================================

def load_model(
    chemin_modele: Path,
    device: torch.device | None = None,
) -> tuple[TransformerHawkes, FeatureStats, list[str], TPPConfig]:
    """Recharge un modèle entraîné depuis un .pth."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(chemin_modele, map_location=device, weights_only=False)
    tpp_cfg = TPPConfig(**blob["tpp_config"])
    model = TransformerHawkes(tpp_cfg).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    feature_stats = FeatureStats.from_dict(blob["feature_stats"])
    event_types = blob["event_types"]
    return model, feature_stats, event_types, tpp_cfg
