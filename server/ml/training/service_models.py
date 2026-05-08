"""
Couche service pour les modèles entraînés.

Orchestration : invoque le trainer (cœur métier) + persiste les métadonnées
en DB. Appelée indifféremment par api_models.py (HTTP) ou cli_models.py (CLI).

L'appelant fournit la connexion DB (cohérence avec le pattern existant
sur les datasets).
"""

from datetime import datetime, timezone
from pathlib import Path

import torch
from psycopg import Connection
from pydantic import BaseModel, Field

from ..dataset import Dataset
from ..service import DatasetIntrouvableError
from ..generator.generate import RACINE_DATASETS
from ..generator.life_model import EVENT_TYPES
from .model_record import Model
from .trainer import TrainingConfig, train


# ============================================================================
# REQUEST / EXCEPTIONS
# ============================================================================

class ModelTrainingRequest(BaseModel):
    """Paramètres pour lancer un entraînement."""
    nom: str = Field(..., description="Identifiant du modèle (clé primaire).")
    dataset_nom: str = Field(..., description="Nom du dataset utilisé pour l'entraînement.")
    # Hyperparamètres : tous optionnels, défauts dans TrainingConfig.
    batch_size: int | None = None
    nb_epochs: int | None = None
    learning_rate: float | None = None
    val_fraction: float | None = None
    d_model: int | None = None
    n_heads: int | None = None
    n_layers: int | None = None
    dim_feedforward: int | None = None
    dropout: float | None = None
    nb_mc_samples: int | None = None
    seed_split: int | None = None


class ModelExisteDejaError(Exception):
    pass


class ModelIntrouvableError(Exception):
    pass


# ============================================================================
# CHEMIN .pth
# ============================================================================

def chemin_modele(nom: str) -> Path:
    return RACINE_DATASETS / f"{nom}.pth"


# ============================================================================
# CRÉATION (= ENTRAÎNEMENT)
# ============================================================================

def _construire_train_config(req: ModelTrainingRequest) -> TrainingConfig:
    """Construit une TrainingConfig en n'écrasant que les champs fournis."""
    cfg = TrainingConfig()
    for champ in [
        "batch_size", "nb_epochs", "learning_rate", "val_fraction",
        "d_model", "n_heads", "n_layers", "dim_feedforward", "dropout",
        "nb_mc_samples", "seed_split",
    ]:
        val = getattr(req, champ)
        if val is not None:
            setattr(cfg, champ, val)
    return cfg


def creer_modele(conn: Connection, request: ModelTrainingRequest) -> Model:
    """
    Lance l'entraînement, persiste le modèle en DB.

    Échoue si :
    - un modèle du même nom existe déjà,
    - le dataset référencé n'existe pas en DB.
    """
    if Model.get(conn, request.nom) is not None:
        raise ModelExisteDejaError(f"Le modèle '{request.nom}' existe déjà.")

    dataset = Dataset.get(conn, request.dataset_nom)
    if dataset is None:
        raise DatasetIntrouvableError(
            f"Le dataset '{request.dataset_nom}' est introuvable."
        )

    train_cfg = _construire_train_config(request)
    chemin_pth = chemin_modele(request.nom)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = train(
        chemin_planets=Path(dataset.chemin_planets),
        chemin_events=Path(dataset.chemin_events),
        event_types=EVENT_TYPES,
        chemin_modele_out=chemin_pth,
        train_cfg=train_cfg,
        device=device,
    )

    # Reconstruction du tpp_config depuis le .pth pour stockage en DB
    # (cohérent avec ce qui est sérialisé dans le fichier).
    blob = torch.load(chemin_pth, map_location="cpu", weights_only=False)
    tpp_config_dict = blob["tpp_config"]

    modele = Model(
        nom=request.nom,
        date_creation=datetime.now(timezone.utc),
        dataset_nom=request.dataset_nom,
        chemin_fichier=str(chemin_pth),
        taille_octets=chemin_pth.stat().st_size,
        duree_entrainement_s=result.duree_totale_s,
        training_config=train_cfg.__dict__,
        tpp_config=tpp_config_dict,
        metrics_par_epoch=result.metrics_par_epoch,
        final_train_loss=result.final_train_loss,
        final_val_loss=result.final_val_loss,
    )
    modele.create(conn)
    conn.commit()
    return modele


# ============================================================================
# LISTE / LECTURE / SUPPRESSION
# ============================================================================

def lister_modeles(conn: Connection) -> list[Model]:
    return Model.list_all(conn)


def lire_modele(conn: Connection, nom: str) -> Model:
    modele = Model.get(conn, nom)
    if modele is None:
        raise ModelIntrouvableError(f"Le modèle '{nom}' est introuvable.")
    return modele


def supprimer_modele(conn: Connection, nom: str) -> None:
    """Supprime le modèle en DB et le fichier .pth associé."""
    modele = Model.get(conn, nom)
    if modele is None:
        raise ModelIntrouvableError(f"Le modèle '{nom}' est introuvable.")

    chemin = chemin_modele(nom)
    if chemin.exists():
        chemin.unlink()
    Model.delete(conn, nom)
    conn.commit()
