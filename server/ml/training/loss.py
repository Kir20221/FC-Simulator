"""
Log-vraisemblance d'un Transformer Hawkes Process.

Pour une séquence d'événements {(t_i, k_i)} observée sur [0, T_obs] :

  log L = sum_i log lambda_{k_i}*(t_i)  -  integrale sur [0, T_obs] de Lambda*(s) ds

où Lambda*(s) = sum_k lambda_k*(s).

Le terme intégral est approximé par Monte Carlo : on tire N points
uniformément sur [0, T_obs] et on estime l'intégrale par
T_obs * mean(Lambda(t_n)).

La loss = -log L moyenné sur le batch (par planète, pas par événement,
pour ne pas désavantager les planètes longues).
"""

import torch

from .data import Batch
from .model import TransformerHawkes


def tpp_negative_log_likelihood(
    model: TransformerHawkes,
    batch: Batch,
    nb_mc_samples: int = 32,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Retourne :
    - la loss scalaire (- log L moyennée sur le batch),
    - un dict de métriques pour suivi (terme événements, terme intégral,
      log-vraisemblance par événement et par planète).
    """
    device = batch.cont.device
    B = batch.cont.shape[0]

    # ----------------------------------------------------------------
    # Terme 1 : sum_i log lambda_{k_i}*(t_i)
    # ----------------------------------------------------------------
    intensities_evt = model.intensities_at_events(batch)             # (B, L, K)

    # Sélectionner pour chaque position i l'intensité du type observé.
    # batch.types : (B, L) ; on gather sur la dernière dim.
    log_lambda_obs = torch.log(intensities_evt + 1e-12)              # (B, L, K)
    log_lambda_at_type = torch.gather(
        log_lambda_obs, dim=-1, index=batch.types.unsqueeze(-1)
    ).squeeze(-1)                                                    # (B, L)
    # Masquer les positions de padding.
    log_lambda_at_type = log_lambda_at_type * batch.seq_mask.float()
    terme_events = log_lambda_at_type.sum(dim=1)                     # (B,)

    # ----------------------------------------------------------------
    # Terme 2 : integrale_0^T_obs Lambda*(s) ds  (Monte Carlo)
    # ----------------------------------------------------------------
    T_obs_f = batch.T_obs.float()                                    # (B,)
    # Tirage uniforme sur [0, T_obs_b] pour chaque b du batch.
    u = torch.rand(B, nb_mc_samples, device=device)                  # (B, M)
    query_times = u * T_obs_f.unsqueeze(1)                           # (B, M)

    intensities_mc = model.intensities_at_query(batch, query_times)  # (B, M, K)
    Lambda_mc = intensities_mc.sum(dim=-1)                           # (B, M)
    # MC : E[f(U)] * T = integrale sur [0, T]. Donc T * mean(Lambda(U)).
    terme_integral = T_obs_f * Lambda_mc.mean(dim=1)                 # (B,)

    # ----------------------------------------------------------------
    # Log-vraisemblance et loss
    # ----------------------------------------------------------------
    log_L = terme_events - terme_integral                            # (B,)
    loss = -log_L.mean()

    # Métriques de suivi.
    nb_events = batch.seq_lengths.sum().item()
    metrics = {
        "loss": loss.item(),
        "terme_events_moy": terme_events.mean().item(),
        "terme_integral_moy": terme_integral.mean().item(),
        "log_L_par_planete": log_L.mean().item(),
        "log_L_par_event": (
            log_L.sum().item() / nb_events if nb_events > 0 else 0.0
        ),
    }
    return loss, metrics
