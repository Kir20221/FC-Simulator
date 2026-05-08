"""
Échantillonnage de séquences à partir d'un modèle TPP entraîné.

Algorithme : thinning d'Ogata (1981). Étant donnée une borne supérieure
locale lambda_max sur l'intensité totale, on tire des candidats à intervalles
exponentiels avec taux lambda_max, puis on accepte chacun avec probabilité
Lambda_total(t) / lambda_max. Si accepté, le type est tiré multinomial sur
{lambda_k(t) / Lambda_total(t)}.

Implémentation : pour chaque planète du batch, on génère une séquence
indépendamment (pas de batching massif sur le thinning, car la longueur
de séquence varie). Pour 100 à 1000 planètes ça reste rapide. Pour des
inférences à très grande échelle, on parallélisera ultérieurement.

Le thinning utilise un lambda_max recalculé périodiquement (toutes les
'rebound_steps' itérations) en évaluant l'intensité courante et en y
appliquant un facteur de sécurité, ce qui est une stratégie standard
quand on n'a pas de borne théorique.
"""

from dataclasses import dataclass
import numpy as np
import torch

from .data import Batch, PlanetSample, collate_fn
from .model import TransformerHawkes


@dataclass
class GeneratedSequence:
    times: list[int]    # timecodes en unités d'unité de timecode
    types: list[int]    # event_type_id


@torch.no_grad()
def sample_sequence(
    model: TransformerHawkes,
    sample: PlanetSample,
    device: torch.device,
    lambda_max_init: float = 1.0,
    safety_factor: float = 1.5,
    max_events: int = 50,
) -> GeneratedSequence:
    """
    Échantillonne une séquence pour une planète donnée par thinning d'Ogata.

    - lambda_max_init : borne initiale sur l'intensité totale. Recalibrée
      périodiquement à partir de l'intensité observée multipliée par
      safety_factor.
    - max_events : sécurité contre les cas pathologiques où le modèle
      diverge et émet trop d'événements.
    """
    model.eval()
    K = model.cfg.nb_types_evt
    T_obs = sample.T_obs

    times: list[int] = list(sample.times.tolist())
    types: list[int] = list(sample.types.tolist())

    t = float(times[-1]) if times else 0.0
    lambda_max = float(lambda_max_init)

    while t < T_obs and len(times) - len(sample.times) < max_events:
        # Intervalle exponentiel jusqu'au prochain candidat.
        dt = float(np.random.exponential(1.0 / max(lambda_max, 1e-8)))
        t_candidate = t + dt
        if t_candidate >= T_obs:
            break

        # Construction du Batch courant (1 planète) avec l'historique à jour.
        current_sample = PlanetSample(
            cont=sample.cont,
            cat=sample.cat,
            times=np.array(times, dtype=np.int64),
            types=np.array(types, dtype=np.int64),
            T_obs=sample.T_obs,
        )
        batch = collate_fn([current_sample]).to(device)

        # Évaluer l'intensité au temps t_candidate.
        q = torch.tensor([[t_candidate]], device=device, dtype=torch.float32)
        lam = model.intensities_at_query(batch, q)[0, 0]             # (K,)
        Lambda_total = lam.sum().item()

        # Si l'intensité observée dépasse lambda_max, on "rebondit" : on
        # rejette et on relève lambda_max pour les prochaines itérations.
        if Lambda_total > lambda_max:
            lambda_max = Lambda_total * safety_factor
            t = t_candidate
            continue

        # Acceptation/rejet.
        u = np.random.uniform()
        if u <= Lambda_total / lambda_max:
            # Accepter : tirer le type selon les intensités.
            probs = (lam / lam.sum()).cpu().numpy()
            k = int(np.random.choice(K, p=probs))
            times.append(int(t_candidate))
            types.append(k)
            # Réajuster lambda_max périodiquement (avec safety) pour ne pas
            # rester bloqué sur une vieille borne.
            lambda_max = max(lambda_max * 0.95, Lambda_total * safety_factor)
        # Que la décision soit accepter ou rejeter, on avance le temps.
        t = t_candidate

    # On retourne uniquement les événements générés (pas l'historique fourni).
    nb_init = len(sample.times)
    return GeneratedSequence(
        times=times[nb_init:],
        types=types[nb_init:],
    )


@torch.no_grad()
def sample_planets(
    model: TransformerHawkes,
    samples: list[PlanetSample],
    device: torch.device,
    **kwargs,
) -> list[GeneratedSequence]:
    """Échantillonne une séquence pour chaque planète d'une liste."""
    return [sample_sequence(model, s, device, **kwargs) for s in samples]
