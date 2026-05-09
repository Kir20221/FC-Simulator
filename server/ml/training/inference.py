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

Calibration de lambda_max :
- Initialement, on sonde l'intensité totale en N_PROBE points uniformément
  répartis sur [t0, T_obs] et on prend le max × safety_factor. Évite de
  démarrer avec une borne 5 ordres de grandeur trop large quand le modèle
  produit des intensités très basses.
- Après chaque acceptation, on resonde l'horizon restant pour réajuster.
- Le rebond classique (lambda_max := Lambda_total * safety_factor si
  Lambda_total > lambda_max) reste en place comme filet de sécurité.

Garde-fou max_iterations : limite stricte du nombre de candidats évalués
par appel, pour éviter qu'une planète pathologique bloque toute la
simulation.
"""

from dataclasses import dataclass
import numpy as np
import torch

from .data import Batch, PlanetSample, collate_fn
from .model import TransformerHawkes


# Nombre de points de sondage pour calibrer lambda_max au début et après
# chaque acceptation. Coût = N_PROBE forwards modèle, négligeable face à
# une boucle de thinning non calibrée.
N_PROBE = 16


@dataclass
class GeneratedSequence:
    times: list[int]    # timecodes en unités d'unité de timecode
    types: list[int]    # event_type_id


def _probe_lambda_max(
    model: TransformerHawkes,
    sample: PlanetSample,
    t_start: float,
    t_end: float,
    device: torch.device,
    safety_factor: float,
) -> float:
    """
    Estime une borne supérieure raisonnable de Lambda_total sur [t_start, t_end]
    en évaluant le modèle en N_PROBE points uniformément répartis. Multiplie
    le max observé par safety_factor pour la marge.

    Si t_end <= t_start (cas dégénéré en fin de boucle), retourne 1e-8 pour
    éviter la division par zéro côté appelant.
    """
    if t_end <= t_start:
        return 1e-8

    batch = collate_fn([sample]).to(device)
    qs = torch.linspace(t_start, t_end, steps=N_PROBE, device=device).unsqueeze(0)
    # intensities_at_query attend shape (B, Q) ; on a B=1, Q=N_PROBE.
    lam = model.intensities_at_query(batch, qs)[0]   # (Q, K)
    Lambda_total = lam.sum(dim=-1)                   # (Q,)
    lambda_max_estime = float(Lambda_total.max().item())
    return max(lambda_max_estime * safety_factor, 1e-8)


@torch.no_grad()
def sample_sequence(
    model: TransformerHawkes,
    sample: PlanetSample,
    device: torch.device,
    safety_factor: float = 1.5,
    max_events: int = 50,
    max_iterations: int = 10_000,
) -> GeneratedSequence:
    """
    Échantillonne une séquence pour une planète donnée par thinning d'Ogata.

    - safety_factor : facteur de marge appliqué au lambda_max estimé.
    - max_events    : nombre max d'événements générés (sécurité).
    - max_iterations: nombre max de candidats évalués (sécurité contre les
                      cas où le modèle produit des intensités si faibles
                      que la boucle n'avance pas).
    """
    model.eval()
    K = model.cfg.nb_types_evt
    T_obs = sample.T_obs

    times: list[int] = list(sample.times.tolist())
    types: list[int] = list(sample.types.tolist())

    t = float(times[-1]) if times else 0.0

    # Calibration initiale de lambda_max par sondage.
    current_sample = PlanetSample(
        cont=sample.cont,
        cat=sample.cat,
        times=np.array(times, dtype=np.int64),
        types=np.array(types, dtype=np.int64),
        T_obs=sample.T_obs,
    )
    lambda_max = _probe_lambda_max(
        model, current_sample, t, float(T_obs), device, safety_factor,
    )

    nb_iterations = 0

    while (
        t < T_obs
        and len(times) - len(sample.times) < max_events
        and nb_iterations < max_iterations
    ):
        nb_iterations += 1

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

        # Rebond : si l'intensité dépasse lambda_max, on relève la borne.
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

            # Recalibrer lambda_max sur le reste de l'horizon, en tenant
            # compte du nouvel historique.
            current_sample = PlanetSample(
                cont=sample.cont,
                cat=sample.cat,
                times=np.array(times, dtype=np.int64),
                types=np.array(types, dtype=np.int64),
                T_obs=sample.T_obs,
            )
            lambda_max = _probe_lambda_max(
                model, current_sample, t_candidate, float(T_obs),
                device, safety_factor,
            )
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
