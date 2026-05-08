"""
Transformer Hawkes Process (Zuo et al. 2020) conditionné sur features
statiques.

Architecture :
- Encodeur de features statiques : MLP qui combine continues + embeddings
  catégoriels en un vecteur de contexte c (B, D_model).
- Encodeur de séquence : embedding du type d'événement + encodage temporel
  sinusoïdal du timecode + injection du contexte c à chaque position.
  Puis nn.TransformerEncoder avec masque causal.
- Tête d'intensité : pour chaque type k, on calcule lambda_k(t | history)
  via projection linéaire + softplus.

Le modèle expose deux interfaces :
- forward_at_event_times(batch) : renvoie l'intensité à chaque événement
  observé (utilisé par la loss pour le terme sum log lambda*(t_i)).
- forward_at_query_times(features_context, history, query_times) :
  renvoie l'intensité à des instants arbitraires (utilisé pour le terme
  intégral Monte Carlo de la loss, et pour le thinning à l'inférence).

Le code est agnostique au nombre de types d'événements (paramètre K).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import Batch


# ============================================================================
# CONFIGURATION DU MODÈLE
# ============================================================================

@dataclass
class TPPConfig:
    # Dimensions
    dim_continues: int                  # nb de features continues normalisées
    dim_categorielles: dict[str, int]   # {nom: nb_categories}
    nb_types_evt: int                   # K, nb de types d'événements
    dim_emb_cat: int = 4                # dim d'embedding par feature catégorielle
    d_model: int = 64                   # dim cachée du Transformer
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    # Encodage temporel
    time_freqs: int = 16                # nb de fréquences sinusoïdales (= d_model/4 typiquement)
    # Normalisation temporelle : tous les timecodes sont divisés par t_ref
    # avant d'entrer dans le modèle. Permet d'éviter que le terme intégral
    # explose sur des fenêtres T_obs grandes (jusqu'à ~5e5 unités).
    # L'intensité apprise est alors lambda'(t') et lambda(t) = lambda'(t/t_ref) / t_ref.
    # La conversion est gérée dans la loss et l'inférence.
    t_ref: float = 1.0e5


# ============================================================================
# ENCODAGE TEMPOREL SINUSOÏDAL
# ============================================================================

class TimeEncoding(nn.Module):
    """
    Encodage sinusoïdal continu du temps (Vaswani-style mais sur des temps
    réels arbitraires, pas des positions discrètes).
    Pour chaque temps t (B, L), retourne un vecteur (B, L, 2 * time_freqs)
    en concaténant sin(omega_k * t) et cos(omega_k * t) pour K fréquences.
    Les omega_k sont log-espacées entre 1 et une fréquence haute, pour
    couvrir plusieurs ordres de grandeur (notre timecode peut aller de 0 à
    ~5e5 unités pour T_sim = 50 Ga).
    """

    def __init__(self, n_freqs: int, t_max: float = 10.0):
        super().__init__()
        # omega_k = 2*pi / lambda_k avec lambda_k log-espacé entre 1e-3 et t_max.
        # Les temps en entrée sont normalisés (typiquement dans [0, 5] après
        # division par t_ref), donc t_max=10 couvre largement la plage.
        log_periods = torch.linspace(math.log(1e-3), math.log(t_max), n_freqs)
        omegas = 2.0 * math.pi / torch.exp(log_periods)
        self.register_buffer("omegas", omegas)
        self.out_dim = 2 * n_freqs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B, L) en float (timecode converti)
        # retour : (B, L, 2 * n_freqs)
        t = t.unsqueeze(-1)                          # (B, L, 1)
        args = t * self.omegas                       # (B, L, n_freqs)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ============================================================================
# ENCODEUR DE FEATURES STATIQUES
# ============================================================================

class StaticFeaturesEncoder(nn.Module):
    """
    MLP qui combine features continues normalisées et embeddings catégoriels
    en un vecteur de contexte de dim d_model.
    """

    def __init__(self, cfg: TPPConfig):
        super().__init__()
        # Embeddings appris pour chaque feature catégorielle.
        self.cat_embs = nn.ModuleDict({
            nom: nn.Embedding(num, cfg.dim_emb_cat)
            for nom, num in cfg.dim_categorielles.items()
        })
        dim_in = cfg.dim_continues + cfg.dim_emb_cat * len(cfg.dim_categorielles)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, cont: torch.Tensor, cat: dict[str, torch.Tensor]) -> torch.Tensor:
        # cont : (B, F_cont) ; cat : {nom: (B,) int64}
        embs = [self.cat_embs[nom](cat[nom]) for nom in self.cat_embs.keys()]
        x = torch.cat([cont] + embs, dim=-1)         # (B, dim_in)
        return self.mlp(x)                           # (B, d_model)


# ============================================================================
# MODÈLE COMPLET
# ============================================================================

class TransformerHawkes(nn.Module):
    def __init__(self, cfg: TPPConfig):
        super().__init__()
        self.cfg = cfg

        self.static_enc = StaticFeaturesEncoder(cfg)
        self.type_emb = nn.Embedding(cfg.nb_types_evt, cfg.d_model)
        self.time_enc = TimeEncoding(cfg.time_freqs)

        # Projection : (type_emb + time_enc + static_context) -> d_model
        dim_in_seq = cfg.d_model + self.time_enc.out_dim + cfg.d_model
        self.input_proj = nn.Linear(dim_in_seq, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,  # plus stable pour les petits modèles
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Tête d'intensité : prédit K intensités positives à partir de l'état caché.
        # On combine : état caché du Transformer (B, L, d_model) + encodage temporel
        # de l'instant interrogé + contexte statique. La sortie passe par softplus
        # pour garantir lambda_k >= 0.
        self.intensity_head = nn.Sequential(
            nn.Linear(cfg.d_model + self.time_enc.out_dim + cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.nb_types_evt),
        )

    # ------------------------------------------------------------------
    # ENCODAGE DE LA SÉQUENCE (HISTORIQUE OBSERVÉ)
    # ------------------------------------------------------------------

    def _encode_sequence(self, batch: Batch, context: torch.Tensor) -> torch.Tensor:
        """
        Encode la séquence d'événements observés en états cachés (B, L, d_model).
        Au-dessus de la longueur réelle, les états sont calculés mais ignorés
        ensuite via le seq_mask.

        Le masque causal du Transformer est strict : la position i ne voit que
        les positions [0..i]. On ajoute aussi un masque de padding pour ignorer
        les positions au-delà de la longueur réelle.
        """
        B, L = batch.times.shape
        device = batch.times.device

        # Embeddings : type, temps (normalisé par t_ref), contexte.
        type_e = self.type_emb(batch.types)                          # (B, L, d_model)
        times_norm = batch.times.float() / self.cfg.t_ref            # (B, L)
        time_e = self.time_enc(times_norm)                           # (B, L, 2*F)
        ctx_e = context.unsqueeze(1).expand(-1, L, -1)               # (B, L, d_model)

        x = torch.cat([type_e, time_e, ctx_e], dim=-1)
        x = self.input_proj(x)                                       # (B, L, d_model)

        # Masque causal triangulaire (L, L), True pour positions futures à masquer.
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )
        # Padding mask : True aux positions de padding (à ignorer par l'attention).
        # batch.seq_mask est True aux positions valides, donc on inverse.
        key_padding_mask = ~batch.seq_mask

        # Cas où seq_lengths est entièrement nul pour certains samples du batch :
        # toutes les positions sont du padding pour ces samples, et le softmax
        # de l'attention sur 0 positions valides produit du NaN. Comme ces
        # positions sont de toute façon ignorées par seq_mask dans la loss,
        # on remplace les NaN par zéro pour ne pas propager.
        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h = torch.nan_to_num(h, nan=0.0)
        return h  # (B, L, d_model)

    # ------------------------------------------------------------------
    # INTENSITÉ AUX TEMPS DE REQUÊTE
    # ------------------------------------------------------------------

    def _intensity_from_state(
        self,
        hidden_state: torch.Tensor,    # (B, M, d_model) état le plus récent par requête
        query_times: torch.Tensor,     # (B, M) float, en timecode brut
        context: torch.Tensor,         # (B, d_model)
    ) -> torch.Tensor:
        """
        Calcule lambda_k(t) à partir de l'état caché et du temps interrogé.

        L'intensité retournée est exprimée *dans l'échelle des timecodes bruts*
        (donc un événement par unité de timecode). En interne, le modèle
        travaille en temps normalisé t' = t / t_ref ; l'intensité dans l'échelle
        normalisée est lambda'(t') ; la conversion vers l'échelle brute est
        lambda(t) = lambda'(t/t_ref) / t_ref. Cette division par t_ref est
        appliquée ici, ce qui rend transparente la normalisation pour la loss
        et l'inférence : tout consommateur reçoit une intensité dans l'échelle
        brute des timecodes.
        """
        times_norm = query_times / self.cfg.t_ref
        time_e = self.time_enc(times_norm)                           # (B, M, 2F)
        ctx_e = context.unsqueeze(1).expand(-1, query_times.shape[1], -1)
        z = torch.cat([hidden_state, time_e, ctx_e], dim=-1)
        logits = self.intensity_head(z)                              # (B, M, K)
        # softplus pour positivité ; division par t_ref pour repasser de
        # l'échelle normalisée à l'échelle brute des timecodes.
        return (torch.nn.functional.softplus(logits) + 1e-8) / self.cfg.t_ref

    # ------------------------------------------------------------------
    # FORWARD : intensités aux temps des événements observés
    # ------------------------------------------------------------------

    def intensities_at_events(self, batch: Batch) -> torch.Tensor:
        """
        Pour chaque événement observé en position i de la séquence :
        renvoie lambda_{1..K}(t_i | t_{<i}, features).

        Important pour la loss : à la position 0, l'historique est vide ;
        à la position i > 0, l'historique est constitué des positions [0..i-1].
        Donc l'intensité à la position i doit utiliser l'état caché de la
        position i-1 (le dernier état "qui sait jusqu'à i-1").

        On construit un état décalé : à chaque position i, on fournit l'état
        h[i-1], et pour i=0 on utilise un état nul (=> intensité de base
        conditionnée seulement sur les features statiques).
        """
        context = self.static_enc(batch.cont, batch.cat)             # (B, d_model)
        h = self._encode_sequence(batch, context)                    # (B, L, d_model)

        # Décalage : on prend h[:, :-1] pour les positions 1..L-1, et un état
        # nul pour la position 0.
        B, L, D = h.shape
        h_prev = torch.zeros_like(h)
        if L > 1:
            h_prev[:, 1:, :] = h[:, :-1, :]

        return self._intensity_from_state(h_prev, batch.times.float(), context)
        # (B, L, K)

    # ------------------------------------------------------------------
    # FORWARD : intensités à des temps quelconques (pour Monte Carlo + thinning)
    # ------------------------------------------------------------------

    def intensities_at_query(
        self,
        batch: Batch,
        query_times: torch.Tensor,    # (B, M) float, timecodes arbitraires
    ) -> torch.Tensor:
        """
        Pour chaque temps de requête t_q (par batch), retourne lambda_{1..K}(t_q).

        L'historique utilisé pour conditionner est l'ensemble des événements
        observés strictement antérieurs à t_q. Implémentation : on encode toute
        la séquence, puis pour chaque t_q on cherche l'index du dernier
        événement observé strictement avant t_q et on prend son état caché.
        Si aucun événement n'est antérieur, on prend un état nul.
        """
        context = self.static_enc(batch.cont, batch.cat)             # (B, d_model)
        h = self._encode_sequence(batch, context)                    # (B, L, d_model)
        B, L, D = h.shape
        M = query_times.shape[1]

        # times_valides[b, l] = batch.times[b, l] si position valide, sinon +inf.
        # On peut alors faire t_event < t_query sans confondre avec le padding.
        times_valides = torch.where(
            batch.seq_mask,
            batch.times.float(),
            torch.full_like(batch.times.float(), float("inf")),
        )                                                            # (B, L)

        # Pour chaque (b, m), compter combien d'événements ont t_event < t_query.
        # Ce nombre - 1 = index du dernier événement antérieur (ou -1 si aucun).
        # times_valides : (B, L) ; query_times : (B, M).
        # On compare par broadcast : (B, L, 1) < (B, 1, M) -> (B, L, M).
        cmp = times_valides.unsqueeze(-1) < query_times.unsqueeze(1) # (B, L, M)
        nb_anterieurs = cmp.sum(dim=1)                               # (B, M)
        last_idx = nb_anterieurs - 1                                 # -1 si aucun

        # On veut : h_prev[b, m] = h[b, last_idx[b, m]] si last_idx >= 0, sinon 0.
        zero_state = torch.zeros(B, 1, D, device=h.device, dtype=h.dtype)
        h_padded = torch.cat([zero_state, h], dim=1)                 # (B, L+1, D)
        gather_idx = (last_idx + 1).clamp(min=0, max=L)              # (B, M) dans [0, L]
        gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, D)
        h_prev = torch.gather(h_padded, dim=1, index=gather_idx_expanded)
        # h_prev : (B, M, D)

        return self._intensity_from_state(h_prev, query_times, context)
        # (B, M, K)
