"""FateModel: Shared policy network with LSTM and mixed action space.

Architecture:
  Encoders: self(128) + ally(128) + enemy(128) + grid(128) + global(32) = 544
  pre_lstm: 544 → 256
  LSTM: 256 → 256 (1 layer, batch_first)
  11× DiscreteHead + 2× ContinuousHead (move, point) + ValueHead(→1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from fateanother_rl.data.constants import DISCRETE_HEADS
from fateanother_rl.model.encoder import SelfEncoder, UnitEncoder, GridEncoder
from fateanother_rl.model.action_utils import apply_mask


class FateModel(nn.Module):
    """Single shared policy for all heroes.

    Uses hero_id one-hot in self_vec for hero-specific behavior.
    LSTM maintains temporal context per agent.
    Mixed action space: 11 discrete + 2 continuous heads.
    """

    def __init__(
        self,
        self_dim: int = 77,
        ally_dim: int = 37,
        enemy_dim: int = 43,
        global_dim: int = 6,
        grid_channels: int = 3,
        hidden_dim: int = 256,
        encoder_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Encoders ---
        self.self_enc = SelfEncoder(input_dim=self_dim, hidden=encoder_dim)
        self.ally_enc = UnitEncoder(input_dim=ally_dim, hidden=encoder_dim)
        self.enemy_enc = UnitEncoder(input_dim=enemy_dim, hidden=encoder_dim)
        self.grid_enc = GridEncoder(in_channels=grid_channels, out_dim=encoder_dim)
        self.global_fc = nn.Sequential(nn.Linear(global_dim, 32), nn.ReLU())

        # --- Core ---
        concat_dim = encoder_dim * 4 + 32  # 128*4 + 32 = 544
        self.pre_lstm = nn.Sequential(nn.Linear(concat_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        # --- Discrete Heads (11) ---
        self.discrete_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, size)
            for name, size in DISCRETE_HEADS.items()
        })

        # --- Continuous Heads (2) ---
        # Move: 2D direction in [-1, 1]
        self.move_mean = nn.Linear(hidden_dim, 2)
        self.move_logstd = nn.Parameter(torch.zeros(2))

        # Point: 2D target in [-1, 1]
        self.point_mean = nn.Linear(hidden_dim, 2)
        self.point_logstd = nn.Parameter(torch.zeros(2))

        # --- Value Head ---
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (PPO standard)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Policy heads: small init for near-uniform initial policy
        for head in self.discrete_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        nn.init.orthogonal_(self.move_mean.weight, gain=0.01)
        nn.init.zeros_(self.move_mean.bias)
        nn.init.orthogonal_(self.point_mean.weight, gain=0.01)
        nn.init.zeros_(self.point_mean.bias)

        # Value head: gain=1.0
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def _encode(self, obs: dict) -> torch.Tensor:
        """Encode observations → pre-LSTM vector.

        Args:
            obs: dict with keys self_vec(B,D), ally_vec(B,5,D),
                 enemy_vec(B,6,D), global_vec(B,D), grid(B,3,H,W)
        Returns:
            (B, hidden_dim) tensor
        """
        s = self.self_enc(obs["self_vec"])        # (B, 128)
        a = self.ally_enc(obs["ally_vec"])         # (B, 128)
        e = self.enemy_enc(obs["enemy_vec"])       # (B, 128)
        g = self.global_fc(obs["global_vec"])      # (B, 32)
        m = self.grid_enc(obs["grid"])             # (B, 128)
        x = torch.cat([s, a, e, m, g], dim=-1)    # (B, 544)
        return self.pre_lstm(x)                     # (B, 256)

    def forward(self, obs, hx, masks, deterministic=False):
        """Single-step forward (inference/rollout).

        Args:
            obs:   dict of (B, ...) tensors
            hx:    (h, c) each (1, B, hidden_dim)
            masks: dict[str, BoolTensor(B, N)]
            deterministic: if True, use mean/argmax instead of sampling

        Returns:
            actions:       dict[str, Tensor]
            total_log_prob: (B,)
            value:          (B,)
            total_entropy:  (B,)
            new_hx:         (h, c)
        """
        x = self._encode(obs).unsqueeze(1)    # (B, 1, 256)
        x, new_hx = self.lstm(x, hx)
        x = x.squeeze(1)                       # (B, 256)

        actions = {}
        log_probs = []
        entropies = []

        # Continuous: move
        move_mu = torch.tanh(self.move_mean(x))
        move_std = self.move_logstd.clamp(-2.0, 0.5).exp().expand_as(move_mu)
        move_dist = Normal(move_mu, move_std)
        if deterministic:
            actions["move"] = move_mu
        else:
            actions["move"] = move_dist.sample()
        log_probs.append(move_dist.log_prob(actions["move"]).sum(-1))
        entropies.append(move_dist.entropy().sum(-1))

        # Continuous: point
        point_mu = torch.tanh(self.point_mean(x))
        point_std = self.point_logstd.clamp(-2.0, 0.5).exp().expand_as(point_mu)
        point_dist = Normal(point_mu, point_std)
        if deterministic:
            actions["point"] = point_mu
        else:
            actions["point"] = point_dist.sample()
        log_probs.append(point_dist.log_prob(actions["point"]).sum(-1))
        entropies.append(point_dist.entropy().sum(-1))

        # Discrete heads (masked)
        for name, head in self.discrete_heads.items():
            logits = head(x)
            if name in masks:
                logits = apply_mask(logits, masks[name])
            dist = Categorical(logits=logits)
            if deterministic:
                actions[name] = dist.probs.argmax(-1)
            else:
                actions[name] = dist.sample()
            log_probs.append(dist.log_prob(actions[name]))
            entropies.append(dist.entropy())

        value = self.value_head(x).squeeze(-1)     # (B,)
        total_lp = torch.stack(log_probs).sum(0)   # (B,)
        total_ent = torch.stack(entropies).sum(0)   # (B,)

        return actions, total_lp, value, total_ent, new_hx

    def forward_sequence(self, obs_seq, hx_init, masks_seq, actions_seq):
        """Sequence forward for PPO training (BPTT).

        Args:
            obs_seq:     dict, each value shape (B, T, ...)
            hx_init:     (h, c) each (1, B, hidden_dim) — detached
            masks_seq:   dict, each value shape (B, T, ...)
            actions_seq: dict, each value shape (B, T, ...)

        Returns:
            log_probs: (B, T)
            values:    (B, T)
            entropies: (B, T)
        """
        B, T = obs_seq["self_vec"].shape[:2]

        # Flatten (B, T, ...) → (B*T, ...) for encoding
        flat_obs = {}
        for k, v in obs_seq.items():
            if k == "masks":
                continue
            flat_obs[k] = v.reshape(B * T, *v.shape[2:])

        # Encode all timesteps
        x_seq = self._encode(flat_obs).reshape(B, T, self.hidden_dim)

        # LSTM over full sequence
        lstm_out, _ = self.lstm(x_seq, hx_init)  # (B, T, hidden_dim)
        x = lstm_out.reshape(B * T, self.hidden_dim)

        log_probs = []
        entropies = []

        # Continuous heads
        for prefix, mean_layer, logstd in [
            ("move", self.move_mean, self.move_logstd),
            ("point", self.point_mean, self.point_logstd),
        ]:
            mu = torch.tanh(mean_layer(x))
            std = logstd.clamp(-2.0, 0.5).exp().expand_as(mu)
            dist = Normal(mu, std)
            act = actions_seq[prefix].reshape(B * T, 2)
            log_probs.append(dist.log_prob(act).sum(-1))
            entropies.append(dist.entropy().sum(-1))

        # Discrete heads
        for name, head in self.discrete_heads.items():
            logits = head(x)
            if name in masks_seq:
                logits = apply_mask(logits, masks_seq[name].reshape(B * T, -1))
            dist = Categorical(logits=logits)
            act = actions_seq[name].reshape(B * T)
            log_probs.append(dist.log_prob(act))
            entropies.append(dist.entropy())

        values = self.value_head(x).squeeze(-1).reshape(B, T)
        total_lp = torch.stack(log_probs).sum(0).reshape(B, T)
        total_ent = torch.stack(entropies).sum(0).reshape(B, T)

        return total_lp, values, total_ent

    @staticmethod
    def init_hidden(batch_size: int, hidden_dim: int = 256,
                    device: str | torch.device = "cpu"):
        """Create zero-initialized LSTM hidden state."""
        h = torch.zeros(1, batch_size, hidden_dim, device=device)
        c = torch.zeros(1, batch_size, hidden_dim, device=device)
        return (h, c)
