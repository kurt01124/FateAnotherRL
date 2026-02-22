"""TorchScript-compatible model export for FateModel.

FateModelExport wraps the same architecture as FateModel but:
- Takes positional tensor args instead of dicts (TorchScript requirement)
- Returns raw logits/means (no sampling, no distributions)
- Is fully scriptable via torch.jit.script()

Usage:
    from fateanother_rl.model.export import export_model
    export_model(trained_fate_model, "fate_model.pt")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from fateanother_rl.model.encoder import SelfEncoder, UnitEncoder, GridEncoder
from fateanother_rl.data.constants import GRID_CHANNELS


class FateModelExport(nn.Module):
    """TorchScript-compatible inference model.

    Takes positional tensor args, returns tuple of logits/means/value/hidden.

    Input shapes (B = batch size):
        self_vec:              (B, 77)
        ally_vec:              (B, N_ally, 37)
        enemy_vec:             (B, N_enemy, 43)
        global_vec:            (B, 6)
        grid:                  (B, 6, H, W)
        hx_h:                  (1, B, 256)
        hx_c:                  (1, B, 256)
        mask_*:                (B, head_size) bool tensors

    Output tuple:
        skill_logits           (B, 8)
        unit_target_logits     (B, 14)
        skill_levelup_logits   (B, 6)
        stat_upgrade_logits    (B, 10)
        attribute_logits       (B, 5)
        item_buy_logits        (B, 18)
        item_use_logits        (B, 7)
        seal_use_logits        (B, 7)
        faire_send_logits      (B, 6)
        faire_request_logits   (B, 6)
        faire_respond_logits   (B, 3)
        move_mean              (B, 2)
        move_logstd            (2,)
        point_mean             (B, 2)
        point_logstd           (2,)
        value                  (B,)
        new_h                  (1, B, 256)
        new_c                  (1, B, 256)
    """

    def __init__(
        self,
        self_dim: int = 77,
        ally_dim: int = 37,
        enemy_dim: int = 43,
        global_dim: int = 6,
        grid_channels: int = GRID_CHANNELS,
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

        # --- Discrete Heads (11) — explicit attributes for TorchScript ---
        self.head_skill = nn.Linear(hidden_dim, 8)
        self.head_unit_target = nn.Linear(hidden_dim, 14)
        self.head_skill_levelup = nn.Linear(hidden_dim, 6)
        self.head_stat_upgrade = nn.Linear(hidden_dim, 10)
        self.head_attribute = nn.Linear(hidden_dim, 5)
        self.head_item_buy = nn.Linear(hidden_dim, 18)
        self.head_item_use = nn.Linear(hidden_dim, 7)
        self.head_seal_use = nn.Linear(hidden_dim, 7)
        self.head_faire_send = nn.Linear(hidden_dim, 6)
        self.head_faire_request = nn.Linear(hidden_dim, 6)
        self.head_faire_respond = nn.Linear(hidden_dim, 3)

        # --- Continuous Heads (2) ---
        self.move_mean = nn.Linear(hidden_dim, 2)
        self.move_logstd = nn.Parameter(torch.zeros(2))

        self.point_mean = nn.Linear(hidden_dim, 2)
        self.point_logstd = nn.Parameter(torch.zeros(2))

        # --- Value Head ---
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        self_vec: torch.Tensor,       # (B, 77)
        ally_vec: torch.Tensor,       # (B, N_ally, 37)
        enemy_vec: torch.Tensor,      # (B, N_enemy, 41)
        global_vec: torch.Tensor,     # (B, 6)
        grid: torch.Tensor,           # (B, 3, H, W)
        hx_h: torch.Tensor,           # (1, B, 256)
        hx_c: torch.Tensor,           # (1, B, 256)
        mask_skill: torch.Tensor,
        mask_unit_target: torch.Tensor,
        mask_skill_levelup: torch.Tensor,
        mask_stat_upgrade: torch.Tensor,
        mask_attribute: torch.Tensor,
        mask_item_buy: torch.Tensor,
        mask_item_use: torch.Tensor,
        mask_seal_use: torch.Tensor,
        mask_faire_send: torch.Tensor,
        mask_faire_request: torch.Tensor,
        mask_faire_respond: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # skill_logits
        torch.Tensor,  # unit_target_logits
        torch.Tensor,  # skill_levelup_logits
        torch.Tensor,  # stat_upgrade_logits
        torch.Tensor,  # attribute_logits
        torch.Tensor,  # item_buy_logits
        torch.Tensor,  # item_use_logits
        torch.Tensor,  # seal_use_logits
        torch.Tensor,  # faire_send_logits
        torch.Tensor,  # faire_request_logits
        torch.Tensor,  # faire_respond_logits
        torch.Tensor,  # move_mean
        torch.Tensor,  # move_logstd
        torch.Tensor,  # point_mean
        torch.Tensor,  # point_logstd
        torch.Tensor,  # value
        torch.Tensor,  # new_h
        torch.Tensor,  # new_c
    ]:
        # --- Encode ---
        s = self.self_enc(self_vec)                   # (B, 128)
        a = self.ally_enc(ally_vec)                   # (B, 128)
        e = self.enemy_enc(enemy_vec)                 # (B, 128)
        g = self.global_fc(global_vec)                # (B, 32)
        m = self.grid_enc(grid)                       # (B, 128)

        x = torch.cat([s, a, e, m, g], dim=-1)       # (B, 544)
        x = self.pre_lstm(x).unsqueeze(1)             # (B, 1, 256)

        # --- LSTM ---
        hx: Tuple[torch.Tensor, torch.Tensor] = (hx_h, hx_c)
        lstm_out, (new_h, new_c) = self.lstm(x, hx)
        x = lstm_out.squeeze(1)                       # (B, 256)

        # --- Discrete Heads with masking ---
        # Mask: zero-out invalid actions by filling with -1e8 before softmax.
        # Use .logical_not() instead of ~mask.bool() for TorchScript compatibility.
        skill_logits = self.head_skill(x).masked_fill(mask_skill.logical_not(), -1e8)
        unit_target_logits = self.head_unit_target(x).masked_fill(mask_unit_target.logical_not(), -1e8)
        skill_levelup_logits = self.head_skill_levelup(x).masked_fill(mask_skill_levelup.logical_not(), -1e8)
        stat_upgrade_logits = self.head_stat_upgrade(x).masked_fill(mask_stat_upgrade.logical_not(), -1e8)
        attribute_logits = self.head_attribute(x).masked_fill(mask_attribute.logical_not(), -1e8)
        item_buy_logits = self.head_item_buy(x).masked_fill(mask_item_buy.logical_not(), -1e8)
        item_use_logits = self.head_item_use(x).masked_fill(mask_item_use.logical_not(), -1e8)
        seal_use_logits = self.head_seal_use(x).masked_fill(mask_seal_use.logical_not(), -1e8)
        faire_send_logits = self.head_faire_send(x).masked_fill(mask_faire_send.logical_not(), -1e8)
        faire_request_logits = self.head_faire_request(x).masked_fill(mask_faire_request.logical_not(), -1e8)
        faire_respond_logits = self.head_faire_respond(x).masked_fill(mask_faire_respond.logical_not(), -1e8)

        # --- Continuous Heads ---
        move_mean = torch.tanh(self.move_mean(x))     # (B, 2)
        point_mean = torch.tanh(self.point_mean(x))   # (B, 2)

        # --- Value ---
        value = self.value_head(x).squeeze(-1)        # (B,)

        return (
            skill_logits,
            unit_target_logits,
            skill_levelup_logits,
            stat_upgrade_logits,
            attribute_logits,
            item_buy_logits,
            item_use_logits,
            seal_use_logits,
            faire_send_logits,
            faire_request_logits,
            faire_respond_logits,
            move_mean,
            self.move_logstd.clamp(-2.0, 0.5),
            point_mean,
            self.point_logstd.clamp(-2.0, 0.5),
            value,
            new_h,
            new_c,
        )


# --- Key mapping from FateModel.discrete_heads to FateModelExport explicit attrs ---
# FateModel stores heads in ModuleDict keyed by DISCRETE_HEADS order.
# We replicate that order here so state_dict keys are predictable.
_HEAD_KEY_MAP = {
    "discrete_heads.skill": "head_skill",
    "discrete_heads.unit_target": "head_unit_target",
    "discrete_heads.skill_levelup": "head_skill_levelup",
    "discrete_heads.stat_upgrade": "head_stat_upgrade",
    "discrete_heads.attribute": "head_attribute",
    "discrete_heads.item_buy": "head_item_buy",
    "discrete_heads.item_use": "head_item_use",
    "discrete_heads.seal_use": "head_seal_use",
    "discrete_heads.faire_send": "head_faire_send",
    "discrete_heads.faire_request": "head_faire_request",
    "discrete_heads.faire_respond": "head_faire_respond",
}


def _convert_state_dict(src_sd: dict) -> dict:
    """Convert FateModel state_dict keys to FateModelExport keys.

    FateModel uses ModuleDict 'discrete_heads.{name}.{weight/bias}'.
    FateModelExport uses 'head_{name}.{weight/bias}'.
    All other keys (encoders, lstm, move_mean, etc.) are identical.
    """
    dst: dict = {}
    for k, v in src_sd.items():
        matched = False
        for src_prefix, dst_prefix in _HEAD_KEY_MAP.items():
            if k.startswith(src_prefix + "."):
                suffix = k[len(src_prefix):]  # e.g. ".weight"
                dst[dst_prefix + suffix] = v
                matched = True
                break
        if not matched:
            dst[k] = v
    return dst


def export_model(
    model_or_state_dict,
    path: str,
    device: str = "cpu",
) -> None:
    """Export FateModel weights to TorchScript (.pt) format.

    Args:
        model_or_state_dict: FateModel instance OR its state_dict (dict).
        path:   Output file path (e.g. "fate_model.pt").
        device: Target device for the exported model ("cpu" or "cuda").

    The exported file can be loaded in any environment (C++, Python without
    fateanother_rl installed) via:
        model = torch.jit.load("fate_model.pt")
    """
    # Obtain state dict
    if isinstance(model_or_state_dict, dict):
        src_sd = model_or_state_dict
    else:
        src_sd = model_or_state_dict.state_dict()

    # Build export model
    export = FateModelExport()
    converted = _convert_state_dict(src_sd)
    export.load_state_dict(converted, strict=True)
    export = export.to(device).eval()
    export.lstm.flatten_parameters()

    # Script and save
    scripted = torch.jit.script(export)
    scripted.save(path)
    print(f"[export] Saved TorchScript model → {path}")
