"""Action utility functions: masking and JSON conversion."""

import torch
from fateanother_rl.data.constants import DISCRETE_HEADS


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply boolean mask to logits.

    Args:
        logits: (B, N) float tensor
        mask:   (B, N) bool tensor, True=allowed, False=blocked

    Returns:
        Masked logits with blocked actions set to -1e8.
    """
    return logits.masked_fill(~mask.bool(), -1e8)


def _to_float(v) -> float:
    return float(v.item() if hasattr(v, 'item') else v)


def _to_int(v) -> int:
    return int(v.item() if hasattr(v, 'item') else v)


def actions_to_json(actions: dict, agent_indices: list[int]) -> dict:
    """Convert actions → C# v2 ACTION JSON format.

    Accepts TWO formats:
      1) Per-agent dict: {agent_idx: {"move": [x,y], "skill": int, ...}}
      2) Batched tensors: {"move": tensor(B,2), "skill": tensor(B,), ...}
    """
    action_list = []

    # Detect format: if first key is int → per-agent dict
    first_key = next(iter(actions), None)
    per_agent = isinstance(first_key, int)

    for i, idx in enumerate(agent_indices):
        entry = {"idx": idx}

        if per_agent:
            a = actions.get(idx, {})
            # Move
            mv = a.get("move")
            if mv is not None:
                entry["move_x"] = _to_float(mv[0])
                entry["move_y"] = _to_float(mv[1])
            else:
                entry["move_x"] = 0.0
                entry["move_y"] = 0.0
            # Point
            pt = a.get("point")
            if pt is not None:
                entry["point_x"] = _to_float(pt[0])
                entry["point_y"] = _to_float(pt[1])
            else:
                entry["point_x"] = 0.0
                entry["point_y"] = 0.0
            # Discrete heads
            for name in DISCRETE_HEADS:
                entry[name] = _to_int(a.get(name, 0))
        else:
            # Batched tensor format
            if "move" in actions:
                mv = actions["move"][i]
                entry["move_x"] = _to_float(mv[0])
                entry["move_y"] = _to_float(mv[1])
            else:
                entry["move_x"] = 0.0
                entry["move_y"] = 0.0

            if "point" in actions:
                pt = actions["point"][i]
                entry["point_x"] = _to_float(pt[0])
                entry["point_y"] = _to_float(pt[1])
            else:
                entry["point_x"] = 0.0
                entry["point_y"] = 0.0

            for name in DISCRETE_HEADS:
                if name in actions:
                    entry[name] = _to_int(actions[name][i])
                else:
                    entry[name] = 0

        action_list.append(entry)

    return {"actions": action_list}
