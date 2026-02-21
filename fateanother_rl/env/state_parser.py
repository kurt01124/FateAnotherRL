"""v2 State parser: JSON → normalized numpy arrays.

Converts C# v2 STATE JSON (49 fields per unit) into per-agent observations.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fateanother_rl.data.constants import (
    N, HERO_IDS, HERO_TO_IDX, NUM_HEROES, DISCRETE_HEADS, SKILL_SLOTS,
    MAP_MIN_X, MAP_MIN_Y, CELL_SIZE, GRID_W, GRID_H,
    SELF_DIM, ALLY_DIM, ENEMY_DIM, GLOBAL_DIM, GRID_CHANNELS,
)
from fateanother_rl.data.item_data import item_index

logger = logging.getLogger(__name__)


@dataclass
class AgentObs:
    """Per-agent observation from one tick."""
    agent_idx: int
    team: int  # 0 or 1
    hero_id: str
    alive: bool
    self_vec: np.ndarray          # (SELF_DIM,)
    ally_vecs: np.ndarray         # (5, ALLY_DIM)
    enemy_vecs: np.ndarray        # (6, ENEMY_DIM)
    global_vec: np.ndarray        # (GLOBAL_DIM,)
    grid: np.ndarray              # (GRID_CHANNELS, GRID_H, GRID_W)
    masks: dict[str, np.ndarray]  # head_name → bool array


def split_perspectives(state: dict) -> list[AgentObs]:
    """Split a full state into 12 per-agent perspectives."""
    units = state.get("units", [])
    if len(units) != 12:
        logger.warning("Expected 12 units, got %d", len(units))
        return []

    global_data = state.get("global", {})
    grids = state.get("grids", {})
    shop = state.get("shop", {})

    obs_list = []
    for i in range(12):
        team = 0 if i < 6 else 1
        my_unit = units[i]

        if team == 0:
            allies = [units[j] for j in range(6) if j != i]
            enemies = [units[j] for j in range(6, 12)]
        else:
            allies = [units[j] for j in range(6, 12) if j != i]
            enemies = [units[j] for j in range(6)]

        self_vec = encode_self(my_unit)
        ally_vecs = np.stack([encode_ally(a) for a in allies])
        enemy_vecs = np.stack([encode_enemy(e) for e in enemies])
        global_vec = encode_global(global_data, team, shop)
        grid = encode_grid(grids, team, units)
        masks = encode_masks(my_unit.get("action_mask", {}))

        obs_list.append(AgentObs(
            agent_idx=i,
            team=team,
            hero_id=my_unit.get("hero_id", "H000"),
            alive=bool(my_unit.get("alive", False)),
            self_vec=self_vec,
            ally_vecs=ally_vecs,
            enemy_vecs=enemy_vecs,
            global_vec=global_vec,
            grid=grid,
            masks=masks,
        ))

    return obs_list


def encode_self(unit: dict) -> np.ndarray:
    """Encode self unit → (SELF_DIM,) float32 vector.

    Dimensions:
      basic(6) + stats(5) + upgrades(9) + combat(3) + growth(4)
      + skill_cd(12) + attributes(4) + buffs(6) + seal(4)
      + items(6) + economy(3) + vel(2) + alive(1) + hero_id(12) = 77
    """
    if not unit.get("alive", False):
        v = np.zeros(SELF_DIM, dtype=np.float32)
        # Still encode hero_id even when dead
        hero_idx = HERO_TO_IDX.get(unit.get("hero_id", ""), 0)
        v[-(NUM_HEROES - hero_idx)] = 0.0  # dead = all zero including hero_id
        return v

    v = []

    # Basic (6)
    v.append(unit.get("hp", 0) / N.hp)
    v.append(unit.get("max_hp", 0) / N.hp)
    v.append(unit.get("mp", 0) / N.mp)
    v.append(unit.get("max_mp", 0) / N.mp)
    v.append(unit.get("x", 0) / N.xy)
    v.append(unit.get("y", 0) / N.xy)

    # Stats (5)
    v.append(unit.get("str", 0) / N.stat)
    v.append(unit.get("agi", 0) / N.stat)
    v.append(unit.get("int", 0) / N.stat)
    v.append(unit.get("atk", 0) / N.atk)
    v.append(unit.get("def", 0) / N.def_)

    # Upgrades (9)
    ups = unit.get("upgrades", [0] * 9)
    for k in range(9):
        v.append(ups[k] / 50.0 if k < len(ups) else 0.0)

    # Combat (3)
    v.append(unit.get("move_spd", 0) / N.move_spd)
    v.append(unit.get("atk_range", 128) / 1000.0)
    v.append(unit.get("atk_spd", 1.5) / 3.0)

    # Growth (4)
    v.append(unit.get("level", 0) / N.level)
    v.append(unit.get("xp", 0) / 50000.0)
    v.append(unit.get("skill_points", 0) / 10.0)
    v.append(unit.get("stat_points", 0) / 200.0)

    # Skill CD (12): 6 slots × (cd_remain, level)
    skills = unit.get("skills", {})
    for slot in SKILL_SLOTS:
        sk = skills.get(slot, {})
        v.append(sk.get("cd_remain", 0) / N.cd)
        v.append(sk.get("level", 0) / 10.0)

    # Attributes (4) - binary
    attrs = unit.get("attributes", [False] * 4)
    for a in attrs[:4]:
        v.append(float(a))

    # Buffs (6) - binary
    buffs = unit.get("buffs", {})
    for b in ["stun", "slow", "silence", "knockback", "root", "invuln"]:
        v.append(float(buffs.get(b, False)))

    # Seal (4)
    v.append(unit.get("seal_charges", 0) / 12.0)
    v.append(unit.get("seal_cd", 0) / 30.0)
    v.append(float(unit.get("seal_first_active", False)))
    v.append(unit.get("seal_first_remain", 0) / 30.0)

    # Items (6) - type_id index
    items = unit.get("items", [{}] * 6)
    for item in items[:6]:
        v.append(item_index(item.get("type_id")) / 20.0)

    # Economy (3)
    v.append(unit.get("faire", 0) / N.faire)
    v.append(0.0)  # faire_regen placeholder (not in C# v2 yet)
    v.append(unit.get("faire_cap", 16000) / 20000.0)

    # Velocity (2)
    v.append(unit.get("vel_x", 0) / 500.0)
    v.append(unit.get("vel_y", 0) / 500.0)

    # Alive (1)
    v.append(1.0)

    # Hero ID one-hot (12)
    hero_onehot = [0.0] * NUM_HEROES
    hero_idx = HERO_TO_IDX.get(unit.get("hero_id", ""), 0)
    hero_onehot[hero_idx] = 1.0
    v.extend(hero_onehot)

    result = np.array(v, dtype=np.float32)
    assert result.shape[0] == SELF_DIM, f"self_vec dim mismatch: {result.shape[0]} != {SELF_DIM}"
    return result


def encode_ally(unit: dict) -> np.ndarray:
    """Encode one ally unit → (ALLY_DIM,) float32 vector.

    Dimensions:
      basic(6) + stats(5) + combat(3) + growth(1) + skill_cd(6)
      + buffs(6) + alive(1) + seal_charges(1) + faire(1) + vel(2) = 32
      Padded to ALLY_DIM=37 with zeros if needed.

    Actual layout: 6+5+3+1+6+6+1+1+1+2 = 32, then 5 zero-padding to reach 37.
    """
    if not unit.get("alive", False):
        return np.zeros(ALLY_DIM, dtype=np.float32)

    v = []

    # Basic (6)
    v.append(unit.get("hp", 0) / N.hp)
    v.append(unit.get("max_hp", 0) / N.hp)
    v.append(unit.get("mp", 0) / N.mp)
    v.append(unit.get("max_mp", 0) / N.mp)
    v.append(unit.get("x", 0) / N.xy)
    v.append(unit.get("y", 0) / N.xy)

    # Stats (5)
    v.append(unit.get("str", 0) / N.stat)
    v.append(unit.get("agi", 0) / N.stat)
    v.append(unit.get("int", 0) / N.stat)
    v.append(unit.get("atk", 0) / N.atk)
    v.append(unit.get("def", 0) / N.def_)

    # Combat (3)
    v.append(unit.get("move_spd", 0) / N.move_spd)
    v.append(unit.get("atk_range", 128) / 1000.0)
    v.append(unit.get("atk_spd", 1.5) / 3.0)

    # Growth (1)
    v.append(unit.get("level", 0) / N.level)

    # Skill CD remain (6)
    skills = unit.get("skills", {})
    for slot in SKILL_SLOTS:
        sk = skills.get(slot, {})
        v.append(sk.get("cd_remain", 0) / N.cd)

    # Buffs (6) - binary
    buffs = unit.get("buffs", {})
    for b in ["stun", "slow", "silence", "knockback", "root", "invuln"]:
        v.append(float(buffs.get(b, False)))

    # Alive (1)
    v.append(1.0)

    # Seal charges (1)
    v.append(unit.get("seal_charges", 0) / 12.0)

    # Faire (1)
    v.append(unit.get("faire", 0) / N.faire)

    # Velocity (2)
    v.append(unit.get("vel_x", 0) / 500.0)
    v.append(unit.get("vel_y", 0) / 500.0)

    # Padding (5) — reserved for future fields (atk_spd, atk_range, etc.)
    v.extend([0.0] * 5)

    result = np.array(v, dtype=np.float32)
    assert result.shape[0] == ALLY_DIM, f"ally_vec dim mismatch: {result.shape[0]} != {ALLY_DIM}"
    return result


def encode_enemy(unit: dict) -> np.ndarray:
    """Encode one enemy unit → (ENEMY_DIM,) float32 vector.

    Dimensions:
      visible(1) + basic(6) + stats(7) + growth(2) + buffs(6) + alive(1)
      + hero_id(12) + vel(2) + belief(4) = 41
    """
    if not unit.get("alive", False):
        v = np.zeros(ENEMY_DIM, dtype=np.float32)
        # Still encode hero_id
        hero_idx = HERO_TO_IDX.get(unit.get("hero_id", ""), 0)
        v[23 + hero_idx] = 1.0  # offset: 1+6+7+2+6+1 = 23
        return v

    v = []

    # Visible (1)
    visible = unit.get("visible", False)
    v.append(float(visible))

    # Basic (6) — C# sends 0 when not visible
    v.append(unit.get("hp", 0) / N.hp)
    v.append(unit.get("max_hp", 0) / N.hp)
    v.append(unit.get("mp", 0) / N.mp)
    v.append(unit.get("max_mp", 0) / N.mp)
    v.append(unit.get("x", 0) / N.xy)
    v.append(unit.get("y", 0) / N.xy)

    # Public stats (7)
    v.append(unit.get("str", 0) / N.stat)
    v.append(unit.get("agi", 0) / N.stat)
    v.append(unit.get("int", 0) / N.stat)
    v.append(unit.get("atk", 0) / N.atk)
    v.append(unit.get("def", 0) / N.def_)
    v.append(unit.get("max_hp", 0) / N.hp)
    v.append(unit.get("max_mp", 0) / N.mp)

    # Growth (2)
    v.append(unit.get("level", 0) / N.level)
    v.append(0.0)  # death_count placeholder

    # Buffs (6) - visible only
    buffs = unit.get("buffs", {})
    for b in ["stun", "slow", "silence", "knockback", "root", "invuln"]:
        v.append(float(buffs.get(b, False)))

    # Alive (1)
    v.append(1.0)

    # Hero ID one-hot (12)
    hero_onehot = [0.0] * NUM_HEROES
    hero_idx = HERO_TO_IDX.get(unit.get("hero_id", ""), 0)
    hero_onehot[hero_idx] = 1.0
    v.extend(hero_onehot)

    # Velocity (2)
    v.append(unit.get("vel_x", 0) / 500.0)
    v.append(unit.get("vel_y", 0) / 500.0)

    # Belief attributes (4) - placeholder, all -1
    v.extend([-1.0] * 4)

    result = np.array(v, dtype=np.float32)
    assert result.shape[0] == ENEMY_DIM, f"enemy_vec dim mismatch: {result.shape[0]} != {ENEMY_DIM}"
    return result


def encode_global(global_data: dict, my_team: int, shop: dict = None) -> np.ndarray:
    """Encode global state → (GLOBAL_DIM,) float32 vector.

    Dimensions: game_time(1) + is_night(1) + scores(2) + c_rank_stock(1) + padding(1) = 6
    """
    if shop is None:
        shop = {}

    v = []
    v.append(global_data.get("game_time", 0) / N.game_time)
    v.append(float(global_data.get("is_night", False)))

    # Scores from my perspective
    if my_team == 0:
        v.append(global_data.get("score_ally", 0) / N.score)
        v.append(global_data.get("score_enemy", 0) / N.score)
    else:
        v.append(global_data.get("score_enemy", 0) / N.score)
        v.append(global_data.get("score_ally", 0) / N.score)

    v.append(shop.get("c_rank_stock", 8) / 8.0)
    v.append(0.0)  # padding to GLOBAL_DIM=6

    result = np.array(v, dtype=np.float32)
    assert result.shape[0] == GLOBAL_DIM, f"global_vec dim mismatch: {result.shape[0]} != {GLOBAL_DIM}"
    return result


def encode_grid(grids: dict, my_team: int, units: list,
                grid_w: int = GRID_W, grid_h: int = GRID_H) -> np.ndarray:
    """Encode 2D grid → (3, GRID_H, GRID_W) float32 array.

    Channel 0: pathability (0=fog, 1=walkable, 2=unwalkable) / 2
    Channel 1: ally positions (1 if ally in cell)
    Channel 2: enemy positions (1 if visible enemy in cell)
    """
    grid = np.zeros((GRID_CHANNELS, grid_h, grid_w), dtype=np.float32)

    # Channel 0: pathability
    path = grids.get("pathability", [])
    if path and len(path) == grid_w * grid_h:
        grid[0] = np.array(path, dtype=np.float32).reshape(grid_h, grid_w) / 2.0

    # Channels 1-2: unit positions
    for idx, unit in enumerate(units):
        if not unit.get("alive", False):
            continue
        gx = int((unit.get("x", 0) - MAP_MIN_X) / CELL_SIZE)
        gy = int((unit.get("y", 0) - MAP_MIN_Y) / CELL_SIZE)
        gx = max(0, min(gx, grid_w - 1))
        gy = max(0, min(gy, grid_h - 1))

        unit_team = 0 if idx < 6 else 1
        if unit_team == my_team:
            grid[1, gy, gx] = 1.0
        else:
            if unit.get("visible", False):
                grid[2, gy, gx] = 1.0

    return grid


def encode_masks(raw_mask: dict) -> dict[str, np.ndarray]:
    """Convert C# action_mask JSON → dict of bool arrays."""
    masks = {}
    for head_name, head_size in DISCRETE_HEADS.items():
        raw = raw_mask.get(head_name, [True] * head_size)
        arr = np.array(raw, dtype=bool)
        if len(arr) != head_size:
            logger.warning("Mask %s: expected %d, got %d", head_name, head_size, len(arr))
            arr = np.ones(head_size, dtype=bool)
        masks[head_name] = arr
    return masks


def batch_observations(obs_list: list[AgentObs], device=None) -> dict:
    """Batch multiple AgentObs into tensor dict for model input.

    Args:
        obs_list: List of AgentObs
        device: torch device (None=cpu)

    Returns:
        dict with keys:
            self_vec: (B, SELF_DIM)
            ally_vec: (B, 5, ALLY_DIM)
            enemy_vec: (B, 6, ENEMY_DIM)
            global_vec: (B, GLOBAL_DIM)
            grid: (B, 3, GRID_H, GRID_W)
            masks: {head_name: (B, head_size) bool tensor}
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    B = len(obs_list)

    batch = {
        "self_vec": torch.from_numpy(
            np.stack([o.self_vec for o in obs_list])
        ).to(device),
        "ally_vec": torch.from_numpy(
            np.stack([o.ally_vecs for o in obs_list])
        ).to(device),
        "enemy_vec": torch.from_numpy(
            np.stack([o.enemy_vecs for o in obs_list])
        ).to(device),
        "global_vec": torch.from_numpy(
            np.stack([o.global_vec for o in obs_list])
        ).to(device),
        "grid": torch.from_numpy(
            np.stack([o.grid for o in obs_list])
        ).to(device),
    }

    # Masks
    masks = {}
    for head_name in DISCRETE_HEADS:
        masks[head_name] = torch.from_numpy(
            np.stack([o.masks[head_name] for o in obs_list])
        ).to(device)
    batch["masks"] = masks

    return batch
