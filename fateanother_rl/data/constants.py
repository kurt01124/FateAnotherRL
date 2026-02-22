"""Normalization constants and ID mappings for FateAnother RL v2."""

# PlayZone coordinates
MAP_MIN_X, MAP_MAX_X = -8416.0, 8320.0
MAP_MIN_Y, MAP_MAX_Y = -2592.0, 6176.0
CELL_SIZE = 350.0
GRID_W, GRID_H = 48, 25

class N:
    """Normalization divisors."""
    hp = 10000.0
    mp = 5000.0
    xy = 10000.0
    stat = 200.0
    atk = 500.0
    def_ = 50.0
    move_spd = 522.0
    level = 25.0
    cd = 120.0
    faire = 16000.0
    score = 70.0
    game_time = 1800.0

HERO_IDS = [
    "H000", "H001", "H002", "H03M", "H028", "H009",  # Team 0
    "H007", "H005", "H003", "H006", "H004", "H008",  # Team 1
]
HERO_TO_IDX = {h: i for i, h in enumerate(HERO_IDS)}
NUM_HEROES = 12

# Discrete action head sizes (must match C# action_mask)
DISCRETE_HEADS = {
    "skill": 8,
    "unit_target": 14,
    "skill_levelup": 6,
    "stat_upgrade": 10,
    "attribute": 5,
    "item_buy": 18,
    "item_use": 7,
    "seal_use": 7,
    "faire_send": 6,
    "faire_request": 6,
    "faire_respond": 3,
}

SKILL_SLOTS = ["Q", "W", "E", "R", "D", "F"]

# Observation dimensions (used by model config)
SELF_DIM = 77    # hero_id(12) included
ALLY_DIM = 37
ENEMY_DIM = 43
GLOBAL_DIM = 6
GRID_CHANNELS = 6  # path, ally, enemy_vis, portal, creep_pos, creep_hp
