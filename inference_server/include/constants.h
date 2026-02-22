#pragma once

#include <array>
#include <string>
#include <unordered_map>

// ============================================================
// Normalization Constants (matches Python data/constants.py)
// ============================================================
namespace N {
    constexpr float hp        = 10000.f;
    constexpr float mp        = 5000.f;
    constexpr float xy        = 10000.f;
    constexpr float stat      = 200.f;
    constexpr float atk       = 500.f;
    constexpr float def_      = 50.f;
    constexpr float move_spd  = 522.f;
    constexpr float level     = 25.f;
    constexpr float cd        = 120.f;
    constexpr float faire     = 16000.f;
    constexpr float score     = 70.f;
    constexpr float game_time = 1800.f;
}

// ============================================================
// Map Constants
// ============================================================
constexpr float MAP_MIN_X = -8416.f;
constexpr float MAP_MAX_X =  8320.f;
constexpr float MAP_MIN_Y = -2592.f;
constexpr float MAP_MAX_Y =  6176.f;
constexpr float CELL_SIZE  = 350.f;

// ============================================================
// Hero IDs (12 heroes, matches Python HERO_IDS)
// ============================================================
constexpr int NUM_HEROES = 12;

inline const std::array<std::string, NUM_HEROES>& hero_ids() {
    static const std::array<std::string, NUM_HEROES> ids = {
        "H000", "H001", "H002", "H03M", "H028", "H009",  // Team 0
        "H007", "H005", "H003", "H006", "H004", "H008",  // Team 1
    };
    return ids;
}

inline const std::unordered_map<std::string, int>& hero_to_idx() {
    static std::unordered_map<std::string, int> m;
    if (m.empty()) {
        const auto& ids = hero_ids();
        for (int i = 0; i < NUM_HEROES; ++i)
            m[ids[i]] = i;
    }
    return m;
}

// ============================================================
// Observation Dimensions (matches Python constants)
// ============================================================
constexpr int SELF_DIM      = 77;
constexpr int ALLY_DIM      = 37;
constexpr int ENEMY_DIM     = 43;
constexpr int GLOBAL_DIM    = 6;
constexpr int GRID_CHANNELS = 3;
constexpr int OBS_GRID_H    = 25;
constexpr int OBS_GRID_W    = 48;
constexpr int HIDDEN_DIM    = 256;

// ============================================================
// Discrete Action Head Sizes
// ============================================================
struct DiscreteHead {
    const char* name;
    int size;
};

constexpr int NUM_DISCRETE_HEADS = 11;

inline const std::array<DiscreteHead, NUM_DISCRETE_HEADS>& discrete_heads() {
    static const std::array<DiscreteHead, NUM_DISCRETE_HEADS> heads = {{
        {"skill",          8},
        {"unit_target",   14},
        {"skill_levelup",  6},
        {"stat_upgrade",  10},
        {"attribute",      5},
        {"item_buy",      17},
        {"item_use",       7},
        {"seal_use",       7},
        {"faire_send",     6},
        {"faire_request",  6},
        {"faire_respond",  3},
    }};
    return heads;
}

// Total discrete action space size
constexpr int TOTAL_DISCRETE_ACTIONS = 8 + 14 + 6 + 10 + 5 + 17 + 7 + 7 + 6 + 6 + 3; // = 89

// Continuous heads: move(2) + point(2)
constexpr int NUM_CONTINUOUS = 4;  // move_x, move_y, point_x, point_y

// ============================================================
// Skill Slot Names
// ============================================================
inline const std::array<std::string, 6>& skill_slots() {
    static const std::array<std::string, 6> slots = {"Q", "W", "E", "R", "D", "F"};
    return slots;
}

// ============================================================
// Reward Constants (OpenAI Five style)
// ============================================================
namespace RewardDefaults {
    // --- Event rewards (raw, before zero-sum/team-spirit) ---
    constexpr float kill_personal    =  3.0f;    // kill bonus (stacks with damage reward)
    constexpr float death            = -1.0f;    // OpenAI Five: -1.0
    constexpr float creep            =  0.16f;   // OpenAI Five last-hit: 0.16
    constexpr float levelup          =  0.5f;
    constexpr float friendly_kill    = -3.0f;    // punish team kills heavily
    constexpr float score_point      =  2.0f;    // team scored a point (70 to win)
    constexpr float damage_ratio     =  3.0f;    // enemy maxHP 100% dealt = 3.0 reward
    constexpr float heal_ratio       =  1.0f;    // self maxHP 100% healed = 1.0 reward
    constexpr float alarm_proximity  =  0.1f;    // per-tick max (closest to enemy)
    constexpr float alarm_duration   = 10.0f;    // seconds after alarm triggers

    // --- Per-tick ---
    constexpr float skill_points_held = -0.02f;
    constexpr float idle_penalty     = -0.003f;  // reduced from -0.01

    // --- Terminal ---
    constexpr float win_reward       = 10.0f;
    constexpr float lose_reward      = -5.0f;
    constexpr float timeout_reward   = -2.0f;

    // --- Team Spirit: τ × team_avg + (1-τ) × individual ---
    constexpr float team_spirit      =  0.5f;    // fixed for now (can anneal later)

    // --- Time decay: reward *= decay_base ^ (game_time / decay_interval) ---
    constexpr float time_decay_base     = 0.7f;
    constexpr float time_decay_interval = 600.f;  // 10 minutes
}
