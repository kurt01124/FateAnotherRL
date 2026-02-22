#include "reward_calc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

// ============================================================
// Constructor
// ============================================================

RewardCalc::RewardCalc()
    : has_prev_pos_(false)
    , prev_game_time_(0.0f)
{
    std::memset(prev_x_, 0, sizeof(prev_x_));
    std::memset(prev_y_, 0, sizeof(prev_y_));
    std::memset(portal_count_, 0, sizeof(portal_count_));
}

// ============================================================
// reset
// ============================================================

void RewardCalc::reset() {
    has_prev_pos_ = false;
    prev_game_time_ = 0.0f;
    std::memset(prev_x_, 0, sizeof(prev_x_));
    std::memset(prev_y_, 0, sizeof(prev_y_));
}

// ============================================================
// apply_team_spirit: τ × team_avg + (1-τ) × individual
// ============================================================

void RewardCalc::apply_team_spirit(std::array<float, MAX_UNITS>& rewards, float tau) {
    // Team 0: indices 0-5, Team 1: indices 6-11
    for (int team = 0; team < 2; ++team) {
        int base = team * 6;
        float team_sum = 0.0f;
        for (int i = 0; i < 6; ++i)
            team_sum += rewards[base + i];
        float team_avg = team_sum / 6.0f;

        for (int i = 0; i < 6; ++i) {
            rewards[base + i] = tau * team_avg + (1.0f - tau) * rewards[base + i];
        }
    }
}

// ============================================================
// apply_zero_sum: subtract enemy team's average
// ============================================================

void RewardCalc::apply_zero_sum(std::array<float, MAX_UNITS>& rewards) {
    // Compute team averages
    float avg[2] = {0.0f, 0.0f};
    for (int i = 0; i < 6; ++i) avg[0] += rewards[i];
    for (int i = 6; i < 12; ++i) avg[1] += rewards[i];
    avg[0] /= 6.0f;
    avg[1] /= 6.0f;

    // Subtract enemy team's average
    for (int i = 0; i < 6; ++i)  rewards[i]     -= avg[1];
    for (int i = 6; i < 12; ++i) rewards[i]     -= avg[0];
}

// ============================================================
// apply_time_decay: reward *= decay_base ^ (game_time / interval)
// ============================================================

void RewardCalc::apply_time_decay(std::array<float, MAX_UNITS>& rewards, float game_time) {
    float decay = std::pow(
        RewardDefaults::time_decay_base,
        game_time / RewardDefaults::time_decay_interval
    );
    for (auto& r : rewards) r *= decay;
}

// ============================================================
// compute: Per-tick rewards (OpenAI Five style)
// ============================================================

std::array<float, MAX_UNITS> RewardCalc::compute(
    const UnitState units[MAX_UNITS],
    const GlobalState& global,
    const std::vector<Event>& events,
    const UnitState prev_units[MAX_UNITS],
    const GlobalState& prev_global,
    bool has_prev)
{
    std::array<float, MAX_UNITS> rewards{};
    rewards.fill(0.0f);

    // ---- 1. Event-based rewards (individual) ----
    for (const auto& ev : events) {
        switch (ev.type) {
        case EVT_KILL: {
            int killer = ev.killer_idx;
            int victim = ev.victim_idx;
            if (killer >= 0 && killer < MAX_UNITS && victim >= 0 && victim < MAX_UNITS) {
                int killer_team = (killer < 6) ? 0 : 1;
                int victim_team = (victim < 6) ? 0 : 1;

                if (killer_team != victim_team) {
                    // Enemy kill: reward to killer
                    rewards[killer] += RewardDefaults::kill_personal;
                } else {
                    // Friendly kill: punish killer
                    rewards[killer] += RewardDefaults::friendly_kill;
                }

                // Death penalty to victim
                rewards[victim] += RewardDefaults::death;
            }
            break;
        }
        case EVT_CREEP_KILL: {
            int killer = ev.killer_idx;
            if (killer >= 0 && killer < MAX_UNITS
                && units[killer].level < RewardDefaults::creep_level_cap) {
                rewards[killer] += RewardDefaults::creep;
            }
            break;
        }
        case EVT_LEVEL_UP: {
            int unit_idx = ev.killer_idx;
            if (unit_idx >= 0 && unit_idx < MAX_UNITS) {
                rewards[unit_idx] += RewardDefaults::levelup;
            }
            break;
        }
        case EVT_PORTAL: {
            int unit_idx = ev.killer_idx;
            if (unit_idx >= 0 && unit_idx < MAX_UNITS) {
                float factor = std::pow(RewardDefaults::portal_decay, portal_count_[unit_idx]);
                rewards[unit_idx] += RewardDefaults::portal_use * factor;
                portal_count_[unit_idx] += 1.0f;
            }
            break;
        }
        default:
            break;
        }
    }

    // ---- 1b. Damage-based rewards (team-distributed) ----
    if (has_prev) {
        for (int i = 0; i < MAX_UNITS; ++i) {
            if (!prev_units[i].alive) continue;
            float hp_delta = prev_units[i].hp - units[i].hp;  // positive = took damage
            if (hp_delta > 0.0f && units[i].max_hp > 0.0f) {
                float ratio = hp_delta / units[i].max_hp;
                int victim_team = (i < 6) ? 0 : 1;
                int attacker_team = 1 - victim_team;
                int attacker_base = attacker_team * 6;
                for (int a = 0; a < 6; ++a)
                    rewards[attacker_base + a] += RewardDefaults::damage_ratio * ratio;
            }
        }

        // Self HP recovery reward
        for (int i = 0; i < MAX_UNITS; ++i) {
            if (!units[i].alive || !prev_units[i].alive) continue;
            float hp_gain = units[i].hp - prev_units[i].hp;  // positive = healed
            if (hp_gain > 0.0f && units[i].max_hp > 0.0f) {
                float ratio = hp_gain / units[i].max_hp;
                rewards[i] += RewardDefaults::heal_ratio * ratio;
            }
        }
    }

    // ---- 2. Score change rewards ----
    if (has_prev) {
        int score_delta_t0 = global.score_team0 - prev_global.score_team0;
        int score_delta_t1 = global.score_team1 - prev_global.score_team1;

        // Team 0 scored
        if (score_delta_t0 > 0) {
            for (int i = 0; i < 6; ++i)
                rewards[i] += RewardDefaults::score_point * score_delta_t0;
        }
        // Team 1 scored
        if (score_delta_t1 > 0) {
            for (int i = 6; i < 12; ++i)
                rewards[i] += RewardDefaults::score_point * score_delta_t1;
        }
    }

    // ---- 3. Per-tick penalties ----
    for (int i = 0; i < MAX_UNITS; ++i) {
        if (!units[i].alive) continue;

        float x = units[i].x;
        float y = units[i].y;

        // Idle penalty — exempt during combat activity
        if (has_prev_pos_) {
            float dx = x - prev_x_[i];
            float dy = y - prev_y_[i];
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 10.0f) {
                bool in_combat = false;
                if (has_prev) {
                    // My HP changed (taking damage)
                    float my_hp_delta = std::abs(units[i].hp - prev_units[i].hp);
                    if (my_hp_delta > 1.0f) in_combat = true;

                    // Nearby enemy HP decreased (dealing damage)
                    int enemy_base = (i < 6) ? 6 : 0;
                    for (int e = 0; e < 6 && !in_combat; ++e) {
                        int eidx = enemy_base + e;
                        float ehp_delta = prev_units[eidx].hp - units[eidx].hp;
                        if (ehp_delta > 1.0f) {
                            float edx = units[i].x - units[eidx].x;
                            float edy = units[i].y - units[eidx].y;
                            float edist = std::sqrt(edx * edx + edy * edy);
                            if (edist < 800.f) in_combat = true;
                        }
                    }
                }
                if (!in_combat) {
                    rewards[i] += RewardDefaults::idle_penalty;
                }
            }
        }

        prev_x_[i] = x;
        prev_y_[i] = y;

        // Skill points held penalty
        int sp = units[i].skill_points;
        if (sp > 0) {
            rewards[i] += RewardDefaults::skill_points_held * static_cast<float>(sp);
        }
    }
    has_prev_pos_ = true;
    prev_game_time_ = global.game_time;

    // ---- 4. Team Spirit blending ----
    apply_team_spirit(rewards, RewardDefaults::team_spirit);

    // ---- 5. Zero-Sum ----
    apply_zero_sum(rewards);

    // ---- 6. Time Decay ----
    apply_time_decay(rewards, global.game_time);

    return rewards;
}

// ============================================================
// compute_terminal: Terminal rewards
// ============================================================

std::array<float, MAX_UNITS> RewardCalc::compute_terminal(uint8_t winner, uint8_t reason) {
    std::array<float, MAX_UNITS> rewards{};
    rewards.fill(0.0f);

    if (winner == 0) {
        // Team 0 wins
        for (int i = 0; i < 6; ++i)  rewards[i] = RewardDefaults::win_reward;
        for (int i = 6; i < 12; ++i) rewards[i] = RewardDefaults::lose_reward;
    } else if (winner == 1) {
        // Team 1 wins
        for (int i = 0; i < 6; ++i)  rewards[i] = RewardDefaults::lose_reward;
        for (int i = 6; i < 12; ++i) rewards[i] = RewardDefaults::win_reward;
    } else {
        // Draw / timeout
        rewards.fill(RewardDefaults::timeout_reward);
    }

    return rewards;
}
