#pragma once

#include <array>
#include <vector>

#include "protocol.h"
#include "constants.h"

class RewardCalc {
public:
    RewardCalc();

    /// Compute per-agent rewards with OpenAI Five-style processing:
    ///   1. Raw event/tick rewards
    ///   2. Team spirit blending
    ///   3. Zero-sum (subtract enemy team avg)
    ///   4. Time decay
    std::array<float, MAX_UNITS> compute(
        const UnitState units[MAX_UNITS],
        const GlobalState& global,
        const std::vector<Event>& events,
        const UnitState prev_units[MAX_UNITS],
        const GlobalState& prev_global,
        bool has_prev);

    /// Compute terminal rewards from DONE packet.
    std::array<float, MAX_UNITS> compute_terminal(uint8_t winner, uint8_t reason);

    /// Reset state for a new episode.
    void reset();

private:
    float prev_x_[MAX_UNITS];
    float prev_y_[MAX_UNITS];
    bool has_prev_pos_;

    /// Apply team spirit: blend individual and team average
    void apply_team_spirit(std::array<float, MAX_UNITS>& rewards, float tau);

    /// Apply zero-sum: subtract enemy team's average from each agent
    void apply_zero_sum(std::array<float, MAX_UNITS>& rewards);

    /// Apply time decay based on game time
    void apply_time_decay(std::array<float, MAX_UNITS>& rewards, float game_time);
};
