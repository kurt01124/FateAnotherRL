#pragma once

#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include <cstring>

#include <torch/torch.h>

#include "state_encoder.h"
#include "constants.h"
#include "protocol.h"

class RolloutWriter {
public:
    explicit RolloutWriter(const std::string& rollout_dir);

    /// Store a single transition for one agent in one instance.
    /// FATE v2 extra parameters are optional (default empty/null/0) for backward compatibility.
    void store(const std::string& instance_id,
               int agent_idx,
               const torch::Tensor& self_vec,      // (77,)
               const torch::Tensor& ally_vec,       // (5, 37)
               const torch::Tensor& enemy_vec,      // (6, 41)
               const torch::Tensor& global_vec,     // (6,)
               const torch::Tensor& grid,           // (3, 25, 48)
               const std::unordered_map<std::string, torch::Tensor>& masks,
               const std::unordered_map<std::string, torch::Tensor>& actions,
               float log_prob,
               float value,
               float reward,
               bool done,
               const torch::Tensor& hx_h,          // (1, 1, 256)
               const torch::Tensor& hx_c,          // (1, 1, 256)
               // FATE v2 extra parameters
               const std::vector<Event>& events = {},
               const UnitState* prev_units = nullptr,
               const UnitState* units = nullptr,
               const GlobalState& prev_global = {},
               const GlobalState& global_state = {},
               int model_version = 0);

    /// Mark last transition as done=true and add terminal rewards.
    /// Must be called BEFORE flush_episode().
    void mark_last_done(const std::string& instance_id,
                        const std::array<float, MAX_UNITS>& terminal_rewards);

    /// Flush all agent buffers for a completed episode.
    void flush_episode(const std::string& instance_id);

    /// Dump accumulated transitions to .pt files if buffer exceeds min_transitions.
    void maybe_dump(int min_transitions);

private:
    struct Transition {
        // Observation tensors (CPU, detached)
        torch::Tensor self_vec;     // (77,)
        torch::Tensor ally_vec;     // (5, 37)
        torch::Tensor enemy_vec;    // (6, 41)
        torch::Tensor global_vec;   // (6,)
        torch::Tensor grid;         // (3, 25, 48)

        // Masks
        std::unordered_map<std::string, torch::Tensor> masks;

        // Actions
        std::unordered_map<std::string, torch::Tensor> actions;

        // Scalars
        float log_prob;
        float value;
        float reward;
        bool done;

        // LSTM hidden state
        torch::Tensor hx_h;        // (1, 1, 256)
        torch::Tensor hx_c;        // (1, 1, 256)

        // === FATE v2 fields ===
        std::vector<Event> events;      // events for this tick
        float prev_hp[MAX_UNITS];       // previous tick HP
        float prev_max_hp[MAX_UNITS];   // previous tick max HP
        int16_t prev_score_t0;          // previous tick team0 score
        int16_t prev_score_t1;          // previous tick team1 score
        float game_time;                // game time (seconds)
        uint8_t unit_alive[MAX_UNITS];  // unit alive flags
        uint8_t unit_level[MAX_UNITS];  // unit levels
        float unit_x[MAX_UNITS];        // unit x positions
        float unit_y[MAX_UNITS];        // unit y positions
        uint8_t skill_points[MAX_UNITS]; // skill points available
        int model_version;              // model version used for this transition

        Transition() : log_prob(0), value(0), reward(0), done(false),
                       prev_score_t0(0), prev_score_t1(0), game_time(0),
                       model_version(0)
        {
            std::memset(prev_hp, 0, sizeof(prev_hp));
            std::memset(prev_max_hp, 0, sizeof(prev_max_hp));
            std::memset(unit_alive, 0, sizeof(unit_alive));
            std::memset(unit_level, 0, sizeof(unit_level));
            std::memset(unit_x, 0, sizeof(unit_x));
            std::memset(unit_y, 0, sizeof(unit_y));
            std::memset(skill_points, 0, sizeof(skill_points));
        }
    };

    // A completed episode: all 12 agents' trajectories together
    struct CompletedEpisode {
        std::array<std::vector<Transition>, MAX_UNITS> agents;
    };

    // instance_id -> per-agent (12) -> list of transitions
    std::map<std::string, std::array<std::vector<Transition>, MAX_UNITS>> buffers_;

    // Completed episodes ready for dumping (aggregated across agents)
    std::vector<CompletedEpisode> completed_;

    std::string rollout_dir_;
    int dump_count_;
    std::mutex mutex_;

    /// Helper: stack a field across agents and timesteps into (T, 12, ...) tensor.
    torch::Tensor stack_field(const CompletedEpisode& ep, int T,
                              std::function<torch::Tensor(const Transition&)> getter,
                              torch::Tensor fallback);

    /// Write a full episode (all 12 agents) to a .pt file with (T, 12, ...) tensors.
    void dump_to_file(const CompletedEpisode& episode);
};
