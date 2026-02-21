#include "rollout_writer.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

// ============================================================
// Constructor
// ============================================================

RolloutWriter::RolloutWriter(const std::string& rollout_dir)
    : rollout_dir_(rollout_dir), dump_count_(0)
{
    fs::create_directories(rollout_dir);
    std::cout << "[RolloutWriter] Output dir: " << rollout_dir << std::endl;
}

// ============================================================
// store: Store a single transition
// ============================================================

void RolloutWriter::store(
    const std::string& instance_id,
    int agent_idx,
    const torch::Tensor& self_vec,
    const torch::Tensor& ally_vec,
    const torch::Tensor& enemy_vec,
    const torch::Tensor& global_vec,
    const torch::Tensor& grid,
    const std::unordered_map<std::string, torch::Tensor>& masks,
    const std::unordered_map<std::string, torch::Tensor>& actions,
    float log_prob,
    float value,
    float reward,
    bool done,
    const torch::Tensor& hx_h,
    const torch::Tensor& hx_c)
{
    std::lock_guard<std::mutex> lock(mutex_);

    Transition t;
    t.self_vec   = self_vec.detach().cpu();
    t.ally_vec   = ally_vec.detach().cpu();
    t.enemy_vec  = enemy_vec.detach().cpu();
    t.global_vec = global_vec.detach().cpu();
    t.grid       = grid.detach().cpu();
    t.hx_h       = hx_h.detach().cpu();
    t.hx_c       = hx_c.detach().cpu();

    for (const auto& [k, v] : masks) {
        t.masks[k] = v.detach().cpu();
    }
    for (const auto& [k, v] : actions) {
        t.actions[k] = v.detach().cpu();
    }

    t.log_prob = log_prob;
    t.value    = value;
    t.reward   = reward;
    t.done     = done;

    if (agent_idx < 0 || agent_idx >= MAX_UNITS) return;
    buffers_[instance_id][agent_idx].push_back(std::move(t));
}

// ============================================================
// mark_last_done: Set done=true on last transition + add terminal rewards
// ============================================================

void RolloutWriter::mark_last_done(
    const std::string& instance_id,
    const std::array<float, MAX_UNITS>& terminal_rewards)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = buffers_.find(instance_id);
    if (it == buffers_.end()) return;

    for (int a = 0; a < MAX_UNITS; ++a) {
        auto& traj = it->second[a];
        if (!traj.empty()) {
            traj.back().done = true;
            traj.back().reward += terminal_rewards[a];
        }
    }
}

// ============================================================
// flush_episode: Group all 12 agents into a single CompletedEpisode
// ============================================================

void RolloutWriter::flush_episode(const std::string& instance_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = buffers_.find(instance_id);
    if (it == buffers_.end()) return;

    CompletedEpisode ep;
    bool has_data = false;
    for (int a = 0; a < MAX_UNITS; ++a) {
        ep.agents[a] = std::move(it->second[a]);
        if (!ep.agents[a].empty()) has_data = true;
    }

    if (has_data) {
        completed_.push_back(std::move(ep));
    }

    buffers_.erase(it);
}

// ============================================================
// maybe_dump: Write .pt files if we have enough transitions
// ============================================================

void RolloutWriter::maybe_dump(int min_transitions) {
    std::lock_guard<std::mutex> lock(mutex_);

    int total = 0;
    for (const auto& ep : completed_) {
        for (const auto& agent_traj : ep.agents) {
            total += static_cast<int>(agent_traj.size());
        }
    }

    if (total < min_transitions) return;

    for (const auto& episode : completed_) {
        dump_to_file(episode);
    }

    std::cout << "[RolloutWriter] Dumped " << completed_.size()
              << " episodes (" << total << " transitions)" << std::endl;

    completed_.clear();
}

// ============================================================
// stack_field: Helper to stack a tensor field across agents & timesteps
// Result: (T, 12, ...) tensor
// ============================================================

torch::Tensor RolloutWriter::stack_field(
    const CompletedEpisode& ep, int T,
    std::function<torch::Tensor(const Transition&)> getter,
    torch::Tensor fallback)
{
    // For each agent, stack their T transitions: (T, ...)
    // Then stack across agents: (12, T, ...)
    // Then permute to (T, 12, ...)
    std::vector<torch::Tensor> per_agent;
    per_agent.reserve(MAX_UNITS);

    for (int a = 0; a < MAX_UNITS; ++a) {
        std::vector<torch::Tensor> steps;
        steps.reserve(T);
        for (int t = 0; t < T; ++t) {
            if (t < static_cast<int>(ep.agents[a].size())) {
                steps.push_back(getter(ep.agents[a][t]));
            } else {
                steps.push_back(fallback.clone());
            }
        }
        per_agent.push_back(torch::stack(steps));  // (T, ...)
    }

    auto stacked = torch::stack(per_agent);  // (12, T, ...)

    // Permute dim 0 (agents=12) and dim 1 (T) -> (T, 12, ...)
    auto ndim = stacked.dim();
    std::vector<int64_t> perm;
    perm.push_back(1);  // T
    perm.push_back(0);  // 12
    for (int64_t d = 2; d < ndim; ++d) perm.push_back(d);

    return stacked.permute(perm).contiguous();
}

// ============================================================
// dump_to_file: Write a full episode as (T, 12, ...) tensors
// ============================================================

void RolloutWriter::dump_to_file(const CompletedEpisode& ep) {
    // Find max T across agents (should be identical for all 12)
    int T = 0;
    for (int a = 0; a < MAX_UNITS; ++a) {
        T = std::max(T, static_cast<int>(ep.agents[a].size()));
    }
    if (T == 0) return;

    // --- Observation tensors ---
    auto self_vecs = stack_field(ep, T,
        [](const Transition& t) { return t.self_vec; },
        torch::zeros({SELF_DIM}));  // (T, 12, 77)

    auto ally_vecs = stack_field(ep, T,
        [](const Transition& t) { return t.ally_vec; },
        torch::zeros({5, ALLY_DIM}));  // (T, 12, 5, 37)

    auto enemy_vecs = stack_field(ep, T,
        [](const Transition& t) { return t.enemy_vec; },
        torch::zeros({6, ENEMY_DIM}));  // (T, 12, 6, 41)

    auto global_vecs = stack_field(ep, T,
        [](const Transition& t) { return t.global_vec; },
        torch::zeros({GLOBAL_DIM}));  // (T, 12, 6)

    auto grids = stack_field(ep, T,
        [](const Transition& t) { return t.grid; },
        torch::zeros({GRID_CHANNELS, GRID_H, GRID_W}));  // (T, 12, 3, 25, 48)

    // --- LSTM hidden states ---
    auto hx_h = stack_field(ep, T,
        [](const Transition& t) { return t.hx_h; },
        torch::zeros({1, 1, HIDDEN_DIM}));  // (T, 12, 1, 1, 256) but we want (T, 12, 1, 256)

    auto hx_c = stack_field(ep, T,
        [](const Transition& t) { return t.hx_c; },
        torch::zeros({1, 1, HIDDEN_DIM}));

    // hx stored as (1, 1, 256) per agent → stacked gives (T, 12, 1, 1, 256)
    // Squeeze the extra dim to get (T, 12, 1, 256)
    if (hx_h.dim() == 5) hx_h = hx_h.squeeze(3);  // (T, 12, 1, 256)
    if (hx_c.dim() == 5) hx_c = hx_c.squeeze(3);

    // --- Scalar sequences: (T, 12) ---
    std::vector<torch::Tensor> lp_per_agent, val_per_agent, rew_per_agent, done_per_agent;
    for (int a = 0; a < MAX_UNITS; ++a) {
        std::vector<float> lps, vals, rews;
        std::vector<int64_t> dns;
        for (int t = 0; t < T; ++t) {
            if (t < static_cast<int>(ep.agents[a].size())) {
                lps.push_back(ep.agents[a][t].log_prob);
                vals.push_back(ep.agents[a][t].value);
                rews.push_back(ep.agents[a][t].reward);
                dns.push_back(ep.agents[a][t].done ? 1 : 0);
            } else {
                lps.push_back(0.0f);
                vals.push_back(0.0f);
                rews.push_back(0.0f);
                dns.push_back(1);  // padded as done
            }
        }
        lp_per_agent.push_back(torch::tensor(lps));     // (T,)
        val_per_agent.push_back(torch::tensor(vals));
        rew_per_agent.push_back(torch::tensor(rews));
        done_per_agent.push_back(torch::tensor(dns, torch::kLong));
    }
    // Stack: (12, T) -> transpose to (T, 12)
    auto log_probs = torch::stack(lp_per_agent).t().contiguous();
    auto values    = torch::stack(val_per_agent).t().contiguous();
    auto rewards   = torch::stack(rew_per_agent).t().contiguous();
    auto dones     = torch::stack(done_per_agent).t().contiguous();

    // --- Masks: per-head, (T, 12, head_size) ---
    // Get head names from first non-empty transition
    std::vector<std::string> mask_names;
    for (int a = 0; a < MAX_UNITS; ++a) {
        if (!ep.agents[a].empty()) {
            for (const auto& [k, _] : ep.agents[a][0].masks) {
                mask_names.push_back(k);
            }
            break;
        }
    }

    std::unordered_map<std::string, torch::Tensor> mask_tensors;
    for (const auto& name : mask_names) {
        // Get fallback shape from first available mask
        torch::Tensor fallback;
        for (int a = 0; a < MAX_UNITS; ++a) {
            if (!ep.agents[a].empty()) {
                auto it = ep.agents[a][0].masks.find(name);
                if (it != ep.agents[a][0].masks.end()) {
                    fallback = torch::zeros_like(it->second);
                    break;
                }
            }
        }

        mask_tensors[name] = stack_field(ep, T,
            [&name](const Transition& t) -> torch::Tensor {
                auto it = t.masks.find(name);
                if (it != t.masks.end()) return it->second;
                return torch::Tensor();  // should not happen
            },
            fallback);
    }

    // --- Actions: per-head, (T, 12) or (T, 12, 2) ---
    std::vector<std::string> action_names;
    for (int a = 0; a < MAX_UNITS; ++a) {
        if (!ep.agents[a].empty()) {
            for (const auto& [k, _] : ep.agents[a][0].actions) {
                action_names.push_back(k);
            }
            break;
        }
    }

    std::unordered_map<std::string, torch::Tensor> action_tensors;
    for (const auto& name : action_names) {
        torch::Tensor fallback;
        for (int a = 0; a < MAX_UNITS; ++a) {
            if (!ep.agents[a].empty()) {
                auto it = ep.agents[a][0].actions.find(name);
                if (it != ep.agents[a][0].actions.end()) {
                    fallback = torch::zeros_like(it->second);
                    break;
                }
            }
        }

        action_tensors[name] = stack_field(ep, T,
            [&name](const Transition& t) -> torch::Tensor {
                auto it = t.actions.find(name);
                if (it != t.actions.end()) return it->second;
                return torch::Tensor();
            },
            fallback);
    }

    // --- Generate filename ---
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    std::ostringstream filename;
    filename << "rollout_" << std::setw(6) << std::setfill('0') << dump_count_
             << "_" << ms << ".pt";

    fs::path filepath = fs::path(rollout_dir_) / filename.str();
    fs::path tmppath  = fs::path(rollout_dir_) / (filename.str() + ".tmp");

    // --- Save as custom binary format (FATE) ---
    // C++ libtorch serialization formats are NOT compatible with Python torch.load():
    //   OutputArchive → TorchScript zip → jit.load → ScriptModule, not dict
    //   pickle_save   → tensor data after STOP opcode → values become ints
    //   pickle()+zip  → PERSID format mismatch with Python's persistent_load
    // Custom binary: FATE magic + named tensors with raw data. Simple & reliable.
    try {
        // Collect all named tensors
        std::vector<std::pair<std::string, torch::Tensor>> entries;
        entries.push_back({"self_vecs", self_vecs});
        entries.push_back({"ally_vecs", ally_vecs});
        entries.push_back({"enemy_vecs", enemy_vecs});
        entries.push_back({"global_vecs", global_vecs});
        entries.push_back({"grids", grids});
        entries.push_back({"log_probs", log_probs});
        entries.push_back({"values", values});
        entries.push_back({"rewards", rewards});
        entries.push_back({"dones", dones});
        entries.push_back({"hx_h", hx_h});
        entries.push_back({"hx_c", hx_c});
        for (const auto& [name, tensor] : mask_tensors) {
            entries.push_back({"mask_" + name, tensor});
        }
        for (const auto& [name, tensor] : action_tensors) {
            entries.push_back({"act_" + name, tensor});
        }

        // Write to .tmp first, then atomic rename
        std::ofstream ofs(tmppath.string(), std::ios::binary);
        if (!ofs) {
            std::cerr << "[RolloutWriter] Cannot open " << tmppath.string() << std::endl;
            return;
        }

        // Header: magic(4) + num_entries(4)
        ofs.write("FATE", 4);
        uint32_t num_entries = static_cast<uint32_t>(entries.size());
        ofs.write(reinterpret_cast<const char*>(&num_entries), 4);

        int64_t total_bytes = 8;
        for (const auto& [name, tensor] : entries) {
            auto t = tensor.contiguous().cpu();

            // name_len(4) + name + dtype(1) + ndim(4) + shape(8*ndim) + nbytes(8) + data
            uint32_t name_len = static_cast<uint32_t>(name.size());
            ofs.write(reinterpret_cast<const char*>(&name_len), 4);
            ofs.write(name.data(), name_len);

            uint8_t dtype = static_cast<uint8_t>(t.scalar_type());
            ofs.write(reinterpret_cast<const char*>(&dtype), 1);

            uint32_t ndim = static_cast<uint32_t>(t.dim());
            ofs.write(reinterpret_cast<const char*>(&ndim), 4);
            for (int64_t d = 0; d < ndim; d++) {
                int64_t s = t.size(d);
                ofs.write(reinterpret_cast<const char*>(&s), 8);
            }

            int64_t nbytes = static_cast<int64_t>(t.nbytes());
            ofs.write(reinterpret_cast<const char*>(&nbytes), 8);
            ofs.write(reinterpret_cast<const char*>(t.data_ptr()), nbytes);

            total_bytes += 4 + name_len + 1 + 4 + 8 * ndim + 8 + nbytes;
        }

        ofs.close();

        // Atomic rename: .tmp → .pt
        fs::rename(tmppath, filepath);
        ++dump_count_;

        std::cout << "[RolloutWriter] Saved " << filepath.string()
                  << " (T=" << T << ", agents=" << MAX_UNITS
                  << ", " << total_bytes / 1024 << " KB)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[RolloutWriter] Failed to save " << filepath.string()
                  << ": " << e.what() << std::endl;
        // Clean up temp file
        fs::remove(tmppath);
    }
}
