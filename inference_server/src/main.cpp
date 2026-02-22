#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <thread>

#include <torch/torch.h>

#include "protocol.h"
#include "constants.h"
#include "udp_server.h"
#include "state_encoder.h"
#include "inference_engine.h"
#include "reward_calc.h"
#include "rollout_writer.h"

// ============================================================
// Per-instance state tracking
// ============================================================
struct InstanceState {
    // LSTM hidden states per hero (keyed by hero_id string)
    std::unordered_map<std::string, torch::Tensor> hx_h;  // (1, 1, 256)
    std::unordered_map<std::string, torch::Tensor> hx_c;

    // Previous state for reward computation
    UnitState prev_units[MAX_UNITS];
    GlobalState prev_global{};
    bool has_prev = false;

    // Reward calculator per instance
    RewardCalc reward_calc;

    // Last tick seen
    uint32_t last_tick = 0;

    // Last time we received a packet from this instance (for timeout detection)
    std::chrono::steady_clock::time_point last_recv_time = std::chrono::steady_clock::now();
};

// ============================================================
// Command-line argument parsing
// ============================================================
struct Config {
    int listen_port  = 7777;
    int send_port    = 7778;
    std::string device_str = "cpu";
    std::string model_dir = "./models";
    std::string rollout_dir = "./rollouts";
    int rollout_size = 4096;
    int reload_interval_sec = 5;
};

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc)
            cfg.listen_port = std::stoi(argv[++i]);
        else if ((arg == "--send-port" || arg == "--action-port") && i + 1 < argc)
            cfg.send_port = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc)
            cfg.device_str = argv[++i];
        else if (arg == "--model-dir" && i + 1 < argc)
            cfg.model_dir = argv[++i];
        else if (arg == "--rollout-dir" && i + 1 < argc)
            cfg.rollout_dir = argv[++i];
        else if (arg == "--rollout-size" && i + 1 < argc)
            cfg.rollout_size = std::stoi(argv[++i]);
        else if (arg == "--reload-interval" && i + 1 < argc)
            cfg.reload_interval_sec = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: fate_inference_server [options]\n"
                      << "  --port <int>           Listen port (default: 7777)\n"
                      << "  --action-port <int>    Reply port (default: 7778)\n"
                      << "  --device <str>         Torch device (default: cpu)\n"
                      << "  --model-dir <path>     Model directory (default: ./models)\n"
                      << "  --rollout-dir <path>   Rollout output dir (default: ./rollouts)\n"
                      << "  --rollout-size <int>   Min transitions before dump (default: 4096)\n"
                      << "  --reload-interval <int> Model reload check seconds (default: 5)\n";
            std::exit(0);
        }
    }
    return cfg;
}

// ============================================================
// Extract IP from "ip:port" string
// ============================================================
static std::string extract_ip(const std::string& addr) {
    auto pos = addr.rfind(':');
    if (pos != std::string::npos)
        return addr.substr(0, pos);
    return addr;
}

// ============================================================
// Build instance key from source address (use IP only,
// since multiple WC3 instances may use different ephemeral ports)
// ============================================================
static std::string instance_key(const std::string& addr) {
    return extract_ip(addr);
}

// ============================================================
// Build and send ACTION packet
// ============================================================
static void send_action_packet(
    UdpServer& server,
    const std::string& addr,
    uint32_t tick,
    const std::array<InferenceEngine::InferResult, MAX_UNITS>& results,
    const UnitState units[MAX_UNITS],
    const EnemySortMapping* sort_map = nullptr)
{
    ActionPacket pkt;
    std::memset(&pkt, 0, sizeof(pkt));

    pkt.header.magic = MAGIC;
    pkt.header.version = PROTO_VERSION;
    pkt.header.msg_type = MSG_ACTION;
    pkt.header.tick = tick;

    for (int i = 0; i < MAX_UNITS; ++i) {
        auto& ua = pkt.actions[i];
        ua.idx = static_cast<uint8_t>(i);
        const auto& r = results[i];

        // Move continuous action -> clamp to [-1, 1]
        auto move_it = r.actions.find("move");
        if (move_it != r.actions.end()) {
            auto mv = move_it->second.cpu().contiguous();
            float mx = mv.dim() > 1 ? mv[0][0].item<float>() : mv[0].item<float>();
            float my = mv.dim() > 1 ? mv[0][1].item<float>() : mv[1].item<float>();
            ua.move_x = std::max(-1.0f, std::min(1.0f, mx));
            ua.move_y = std::max(-1.0f, std::min(1.0f, my));
        }

        // Point continuous action
        auto point_it = r.actions.find("point");
        if (point_it != r.actions.end()) {
            auto pt = point_it->second.cpu().contiguous();
            float px = pt.dim() > 1 ? pt[0][0].item<float>() : pt[0].item<float>();
            float py = pt.dim() > 1 ? pt[0][1].item<float>() : pt[1].item<float>();
            ua.point_x = std::max(-1.0f, std::min(1.0f, px));
            ua.point_y = std::max(-1.0f, std::min(1.0f, py));
        }

        // Discrete actions
        auto get_int = [&](const char* name) -> uint8_t {
            auto it = r.actions.find(name);
            if (it != r.actions.end())
                return static_cast<uint8_t>(it->second.item<int64_t>());
            return 0;
        };

        ua.skill          = get_int("skill");
        ua.unit_target    = get_int("unit_target");

        // Remap enemy target from sorted slot to real player offset
        // unit_target layout: 0-5=allies, 6-7=special(no_target,attack_point), 8-13=enemies
        if (sort_map && ua.unit_target >= 8 && ua.unit_target <= 13) {
            int sorted_slot = ua.unit_target - 8;
            int real_offset = sort_map->sorted_to_real[i][sorted_slot];
            ua.unit_target = static_cast<uint8_t>(8 + real_offset);
        }

        ua.skill_levelup  = get_int("skill_levelup");
        ua.stat_upgrade   = get_int("stat_upgrade");
        ua.attribute      = get_int("attribute");
        ua.item_buy       = get_int("item_buy");
        ua.item_use       = get_int("item_use");
        ua.seal_use       = get_int("seal_use");
        ua.faire_send     = get_int("faire_send");
        ua.faire_request  = get_int("faire_request");
        ua.faire_respond  = get_int("faire_respond");
    }

    server.send_to(addr, reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt));
}

// ============================================================
// Main loop
// ============================================================

int main(int argc, char* argv[]) {
    auto cfg = parse_args(argc, argv);

    // Determine device
    torch::Device device(torch::kCPU);
    if (cfg.device_str == "cuda" && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "[main] Using CUDA" << std::endl;
    } else {
        std::cout << "[main] Using CPU" << std::endl;
    }

    // Initialize components
    UdpServer server(cfg.listen_port, cfg.send_port);
    InferenceEngine engine(cfg.model_dir, device);
    RolloutWriter writer(cfg.rollout_dir);

    // Per-instance state
    std::unordered_map<std::string, InstanceState> instances;

    // Timing for periodic tasks
    auto last_reload = std::chrono::steady_clock::now();
    auto last_stats = std::chrono::steady_clock::now();
    uint64_t total_packets = 0;
    uint64_t total_inferences = 0;

    std::cout << "[main] Inference server running. Press Ctrl+C to stop." << std::endl;

    uint64_t total_skipped = 0;

    while (true) {
        // 1. Receive all pending packets
        auto packets = server.recv_all();

        if (packets.empty()) {
            // No data: sleep briefly to avoid busy-wait
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            // Continue to periodic tasks
        }

        // ============================================================
        // Phase 1: Classify packets — keep DONE + latest STATE per instance
        // This prevents buffer overflow from dropping critical DONE packets.
        // With 3 WC3 containers sending ~100 STATE/s each, processing
        // every STATE is impossible (~12ms per inference × 300 pkt/s).
        // Instead, only process the LATEST STATE per instance per cycle.
        // ============================================================
        struct ClassifiedPacket {
            std::string addr;
            std::string inst_id;
            const uint8_t* data;
            size_t size;
            uint8_t msg_type;
            uint32_t tick;
        };

        std::vector<ClassifiedPacket> done_packets;
        // inst_id → index into packets (latest STATE per instance)
        std::unordered_map<std::string, size_t> latest_state;
        uint64_t skipped_this_cycle = 0;

        for (size_t pi = 0; pi < packets.size(); ++pi) {
            auto& [addr, raw_data] = packets[pi];
            if (raw_data.size() < sizeof(PacketHeader)) continue;

            const PacketHeader* hdr = reinterpret_cast<const PacketHeader*>(raw_data.data());
            if (hdr->magic != MAGIC || hdr->version != PROTO_VERSION) continue;

            std::string inst_id = instance_key(addr);

            if (hdr->msg_type == MSG_DONE) {
                done_packets.push_back({addr, inst_id, raw_data.data(), raw_data.size(),
                                        hdr->msg_type, hdr->tick});
            } else if (hdr->msg_type == MSG_STATE) {
                auto it = latest_state.find(inst_id);
                if (it != latest_state.end()) {
                    // Already have a STATE for this instance — keep the newer one
                    auto& prev = packets[it->second];
                    const PacketHeader* prev_hdr = reinterpret_cast<const PacketHeader*>(prev.second.data());
                    if (hdr->tick >= prev_hdr->tick) {
                        it->second = pi;  // replace with newer
                    }
                    ++skipped_this_cycle;
                } else {
                    latest_state[inst_id] = pi;
                }
            }
        }
        total_skipped += skipped_this_cycle;

        // ============================================================
        // Phase 2: Process DONE packets first (critical, never skip)
        // ============================================================
        for (auto& dp : done_packets) {
            if (dp.size < sizeof(DonePacket)) continue;
            const DonePacket* done = reinterpret_cast<const DonePacket*>(dp.data);

            std::cout << "[main] DONE from " << dp.inst_id
                      << " winner=" << (int)done->winner
                      << " reason=" << (int)done->reason
                      << " score=" << done->score_team0 << "-" << done->score_team1
                      << " tick=" << dp.tick
                      << std::endl;

            // Compute terminal rewards
            auto it = instances.find(dp.inst_id);
            if (it != instances.end()) {
                auto terminal_r = it->second.reward_calc.compute_terminal(
                    done->winner, done->reason);

                writer.mark_last_done(dp.inst_id, terminal_r);
                writer.flush_episode(dp.inst_id);
                instances.erase(it);
            }

            // Remove from latest_state if present (don't process STATE after DONE)
            latest_state.erase(dp.inst_id);
        }

        // ============================================================
        // Phase 3: Process latest STATE per instance
        // ============================================================
        for (auto& [inst_id, pkt_idx] : latest_state) {
            auto& [addr, raw_data] = packets[pkt_idx];

            // Parse binary state
            PacketHeader header;
            GlobalState global;
            UnitState units[MAX_UNITS];
            std::vector<Event> events;
            std::vector<uint8_t> pathability, vis_t0, vis_t1;
            std::vector<CreepState> creeps;

            if (!state_encoder::parse_packet(raw_data.data(), raw_data.size(),
                                             header, global, units,
                                             events, pathability, vis_t0, vis_t1,
                                             creeps)) {
                std::cerr << "[main] Failed to parse STATE from " << addr << std::endl;
                continue;
            }

            ++total_packets;

            // Get or create instance state
            auto& inst = instances[inst_id];
            if (inst.last_tick == 0 && header.tick > 0) {
                std::cout << "[main] New instance: " << inst_id
                          << " tick=" << header.tick << std::endl;
            } else if (inst.last_tick > 0 && header.tick < inst.last_tick) {
                // Tick went backwards → new episode from same IP
                std::cout << "[main] Tick reset: " << inst_id
                          << " old_tick=" << inst.last_tick
                          << " new_tick=" << header.tick << std::endl;
                std::array<float, MAX_UNITS> zero_rewards{};
                writer.mark_last_done(inst_id, zero_rewards);
                writer.flush_episode(inst_id);
                inst = InstanceState{};  // reset
            }
            inst.last_tick = header.tick;
            inst.last_recv_time = std::chrono::steady_clock::now();

            // Encode state -> tensors (with distance-sorted enemies)
            std::cerr << "[main] Encoding state..." << std::endl;
            EncodedObs obs = state_encoder::encode(units, global, pathability, vis_t0, vis_t1, creeps);
            MaskSet masks = state_encoder::encode_masks(units, &obs.sort_map);

            // Compute rewards (from previous state to current)
            std::cerr << "[main] Computing rewards..." << std::endl;
            auto rewards = inst.reward_calc.compute(
                units, global, events, inst.prev_units, inst.prev_global, inst.has_prev);
            std::cerr << "[main] Rewards done, starting inference..." << std::endl;

            // Save INPUT hidden states BEFORE inference (for rollout storage)
            std::array<torch::Tensor, MAX_UNITS> input_hx_h;
            std::array<torch::Tensor, MAX_UNITS> input_hx_c;

            // Run inference for all 12 heroes
            std::array<InferenceEngine::InferResult, MAX_UNITS> results;

            for (int i = 0; i < MAX_UNITS; ++i) {
                std::string hero_id(units[i].hero_id, 4);

                // Get or init LSTM hidden state
                if (inst.hx_h.find(hero_id) == inst.hx_h.end()) {
                    auto [h, c] = engine.init_hidden();
                    inst.hx_h[hero_id] = h;
                    inst.hx_c[hero_id] = c;
                }

                // Save INPUT hx before inference (detach + cpu for storage)
                input_hx_h[i] = inst.hx_h[hero_id].detach().cpu();
                input_hx_c[i] = inst.hx_c[hero_id].detach().cpu();

                // Slice per-agent tensors: (12, ...) -> (1, ...)
                auto self_i   = obs.self_vec[i].unsqueeze(0).to(device);
                auto ally_i   = obs.ally_vec[i].unsqueeze(0).to(device);
                auto enemy_i  = obs.enemy_vec[i].unsqueeze(0).to(device);
                auto global_i = obs.global_vec[i].unsqueeze(0).to(device);
                auto grid_i   = obs.grid[i].unsqueeze(0).to(device);

                // Per-agent masks
                std::unordered_map<std::string, torch::Tensor> agent_masks;
                for (const auto& [name, mask_tensor] : masks.masks) {
                    agent_masks[name] = mask_tensor[i].unsqueeze(0).to(device);
                }

                try {
                    results[i] = engine.infer_hero(
                        hero_id, self_i, ally_i, enemy_i, global_i, grid_i,
                        inst.hx_h[hero_id], inst.hx_c[hero_id],
                        agent_masks
                    );
                } catch (const std::exception& e) {
                    std::cerr << "[main] Inference error hero=" << hero_id
                              << " i=" << i << ": " << e.what() << std::endl;
                    // Print mask shapes for debugging
                    for (const auto& [mname, mt] : agent_masks) {
                        std::cerr << "  mask[" << mname << "] shape=";
                        for (int d = 0; d < mt.dim(); ++d) std::cerr << mt.size(d) << (d+1<mt.dim()? "x" : "");
                        std::cerr << std::endl;
                    }
                    // Use default (no-op) result
                    results[i].actions["move"] = torch::zeros({1, 2});
                    results[i].actions["point"] = torch::zeros({1, 2});
                    results[i].log_prob = torch::zeros({1});
                    results[i].value = torch::zeros({1});
                    results[i].new_h = inst.hx_h[hero_id];
                    results[i].new_c = inst.hx_c[hero_id];
                    const auto& heads = discrete_heads();
                    for (int h = 0; h < NUM_DISCRETE_HEADS; ++h)
                        results[i].actions[heads[h].name] = torch::zeros({1}, torch::kLong);
                }

                // Update LSTM hidden state
                inst.hx_h[hero_id] = results[i].new_h;
                inst.hx_c[hero_id] = results[i].new_c;

                ++total_inferences;
            }

            // Store transitions in rollout buffer
            if (inst.has_prev) {
                for (int i = 0; i < MAX_UNITS; ++i) {
                    writer.store(
                        inst_id, i,
                        obs.self_vec[i],
                        obs.ally_vec[i],
                        obs.enemy_vec[i],
                        obs.global_vec[i],
                        obs.grid[i],
                        [&]() -> std::unordered_map<std::string, torch::Tensor> {
                            std::unordered_map<std::string, torch::Tensor> m;
                            for (const auto& [name, t] : masks.masks) {
                                m[name] = t[i];
                            }
                            return m;
                        }(),
                        results[i].actions,
                        results[i].log_prob.item<float>(),
                        results[i].value.item<float>(),
                        rewards[i],
                        false,  // not done
                        input_hx_h[i],
                        input_hx_c[i]
                    );
                }
            }

            // Save current state as previous for next tick
            std::memcpy(inst.prev_units, units, sizeof(UnitState) * MAX_UNITS);
            inst.prev_global = global;
            inst.has_prev = true;

            // Send ACTION packet back (with enemy sort mapping for target remapping)
            send_action_packet(server, addr, header.tick, results, units, &obs.sort_map);
        }

        // --------------------------------------------------------
        // Periodic tasks
        // --------------------------------------------------------
        auto now = std::chrono::steady_clock::now();

        // Model hot-reload check
        auto reload_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_reload).count();
        if (reload_elapsed >= cfg.reload_interval_sec) {
            engine.maybe_reload();
            last_reload = now;
        }

        // No timeout — episodes end only via DONE packet or tick reset

        // Rollout dump check
        writer.maybe_dump(cfg.rollout_size);

        // Stats logging every 30 seconds
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_stats).count();
        if (stats_elapsed >= 30) {
            std::cout << "[main] Stats: " << total_packets << " packets, "
                      << total_inferences << " inferences, "
                      << instances.size() << " active instances, "
                      << total_skipped << " skipped" << std::endl;
            last_stats = now;
        }
    }

    return 0;
}
