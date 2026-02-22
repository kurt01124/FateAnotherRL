#include "state_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace state_encoder {

// ============================================================
// Binary packet parsing
// ============================================================

bool parse_packet(const uint8_t* data, size_t len,
                  PacketHeader& header,
                  GlobalState& global,
                  UnitState units[MAX_UNITS],
                  std::vector<Event>& events,
                  std::vector<uint8_t>& pathability,
                  std::vector<uint8_t>& vis_t0,
                  std::vector<uint8_t>& vis_t1,
                  std::vector<CreepState>& creeps)
{
    // Minimum size: fixed portion
    constexpr size_t FIXED_SIZE = sizeof(StatePacketFixed);
    if (len < FIXED_SIZE) {
        std::cerr << "[state_encoder] Packet too small: " << len
                  << " < " << FIXED_SIZE << std::endl;
        return false;
    }

    const StatePacketFixed* fixed = reinterpret_cast<const StatePacketFixed*>(data);

    // Validate magic and version
    if (fixed->header.magic != MAGIC) {
        std::cerr << "[state_encoder] Bad magic: 0x" << std::hex
                  << fixed->header.magic << std::dec << std::endl;
        return false;
    }
    if (fixed->header.version != PROTO_VERSION) {
        std::cerr << "[state_encoder] Bad version: " << (int)fixed->header.version << std::endl;
        return false;
    }
    if (fixed->header.msg_type != MSG_STATE) {
        std::cerr << "[state_encoder] Not a STATE packet: type="
                  << (int)fixed->header.msg_type << std::endl;
        return false;
    }

    header = fixed->header;
    global = fixed->global;
    std::memcpy(units, fixed->units, sizeof(UnitState) * MAX_UNITS);

    // Parse events
    uint8_t num_events = fixed->num_events;
    if (num_events > MAX_EVENTS) num_events = MAX_EVENTS;

    size_t offset = FIXED_SIZE;
    size_t events_size = num_events * sizeof(Event);
    if (offset + events_size > len) {
        std::cerr << "[state_encoder] Packet truncated at events" << std::endl;
        return false;
    }

    events.clear();
    events.resize(num_events);
    if (num_events > 0) {
        std::memcpy(events.data(), data + offset, events_size);
    }
    offset += events_size;

    // Parse has_pathability flag
    if (offset + 1 > len) {
        std::cerr << "[state_encoder] Packet truncated at has_pathability" << std::endl;
        return false;
    }
    uint8_t has_path = data[offset++];

    // Parse pathability grid (if present)
    pathability.clear();
    if (has_path) {
        if (offset + GRID_CELLS > len) {
            std::cerr << "[state_encoder] Packet truncated at pathability grid" << std::endl;
            return false;
        }
        pathability.assign(data + offset, data + offset + GRID_CELLS);
        offset += GRID_CELLS;
    }

    // Parse visibility grids (team 0 and team 1)
    vis_t0.clear();
    vis_t1.clear();

    if (offset + GRID_CELLS > len) {
        std::cerr << "[state_encoder] Packet truncated at visibility_t0" << std::endl;
        return false;
    }
    vis_t0.assign(data + offset, data + offset + GRID_CELLS);
    offset += GRID_CELLS;

    if (offset + GRID_CELLS > len) {
        std::cerr << "[state_encoder] Packet truncated at visibility_t1" << std::endl;
        return false;
    }
    vis_t1.assign(data + offset, data + offset + GRID_CELLS);
    offset += GRID_CELLS;

    // Parse creep data (optional, for backwards compatibility)
    creeps.clear();
    if (offset + 1 <= len) {
        uint8_t num_creeps = data[offset++];
        if (num_creeps > MAX_CREEPS) num_creeps = MAX_CREEPS;
        size_t creeps_size = num_creeps * sizeof(CreepState);
        if (offset + creeps_size <= len) {
            creeps.resize(num_creeps);
            if (num_creeps > 0) {
                std::memcpy(creeps.data(), data + offset, creeps_size);
            }
            offset += creeps_size;
        }
    }

    return true;
}

// ============================================================
// Helper: get hero_id string from char[4]
// ============================================================
static std::string hero_id_str(const char id[4]) {
    return std::string(id, 4);
}

// ============================================================
// Helper: grid cell from world coordinates
// ============================================================
static std::pair<int, int> world_to_grid(float x, float y) {
    int gx = static_cast<int>((x - MAP_MIN_X) / CELL_SIZE);
    int gy = static_cast<int>((y - MAP_MIN_Y) / CELL_SIZE);
    gx = std::max(0, std::min(gx, OBS_GRID_W - 1));
    gy = std::max(0, std::min(gy, OBS_GRID_H - 1));
    return {gx, gy};
}

// ============================================================
// Encode self unit -> (SELF_DIM,) float vector
// ============================================================
static void encode_self(const UnitState& u, float* out) {
    std::memset(out, 0, SELF_DIM * sizeof(float));

    if (!u.alive) {
        // Dead: all zeros (hero_id also zero for dead units, matching Python)
        return;
    }

    int idx = 0;

    // Basic (6)
    out[idx++] = u.hp / N::hp;
    out[idx++] = u.max_hp / N::hp;
    out[idx++] = u.mp / N::mp;
    out[idx++] = u.max_mp / N::mp;
    out[idx++] = u.x / N::xy;
    out[idx++] = u.y / N::xy;

    // Stats (5)
    out[idx++] = static_cast<float>(u.str) / N::stat;
    out[idx++] = static_cast<float>(u.agi) / N::stat;
    out[idx++] = static_cast<float>(u.int_) / N::stat;
    out[idx++] = u.atk / N::atk;
    out[idx++] = u.def_ / N::def_;

    // Upgrades (9)
    for (int k = 0; k < 9; ++k) {
        out[idx++] = u.upgrades[k] / 50.0f;
    }

    // Combat (3)
    out[idx++] = u.move_spd / N::move_spd;
    out[idx++] = u.atk_range / 1000.0f;
    out[idx++] = u.atk_spd / 3.0f;

    // Growth (4)
    out[idx++] = static_cast<float>(u.level) / N::level;
    out[idx++] = static_cast<float>(u.xp) / 50000.0f;
    out[idx++] = static_cast<float>(u.skill_points) / 10.0f;
    out[idx++] = static_cast<float>(u.stat_points) / 200.0f;

    // Skill CD (12): 6 slots x (cd_remain/N.cd, level/10.0)
    for (int s = 0; s < 6; ++s) {
        out[idx++] = u.skills[s].cd_remain / N::cd;
        out[idx++] = static_cast<float>(u.skills[s].level) / 10.0f;
    }

    // Attributes (4): bit-unpack
    for (int b = 0; b < 4; ++b) {
        out[idx++] = static_cast<float>((u.attributes >> b) & 1);
    }

    // Buffs (6): bit-unpack (stun|slow|silence|knockback|root|invuln)
    for (int b = 0; b < 6; ++b) {
        out[idx++] = static_cast<float>((u.buffs >> b) & 1);
    }

    // Seal (4)
    out[idx++] = static_cast<float>(u.seal_charges) / 12.0f;
    out[idx++] = static_cast<float>(u.seal_cd) / 30.0f;
    out[idx++] = static_cast<float>(u.seal_first_active);
    out[idx++] = u.seal_first_remain / 30.0f;

    // Items (6): type_id / 20.0
    for (int i = 0; i < 6; ++i) {
        out[idx++] = static_cast<float>(u.items[i].type_id) / 20.0f;
    }

    // Economy (3)
    out[idx++] = static_cast<float>(u.faire) / N::faire;
    out[idx++] = 0.0f;  // faire_regen placeholder
    out[idx++] = static_cast<float>(u.faire_cap) / 20000.0f;

    // Velocity (2)
    out[idx++] = u.vel_x / 500.0f;
    out[idx++] = u.vel_y / 500.0f;

    // Alive (1)
    out[idx++] = 1.0f;

    // Hero ID one-hot (12)
    std::string hid = hero_id_str(u.hero_id);
    const auto& h2i = hero_to_idx();
    auto it = h2i.find(hid);
    int hero_idx = (it != h2i.end()) ? it->second : 0;
    for (int h = 0; h < NUM_HEROES; ++h) {
        out[idx++] = (h == hero_idx) ? 1.0f : 0.0f;
    }
}

// ============================================================
// Encode ally unit -> (ALLY_DIM,) float vector
// ============================================================
static void encode_ally(const UnitState& u, float my_x, float my_y, float* out) {
    std::memset(out, 0, ALLY_DIM * sizeof(float));

    if (!u.alive) {
        return;
    }

    int idx = 0;

    // Basic (6)
    out[idx++] = u.hp / N::hp;
    out[idx++] = u.max_hp / N::hp;
    out[idx++] = u.mp / N::mp;
    out[idx++] = u.max_mp / N::mp;
    out[idx++] = u.x / N::xy;
    out[idx++] = u.y / N::xy;

    // Stats (5)
    out[idx++] = static_cast<float>(u.str) / N::stat;
    out[idx++] = static_cast<float>(u.agi) / N::stat;
    out[idx++] = static_cast<float>(u.int_) / N::stat;
    out[idx++] = u.atk / N::atk;
    out[idx++] = u.def_ / N::def_;

    // Combat (3)
    out[idx++] = u.move_spd / N::move_spd;
    out[idx++] = u.atk_range / 1000.0f;
    out[idx++] = u.atk_spd / 3.0f;

    // Growth (1)
    out[idx++] = static_cast<float>(u.level) / N::level;

    // Skill CD remain (6)
    for (int s = 0; s < 6; ++s) {
        out[idx++] = u.skills[s].cd_remain / N::cd;
    }

    // Buffs (6)
    for (int b = 0; b < 6; ++b) {
        out[idx++] = static_cast<float>((u.buffs >> b) & 1);
    }

    // Alive (1)
    out[idx++] = 1.0f;

    // Seal charges (1)
    out[idx++] = static_cast<float>(u.seal_charges) / 12.0f;

    // Faire (1)
    out[idx++] = static_cast<float>(u.faire) / N::faire;

    // Velocity (2)
    out[idx++] = u.vel_x / 500.0f;
    out[idx++] = u.vel_y / 500.0f;

    // idx = 32 here. Padding slots [32],[33] used for relative polar coords.
    {
        float dx = u.x - my_x;
        float dy = u.y - my_y;
        out[idx++] = std::atan2(dy, dx) / static_cast<float>(M_PI);  // rel_angle [-1, 1]
        out[idx++] = std::sqrt(dx * dx + dy * dy) / 10000.f;          // rel_dist normalized
    }

    // Padding (3) to reach ALLY_DIM = 37
    // idx = 34 here, remaining 3 are zeros from memset
}

// ============================================================
// Encode enemy unit -> (ENEMY_DIM,) float vector
// ============================================================
static void encode_enemy(const UnitState& u, float my_x, float my_y, int observer_pid, float* out) {
    std::memset(out, 0, ENEMY_DIM * sizeof(float));

    std::string hid = hero_id_str(u.hero_id);
    const auto& h2i = hero_to_idx();
    auto it = h2i.find(hid);
    int hero_idx = (it != h2i.end()) ? it->second : 0;

    // Per-observer visibility: check if observer can see this enemy
    bool vis = (u.visible_mask >> observer_pid) & 1;

    if (!u.alive) {
        // Dead: only encode hero_id at offset 23 (1+6+7+2+6+1 = 23)
        out[23 + hero_idx] = 1.0f;
        return;
    }

    if (!vis) {
        // Alive but NOT visible: only hero_id + alive (rest stays 0 from memset)
        out[22] = 1.0f;                  // alive
        out[23 + hero_idx] = 1.0f;       // hero_id one-hot
        return;
    }

    // --- Visible enemy: full data ---
    int idx = 0;

    // Visible (1)
    out[idx++] = 1.0f;

    // Basic (6)
    out[idx++] = u.hp / N::hp;
    out[idx++] = u.max_hp / N::hp;
    out[idx++] = u.mp / N::mp;
    out[idx++] = u.max_mp / N::mp;
    out[idx++] = u.x / N::xy;
    out[idx++] = u.y / N::xy;

    // Public stats (7)
    out[idx++] = static_cast<float>(u.str) / N::stat;
    out[idx++] = static_cast<float>(u.agi) / N::stat;
    out[idx++] = static_cast<float>(u.int_) / N::stat;
    out[idx++] = u.atk / N::atk;
    out[idx++] = u.def_ / N::def_;
    out[idx++] = u.max_hp / N::hp;
    out[idx++] = u.max_mp / N::mp;

    // Growth (2)
    out[idx++] = static_cast<float>(u.level) / N::level;
    out[idx++] = 0.0f;  // death_count placeholder

    // Buffs (6)
    for (int b = 0; b < 6; ++b) {
        out[idx++] = static_cast<float>((u.buffs >> b) & 1);
    }

    // Alive (1)
    out[idx++] = 1.0f;

    // Hero ID one-hot (12)
    for (int h = 0; h < NUM_HEROES; ++h) {
        out[idx++] = (h == hero_idx) ? 1.0f : 0.0f;
    }

    // Velocity (2)
    out[idx++] = u.vel_x / 500.0f;
    out[idx++] = u.vel_y / 500.0f;

    // Belief attributes (4) - placeholder, all -1
    out[idx++] = -1.0f;
    out[idx++] = -1.0f;
    out[idx++] = -1.0f;
    out[idx++] = -1.0f;

    // Relative polar coordinates (2) — angle and distance from self
    {
        float dx = u.x - my_x;
        float dy = u.y - my_y;
        out[idx++] = std::atan2(dy, dx) / static_cast<float>(M_PI);  // [-1, 1]
        out[idx++] = std::sqrt(dx * dx + dy * dy) / 10000.f;          // normalized
    }
}

// ============================================================
// Encode global state -> (GLOBAL_DIM,) float vector
// ============================================================
static void encode_global(const GlobalState& g, int my_team, float* out) {
    int idx = 0;
    out[idx++] = g.game_time / N::game_time;
    out[idx++] = static_cast<float>(g.is_night);

    // Scores from agent's perspective
    if (my_team == 0) {
        out[idx++] = static_cast<float>(g.score_team0) / N::score;
        out[idx++] = static_cast<float>(g.score_team1) / N::score;
    } else {
        out[idx++] = static_cast<float>(g.score_team1) / N::score;
        out[idx++] = static_cast<float>(g.score_team0) / N::score;
    }

    out[idx++] = static_cast<float>(g.c_rank_stock) / 8.0f;
    out[idx++] = 0.0f;  // padding
}

// ============================================================
// Portal definitions (entrance <-> exit, bidirectional)
// ============================================================
struct PortalDef {
    float x, y;     // center of entrance rect
    float ex, ey;   // center of exit rect
};
static const PortalDef PORTALS[] = {
    // Fuyuki <-> Ryudou Temple
    {-7328.f, 2128.f,  -2048.f, 7296.f},
    // Fuyuki <-> Tohsaka Mansion
    {-2288.f, -512.f,  -8000.f, -5376.f},
    // Fuyuki <-> Matou Mansion
    {-2800.f, -208.f,  -3328.f, -8256.f},
    // Fuyuki <-> Einzbern Castle
    {-6816.f, -1568.f,  6288.f, -6208.f},
    // Fuyuki <-> Emiya Mansion
    {-5920.f, 4800.f,  -6912.f, 7488.f},
    // Fuyuki <-> School
    {-3856.f, 1152.f,   3072.f, 7360.f},
    // Fuyuki <-> Church
    { 6816.f,  -80.f,   7232.f, 9984.f},
    // Internal warp
    { 2576.f, 5031.f,   2125.f, 5155.f},
};
static constexpr int NUM_PORTALS = sizeof(PORTALS) / sizeof(PORTALS[0]);

// ============================================================
// Encode grid -> (6, GRID_H, GRID_W) float
// ch0: pathability, ch1: ally, ch2: visible enemy,
// ch3: portal, ch4: creep position, ch5: creep HP (visible only)
// ============================================================
static void encode_grid(int my_team,
                        int observer_pid,
                        const UnitState units[MAX_UNITS],
                        const std::vector<uint8_t>& pathability,
                        const std::vector<uint8_t>& vis_t0,
                        const std::vector<uint8_t>& vis_t1,
                        const std::vector<CreepState>& creeps,
                        float* out)
{
    const int plane_size = OBS_GRID_H * OBS_GRID_W;  // 1200
    std::memset(out, 0, GRID_CHANNELS * plane_size * sizeof(float));

    // Channel 0: pathability / 2.0
    if (!pathability.empty() && static_cast<int>(pathability.size()) == plane_size) {
        for (int i = 0; i < plane_size; ++i) {
            out[i] = static_cast<float>(pathability[i]) / 2.0f;
        }
    }

    // Channel 1: ally positions, Channel 2: visible enemy positions (per-observer)
    float* ch1 = out + plane_size;
    float* ch2 = out + 2 * plane_size;

    for (int i = 0; i < MAX_UNITS; ++i) {
        if (!units[i].alive) continue;

        auto [gx, gy] = world_to_grid(units[i].x, units[i].y);
        int cell = gy * OBS_GRID_W + gx;

        int unit_team = (i < 6) ? 0 : 1;
        if (unit_team == my_team) {
            ch1[cell] = 1.0f;
        } else {
            bool vis_to_me = (units[i].visible_mask >> observer_pid) & 1;
            if (vis_to_me) {
                ch2[cell] = 1.0f;
            }
        }
    }

    // Channel 3: portal locations (static, both entrance and exit)
    float* ch3 = out + 3 * plane_size;
    for (int p = 0; p < NUM_PORTALS; ++p) {
        auto [gx1, gy1] = world_to_grid(PORTALS[p].x, PORTALS[p].y);
        ch3[gy1 * OBS_GRID_W + gx1] = 1.0f;
        auto [gx2, gy2] = world_to_grid(PORTALS[p].ex, PORTALS[p].ey);
        ch3[gy2 * OBS_GRID_W + gx2] = 1.0f;
    }

    // Channel 4: creep positions (always visible), Channel 5: creep HP (visible only)
    float* ch4 = out + 4 * plane_size;
    float* ch5 = out + 5 * plane_size;

    // Use observer's team visibility grid to check creep visibility
    const auto& vis_grid = (my_team == 0) ? vis_t0 : vis_t1;

    for (size_t c = 0; c < creeps.size(); ++c) {
        if (creeps[c].max_hp <= 0.0f) continue;  // skip invalid
        float hp_ratio = creeps[c].hp / creeps[c].max_hp;
        if (hp_ratio <= 0.0f) continue;  // dead creep

        auto [gx, gy] = world_to_grid(creeps[c].x, creeps[c].y);
        int cell = gy * OBS_GRID_W + gx;

        // Always mark position
        ch4[cell] = 1.0f;

        // HP only if observer's team has visibility of that cell
        if (!vis_grid.empty() && vis_grid[cell]) {
            ch5[cell] = hp_ratio;
        }
    }
}

// ============================================================
// encode(): Full state -> EncodedObs for all 12 agents
// ============================================================

EncodedObs encode(const UnitState units[MAX_UNITS],
                  const GlobalState& global,
                  const std::vector<uint8_t>& pathability,
                  const std::vector<uint8_t>& vis_t0,
                  const std::vector<uint8_t>& vis_t1,
                  const std::vector<CreepState>& creeps)
{
    EncodedObs obs;

    // Pre-allocate tensors
    obs.self_vec   = torch::zeros({MAX_UNITS, SELF_DIM});
    obs.ally_vec   = torch::zeros({MAX_UNITS, 5, ALLY_DIM});
    obs.enemy_vec  = torch::zeros({MAX_UNITS, 6, ENEMY_DIM});
    obs.global_vec = torch::zeros({MAX_UNITS, GLOBAL_DIM});
    obs.grid       = torch::zeros({MAX_UNITS, GRID_CHANNELS, OBS_GRID_H, OBS_GRID_W});

    auto self_acc   = obs.self_vec.accessor<float, 2>();
    auto ally_acc   = obs.ally_vec.accessor<float, 3>();
    auto enemy_acc  = obs.enemy_vec.accessor<float, 3>();
    auto global_acc = obs.global_vec.accessor<float, 2>();

    for (int i = 0; i < MAX_UNITS; ++i) {
        int team = (i < 6) ? 0 : 1;

        // Self vector
        encode_self(units[i], &self_acc[i][0]);

        // Self position for relative features
        float my_x = units[i].x;
        float my_y = units[i].y;

        // Allies (5 other same-team units)
        int ally_idx = 0;
        for (int j = team * 6; j < team * 6 + 6; ++j) {
            if (j == i) continue;
            encode_ally(units[j], my_x, my_y, &ally_acc[i][ally_idx][0]);
            ++ally_idx;
        }

        // Enemies (6 opposite-team units) — sorted by distance (visible first)
        int enemy_start = (team == 0) ? 6 : 0;

        // Build sortable array: (offset 0-5, distance, visible, alive)
        struct EnemySort {
            int offset;       // 0-5 within enemy team
            float dist_sq;
            bool visible;
            bool alive;
        };
        std::array<EnemySort, 6> enemy_sort;
        for (int j = 0; j < 6; ++j) {
            const auto& eu = units[enemy_start + j];
            float dx = eu.x - my_x;
            float dy = eu.y - my_y;
            bool vis_to_me = (eu.visible_mask >> i) & 1;
            enemy_sort[j] = {j, dx * dx + dy * dy, vis_to_me, eu.alive != 0};
        }

        // Sort: visible+alive first → then by distance → tiebreak by player_id
        std::stable_sort(enemy_sort.begin(), enemy_sort.end(),
            [](const EnemySort& a, const EnemySort& b) {
                int a_rank = (a.alive && a.visible) ? 0 : (a.alive ? 1 : 2);
                int b_rank = (b.alive && b.visible) ? 0 : (b.alive ? 1 : 2);
                if (a_rank != b_rank) return a_rank < b_rank;
                if (a.dist_sq != b.dist_sq) return a.dist_sq < b.dist_sq;
                return a.offset < b.offset;
            });

        // Record sort mapping and encode in sorted order
        for (int j = 0; j < 6; ++j) {
            obs.sort_map.sorted_to_real[i][j] = enemy_sort[j].offset;
            encode_enemy(units[enemy_start + enemy_sort[j].offset],
                         my_x, my_y, i, &enemy_acc[i][j][0]);
        }

        // Global
        encode_global(global, team, &global_acc[i][0]);

        // Grid (per-observer perspective)
        float* grid_ptr = obs.grid[i].data_ptr<float>();
        encode_grid(team, i, units, pathability, vis_t0, vis_t1, creeps, grid_ptr);
    }

    return obs;
}

// ============================================================
// encode_masks(): Extract action masks from bit-packed fields
// ============================================================

MaskSet encode_masks(const UnitState units[MAX_UNITS],
                     const EnemySortMapping* sort_map) {
    MaskSet ms;

    const auto& heads = discrete_heads();

    // Initialize all mask tensors
    for (int h = 0; h < NUM_DISCRETE_HEADS; ++h) {
        ms.masks[heads[h].name] = torch::ones({MAX_UNITS, heads[h].size},
                                               torch::kBool);
    }

    for (int i = 0; i < MAX_UNITS; ++i) {
        const auto& u = units[i];

        // skill: 8 bits from mask_skill
        {
            auto acc = ms.masks["skill"].accessor<bool, 2>();
            for (int b = 0; b < 8; ++b)
                acc[i][b] = mask_bit(u.mask_skill, b);
        }

        // unit_target: 14 bits from mask_unit_target (16-bit field)
        // Bits 0-5: allies (self team), bits 6-7: special, bits 8-13: enemies
        // If sort_map provided, remap enemy bits 8-13 from real → sorted order
        // unit_target layout: [self_allies(6) | no_target(1) | attack_point(1) | enemies(6)]
        //   bits 0-5: same-team targets (fixed order)
        //   bit 6: no-target (12)
        //   bit 7: attack-point (13)
        //   bits 8-13: enemy targets (remapped by distance sort)
        {
            auto acc = ms.masks["unit_target"].accessor<bool, 2>();
            // First 8 bits: allies + special — no remapping
            for (int b = 0; b < 8; ++b)
                acc[i][b] = mask_bit16(u.mask_unit_target, b);

            if (sort_map) {
                // Remap enemy bits 8-13: sorted_slot → real enemy offset
                for (int sorted_slot = 0; sorted_slot < 6; ++sorted_slot) {
                    int real_offset = sort_map->sorted_to_real[i][sorted_slot];
                    acc[i][8 + sorted_slot] = mask_bit16(u.mask_unit_target, 8 + real_offset);
                }
            } else {
                for (int b = 8; b < 14; ++b)
                    acc[i][b] = mask_bit16(u.mask_unit_target, b);
            }
        }

        // skill_levelup: 6 bits from mask_skill_levelup
        {
            auto acc = ms.masks["skill_levelup"].accessor<bool, 2>();
            for (int b = 0; b < 6; ++b)
                acc[i][b] = mask_bit(u.mask_skill_levelup, b);
        }

        // stat_upgrade: 10 bits from mask_stat_upgrade (16-bit field)
        {
            auto acc = ms.masks["stat_upgrade"].accessor<bool, 2>();
            for (int b = 0; b < 10; ++b)
                acc[i][b] = mask_bit16(u.mask_stat_upgrade, b);
        }

        // attribute: 5 bits from mask_attribute
        {
            auto acc = ms.masks["attribute"].accessor<bool, 2>();
            for (int b = 0; b < 5; ++b)
                acc[i][b] = mask_bit(u.mask_attribute, b);
        }

        // item_buy: 18 bits from mask_item_buy (32-bit field)
        {
            auto acc = ms.masks["item_buy"].accessor<bool, 2>();
            for (int b = 0; b < 18; ++b)
                acc[i][b] = mask_bit32(u.mask_item_buy, b);
        }

        // item_use: 7 bits from mask_item_use
        {
            auto acc = ms.masks["item_use"].accessor<bool, 2>();
            for (int b = 0; b < 7; ++b)
                acc[i][b] = mask_bit(u.mask_item_use, b);
        }

        // seal_use: 7 bits from mask_seal_use
        {
            auto acc = ms.masks["seal_use"].accessor<bool, 2>();
            for (int b = 0; b < 7; ++b)
                acc[i][b] = mask_bit(u.mask_seal_use, b);
        }

        // faire_send: 6 bits from mask_faire_send
        {
            auto acc = ms.masks["faire_send"].accessor<bool, 2>();
            for (int b = 0; b < 6; ++b)
                acc[i][b] = mask_bit(u.mask_faire_send, b);
        }

        // faire_request: 6 bits from mask_faire_request
        {
            auto acc = ms.masks["faire_request"].accessor<bool, 2>();
            for (int b = 0; b < 6; ++b)
                acc[i][b] = mask_bit(u.mask_faire_request, b);
        }

        // faire_respond: 3 bits from mask_faire_respond
        {
            auto acc = ms.masks["faire_respond"].accessor<bool, 2>();
            for (int b = 0; b < 3; ++b)
                acc[i][b] = mask_bit(u.mask_faire_respond, b);
        }
    }

    return ms;
}

} // namespace state_encoder
