#pragma once

#include <cstdint>

// Binary protocol for FateAnother RL communication
// C# (BinaryWriter, little-endian) <-> C++ (little-endian structs)
// All structs are packed to match C# BinaryWriter output exactly.

#pragma pack(push, 1)

// ============================================================
// Packet Header (8 bytes)
// ============================================================
constexpr uint16_t MAGIC = 0xFA7E;
constexpr uint8_t  PROTO_VERSION = 1;

enum MsgType : uint8_t {
    MSG_STATE  = 1,
    MSG_ACTION = 2,
    MSG_DONE   = 3,
};

struct PacketHeader {
    uint16_t magic;      // 0xFA7E
    uint8_t  version;    // 1
    uint8_t  msg_type;   // MsgType
    uint32_t tick;
};
static_assert(sizeof(PacketHeader) == 8, "PacketHeader must be 8 bytes");

// ============================================================
// Skill Slot (14 bytes)
// ============================================================
struct SkillSlot {
    int32_t  abil_id;    // FourCC ability id (0 if empty)
    uint8_t  level;      // 0-5
    float    cd_remain;  // seconds
    float    cd_max;     // seconds
    uint8_t  exists;     // 1 if skill exists
};
static_assert(sizeof(SkillSlot) == 14, "SkillSlot must be 14 bytes");

// ============================================================
// Item Slot (4 bytes)
// ============================================================
struct ItemSlot {
    int16_t  type_id;    // item type index (0 = empty)
    uint8_t  charges;
    uint8_t  padding;
};
static_assert(sizeof(ItemSlot) == 4, "ItemSlot must be 4 bytes");

// ============================================================
// Unit State (~220 bytes)
// ============================================================
struct UnitState {
    // Identity (6 bytes)
    uint8_t  idx;            // 0-11
    char     hero_id[4];     // "H000" etc
    uint8_t  team;           // 0 or 1

    // Basic (37 bytes)
    float    hp;
    float    max_hp;
    float    mp;
    float    max_mp;
    float    x;
    float    y;
    float    vel_x;
    float    vel_y;
    uint8_t  alive;
    float    revive_remain;

    // Stats (24 bytes)
    int16_t  str;
    int16_t  agi;
    int16_t  int_;
    float    atk;
    float    def_;
    float    move_spd;
    float    atk_range;
    float    atk_spd;

    // Progression (8 bytes)
    uint8_t  level;
    uint8_t  skill_points;
    uint8_t  stat_points;
    uint8_t  _pad_prog;
    int32_t  xp;

    // Skills: 6 slots (84 bytes)
    SkillSlot skills[6];

    // Upgrades (9 bytes)
    uint8_t  upgrades[9];

    // Attributes (1 byte, bit-packed: bits 0-3 = A,B,C,D)
    uint8_t  attributes;

    // Buffs (1 byte, bit-packed: stun|slow|silence|knockback|root|invuln)
    uint8_t  buffs;

    // Seal (8 bytes)
    uint8_t  seal_charges;
    int16_t  seal_cd;
    uint8_t  seal_first_active;
    float    seal_first_remain;

    // Items: 6 slots (24 bytes)
    ItemSlot items[6];

    // Economy (8 bytes)
    int32_t  faire;
    int16_t  faire_cap;
    uint8_t  _pad_econ[2];

    // Flags (3 bytes)
    uint8_t  enemy_alarm;
    uint16_t visible_mask;       // bit i = visible to player i (12 bits used)

    // Action Masks (bit-packed, 13 bytes)
    // skill(8 bits), unit_target(16 bits), skill_levelup(8 bits),
    // stat_upgrade(16 bits), attribute(8 bits), item_buy(32 bits),
    // item_use(8 bits), seal_use(8 bits), faire_send(8 bits),
    // faire_request(8 bits), faire_respond(8 bits)
    uint8_t  mask_skill;            // 8 bits  (indices 0-7)
    uint16_t mask_unit_target;      // 16 bits (indices 0-13, 2 spare)
    uint8_t  mask_skill_levelup;    // 8 bits  (indices 0-5, 2 spare)
    uint16_t mask_stat_upgrade;     // 16 bits (indices 0-9, 6 spare)
    uint8_t  mask_attribute;        // 8 bits  (indices 0-4, 3 spare)
    uint32_t mask_item_buy;         // 32 bits (indices 0-16, 17 used)
    uint8_t  mask_item_use;         // 8 bits  (indices 0-6, 1 spare)
    uint8_t  mask_seal_use;         // 8 bits  (indices 0-6, 1 spare)
    uint8_t  mask_faire_send;       // 8 bits  (indices 0-5, 2 spare)
    uint8_t  mask_faire_request;    // 8 bits  (indices 0-5, 2 spare)
    uint8_t  mask_faire_respond;    // 8 bits  (indices 0-2, 5 spare)
};

// ============================================================
// Event (8 bytes)
// ============================================================
enum EventType : uint8_t {
    EVT_KILL       = 1,
    EVT_CREEP_KILL = 2,
    EVT_LEVEL_UP   = 3,
};

struct Event {
    uint8_t  type;          // EventType
    uint8_t  killer_idx;    // or unit_idx for LEVEL_UP
    uint8_t  victim_idx;    // or new_level for LEVEL_UP
    uint8_t  padding;
    uint32_t tick;
};
static_assert(sizeof(Event) == 8, "Event must be 8 bytes");

// ============================================================
// Global State (28 bytes)
// ============================================================
struct GlobalState {
    float    game_time;
    float    time_of_day;
    float    next_point_time;
    uint8_t  is_night;
    uint8_t  _pad_global[3];
    int16_t  score_team0;
    int16_t  score_team1;
    int16_t  target_score;
    int16_t  c_rank_stock;
    float    _reserved;     // padding to 28 bytes
};
static_assert(sizeof(GlobalState) == 28, "GlobalState must be 28 bytes");

// ============================================================
// State Packet (variable length)
// ============================================================
constexpr int MAX_UNITS   = 12;
constexpr int MAX_EVENTS  = 32;
constexpr int GRID_W      = 48;
constexpr int GRID_H      = 25;
constexpr int GRID_CELLS  = GRID_W * GRID_H;  // 1200

// Fixed portion of state packet (before variable-length events and grids)
struct StatePacketFixed {
    PacketHeader header;            // 8
    GlobalState  global;            // 28
    UnitState    units[MAX_UNITS];  // 12 * sizeof(UnitState)
    uint8_t      num_events;        // 0-32
};

// Full state packet is parsed incrementally:
//   StatePacketFixed + Event[num_events]
//   + uint8_t has_pathability
//   + (if has_pathability) uint8_t pathability[1200]
//   + uint8_t visibility_t0[1200]
//   + uint8_t visibility_t1[1200]

// ============================================================
// Unit Action (28 bytes)
// ============================================================
struct UnitAction {
    uint8_t  idx;               // 0-11
    uint8_t  _pad;
    float    move_x;            // [-1, 1]
    float    move_y;            // [-1, 1]
    float    point_x;           // [-1, 1]
    float    point_y;           // [-1, 1]
    uint8_t  skill;             // 0-7
    uint8_t  unit_target;       // 0-13
    uint8_t  skill_levelup;     // 0-5
    uint8_t  stat_upgrade;      // 0-9
    uint8_t  attribute;         // 0-4
    uint8_t  item_buy;          // 0-16
    uint8_t  item_use;          // 0-6
    uint8_t  seal_use;          // 0-6
    uint8_t  faire_send;        // 0-5
    uint8_t  faire_request;     // 0-5
    uint8_t  faire_respond;     // 0-2
    uint8_t  _pad2;
};
static_assert(sizeof(UnitAction) == 30, "UnitAction must be 30 bytes");

// ============================================================
// Action Packet (368 bytes)
// ============================================================
struct ActionPacket {
    PacketHeader header;            // 8
    UnitAction   actions[MAX_UNITS]; // 12 * 30 = 360
};
static_assert(sizeof(ActionPacket) == 368, "ActionPacket must be 368 bytes");

// ============================================================
// Done Packet (16 bytes)
// ============================================================
struct DonePacket {
    PacketHeader header;    // 8
    uint8_t  winner;        // 0=team0, 1=team1, 2=draw
    uint8_t  reason;        // 1=team_wipe, 2=timeout, 3=score
    int16_t  score_team0;
    int16_t  score_team1;
    uint8_t  _pad[2];
};
static_assert(sizeof(DonePacket) == 16, "DonePacket must be 16 bytes");

#pragma pack(pop)

// ============================================================
// Helper: extract mask bit
// ============================================================
inline bool mask_bit(uint8_t mask, int bit) {
    return (mask >> bit) & 1;
}
inline bool mask_bit16(uint16_t mask, int bit) {
    return (mask >> bit) & 1;
}
inline bool mask_bit32(uint32_t mask, int bit) {
    return (mask >> bit) & 1;
}
