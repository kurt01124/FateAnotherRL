#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "protocol.h"
#include "constants.h"

// ============================================================
// Encoded observation tensors for all 12 agents
// ============================================================
struct EncodedObs {
    torch::Tensor self_vec;    // (12, 77) float
    torch::Tensor ally_vec;    // (12, 5, 37) float
    torch::Tensor enemy_vec;   // (12, 6, 41) float
    torch::Tensor global_vec;  // (12, 6) float
    torch::Tensor grid;        // (12, 3, 25, 48) float
};

// ============================================================
// Action masks for all 12 agents
// ============================================================
struct MaskSet {
    // One tensor per discrete head, each (12, head_size) bool
    std::unordered_map<std::string, torch::Tensor> masks;
};

// ============================================================
// State Encoder namespace
// ============================================================
namespace state_encoder {

    /// Parse a raw binary STATE packet into structured data.
    /// Returns true on success.
    bool parse_packet(const uint8_t* data, size_t len,
                      PacketHeader& header,
                      GlobalState& global,
                      UnitState units[MAX_UNITS],
                      std::vector<Event>& events,
                      std::vector<uint8_t>& pathability,
                      std::vector<uint8_t>& vis_t0,
                      std::vector<uint8_t>& vis_t1);

    /// Encode parsed state into per-agent observation tensors (12 perspectives).
    EncodedObs encode(const UnitState units[MAX_UNITS],
                      const GlobalState& global,
                      const std::vector<uint8_t>& pathability,
                      const std::vector<uint8_t>& vis_t0,
                      const std::vector<uint8_t>& vis_t1);

    /// Extract action masks from unit state bit-packed fields.
    MaskSet encode_masks(const UnitState units[MAX_UNITS]);

} // namespace state_encoder
