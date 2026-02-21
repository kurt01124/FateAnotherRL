#include "inference_engine.h"

#include <cmath>
#include <iostream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// ============================================================
// Constructor
// ============================================================

InferenceEngine::InferenceEngine(const std::string& model_dir, torch::Device device)
    : model_dir_(model_dir), device_(device)
{
    std::cout << "[InferenceEngine] Model dir: " << model_dir
              << ", device: " << device << std::endl;

    // Try to load model_latest.pt at startup
    if (load_shared_model()) {
        std::cout << "[InferenceEngine] Shared model loaded at startup" << std::endl;
    } else {
        std::cout << "[InferenceEngine] No model_latest.pt found at startup (will retry)" << std::endl;
    }
}

// ============================================================
// load_shared_model: Load model_latest.pt
// ============================================================

bool InferenceEngine::load_shared_model() {
    fs::path model_path = fs::path(model_dir_) / "model_latest.pt";

    if (!fs::exists(model_path)) {
        return false;
    }

    try {
        shared_model_ = torch::jit::load(model_path.string(), device_);
        shared_model_.eval();
        model_loaded_ = true;
        model_time_ = fs::last_write_time(model_path);

        std::cout << "[InferenceEngine] Loaded shared model: " << model_path.string() << std::endl;
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "[InferenceEngine] Failed to load " << model_path.string()
                  << ": " << e.what() << std::endl;
        return false;
    }
}

// ============================================================
// maybe_reload: Check model_latest.pt for changes
// ============================================================

void InferenceEngine::maybe_reload() {
    fs::path model_path = fs::path(model_dir_) / "model_latest.pt";
    if (!fs::exists(model_path)) return;

    auto new_time = fs::last_write_time(model_path);
    if (!model_loaded_ || model_time_ != new_time) {
        std::cout << "[InferenceEngine] Reloading model_latest.pt..." << std::endl;
        load_shared_model();
    }
}

// ============================================================
// has_model
// ============================================================

bool InferenceEngine::has_model(const std::string& hero_id) const {
    return model_loaded_;
}

// ============================================================
// init_hidden: Zero LSTM hidden state
// ============================================================

std::pair<torch::Tensor, torch::Tensor> InferenceEngine::init_hidden() {
    auto h = torch::zeros({1, 1, HIDDEN_DIM}, torch::TensorOptions().device(device_));
    auto c = torch::zeros({1, 1, HIDDEN_DIM}, torch::TensorOptions().device(device_));
    return {h, c};
}

// ============================================================
// sample_categorical: Masked categorical sampling
// ============================================================

std::pair<torch::Tensor, torch::Tensor>
InferenceEngine::sample_categorical(torch::Tensor logits, torch::Tensor mask) {
    // logits: (1, N), mask: (1, N) bool
    // Apply mask: set disallowed actions to -inf
    auto masked_logits = logits.masked_fill(~mask, -1e8f);

    // Softmax -> probabilities
    auto probs = torch::softmax(masked_logits, /*dim=*/-1);

    // Sample
    auto action = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/false);
    // action: (1, 1) -> squeeze to (1,)
    action = action.squeeze(-1);

    // Log probability
    auto log_probs = torch::log_softmax(masked_logits, /*dim=*/-1);
    auto log_prob = log_probs.gather(/*dim=*/-1, action.unsqueeze(-1)).squeeze(-1);

    return {action, log_prob};
}

// ============================================================
// sample_normal: Normal distribution sampling
// ============================================================

std::pair<torch::Tensor, torch::Tensor>
InferenceEngine::sample_normal(torch::Tensor mean, torch::Tensor logstd) {
    // mean: (1, 2), logstd: (2,)
    auto stddev = logstd.exp().expand_as(mean);
    auto noise = torch::randn_like(mean);
    auto sample = mean + stddev * noise;

    // Log prob: sum over dimensions
    // log_prob = -0.5 * ((sample - mean) / std)^2 - log(std) - 0.5 * log(2*pi)
    constexpr float log2pi = 1.8378770664093453f;  // log(2 * pi)
    auto var = stddev * stddev;
    auto log_prob = -0.5f * ((sample - mean).pow(2) / var) - logstd.expand_as(mean)
                    - 0.5f * log2pi;
    auto total_log_prob = log_prob.sum(-1);  // (1,)

    return {sample, total_log_prob};
}

// ============================================================
// infer_hero: Run inference for a single hero (shared model)
// ============================================================

InferenceEngine::InferResult InferenceEngine::infer_hero(
    const std::string& hero_id,
    torch::Tensor self_vec,
    torch::Tensor ally_vec,
    torch::Tensor enemy_vec,
    torch::Tensor global_vec,
    torch::Tensor grid,
    torch::Tensor hx_h,
    torch::Tensor hx_c,
    const std::unordered_map<std::string, torch::Tensor>& masks)
{
    InferResult result;

    if (!model_loaded_) {
        // No model loaded: return random/default actions
        std::cerr << "[InferenceEngine] No model loaded, returning defaults" << std::endl;

        const auto& heads = discrete_heads();
        for (int h = 0; h < NUM_DISCRETE_HEADS; ++h) {
            result.actions[heads[h].name] = torch::zeros({1}, torch::kLong).to(device_);
        }
        result.actions["move"] = torch::zeros({1, 2}).to(device_);
        result.actions["point"] = torch::zeros({1, 2}).to(device_);
        result.log_prob = torch::zeros({1}).to(device_);
        result.value = torch::zeros({1}).to(device_);
        result.new_h = hx_h;
        result.new_c = hx_c;
        return result;
    }

    // Build input vector matching FateModelExport.forward() signature
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(self_vec);
    inputs.push_back(ally_vec);
    inputs.push_back(enemy_vec);
    inputs.push_back(global_vec);
    inputs.push_back(grid);
    inputs.push_back(hx_h);
    inputs.push_back(hx_c);

    // Add mask tensors in the exact order expected by FateModelExport
    const char* mask_names[] = {
        "skill", "unit_target", "skill_levelup", "stat_upgrade",
        "attribute", "item_buy", "item_use", "seal_use",
        "faire_send", "faire_request", "faire_respond"
    };
    for (const char* name : mask_names) {
        auto it = masks.find(name);
        if (it != masks.end()) {
            inputs.push_back(it->second);
        } else {
            // Default: all allowed
            int size = 1;
            for (int h = 0; h < NUM_DISCRETE_HEADS; ++h) {
                if (std::string(discrete_heads()[h].name) == name) {
                    size = discrete_heads()[h].size;
                    break;
                }
            }
            inputs.push_back(torch::ones({1, size}, torch::kBool).to(device_));
        }
    }

    // Run forward pass
    torch::NoGradGuard no_grad;
    auto output = shared_model_.forward(inputs);

    // Parse output tuple (18 tensors):
    // 0-10: discrete logits (already masked by model)
    // 11: move_mean (B,2), 12: move_logstd (2,)
    // 13: point_mean (B,2), 14: point_logstd (2,)
    // 15: value (B,)
    // 16: new_h (1,B,256), 17: new_c (1,B,256)
    auto tuple = output.toTuple();
    auto elements = tuple->elements();

    // Discrete head outputs (indices 0-10) - these are already masked logits
    float total_log_prob = 0.0f;
    const auto& heads = discrete_heads();
    for (int h = 0; h < NUM_DISCRETE_HEADS; ++h) {
        auto logits = elements[h].toTensor();  // (1, head_size)
        auto mask_it = masks.find(heads[h].name);
        torch::Tensor mask_t;
        if (mask_it != masks.end()) {
            mask_t = mask_it->second;
        } else {
            mask_t = torch::ones({1, heads[h].size}, torch::kBool).to(device_);
        }

        auto [action, lp] = sample_categorical(logits, mask_t);
        result.actions[heads[h].name] = action;
        total_log_prob += lp.item<float>();
    }

    // Continuous: move (indices 11, 12)
    auto move_mean = elements[11].toTensor();    // (1, 2)
    auto move_logstd = elements[12].toTensor();  // (2,)
    auto [move_sample, move_lp] = sample_normal(move_mean, move_logstd);
    result.actions["move"] = move_sample;
    total_log_prob += move_lp.item<float>();

    // Continuous: point (indices 13, 14)
    auto point_mean = elements[13].toTensor();    // (1, 2)
    auto point_logstd = elements[14].toTensor();  // (2,)
    auto [point_sample, point_lp] = sample_normal(point_mean, point_logstd);
    result.actions["point"] = point_sample;
    total_log_prob += point_lp.item<float>();

    // Value (index 15)
    result.value = elements[15].toTensor();  // (1,)

    // LSTM hidden state (indices 16, 17)
    result.new_h = elements[16].toTensor();  // (1, 1, 256)
    result.new_c = elements[17].toTensor();  // (1, 1, 256)

    result.log_prob = torch::tensor({total_log_prob}).to(device_);

    return result;
}
