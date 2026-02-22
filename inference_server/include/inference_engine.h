#pragma once

#include <string>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <utility>

#include <torch/torch.h>
#include <torch/script.h>

#include "constants.h"

class InferenceEngine {
public:
    struct InferResult {
        std::unordered_map<std::string, torch::Tensor> actions;  // sampled discrete + continuous
        torch::Tensor log_prob;   // scalar
        torch::Tensor value;      // scalar
        torch::Tensor new_h;      // (1, 1, 256)
        torch::Tensor new_c;      // (1, 1, 256)
    };

    /// model_dir: directory containing per-hero .pt files (H000.pt, H001.pt, ...)
    /// device: torch::kCPU or torch::kCUDA
    InferenceEngine(const std::string& model_dir, torch::Device device);

    /// Run inference for a single hero using its dedicated model.
    InferResult infer_hero(
        const std::string& hero_id,
        torch::Tensor self_vec,      // (1, 77)
        torch::Tensor ally_vec,      // (1, 5, 37)
        torch::Tensor enemy_vec,     // (1, 6, 43)
        torch::Tensor global_vec,    // (1, 6)
        torch::Tensor grid,          // (1, 3, 25, 48)
        torch::Tensor hx_h,         // (1, 1, 256)
        torch::Tensor hx_c,         // (1, 1, 256)
        const std::unordered_map<std::string, torch::Tensor>& masks
    );

    /// Check per-hero .pt files for changes and reload if updated.
    void maybe_reload();

    /// Create a zero-initialized LSTM hidden state pair.
    std::pair<torch::Tensor, torch::Tensor> init_hidden();

    /// Check if a model is loaded for the given hero.
    bool has_model(const std::string& hero_id) const;

private:
    // Per-hero models: hero_id → TorchScript module
    std::unordered_map<std::string, torch::jit::script::Module> hero_models_;
    std::unordered_map<std::string, std::filesystem::file_time_type> model_times_;
    std::string model_dir_;
    torch::Device device_;

    /// Try to load all hero .pt files. Returns number of models loaded.
    int load_hero_models();

    /// Try to load a single hero model. Returns true on success.
    bool load_hero_model(const std::string& hero_id);

    /// Sample from categorical distribution with mask applied.
    std::pair<torch::Tensor, torch::Tensor> sample_categorical(
        torch::Tensor logits, torch::Tensor mask);

    /// Sample from normal distribution.
    std::pair<torch::Tensor, torch::Tensor> sample_normal(
        torch::Tensor mean, torch::Tensor logstd);
};
