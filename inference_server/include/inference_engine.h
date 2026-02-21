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

    /// model_dir: directory containing model_latest.pt (shared model for all heroes)
    /// device: torch::kCPU or torch::kCUDA
    InferenceEngine(const std::string& model_dir, torch::Device device);

    /// Run inference for a single hero using the shared model.
    /// hero_id is encoded in self_vec one-hot; same model handles all heroes.
    InferResult infer_hero(
        const std::string& hero_id,
        torch::Tensor self_vec,      // (1, 77)
        torch::Tensor ally_vec,      // (1, 5, 37)
        torch::Tensor enemy_vec,     // (1, 6, 41)
        torch::Tensor global_vec,    // (1, 6)
        torch::Tensor grid,          // (1, 3, 25, 48)
        torch::Tensor hx_h,         // (1, 1, 256)
        torch::Tensor hx_c,         // (1, 1, 256)
        const std::unordered_map<std::string, torch::Tensor>& masks
    );

    /// Check model_latest.pt for changes and reload if updated.
    void maybe_reload();

    /// Create a zero-initialized LSTM hidden state pair.
    std::pair<torch::Tensor, torch::Tensor> init_hidden();

    /// Check if the shared model is loaded.
    bool has_model(const std::string& hero_id) const;

private:
    // Single shared model for all heroes (hero_id encoded in self_vec one-hot)
    torch::jit::script::Module shared_model_;
    bool model_loaded_ = false;
    std::filesystem::file_time_type model_time_;
    std::string model_dir_;
    torch::Device device_;

    /// Try to load model_latest.pt. Returns true on success.
    bool load_shared_model();

    /// Sample from categorical distribution with mask applied.
    std::pair<torch::Tensor, torch::Tensor> sample_categorical(
        torch::Tensor logits, torch::Tensor mask);

    /// Sample from normal distribution.
    std::pair<torch::Tensor, torch::Tensor> sample_normal(
        torch::Tensor mean, torch::Tensor logstd);
};
