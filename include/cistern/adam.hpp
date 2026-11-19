#pragma once
// Seam header for the owner-implemented CPU Adam step over 1B-param state.
#include <cstdint>

#include "cistern/state.hpp"

namespace cistern {

struct AdamParams {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
};

// One CPU Adam step: reads Role::grad buffers, updates Role::weight, Role::exp_avg and
// Role::exp_avg_sq in place (memory-bound). `step` is 1-based.
void adam_step(State& state, const AdamParams& params, std::uint64_t step);

}  // namespace cistern
