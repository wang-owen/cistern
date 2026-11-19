#pragma once
// Seam header for the owner-implemented checkpoint writer. Declarations only; reshape freely.
#include <cstddef>
#include <filesystem>

#include "cistern/state.hpp"

namespace cistern {

struct WriterConfig {
    unsigned queue_depth = 16;
    std::size_t chunk_bytes = 1 << 20;
    bool registered_buffers = false;
    bool fallocate = false;
};

// Writes the checkpointed buffers of `state` to `dst` and commits atomically.
// On return the checkpoint is durable. Throws on failure.
void write_checkpoint(const State& state, const std::filesystem::path& dst,
                      const WriterConfig& cfg);

}  // namespace cistern
