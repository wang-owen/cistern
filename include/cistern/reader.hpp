#pragma once
// Seam header for the owner-implemented restore reader. Declarations only; reshape freely.
#include <cstddef>
#include <filesystem>

#include "cistern/state.hpp"

namespace cistern {

struct ReaderConfig {
    unsigned queue_depth = 16;
    std::size_t chunk_bytes = 1 << 20;
};

// Restores the checkpointed buffers of `into` from `src`. Throws on any failure
// (including a checksum mismatch, which should name the chunk index).
void restore_checkpoint(const std::filesystem::path& src, State& into, const ReaderConfig& cfg);

}  // namespace cistern
