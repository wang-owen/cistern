#pragma once

#include <filesystem>

#include "state.h"

namespace pitstop {

// Reads `dir`/ckpt (as written by write_checkpoint) into `out`, which must have the
// checkpoint's table. Throws std::runtime_error on a corrupt or truncated file and
// std::invalid_argument when `out` does not match the checkpoint layout.
void restore_checkpoint(const std::filesystem::path& dir, State& out);

}  // namespace pitstop
