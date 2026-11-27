#pragma once

#include <filesystem>

namespace cistern {

class State;
struct Layout;

void write_checkpoint(const State& state, const Layout& layout, const std::filesystem::path& dir);

}  // namespace cistern
