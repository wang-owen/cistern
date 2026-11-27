#pragma once

#include <filesystem>
#include <optional>

#include "ring.h"

namespace pitstop {

class State;
struct Layout;

class Writer {
public:
    explicit Writer(const State& state);
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    void write(const Layout& layout, const std::filesystem::path& dir);

    // false: registration failed and writes use plain (unregistered) buffers
    bool fixed_buffers() const noexcept { return fixed_; }

private:
    void reset_ring();
    void register_buffers();
    void write_impl(const Layout& layout, const std::filesystem::path& dir);

    const State& state_;
    std::optional<Ring> ring_;
    bool fixed_ = false;
};

// Prefer a Writer when checkpointing the same State repeatedly
void write_checkpoint(const State& state, const Layout& layout, const std::filesystem::path& dir);

}  // namespace pitstop
