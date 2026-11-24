#pragma once
// Synthetic training state: bf16 weights, two Adam moments and (optionally) a gradient per tensor,
// plus an optional pre-faulted staging buffer. Each buffer is its own page-aligned anonymous mmap
// whose length is the tensor size rounded up to DioAlign::unit(); padding is zero. No file offsets.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/align.h"
#include "table/tensor_table.h"

namespace cistern {

enum class Role : unsigned { weight = 0, exp_avg = 1, exp_avg_sq = 2, grad = 3 };
inline constexpr std::size_t kRoleCount = 4;

// Checkpointed bytes (weight, exp_avg, exp_avg_sq per tensor), raw and padded to `unit`.
struct PaddingReport {
    std::uint64_t payload_bytes = 0;
    std::uint64_t padded_bytes = 0;
};
PaddingReport compute_padding(const TensorTable& table, std::size_t unit);

struct StateOptions {
    std::uint64_t seed = 1234;
    std::filesystem::path align_dir = ".";  // probed with statx(STATX_DIOALIGN)
    std::optional<DioAlign> align_override;
    bool gradients = false;  // one grad tensor per weight
    bool staging = false;    // pre-faulted buffer of checkpoint size
    unsigned threads = 0;    // 0 = hardware concurrency
    std::optional<std::uint64_t> available_ram_override;
};

inline constexpr std::uint64_t kRamMarginBytes = 512ull << 20;

// What a state of this shape needs, before anything is allocated.
struct RamPlan {
    DioAlign align;
    PaddingReport padding;
    std::uint64_t gradient_bytes = 0;
    std::uint64_t staging_bytes = 0;
    std::uint64_t required() const noexcept {
        return padding.padded_bytes + gradient_bytes + staging_bytes + kRamMarginBytes;
    }
};
RamPlan plan_state(const TensorTable& table, const DioAlign& align, const StateOptions& opts);

struct TensorBuffer {
    std::size_t tensor = 0;  // index into State::table().tensors
    Role role = Role::weight;
    std::size_t bytes = 0;  // payload bytes
    AlignedBuffer mem;      // bytes rounded up to the padding unit
    std::span<std::byte> padded() const noexcept { return {mem.data(), mem.size()}; }
    std::span<std::byte> data() const noexcept { return padded().first(bytes); }
};

class State {
public:
    State(State&&) = default;
    State& operator=(State&&) = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    static State generate(const TensorTable& table, const StateOptions& opts);
    // All-zero state (restore target); ignores opts.gradients, staging and threads.
    static State zeros(const TensorTable& table, const StateOptions& opts);
    // zeros() with other's table, alignment and seed.
    static State zeros_like(const State& other, StateOptions opts = {});

    const TensorTable& table() const noexcept { return table_; }
    std::span<const TensorBuffer> buffers(Role r) const noexcept {
        return by_role_[static_cast<std::size_t>(r)];
    }
    // weight, exp_avg, exp_avg_sq per tensor, in table order.
    std::span<TensorBuffer* const> checkpointed() const noexcept { return checkpoint_; }
    std::span<std::byte> staging() const noexcept { return {staging_.data(), staging_.size()}; }
    const DioAlign& alignment() const noexcept { return align_; }
    // Sum of the padded checkpointed buffers.
    std::uint64_t checkpoint_bytes() const noexcept { return padding_.padded_bytes; }
    std::uint64_t payload_bytes() const noexcept { return padding_.payload_bytes; }
    bool has_gradients() const noexcept { return !buffers(Role::grad).empty(); }

    // 64-bit hash of all payload bytes (checkpointed buffers, plus gradients if present).
    std::uint64_t content_hash() const;

private:
    State() = default;
    // Zero-filled; checks RAM first.
    static State allocate(const TensorTable& table, const StateOptions& opts);
    void fill(unsigned threads);

    TensorTable table_;
    DioAlign align_;
    PaddingReport padding_;
    std::uint64_t seed_ = 0;
    std::array<std::vector<TensorBuffer>, kRoleCount> by_role_;
    std::vector<TensorBuffer*> checkpoint_;
    AlignedBuffer staging_;
};

// "<tensor>:<role>" of the first checkpointed buffer whose payload differs, if any.
std::optional<std::string> first_mismatch(const State& a, const State& b);

}  // namespace cistern
