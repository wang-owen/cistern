#pragma once
// Synthetic training state: bf16 weights, two Adam moments per weight, one gradient tensor per
// weight (Adam step input, never checkpointed), and an optional pre-faulted staging buffer.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cistern/align.hpp"
#include "cistern/tensor_table.hpp"

namespace cistern {

enum class Role : unsigned { weight = 0, exp_avg = 1, exp_avg_sq = 2, grad = 3 };
inline constexpr std::size_t kRoleCount = 4;
const char* role_name(Role r) noexcept;

// Padding added to satisfy direct-I/O alignment. Tensor padding only (no header block).
struct PaddingReport {
    std::uint64_t unit = 0;            // padding granularity in bytes
    std::uint64_t payload_bytes = 0;   // sum of tensor bytes in the checkpoint
    std::uint64_t padded_bytes = 0;    // sum of padded buffer sizes
    std::uint64_t overhead_bytes = 0;  // padded - payload
    double overhead_pct = 0.0;         // overhead / payload * 100
};
// Pure: each tensor is stored three times (weight, exp_avg, exp_avg_sq), each padded to `unit`.
PaddingReport compute_padding(const TensorTable& table, std::size_t unit);

struct StateOptions {
    std::uint64_t seed = 1234;
    std::filesystem::path align_dir = ".";   // directory probed with statx(STATX_DIOALIGN)
    std::optional<DioAlign> align_override;  // tests / --align
    bool gradients = false;                  // allocate + fill one grad tensor per weight
    bool staging = false;                    // pre-faulted staging buffer of checkpoint size
    unsigned threads = 0;                    // 0 = hardware concurrency
    bool check_ram = true;
    std::optional<std::uint64_t> available_ram_override;  // tests
};

// What a state of this shape needs, before anything is allocated.
struct RamPlan {
    DioAlign align;
    PaddingReport padding;
    std::uint64_t checkpoint_bytes = 0;  // padded weights + moments
    std::uint64_t gradient_bytes = 0;    // padded
    std::uint64_t staging_bytes = 0;
    std::uint64_t margin_bytes = 0;
    std::uint64_t required() const noexcept {
        return checkpoint_bytes + gradient_bytes + staging_bytes + margin_bytes;
    }
};
inline constexpr std::uint64_t kRamMarginBytes = 512ull << 20;
RamPlan plan_state(const TensorTable& table, const DioAlign& align, const StateOptions& opts);

struct TensorBuffer {
    std::size_t tensor = 0;  // index into State::table().tensors
    Role role = Role::weight;
    std::size_t bytes = 0;        // payload bytes
    std::span<std::byte> padded;  // bytes rounded up to the padding unit (zero padding)
    AlignedBuffer mem;
    std::span<std::byte> data() const noexcept { return padded.first(bytes); }
};

// One checkpointed buffer and its offset within the data region (padded, contiguous).
struct CheckpointEntry {
    TensorBuffer* buf = nullptr;
    std::uint64_t offset = 0;
};

class State {
public:
    State(State&&) = default;
    State& operator=(State&&) = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    static State generate(const TensorTable& table, const StateOptions& opts);
    // Zero-filled state of the same shape and alignment (restore target): no grads, no staging.
    static State zeros_like(const State& other, unsigned threads = 0, bool check_ram = true);

    const TensorTable& table() const noexcept { return table_; }
    std::span<TensorBuffer> buffers(Role r) noexcept {
        return by_role_[static_cast<std::size_t>(r)];
    }
    std::span<const TensorBuffer> buffers(Role r) const noexcept {
        return by_role_[static_cast<std::size_t>(r)];
    }
    // w, m, v per tensor, in table order; offsets are contiguous from 0.
    std::span<const CheckpointEntry> checkpointed() const noexcept { return checkpoint_; }
    // Pre-faulted staging buffer of checkpoint_bytes() (empty if not requested).
    std::span<std::byte> staging() const noexcept { return {staging_.data(), staging_.size()}; }
    const DioAlign& alignment() const noexcept { return align_; }
    const PaddingReport& padding_report() const noexcept { return padding_; }
    std::uint64_t checkpoint_bytes() const noexcept { return padding_.padded_bytes; }
    std::uint64_t payload_bytes() const noexcept { return padding_.payload_bytes; }
    std::uint64_t seed() const noexcept { return seed_; }
    bool has_gradients() const noexcept { return !buffers(Role::grad).empty(); }

    // 64-bit hash of all payload bytes (checkpointed buffers, plus gradients if present).
    std::uint64_t content_hash() const;

private:
    State() = default;
    static State build(const TensorTable& table, const DioAlign& align, std::uint64_t seed,
                       bool gradients, bool staging);
    void fill(unsigned threads);

    TensorTable table_;
    DioAlign align_;
    PaddingReport padding_;
    std::uint64_t seed_ = 0;
    std::array<std::vector<TensorBuffer>, kRoleCount> by_role_;
    std::vector<CheckpointEntry> checkpoint_;
    AlignedBuffer staging_;
};

// Name of the first checkpointed buffer whose payload differs ("<tensor>:<role>"), if any.
std::optional<std::string> first_mismatch(const State& a, const State& b);

}  // namespace cistern
