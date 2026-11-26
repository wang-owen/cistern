#pragma once

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

enum class Role : unsigned { weight = 0, exp_avg = 1, exp_avg_sq = 2 };

constexpr unsigned kRoleCount = 3;

struct StateOptions {
    std::uint64_t seed = 1234;
    std::filesystem::path align_dir = ".";  // probed with statx(STATX_DIOALIGN)
    std::optional<DioAlign> align_override;
    unsigned threads = 0;
};

struct TensorBuffer {
    std::size_t tensor = 0;  // index into State::table().tensors
    Role role = Role::weight;
    std::size_t bytes = 0;  // payload bytes
    AlignedBuffer mem;      // `bytes` rounded up to DioAlign::unit(); padding is zero
    std::span<std::byte> padded() const noexcept { return {mem.data(), mem.size()}; }
    std::span<std::byte> data() const noexcept { return padded().first(bytes); }
};

class State {
public:
    static State generate(const TensorTable& table, const StateOptions& opts);

    const DioAlign& alignment() const noexcept { return align_; }

    // weight, exp_avg, exp_avg_sq per tensor, in table order
    std::span<const TensorBuffer> checkpointed() const noexcept { return buffers_; }

    // Sum of padded().size() over checkpointed()
    std::uint64_t checkpoint_bytes() const noexcept { return checkpoint_bytes_; }

    // All-zero state (restore target) with other's table and alignment
    static State zeros_like(const State& other);

    const TensorTable& table() const noexcept { return table_; }

private:
    State() = default;

    // Zero-filled; checks RAM first
    static State allocate(const TensorTable& table, const DioAlign& align);

    void fill(std::uint64_t seed, unsigned threads);

    TensorTable table_;
    DioAlign align_;
    std::uint64_t checkpoint_bytes_ = 0;
    std::vector<TensorBuffer> buffers_;
};

// "<tensor>:<role>" of the first checkpointed buffer whose payload differs
std::optional<std::string> first_mismatch(const State& a, const State& b);

}  // namespace cistern
