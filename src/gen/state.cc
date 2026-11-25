#include "gen/state.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "common/env.h"

namespace cistern {

namespace {

// Checkpointed bytes (weight, exp_avg, exp_avg_sq per tensor), each buffer padded to `unit`.
std::uint64_t padded_checkpoint_bytes(const TensorTable& table, std::size_t unit) {
    std::uint64_t n = 0;
    for (const auto& t : table.tensors) n += 3 * round_up(t.bytes, unit);
    return n;
}

constexpr std::uint64_t kGamma = 0x9E3779B97F4A7C15ull;

constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

constexpr std::uint64_t stream_key(std::uint64_t seed, std::size_t tensor, Role role) noexcept {
    return mix64(seed ^ mix64(static_cast<std::uint64_t>(tensor) * 4 +
                              static_cast<std::uint64_t>(role) + 1));
}

std::uint16_t to_bf16(float f) noexcept {
    std::uint32_t b = std::bit_cast<std::uint32_t>(f);
    b += 0x7FFFu + ((b >> 16) & 1u);  // round to nearest even; inputs are small and finite
    return static_cast<std::uint16_t>(b >> 16);
}

// Values are small, finite and normal in bf16 (never 0, denormal, Inf or NaN); exp_avg_sq > 0.
float value(Role role, float u1, float u2) noexcept {
    const float tri = u1 + u2 - 1.0f;  // triangular on (-1, 1)
    float x = 0.0f;
    switch (role) {
        case Role::weight:
            x = 0.04f * tri;
            break;
        case Role::exp_avg:
            x = 1e-3f * tri;
            break;
        case Role::exp_avg_sq:
            return 1e-6f * (u1 + u2 + 0.05f);
    }
    constexpr float kMin = 1e-9f;
    if (std::fabs(x) < kMin) x = std::copysign(kMin, tri);
    return x;
}

// Counter-based: element i depends only on (key, i), so output is independent of threading.
// Requires even `first`.
void fill_range(Role role, std::uint64_t key, std::uint16_t* out, std::size_t first,
                std::size_t last) {
    constexpr float k = 1.0f / 65536.0f;
    for (std::size_t i = first; i < last; i += 2) {
        const std::uint64_t r = mix64(key + (static_cast<std::uint64_t>(i / 2) + 1) * kGamma);
        const auto u = [&](unsigned shift) {
            return static_cast<float>((r >> shift) & 0xFFFFu) * k;
        };
        out[i] = to_bf16(value(role, u(0), u(16)));
        if (i + 1 < last) out[i + 1] = to_bf16(value(role, u(32), u(48)));
    }
}

// Runs f(0..n-1) on up to `threads` threads (0 = hardware concurrency), the calling thread
// included, pulling indices from a shared counter.
template <class F>
void parallel_for(std::size_t n, unsigned threads, F&& f) {
    std::atomic<std::size_t> next{0};
    auto work = [&] {
        for (std::size_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) f(i);
    };
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const auto t =
        static_cast<unsigned>(std::min<std::size_t>(threads, std::max<std::size_t>(n, 1)));
    std::vector<std::jthread> pool;
    for (unsigned i = 1; i < t; ++i) pool.emplace_back(work);
    work();
}

}  // namespace

void State::fill(std::uint64_t seed, unsigned threads) {
    struct Task {
        TensorBuffer* b;
        std::size_t first, last;  // element range
    };

    // 2 MiB of bf16 per task; even, as fill_range requires.
    constexpr std::size_t kElems = 1u << 20;
    std::vector<Task> tasks;
    for (auto& b : buffers_) {
        const std::size_t n = b.bytes / 2;
        for (std::size_t f = 0; f < n; f += kElems)
            tasks.push_back({&b, f, std::min(n, f + kElems)});
    }
    parallel_for(tasks.size(), threads, [&](std::size_t i) {
        const Task& t = tasks[i];
        fill_range(t.b->role, stream_key(seed, t.b->tensor, t.b->role),
                   reinterpret_cast<std::uint16_t*>(t.b->mem.data()), t.first, t.last);
    });
}

State State::allocate(const TensorTable& table, const DioAlign& align) {
    for (const auto& t : table.tensors) {
        if (t.dtype != DType::bf16)
            throw std::runtime_error("state generator supports bf16 only: " + t.name);
    }

    // Overcommit lets mmap succeed regardless; refuse here rather than OOM partway through fill().
    const std::size_t unit = align.unit();
    const std::uint64_t bytes = padded_checkpoint_bytes(table, unit);
    constexpr std::uint64_t kRamMarginBytes = 512ull << 20;
    env::check_free_ram(bytes + kRamMarginBytes, env::mem_available_bytes());

    State s;
    s.table_ = table;
    s.align_ = align;
    s.checkpoint_bytes_ = bytes;
    s.buffers_.reserve(3 * table.tensors.size());
    for (std::size_t i = 0; i < table.tensors.size(); ++i) {
        for (Role role : {Role::weight, Role::exp_avg, Role::exp_avg_sq}) {
            TensorBuffer b;
            b.tensor = i;
            b.role = role;
            b.bytes = static_cast<std::size_t>(table.tensors[i].bytes);
            b.mem = AlignedBuffer(round_up(b.bytes, unit), align.mem, false);
            s.buffers_.push_back(std::move(b));
        }
    }
    return s;
}

State State::generate(const TensorTable& table, const StateOptions& opts) {
    const DioAlign align =
        opts.align_override ? *opts.align_override : probe_dio_align(opts.align_dir);
    State s = allocate(table, align);
    s.fill(opts.seed, opts.threads);
    return s;
}

// Used only by tests.

State State::zeros_like(const State& other) {
    // Anonymous mmap pages are zero.
    return allocate(other.table_, other.align_);
}

namespace {

const char* role_name(Role r) noexcept {
    switch (r) {
        case Role::weight:
            return "weight";
        case Role::exp_avg:
            return "exp_avg";
        case Role::exp_avg_sq:
            return "exp_avg_sq";
    }
    return "?";
}

}  // namespace

std::optional<std::string> first_mismatch(const State& a, const State& b) {
    const auto ea = a.checkpointed();
    const auto eb = b.checkpointed();
    if (ea.size() != eb.size()) return std::string("<buffer count differs>");
    std::atomic<std::size_t> first{ea.size()};  // lowest mismatching index found so far
    parallel_for(ea.size(), 0, [&](std::size_t i) {
        std::size_t cur = first.load(std::memory_order_relaxed);
        if (i >= cur) return;
        const auto da = ea[i].data();
        const auto db = eb[i].data();
        if (da.size() == db.size() && std::memcmp(da.data(), db.data(), da.size()) == 0) return;
        while (i < cur && !first.compare_exchange_weak(cur, i)) {
        }
    });
    const std::size_t i = first.load();
    if (i == ea.size()) return std::nullopt;
    return a.table().tensors[ea[i].tensor].name + ":" + role_name(ea[i].role);
}

}  // namespace cistern
