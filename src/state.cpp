#include "cistern/state.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "cistern/env.hpp"

namespace cistern {

const char* role_name(Role r) noexcept {
    switch (r) {
        case Role::weight:
            return "weight";
        case Role::exp_avg:
            return "exp_avg";
        case Role::exp_avg_sq:
            return "exp_avg_sq";
        case Role::grad:
            return "grad";
    }
    return "?";
}

PaddingReport compute_padding(const TensorTable& table, std::size_t unit) {
    PaddingReport r;
    r.unit = unit;
    for (const auto& t : table.tensors) {
        r.payload_bytes += 3 * t.bytes;
        r.padded_bytes += 3 * round_up(t.bytes, unit);
    }
    r.overhead_bytes = r.padded_bytes - r.payload_bytes;
    r.overhead_pct = r.payload_bytes == 0 ? 0.0
                                          : 100.0 * static_cast<double>(r.overhead_bytes) /
                                                static_cast<double>(r.payload_bytes);
    return r;
}

RamPlan plan_state(const TensorTable& table, const DioAlign& align, const StateOptions& opts) {
    RamPlan p;
    p.align = align;
    p.padding = compute_padding(table, align.unit());
    p.checkpoint_bytes = p.padding.padded_bytes;
    if (opts.gradients) {
        for (const auto& t : table.tensors) p.gradient_bytes += round_up(t.bytes, align.unit());
    }
    if (opts.staging) p.staging_bytes = p.checkpoint_bytes;
    p.margin_bytes = kRamMarginBytes;
    return p;
}

namespace {

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
        case Role::grad:
            x = 5e-3f * tri;
            break;
        case Role::exp_avg_sq:
            return 1e-6f * (u1 + u2 + 0.05f);
    }
    constexpr float kMin = 1e-9f;
    if (std::fabs(x) < kMin) x = std::copysign(kMin, tri);
    return x;
}

// Counter-based: element i depends only on (key, i), so output is independent of threading.
// `first` must be even.
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

std::uint64_t hash_span(std::span<const std::byte> s) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ull ^ s.size();
    std::size_t i = 0;
    for (; i + 8 <= s.size(); i += 8) {
        std::uint64_t w;
        std::memcpy(&w, s.data() + i, 8);
        h = std::rotl((h ^ w) * 0x100000001B3ull, 29) + kGamma;
    }
    for (; i < s.size(); ++i) h = (h ^ static_cast<std::uint64_t>(s[i])) * 0x100000001B3ull;
    return mix64(h);
}

unsigned resolve_threads(unsigned t) {
    if (t != 0) return t;
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : hc;
}

}  // namespace

State State::build(const TensorTable& table, const DioAlign& align, std::uint64_t seed,
                   bool gradients, bool staging) {
    for (const auto& t : table.tensors) {
        if (t.dtype != DType::bf16)
            throw std::runtime_error("state generator supports bf16 only: " + t.name);
    }
    State s;
    s.table_ = table;
    s.align_ = align;
    s.seed_ = seed;
    s.padding_ = compute_padding(table, align.unit());
    const std::size_t unit = align.unit();
    auto alloc = [&](Role role) {
        auto& v = s.by_role_[static_cast<std::size_t>(role)];
        v.reserve(table.tensors.size());
        for (std::size_t i = 0; i < table.tensors.size(); ++i) {
            TensorBuffer b;
            b.tensor = i;
            b.role = role;
            b.bytes = static_cast<std::size_t>(table.tensors[i].bytes);
            b.mem = AlignedBuffer(round_up(b.bytes, unit), align.mem, false);
            b.padded = {b.mem.data(), b.mem.size()};
            v.push_back(std::move(b));
        }
    };
    alloc(Role::weight);
    alloc(Role::exp_avg);
    alloc(Role::exp_avg_sq);
    if (gradients) alloc(Role::grad);
    s.checkpoint_.reserve(3 * table.tensors.size());
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < table.tensors.size(); ++i) {
        for (Role r : {Role::weight, Role::exp_avg, Role::exp_avg_sq}) {
            TensorBuffer* b = &s.by_role_[static_cast<std::size_t>(r)][i];
            s.checkpoint_.push_back({b, off});
            off += b->padded.size();
        }
    }
    if (staging) s.staging_ = AlignedBuffer(static_cast<std::size_t>(off), align.mem, true);
    return s;
}

void State::fill(unsigned threads) {
    struct Task {
        TensorBuffer* b;
        std::size_t first, last;  // element range
    };
    constexpr std::size_t kElems = 1u << 20;  // 2 MiB of bf16 per task (even)
    std::vector<Task> tasks;
    for (auto& v : by_role_) {
        for (auto& b : v) {
            const std::size_t n = b.bytes / 2;
            for (std::size_t f = 0; f < n; f += kElems)
                tasks.push_back({&b, f, std::min(n, f + kElems)});
        }
    }
    std::atomic<std::size_t> next{0};
    auto work = [&] {
        for (std::size_t i = next.fetch_add(1); i < tasks.size(); i = next.fetch_add(1)) {
            const Task& t = tasks[i];
            fill_range(t.b->role, stream_key(seed_, t.b->tensor, t.b->role),
                       reinterpret_cast<std::uint16_t*>(t.b->padded.data()), t.first, t.last);
        }
    };
    const auto n = static_cast<unsigned>(
        std::min<std::size_t>(resolve_threads(threads), std::max<std::size_t>(tasks.size(), 1)));
    std::vector<std::jthread> pool;
    for (unsigned i = 1; i < n; ++i) pool.emplace_back(work);
    work();
}

State State::generate(const TensorTable& table, const StateOptions& opts) {
    const DioAlign align =
        opts.align_override ? *opts.align_override : probe_dio_align(opts.align_dir);
    const RamPlan plan = plan_state(table, align, opts);
    if (opts.check_ram) {
        env::check_free_ram(plan.required(), opts.available_ram_override
                                                 ? *opts.available_ram_override
                                                 : env::mem_available_bytes());
    }
    State s = build(table, align, opts.seed, opts.gradients, opts.staging);
    s.fill(opts.threads);
    return s;
}

State State::zeros_like(const State& other, unsigned, bool check_ram) {
    StateOptions o;
    o.align_override = other.align_;
    const RamPlan plan = plan_state(other.table_, other.align_, o);
    if (check_ram) env::check_free_ram(plan.required(), env::mem_available_bytes());
    return build(other.table_, other.align_, other.seed_, false, false);
}

std::uint64_t State::content_hash() const {
    std::uint64_t h = 0;
    for (const auto& e : checkpoint_) h = mix64(h ^ hash_span(e.buf->data()));
    for (const auto& b : buffers(Role::grad)) h = mix64(h ^ hash_span(b.data()));
    return h;
}

std::optional<std::string> first_mismatch(const State& a, const State& b) {
    const auto ea = a.checkpointed();
    const auto eb = b.checkpointed();
    if (ea.size() != eb.size()) return std::string("<buffer count differs>");
    for (std::size_t i = 0; i < ea.size(); ++i) {
        const auto da = ea[i].buf->data();
        const auto db = eb[i].buf->data();
        if (da.size() != db.size() || std::memcmp(da.data(), db.data(), da.size()) != 0) {
            return a.table().tensors[ea[i].buf->tensor].name + ":" + role_name(ea[i].buf->role);
        }
    }
    return std::nullopt;
}

}  // namespace cistern
