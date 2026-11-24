#include "gen/state.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

#include "common/env.h"
#include "test_util.h"

namespace cistern {

namespace {

StateOptions opts(std::uint64_t seed, unsigned threads, bool grads = false) {
    StateOptions o;
    o.seed = seed;
    o.threads = threads;
    o.gradients = grads;
    o.align_override = DioAlign{4096, 4096, false};
    return o;
}

}  // namespace

TEST_CASE("generator is deterministic and thread-count independent", "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto a = State::generate(t, opts(7, 1, true));
    const auto b = State::generate(t, opts(7, 5, true));
    const auto c = State::generate(t, opts(8, 3, true));
    CHECK(a.content_hash() == b.content_hash());
    CHECK(a.content_hash() != c.content_hash());
    CHECK(!first_mismatch(a, b).has_value());
    CHECK(first_mismatch(a, c).has_value());
}

TEST_CASE("generated values are finite, normal, and v is positive", "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto s = State::generate(t, opts(1, 2, true));
    for (Role r : {Role::weight, Role::exp_avg, Role::exp_avg_sq, Role::grad}) {
        for (const auto& b : s.buffers(r)) {
            const auto d = b.data();
            REQUIRE(d.size() % 2 == 0);
            for (std::size_t i = 0; i < d.size(); i += 2) {
                std::uint16_t v;
                std::memcpy(&v, d.data() + i, 2);
                const unsigned exp = (v >> 7) & 0xFFu;
                REQUIRE(exp != 0);      // no zero/denormal
                REQUIRE(exp != 0xFFu);  // no Inf/NaN
                if (r == Role::exp_avg_sq) REQUIRE((v >> 15) == 0);
            }
        }
    }
}

TEST_CASE("padding is zero and aligned, and checkpoint order is w, m, v per tensor",
          "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto s = State::generate(t, opts(1, 2));
    std::uint64_t total = 0;
    std::size_t k = 0;
    for (const TensorBuffer* b : s.checkpointed()) {
        CHECK(b->padded().size() % 4096 == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(b->padded().data()) % 4096 == 0);
        for (std::size_t i = b->bytes; i < b->padded().size(); ++i) {
            REQUIRE(b->padded()[i] == std::byte{0});
        }
        CHECK(b->tensor == k / 3);
        CHECK(b->role == (k % 3 == 0   ? Role::weight
                          : k % 3 == 1 ? Role::exp_avg
                                       : Role::exp_avg_sq));
        total += b->padded().size();
        ++k;
    }
    CHECK(total == s.checkpoint_bytes());
    CHECK(s.checkpointed().size() == 3 * t.tensors.size());
}

TEST_CASE("padding report math", "[unit][state]") {
    TensorTable t;
    t.name = "x";
    t.tensors = {{"a", {1000}, DType::bf16, 2000}, {"b", {2048}, DType::bf16, 4096}};
    const auto r = compute_padding(t, 4096);
    CHECK(r.payload_bytes == 3 * 6096);
    CHECK(r.padded_bytes == 3 * (4096 + 4096));
    CHECK(compute_padding(t, 1).padded_bytes == r.payload_bytes);
}

TEST_CASE("gradients and staging are optional and sized right", "[unit][state]") {
    const auto t = testutil::tiny_table();
    auto o = opts(1, 2, true);
    o.staging = true;
    const auto s = State::generate(t, o);
    CHECK(s.has_gradients());
    CHECK(s.buffers(Role::grad).size() == t.tensors.size());
    CHECK(s.staging().size() == s.checkpoint_bytes());
    const auto z = State::zeros_like(s);
    CHECK(!z.has_gradients());
    CHECK(z.staging().empty());
    CHECK(first_mismatch(s, z).has_value());
}

TEST_CASE("zeros matches zeros_like(generate) in buffer sizes and alignment", "[unit][state]") {
    const auto t = testutil::tiny_table();
    auto o = opts(3, 2, true);
    o.staging = true;
    const auto z = State::zeros(t, o);
    const auto zl = State::zeros_like(State::generate(t, o));
    CHECK(z.alignment().mem == zl.alignment().mem);
    CHECK(z.alignment().offset == zl.alignment().offset);
    CHECK(z.checkpoint_bytes() == zl.checkpoint_bytes());
    CHECK(z.payload_bytes() == zl.payload_bytes());
    CHECK(!z.has_gradients());
    CHECK(z.staging().empty());
    const std::size_t mem_align = z.alignment().mem;
    const std::size_t unit = z.alignment().unit();
    REQUIRE(z.checkpointed().size() == zl.checkpointed().size());
    for (std::size_t i = 0; i < z.checkpointed().size(); ++i) {
        const TensorBuffer* a = z.checkpointed()[i];
        const TensorBuffer* b = zl.checkpointed()[i];
        CHECK(a->tensor == b->tensor);
        CHECK(a->role == b->role);
        CHECK(a->bytes == b->bytes);
        CHECK(a->padded().size() == b->padded().size());
        CHECK(a->padded().size() % unit == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(a->padded().data()) % mem_align == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(b->padded().data()) % mem_align == 0);
        for (std::byte x : a->padded()) REQUIRE(x == std::byte{0});
    }
}

TEST_CASE("RAM check refuses before allocating", "[unit][state][ram]") {
    CHECK_NOTHROW(env::check_free_ram(100, 100));
    CHECK_THROWS_AS(env::check_free_ram(101, 100), env::InsufficientRam);
    try {
        env::check_free_ram(18'000'000'000ull, 7'000'000'000ull);
        FAIL("expected throw");
    } catch (const env::InsufficientRam& e) {
        const std::string m = e.what();
        CHECK(m.find("18000000000") != std::string::npos);
        CHECK(m.find("7000000000") != std::string::npos);
    }
    auto o = opts(1, 1);
    o.available_ram_override = 1 << 20;
    CHECK_THROWS_AS(State::generate(testutil::tiny_table(), o), env::InsufficientRam);
}

TEST_CASE("full real layout requirement is about 18 GB", "[unit][state][ram]") {
    const auto t = TensorTable::load(testutil::real_table());
    StateOptions o;
    o.gradients = true;
    o.staging = true;
    const auto p = plan_state(t, DioAlign{4096, 4096, false}, o);
    const double gb = static_cast<double>(p.required()) / 1e9;
    CHECK(gb > 17.0);
    CHECK(gb < 18.5);
}

}  // namespace cistern
