#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

#include "cistern/env.hpp"
#include "cistern/state.hpp"
#include "test_util.hpp"

using namespace cistern;

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

TEST_CASE("padding is zero, aligned, and offsets are contiguous", "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto s = State::generate(t, opts(1, 2));
    std::uint64_t expect = 0;
    for (const auto& e : s.checkpointed()) {
        CHECK(e.offset == expect);
        CHECK(e.offset % 4096 == 0);
        CHECK(e.buf->padded.size() % 4096 == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(e.buf->padded.data()) % 4096 == 0);
        for (std::size_t i = e.buf->bytes; i < e.buf->padded.size(); ++i) {
            REQUIRE(e.buf->padded[i] == std::byte{0});
        }
        expect += e.buf->padded.size();
    }
    CHECK(expect == s.checkpoint_bytes());
    CHECK(s.checkpointed().size() == 3 * t.tensors.size());
}

TEST_CASE("padding report math", "[unit][state]") {
    TensorTable t;
    t.name = "x";
    t.tensors = {{"a", {1000}, DType::bf16, 2000}, {"b", {2048}, DType::bf16, 4096}};
    const auto r = compute_padding(t, 4096);
    CHECK(r.payload_bytes == 3 * 6096);
    CHECK(r.padded_bytes == 3 * (4096 + 4096));
    CHECK(r.overhead_bytes == r.padded_bytes - r.payload_bytes);
    CHECK(r.overhead_pct == Catch::Approx(100.0 * static_cast<double>(r.overhead_bytes) /
                                          static_cast<double>(r.payload_bytes)));
    CHECK(compute_padding(t, 1).overhead_bytes == 0);
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
    const auto t = TensorTable::load(testutil::data_file("llama-3.2-1b.tensors.json"));
    StateOptions o;
    o.gradients = true;
    o.staging = true;
    const auto p = plan_state(t, DioAlign{4096, 4096, false}, o);
    const double gb = static_cast<double>(p.required()) / 1e9;
    CHECK(gb > 17.0);
    CHECK(gb < 18.5);
}
