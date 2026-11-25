#include "gen/state.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

#include "common/env.h"
#include "test_util.h"

namespace cistern {

namespace {

StateOptions opts(std::uint64_t seed, unsigned threads) {
    StateOptions o;
    o.seed = seed;
    o.threads = threads;
    o.align_override = DioAlign{4096, 4096, false};
    return o;
}

}  // namespace

TEST_CASE("generator is deterministic and thread-count independent", "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto a = State::generate(t, opts(7, 1));
    const auto b = State::generate(t, opts(7, 5));
    const auto c = State::generate(t, opts(8, 3));
    CHECK(!first_mismatch(a, b).has_value());
    CHECK(first_mismatch(a, c).has_value());
}

TEST_CASE("generated values are finite, normal, and v is positive", "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto s = State::generate(t, opts(1, 2));
    for (const TensorBuffer& b : s.checkpointed()) {
        const auto d = b.data();
        REQUIRE(d.size() % 2 == 0);
        for (std::size_t i = 0; i < d.size(); i += 2) {
            std::uint16_t v;
            std::memcpy(&v, d.data() + i, 2);
            const unsigned exp = (v >> 7) & 0xFFu;
            REQUIRE(exp != 0);      // no zero/denormal
            REQUIRE(exp != 0xFFu);  // no Inf/NaN
            if (b.role == Role::exp_avg_sq) REQUIRE((v >> 15) == 0);
        }
    }
}

TEST_CASE("padding is zero and aligned, and checkpoint order is w, m, v per tensor",
          "[unit][state]") {
    const auto t = testutil::tiny_table();
    const auto s = State::generate(t, opts(1, 2));
    std::uint64_t total = 0;
    std::size_t k = 0;
    for (const TensorBuffer& b : s.checkpointed()) {
        CHECK(b.padded().size() % 4096 == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(b.padded().data()) % 4096 == 0);
        for (std::size_t i = b.bytes; i < b.padded().size(); ++i) {
            REQUIRE(b.padded()[i] == std::byte{0});
        }
        CHECK(b.tensor == k / 3);
        CHECK(b.role == (k % 3 == 0   ? Role::weight
                         : k % 3 == 1 ? Role::exp_avg
                                      : Role::exp_avg_sq));
        total += b.padded().size();
        ++k;
    }
    CHECK(total == s.checkpoint_bytes());
    CHECK(s.checkpointed().size() == 3 * t.tensors.size());
}

TEST_CASE("zeros_like is all-zero with the source's layout and alignment", "[unit][state]") {
    const auto s = State::generate(testutil::tiny_table(), opts(3, 2));
    const auto z = State::zeros_like(s);
    CHECK(z.alignment().mem == s.alignment().mem);
    CHECK(z.alignment().offset == s.alignment().offset);
    CHECK(z.checkpoint_bytes() == s.checkpoint_bytes());
    CHECK(first_mismatch(s, z).has_value());
    const std::size_t mem_align = z.alignment().mem;
    const std::size_t unit = z.alignment().unit();
    REQUIRE(z.checkpointed().size() == s.checkpointed().size());
    for (std::size_t i = 0; i < z.checkpointed().size(); ++i) {
        const TensorBuffer& a = z.checkpointed()[i];
        const TensorBuffer& b = s.checkpointed()[i];
        CHECK(a.tensor == b.tensor);
        CHECK(a.role == b.role);
        CHECK(a.bytes == b.bytes);
        CHECK(a.padded().size() == b.padded().size());
        CHECK(a.padded().size() % unit == 0);
        CHECK(reinterpret_cast<std::uintptr_t>(a.padded().data()) % mem_align == 0);
        for (std::byte x : a.padded()) REQUIRE(x == std::byte{0});
    }
}

TEST_CASE("free-RAM check reports required and available bytes", "[unit][state][ram]") {
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
}

}  // namespace cistern
