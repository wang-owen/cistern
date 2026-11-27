#include "state.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

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

}  // namespace cistern
