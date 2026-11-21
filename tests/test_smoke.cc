#include <catch2/catch_test_macros.hpp>

TEST_CASE("test runner is wired", "[smoke]") {
    CHECK(1 + 1 == 2);
}
