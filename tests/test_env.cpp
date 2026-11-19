#include <catch2/catch_test_macros.hpp>

#include "cistern/env.hpp"

using namespace cistern::env;

TEST_CASE("kernel release parsing", "[unit][env]") {
    CHECK(parse_kernel_release("6.1.0-18-arm64") == KernelVersion{6, 1, 0});
    CHECK(parse_kernel_release("7.2.12-orbstack-00394-gbecdd6130068") == KernelVersion{7, 2, 12});
    CHECK(parse_kernel_release("5.15") == KernelVersion{5, 15, 0});
    CHECK(parse_kernel_release("6.8-rc1") == KernelVersion{6, 8, 0});
    CHECK(!parse_kernel_release("garbage").has_value());
    CHECK(!parse_kernel_release("6").has_value());
    CHECK(!parse_kernel_release("").has_value());
    CHECK(KernelVersion{6, 0, 99} < KernelVersion{6, 1, 0});
}

TEST_CASE("kernel requirement", "[unit][env]") {
    CHECK_NOTHROW(require_kernel(2, 6));
    CHECK_THROWS_AS(require_kernel(999, 0), std::runtime_error);
}

TEST_CASE("meminfo parsing", "[unit][env]") {
    CHECK(parse_mem_available("MemTotal: 1 kB\nMemAvailable:   2048 kB\n") == 2048ull * 1024);
    CHECK(!parse_mem_available("MemTotal: 1 kB\n").has_value());
    CHECK(mem_available_bytes() > 0);
}

TEST_CASE("memlock is readable", "[unit][env]") {
    CHECK(memlock_limit_bytes() > 0);
}
