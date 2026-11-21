#include <hip/hip_runtime_api.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>

extern "C" int cistern_hip_add_one(int* host, int n);

TEST_CASE("a HIP device is visible", "[hip]") {
    int n = 0;
    REQUIRE(hipGetDeviceCount(&n) == hipSuccess);
    CHECK(n >= 1);
}

TEST_CASE("a hipcc-built kernel round trips", "[hip]") {
    std::vector<int> v(1000);
    for (int i = 0; i < 1000; ++i) v[static_cast<std::size_t>(i)] = i;
    REQUIRE(cistern_hip_add_one(v.data(), 1000) == 0);
    for (int i = 0; i < 1000; ++i) CHECK(v[static_cast<std::size_t>(i)] == i + 1);
}
