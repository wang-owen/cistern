#include "common/crc32c.h"

#include <catch2/catch_test_macros.hpp>

namespace cistern {

TEST_CASE("basic", "[unit][crc32c]") {
    constexpr std::string_view s = "123456789";
    const auto bytes = std::as_bytes(std::span(s));
    CHECK(crc32c(bytes.data(), bytes.size()) == 0xE3069283u);
}

}  // namespace cistern
