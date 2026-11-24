#include "common/align.h"

#include <catch2/catch_test_macros.hpp>

#include "test_util.h"

namespace cistern {

TEST_CASE("round_up", "[unit][align]") {
    CHECK(round_up(0, 4096) == 0);
    CHECK(round_up(1, 4096) == 4096);
    CHECK(round_up(4096, 4096) == 4096);
    CHECK(round_up(4097, 4096) == 8192);
    CHECK(round_up(5, 1) == 5);
}

TEST_CASE("DIO alignment fallback", "[unit][align]") {
    auto a = resolve_dio_align(true, 512, 512);
    CHECK(a.mem == 512);
    CHECK(a.offset == 512);
    CHECK(a.from_statx);
    a = resolve_dio_align(false, 0, 0);
    CHECK(a.mem == 4096);
    CHECK(a.offset == 4096);
    CHECK(!a.from_statx);
    CHECK(!resolve_dio_align(true, 0, 512).from_statx);
    CHECK(!resolve_dio_align(true, 512, 3).from_statx);  // not a power of two
    CHECK(resolve_dio_align(true, 512, 4096).unit() == 4096);
}

TEST_CASE("probe on a temp dir yields a usable alignment", "[unit][align]") {
    testutil::TempDir d;
    const auto a = probe_dio_align(d.path());
    CHECK(a.mem >= 1);
    CHECK(a.offset >= 1);
    CHECK(a.unit() >= 512);
    // The probe file is cleaned up.
    CHECK(std::filesystem::is_empty(d.path()));
}

TEST_CASE("AlignedBuffer is page aligned and guards alignment", "[unit][align]") {
    AlignedBuffer b(10000, 4096, true);
    CHECK(reinterpret_cast<std::uintptr_t>(b.data()) % page_size() == 0);
    CHECK(b.size() == 10000);
    AlignedBuffer moved = std::move(b);
    CHECK(b.data() == nullptr);
    CHECK(moved.data() != nullptr);
    CHECK_THROWS_AS(AlignedBuffer(4096, page_size() * 2, false), std::invalid_argument);
}

}  // namespace cistern
