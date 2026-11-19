// Seam harness: byte-for-byte round trip through the owner's writer and reader.
#include <catch2/catch_test_macros.hpp>

#include "cistern/reader.hpp"
#include "cistern/state.hpp"
#include "cistern/writer.hpp"
#include "test_util.hpp"

using namespace cistern;

TEST_CASE("write -> restore is byte for byte", "[seam][roundtrip]") {
    testutil::TempDir dir;
    StateOptions o;
    o.seed = 42;
    o.align_dir = dir.path();
    const State orig = State::generate(testutil::tiny_table(), o);
    const auto dst = dir.path() / "ckpt.bin";

    WriterConfig w;
    w.queue_depth = 8;
    w.chunk_bytes = 256 << 10;
    write_checkpoint(orig, dst, w);
    REQUIRE(std::filesystem::exists(dst));

    State back = State::zeros_like(orig);
    REQUIRE(first_mismatch(orig, back).has_value());  // the target really starts different
    ReaderConfig r;
    r.queue_depth = 8;
    r.chunk_bytes = 256 << 10;
    restore_checkpoint(dst, back, r);

    const auto bad = first_mismatch(orig, back);
    INFO("first mismatching buffer: " << bad.value_or("none"));
    CHECK(!bad.has_value());
}
