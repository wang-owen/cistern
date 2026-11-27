#include "restore.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>

#include "layout.h"
#include "state.h"
#include "test_util.h"
#include "writer.h"

namespace cistern {

namespace {

constexpr std::uint64_t kAlign = 4096;

State make_state(std::uint64_t seed) {
    StateOptions o;
    o.seed = seed;
    o.threads = 2;
    o.align_override = DioAlign{kAlign, kAlign, false};
    return State::generate(testutil::tiny_table(), o);
}

// XORs the byte at `offset` in `path` with `mask`, in place.
void corrupt_byte(const std::filesystem::path& path, std::uint64_t offset,
                  std::uint8_t mask = 0xff) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(f);
    f.seekg(static_cast<std::streamoff>(offset));
    char c;
    REQUIRE(f.get(c));
    f.seekp(static_cast<std::streamoff>(offset));
    f.put(static_cast<char>(static_cast<std::uint8_t>(c) ^ mask));
    REQUIRE(f.flush());
}

// Payload matches `expect` and every byte of padding is zero.
void check_restored(const State& expect, const State& got) {
    const auto mismatch = first_mismatch(expect, got);
    INFO("first mismatch: " << mismatch.value_or("<none>"));
    CHECK_FALSE(mismatch.has_value());

    for (const TensorBuffer& b : got.checkpointed()) {
        const auto padded = b.padded();
        for (std::size_t i = b.bytes; i < padded.size(); ++i) {
            INFO("tensor " << b.tensor << " role " << static_cast<unsigned>(b.role)
                           << " padding byte " << i);
            REQUIRE(padded[i] == std::byte{0});
        }
    }
}

}  // namespace

TEST_CASE("restore_checkpoint round-trips across piece sizes", "[unit][restore]") {
    // align-sized pieces, split buffers with a short tail, a size make_layout must round up,
    // and one piece per buffer
    const std::uint64_t piece_size =
        GENERATE(kAlign, 3 * kAlign, kAlign + 1, std::uint64_t{1} << 20);
    INFO("piece_size " << piece_size);

    testutil::TempDir dir;
    const State src = make_state(7);
    const Layout layout = make_layout(src.table(), kAlign, piece_size);
    write_checkpoint(src, layout, dir.path());

    SECTION("into a zeroed state") {
        State dst = State::zeros_like(src);
        restore_checkpoint(dir.path(), dst);
        check_restored(src, dst);
    }
    SECTION("into a state holding other data") {
        // any region restore fails to read keeps seed-8 data and shows up as a mismatch
        State dst = make_state(8);
        REQUIRE(first_mismatch(src, dst).has_value());
        restore_checkpoint(dir.path(), dst);
        check_restored(src, dst);
    }
}

TEST_CASE("restore_checkpoint reads the latest of two checkpoints", "[unit][restore]") {
    testutil::TempDir dir;
    const State first = make_state(1);
    const State second = make_state(2);
    const Layout layout = make_layout(first.table(), kAlign, 4 * kAlign);
    write_checkpoint(first, layout, dir.path());
    write_checkpoint(second, layout, dir.path());

    State dst = State::zeros_like(first);
    restore_checkpoint(dir.path(), dst);
    check_restored(second, dst);
}

TEST_CASE("restore_checkpoint throws when the checkpoint is missing", "[unit][restore]") {
    testutil::TempDir dir;
    State dst = State::zeros_like(make_state(1));
    CHECK_THROWS(restore_checkpoint(dir.path(), dst));
}

TEST_CASE("restore_checkpoint rejects a damaged checkpoint", "[unit][restore]") {
    testutil::TempDir dir;
    const State src = make_state(7);
    const Layout layout = make_layout(src.table(), kAlign, 3 * kAlign);
    write_checkpoint(src, layout, dir.path());

    const std::filesystem::path ckpt = dir.path() / "ckpt";
    State dst = State::zeros_like(src);

    // Each SECTION reruns the setup above, so it starts from a fresh, valid checkpoint.
    // Header field offsets: magic 0, version 7, tensor_count 8, piece_count 12, align 16,
    // piece_size 24, file_size 32; the crc table follows at sizeof(Header).

    SECTION("flipped byte in a piece") {
        corrupt_byte(ckpt, layout.pieces[1].file_offset + 17);
        REQUIRE_THROWS_WITH(restore_checkpoint(dir.path(), dst),
                            Catch::Matchers::ContainsSubstring("crc mismatch in piece 1"));
    }

    SECTION("bad magic") {
        corrupt_byte(ckpt, 1);
        REQUIRE_THROWS_WITH(
            restore_checkpoint(dir.path(), dst),
            Catch::Matchers::ContainsSubstring("invalid checkpoint file: bad magic"));
    }

    SECTION("flipped byte in the crc table") {
        // the header crc covers the table, so this fails before any piece is read; a
        // "crc mismatch in piece 1" here would mean the table went unchecked
        corrupt_byte(ckpt, sizeof(Header) + 1 * sizeof(std::uint32_t));
        REQUIRE_THROWS_WITH(restore_checkpoint(dir.path(), dst),
                            Catch::Matchers::ContainsSubstring("header crc mismatch"));
    }

    SECTION("truncated file") {
        // still a whole number of align blocks, so only the file_size check can catch it
        std::filesystem::resize_file(ckpt, layout.file_size - kAlign);
        REQUIRE_THROWS_WITH(
            restore_checkpoint(dir.path(), dst),
            Catch::Matchers::ContainsSubstring(std::format(
                "header says {} bytes, file has {}", layout.file_size, layout.file_size - kAlign)));
    }

    // align is 4096 = 0x1000 little-endian, so 0x10 sits in byte 17
    SECTION("align of zero") {
        corrupt_byte(ckpt, 17, 0x10);  // 0x1000 -> 0x0000
        REQUIRE_THROWS_WITH(restore_checkpoint(dir.path(), dst),
                            Catch::Matchers::ContainsSubstring("align 0 is not a power of two"));
    }

    SECTION("piece_size that does not match the piece count") {
        // 0x3000 -> 0x2000: still in range, but make_layout now yields more pieces than the
        // header records. make_header is compared before the header crc, so no crc fixup is
        // needed to reach that check.
        corrupt_byte(ckpt, 25, 0x10);
        REQUIRE_THROWS_WITH(
            restore_checkpoint(dir.path(), dst),
            Catch::Matchers::ContainsSubstring("header does not match the layout for this table"));
    }

    SECTION("align not a power of two") {
        corrupt_byte(ckpt, 17, 0x20);  // 0x1000 -> 0x3000
        REQUIRE_THROWS_WITH(
            restore_checkpoint(dir.path(), dst),
            Catch::Matchers::ContainsSubstring("align 12288 is not a power of two"));
    }
}

TEST_CASE("restore_checkpoint rejects a state built from another table", "[unit][restore]") {
    testutil::TempDir dir;
    const State src = make_state(7);
    write_checkpoint(src, make_layout(src.table(), kAlign, 3 * kAlign), dir.path());

    const TensorTable other = TensorTable::parse(R"({
        "schema": "cistern.tensors/1", "name": "other", "verified": false,
        "tensor_count": 1, "total_params": 1000, "total_bytes": 2000,
        "tensors": [{"name": "w", "shape": [10, 100], "dtype": "bf16", "bytes": 2000}]
    })");
    StateOptions o;
    o.threads = 2;
    o.align_override = DioAlign{kAlign, kAlign, false};
    State dst = State::generate(other, o);

    // The layout is rebuilt from dst's table, so a different table shows up as a header
    // mismatch (tensor_count, piece_count, file_size) before any data is read.
    REQUIRE_THROWS_WITH(
        restore_checkpoint(dir.path(), dst),
        Catch::Matchers::ContainsSubstring("header does not match the layout for this table"));
}

}  // namespace cistern
