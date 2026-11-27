#include "writer.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "crc32c.h"
#include "layout.h"
#include "state.h"
#include "test_util.h"

namespace pitstop {

namespace {

constexpr std::uint64_t kAlign = 4096;

State make_state(std::uint64_t seed) {
    StateOptions o;
    o.seed = seed;
    o.threads = 2;
    o.align_override = DioAlign{kAlign, kAlign, false};
    return State::generate(testutil::tiny_table(), o);
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

std::uint32_t load_u32(const std::vector<std::byte>& file, std::uint64_t offset) {
    std::uint32_t v;
    std::memcpy(&v, file.data() + offset, sizeof(v));
    return v;
}

// Checks `file` is a complete, valid checkpoint of `state` laid out as `layout`.
void check_checkpoint(const std::vector<std::byte>& file, const State& state,
                      const Layout& layout) {
    REQUIRE(file.size() == layout.file_size);

    // --- fixed header ---
    Header h;
    std::memcpy(&h, file.data(), sizeof(h));
    const Header expect = make_header(layout);
    CHECK(std::memcmp(h.magic, kMagic, sizeof(kMagic)) == 0);
    CHECK(h.version == kFormatVersion);
    CHECK(h.tensor_count == expect.tensor_count);
    CHECK(h.piece_count == expect.piece_count);
    CHECK(h.align == expect.align);
    CHECK(h.piece_size == expect.piece_size);
    CHECK(h.file_size == expect.file_size);

    // --- header crc covers fixed header + crc table, stored right after ---
    const std::uint64_t table_bytes = layout.pieces.size() * sizeof(std::uint32_t);
    const std::uint64_t crc_offset = sizeof(Header) + table_bytes;
    CHECK(load_u32(file, crc_offset) == crc32c(file.data(), crc_offset));

    // --- header padding is zero ---
    for (std::uint64_t i = crc_offset + sizeof(std::uint32_t); i < layout.header_size; ++i) {
        INFO("header byte " << i);
        REQUIRE(file[i] == std::byte{0});
    }

    // --- each piece: stored crc matches the file and the in-memory source ---
    const auto buffers = state.checkpointed();
    for (std::size_t i = 0; i < layout.pieces.size(); ++i) {
        INFO("piece " << i);
        const Piece& p = layout.pieces[i];
        const std::byte* on_disk = file.data() + p.file_offset;
        const std::byte* in_mem = buffers[p.buffer_idx].padded().data() + p.offset_in_buffer;

        REQUIRE(std::memcmp(on_disk, in_mem, p.size) == 0);
        CHECK(load_u32(file, sizeof(Header) + i * sizeof(std::uint32_t)) ==
              crc32c(on_disk, p.size));
    }
}

}  // namespace

TEST_CASE("write_checkpoint round-trips the tiny table", "[unit][writer]") {
    testutil::TempDir dir;
    const State state = make_state(7);
    // small pieces so buffers split into several, with a short last piece
    const Layout layout = make_layout(state.table(), kAlign, 3 * kAlign);
    REQUIRE(layout.pieces.size() > layout.buffers.size());

    write_checkpoint(state, layout, dir.path());

    CHECK(std::filesystem::exists(dir.path() / "ckpt"));
    CHECK_FALSE(std::filesystem::exists(dir.path() / "ckpt.tmp"));
    check_checkpoint(read_file(dir.path() / "ckpt"), state, layout);
}

TEST_CASE("one piece per buffer when piece_size exceeds every buffer", "[unit][writer]") {
    testutil::TempDir dir;
    const State state = make_state(3);
    const Layout layout = make_layout(state.table(), kAlign, 1 << 20);
    REQUIRE(layout.pieces.size() == layout.buffers.size());

    write_checkpoint(state, layout, dir.path());
    check_checkpoint(read_file(dir.path() / "ckpt"), state, layout);
}

}  // namespace pitstop
