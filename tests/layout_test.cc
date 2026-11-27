#include "layout.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor_table.h"
#include "test_util.h"

namespace pitstop {

namespace {

TensorTable table_of(const std::vector<std::uint64_t>& sizes) {
    TensorTable t;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        TensorInfo info;
        info.name = "t" + std::to_string(i);
        info.bytes = sizes[i];
        t.tensors.push_back(info);
    }
    return t;
}

// Checks every structural invariant of a layout against the table it was built from.
void check_invariants(const Layout& l, const TensorTable& t, std::uint64_t align,
                      std::uint64_t piece_size) {
    INFO("align=" << align << " piece_size=" << piece_size << " tensors=" << t.tensors.size());

    // --- header / globals ---
    REQUIRE(l.align == align);
    CHECK(l.piece_size >= piece_size);
    CHECK(l.piece_size % align == 0);
    CHECK(l.piece_size < piece_size + align);  // minimal round-up
    CHECK(l.header_size % align == 0);
    CHECK(l.file_size % align == 0);
    // The header (fixed part + one crc per piece + header crc) fits in the reserved block.
    const std::uint64_t header_bytes =
        kHeaderFixedSize + l.pieces.size() * sizeof(std::uint32_t) + sizeof(std::uint32_t);
    CHECK(l.header_size >= header_bytes);
    CHECK(l.header_size < header_bytes + align);  // and is no bigger than needed

    // --- buffers ---
    REQUIRE(l.buffers.size() == kBuffersPerTensor * t.tensors.size());
    std::uint64_t cursor = l.header_size;  // buffers must be contiguous, in order, after header
    std::uint32_t piece_cursor = 0;
    std::uint64_t sum_disk = 0;
    for (std::size_t b = 0; b < l.buffers.size(); ++b) {
        INFO("buffer " << b);
        const BufferSlot& buf = l.buffers[b];
        const std::size_t ti = b / kBuffersPerTensor;
        const Role expect_role = static_cast<Role>(b % kBuffersPerTensor);

        CHECK(buf.tensor_idx == ti);
        CHECK(buf.role == expect_role);
        CHECK(buf.data_size == t.tensors[ti].bytes);
        CHECK(buf.disk_size >= buf.data_size);  // covers all tensor bytes
        CHECK(buf.disk_size < buf.data_size + align);
        CHECK(buf.disk_size % align == 0);
        CHECK(buf.file_offset % align == 0);
        CHECK(buf.file_offset >= l.header_size);  // never overlaps header
        CHECK(buf.file_offset == cursor);         // no overlap, no gap with previous buffer
        cursor = buf.file_offset + buf.disk_size;
        sum_disk += buf.disk_size;

        CHECK(buf.first_piece_idx == piece_cursor);
        CHECK(buf.piece_count >= 1);
        CHECK(buf.piece_count == (buf.disk_size + l.piece_size - 1) / l.piece_size);
        piece_cursor += buf.piece_count;

        // --- pieces of this buffer tile [0, disk_size) exactly ---
        REQUIRE(std::uint64_t{buf.first_piece_idx} + buf.piece_count <= l.pieces.size());
        std::uint64_t in_buf = 0;
        for (std::uint32_t j = 0; j < buf.piece_count; ++j) {
            INFO("piece " << (buf.first_piece_idx + j));
            const Piece& p = l.pieces[buf.first_piece_idx + j];
            CHECK(p.buffer_idx == b);
            CHECK(p.offset_in_buffer == in_buf);
            CHECK(p.file_offset == buf.file_offset + p.offset_in_buffer);
            CHECK(p.size > 0);
            CHECK(p.size <= l.piece_size);
            if (j + 1 < buf.piece_count) CHECK(p.size == l.piece_size);  // only last is short
            CHECK(p.offset_in_buffer % align == 0);
            CHECK(p.file_offset % align == 0);
            CHECK(p.size % align == 0);
            in_buf += p.size;
        }
        CHECK(in_buf == buf.disk_size);
    }
    CHECK(piece_cursor == l.pieces.size());
    CHECK(l.file_size == cursor);
    CHECK(l.file_size == l.header_size + sum_disk);

    // --- every piece belongs to exactly one buffer and pieces are file-ordered, disjoint ---
    std::uint64_t file_cursor = l.header_size;
    for (const Piece& p : l.pieces) {
        REQUIRE(p.buffer_idx < l.buffers.size());
        CHECK(p.file_offset == file_cursor);
        file_cursor = p.file_offset + p.size;
    }
    CHECK(file_cursor == l.file_size);
}

}  // namespace

TEST_CASE("layout invariants over a grid of sizes", "[unit][layout]") {
    const std::uint64_t align = GENERATE(1u, 512u, 4096u);
    const std::uint64_t piece_size = GENERATE(1u, 511u, 512u, 4096u, 5000u, 1u << 20);
    std::vector<std::uint64_t> sizes{1, 511, 512, 513, 4095, 4096, 4097, 12345};
    if (piece_size >= 512) {  // keep byte-sized pieces cheap; ~6M pieces otherwise
        sizes.push_back(1u << 20);
        sizes.push_back((1u << 20) + 1);
    }
    const auto t = table_of(sizes);
    const Layout l = make_layout(t, align, piece_size);
    check_invariants(l, t, align, piece_size);
}

TEST_CASE("layout on the tiny fixture table", "[unit][layout]") {
    const auto t = testutil::tiny_table();
    const Layout l = make_layout(t, 4096, 1u << 20);
    check_invariants(l, t, 4096, 1u << 20);
}

TEST_CASE("layout on the real Llama table", "[unit][layout]") {
    const auto t = TensorTable::load(default_table_path());
    const std::uint64_t piece_size = GENERATE(1u << 20, 8u << 20);
    const Layout l = make_layout(t, 4096, piece_size);
    check_invariants(l, t, 4096, piece_size);
    const double gb = static_cast<double>(l.file_size - l.header_size) / 1e9;
    CHECK(gb > 7.0);  // ~7.4 GB of padded weight + Adam moments
    CHECK(gb < 8.0);
}

TEST_CASE("layout concrete values", "[unit][layout]") {
    // one 5000-byte tensor, align 4096, piece 4096 -> disk 8192, 2 pieces per buffer
    const auto t = table_of({5000});
    const Layout l = make_layout(t, 4096, 4096);
    CHECK(l.piece_size == 4096);
    CHECK(l.header_size == 4096);  // 40 + 6*4 + 4 = 68 -> 4096
    REQUIRE(l.buffers.size() == 3);
    REQUIRE(l.pieces.size() == 6);
    CHECK(l.buffers[0].file_offset == 4096);
    CHECK(l.buffers[1].file_offset == 4096 + 8192);
    CHECK(l.buffers[2].file_offset == 4096 + 2 * 8192);
    CHECK(l.file_size == 4096 + 3 * 8192);
    CHECK(l.pieces[1].offset_in_buffer == 4096);
    CHECK(l.pieces[1].size == 4096);
    CHECK(l.pieces[5].file_offset == l.file_size - 4096);
}

TEST_CASE("header grows past one block when there are many pieces", "[unit][layout]") {
    // header_size tracks the crc table: one block for few pieces, more for many.
    const auto t = table_of({4096});
    const Layout l = make_layout(t, 4096, 4096);
    CHECK(l.header_size == 4096);
    const Layout big = make_layout(t, 512, 512);  // 24 pieces: 40 + 96 + 4 = 140 -> 512
    CHECK(big.header_size == 512);
    const Layout many = make_layout(table_of({1u << 20}), 512, 512);  // 3*2048 pieces
    CHECK(many.header_size == round_up(kHeaderFixedSize + 3 * 2048 * 4 + 4, 512));
    CHECK(many.buffers[0].file_offset == many.header_size);
    check_invariants(many, table_of({1u << 20}), 512, 512);
}

TEST_CASE("zero tensors gives a header-only layout", "[unit][layout][edge]") {
    const TensorTable t{};
    const Layout l = make_layout(t, 4096, 1u << 20);
    CHECK(l.buffers.empty());
    CHECK(l.pieces.empty());
    CHECK(l.header_size == 4096);
    CHECK(l.file_size == l.header_size);
    check_invariants(l, t, 4096, 1u << 20);
}

TEST_CASE("zero-byte tensor throws", "[unit][layout][edge]") {
    CHECK_THROWS_AS(make_layout(table_of({0}), 4096, 4096), std::invalid_argument);
    CHECK_THROWS_AS(make_layout(table_of({4096, 0, 4096}), 4096, 4096), std::invalid_argument);
}

TEST_CASE("more than 2^32 pieces throws", "[unit][layout][edge]") {
    // 3 * 2^31 one-byte pieces > UINT32_MAX; must throw before allocating pieces.
    CHECK_THROWS_AS(make_layout(table_of({std::uint64_t{1} << 31}), 1, 1), std::invalid_argument);
    // Same, summed over several tensors that are each individually fine.
    CHECK_THROWS_AS(make_layout(table_of({std::uint64_t{1} << 30, std::uint64_t{1} << 30}), 1, 1),
                    std::invalid_argument);
}

TEST_CASE("single-byte tensors with align 1", "[unit][layout][edge]") {
    const auto t = table_of({1, 1, 1});
    const Layout l = make_layout(t, 1, 1);
    CHECK(l.pieces.size() == 9);
    CHECK(l.file_size == l.header_size + 9);
    check_invariants(l, t, 1, 1);
}

}  // namespace pitstop
