#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "gen/state.h"

namespace cistern {

struct TensorTable;

inline constexpr std::uint64_t kBuffersPerTensor = 3;

inline constexpr char kMagic[7] = {'C', 'I', 'S', 'T', 'E', 'R', 'N'};
inline constexpr std::uint8_t kFormatVersion = 1;

struct Header {
    char magic[7];         // kMagic
    std::uint8_t version;  // kFormatVersion
    std::uint32_t tensor_count;
    std::uint32_t piece_count;  // length of the crc table that follows
    std::uint64_t align;        // Layout::align at write time
    std::uint64_t piece_size;   // Layout::piece_size
    std::uint64_t file_size;    // Layout::file_size; detects truncation
};
static_assert(sizeof(Header) == 40, "Header has implicit padding or changed size");
static_assert(std::is_trivially_copyable_v<Header>);

inline constexpr std::uint64_t kHeaderFixedSize = sizeof(Header);

struct BufferSlot {
    std::uint32_t tensor_idx = 0;  // index into TensorTable::tensors
    Role role = Role::weight;
    std::uint64_t data_size = 0;  // TensorInfo::bytes
    std::uint64_t disk_size = 0;  // round_up(data_size, align)
    std::uint64_t file_offset = 0;
    std::uint32_t first_piece_idx = 0;  // index into Layout::pieces
    std::uint32_t piece_count = 0;
};

struct Piece {
    std::uint32_t buffer_idx = 0;  // index into Layout::buffers
    std::uint64_t offset_in_buffer = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t size = 0;  // bytes to write; multiple of align
};

struct Layout {
    std::uint64_t align = 0;        // alignment from statx
    std::uint64_t piece_size = 0;   // max piece size
    std::uint64_t header_size = 0;  // rounded up to align
    std::uint64_t file_size = 0;    // header_size + sum of disk_size

    std::vector<BufferSlot> buffers;
    std::vector<Piece> pieces;
};

// Places the header, then weight/exp_avg/exp_avg_sq buffers for each tensor, split into
// pieces of at most piece_size (rounded up to align). Throws std::invalid_argument on a
// zero-byte tensor or a layout exceeding 2^32 pieces.
Layout make_layout(const TensorTable& table, std::uint64_t align, std::uint64_t piece_size);

// Fixed-size header fields for `layout`. The crc table and header crc that follow it on disk
// are computed by the writer.
Header make_header(const Layout& layout);

}  // namespace cistern
