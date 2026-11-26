#include "layout/layout.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "common/align.h"
#include "gen/state.h"
#include "table/tensor_table.h"

namespace cistern {

Layout make_layout(const TensorTable& table, std::uint64_t align, std::uint64_t piece_size) {
    assert(align != 0 && piece_size != 0);

    Layout layout;
    layout.align = align;
    const std::uint64_t aligned_piece_size = round_up(piece_size, align);
    layout.piece_size = aligned_piece_size;
    auto piece_count_for = [aligned_piece_size](std::uint64_t disk_size) -> std::uint64_t {
        return (disk_size + aligned_piece_size - 1) / aligned_piece_size;  // ceil
    };

    // get total piece count for header size
    std::uint64_t total_piece_count = 0;
    for (const auto& tensor : table.tensors) {
        if (tensor.bytes == 0) {
            throw std::invalid_argument("tensor '" + tensor.name + "' has zero bytes");
        }
        total_piece_count += kBuffersPerTensor * piece_count_for(round_up(tensor.bytes, align));
    }
    if (total_piece_count > UINT32_MAX) {
        throw std::invalid_argument("layout exceeds 2^32 pieces");
    }
    layout.header_size =
        round_up(kHeaderFixedSize                                 // header
                     + total_piece_count * sizeof(std::uint32_t)  // one crc per piece
                     + sizeof(std::uint32_t),                     // header crc
                 align);

    // place buffers and pieces
    std::uint64_t next_file_offset = layout.header_size;
    std::uint32_t next_piece_idx = 0;
    for (std::uint32_t tensor_idx = 0; tensor_idx < table.tensors.size(); ++tensor_idx) {
        const std::uint64_t data_size = table.tensors[tensor_idx].bytes;
        const std::uint64_t disk_size = round_up(data_size, align);
        // safe: each buffer's count <= total_piece_count <= UINT32_MAX (checked above)
        const auto piece_count = static_cast<std::uint32_t>(piece_count_for(disk_size));

        for (Role role : {Role::weight, Role::exp_avg, Role::exp_avg_sq}) {
            // claim space: returns the start, advances the cursor
            const std::uint64_t buf_file_offset =
                std::exchange(next_file_offset, next_file_offset + disk_size);
            const std::uint32_t first_piece_idx =
                std::exchange(next_piece_idx, next_piece_idx + piece_count);

            // build buffer
            BufferSlot buf{.tensor_idx = tensor_idx,
                           .role = role,
                           .data_size = data_size,
                           .disk_size = disk_size,
                           .file_offset = buf_file_offset,
                           .first_piece_idx = first_piece_idx,
                           .piece_count = piece_count};
            layout.buffers.push_back(buf);
            const auto buffer_idx = static_cast<std::uint32_t>(layout.buffers.size() - 1);

            // build pieces; last one may be shorter (still a multiple of align)
            for (std::uint32_t j = 0; j < piece_count; ++j) {
                const std::uint64_t offset_in_buffer = j * aligned_piece_size;
                Piece piece{.buffer_idx = buffer_idx,
                            .offset_in_buffer = offset_in_buffer,
                            .file_offset = buf_file_offset + offset_in_buffer,
                            .size = std::min(aligned_piece_size, disk_size - offset_in_buffer)};
                layout.pieces.push_back(piece);
            }
        }
    }

    layout.file_size = next_file_offset;

    assert(layout.pieces.size() == total_piece_count);
    assert(next_file_offset % align == 0);

    return layout;
}

Header make_header(const Layout& layout) {
    assert(layout.buffers.size() % kBuffersPerTensor == 0);

    Header header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kFormatVersion;
    header.tensor_count = static_cast<std::uint32_t>(layout.buffers.size() / kBuffersPerTensor);
    header.piece_count = static_cast<std::uint32_t>(layout.pieces.size());
    header.align = layout.align;
    header.piece_size = layout.piece_size;
    header.file_size = layout.file_size;
    return header;
}

}  // namespace cistern
