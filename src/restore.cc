#include "restore.h"

#include <fcntl.h>
#include <liburing.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "align.h"
#include "crc32c.h"
#include "direct_file.h"
#include "layout.h"
#include "ring.h"
#include "state.h"

namespace pitstop {

namespace {

constexpr unsigned kQueueDepth = 16;
constexpr unsigned kMaxRetries = 10;

void aligned_pread(int fd, std::size_t unit, void* out, std::size_t out_size, std::size_t offset) {
    const std::size_t start = round_down(offset, unit);
    const std::size_t skip = offset - start;
    const std::size_t want = skip + out_size;
    const std::size_t aligned_size = round_up(want, unit);
    AlignedBuffer buf(aligned_size, unit, false);
    std::size_t bytes_read = 0;  // always a multiple of unit
    while (bytes_read < want) {
        const ssize_t ret = ::pread(fd, buf.data() + bytes_read, aligned_size - bytes_read,
                                    static_cast<off_t>(start + bytes_read));
        if (ret == -1) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "::pread");
        }
        const std::size_t end = bytes_read + static_cast<std::size_t>(ret);
        if (end >= want) {
            break;
        }
        const std::size_t next = round_down(end, unit);
        if (next == bytes_read) {
            throw std::runtime_error("invalid checkpoint file: short read");
        }
        bytes_read = next;
    }
    std::memcpy(out, buf.data() + skip, out_size);
}

// Every format error starts with "invalid checkpoint file" so callers can match on it.
std::runtime_error invalid_checkpoint(const std::string& reason) {
    return std::runtime_error("invalid checkpoint file: " + reason);
}

}  // namespace

void restore_checkpoint(const std::filesystem::path& dir, State& state) {
    const std::filesystem::path filepath = dir / "ckpt";
    DirectFile file = DirectFile::open_read(filepath);
    std::size_t unit = probe_dio_align(filepath).unit();

    Header header;
    aligned_pread(file.get(), unit, &header, sizeof(Header), 0);

    struct stat file_info;
    if (::fstat(file.get(), &file_info) == -1) {
        throw std::system_error(errno, std::generic_category(), "::fstat");
    }
    const auto file_size = static_cast<std::uint64_t>(file_info.st_size);

    // verify header; bounding align and piece_size by the real file size keeps
    // round_up in make_layout from overflowing
    if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
        throw invalid_checkpoint("bad magic");
    }
    if (header.file_size != file_size) {
        throw invalid_checkpoint(
            std::format("header says {} bytes, file has {}", header.file_size, file_size));
    }
    if (header.align == 0 || (header.align & (header.align - 1)) != 0) {
        throw invalid_checkpoint(std::format("align {} is not a power of two", header.align));
    }
    if (header.align % unit != 0 || header.align > file_size) {
        throw invalid_checkpoint(std::format(
            "align {} incompatible with device unit {} or file size", header.align, unit));
    }
    if (header.piece_size == 0 || header.piece_size > file_size) {
        throw invalid_checkpoint(std::format("piece_size {} out of range", header.piece_size));
    }
    Layout layout = make_layout(state.table(), header.align, header.piece_size);
    if (make_header(layout) != header) {
        throw invalid_checkpoint("header does not match the layout for this table");
    }

    // validate crcs
    std::vector<std::uint32_t> crcs(header.piece_count);
    std::vector<std::byte> hdr(layout.header_size);
    aligned_pread(file.get(), unit, hdr.data(), hdr.size(), 0);
    const std::size_t crc_bytes = header.piece_count * sizeof(std::uint32_t);
    std::uint32_t stored;
    std::memcpy(&stored, hdr.data() + sizeof(Header) + crc_bytes, sizeof(stored));
    if (crc32c(hdr.data(), sizeof(Header) + crc_bytes) != stored) {
        throw invalid_checkpoint("header crc mismatch");
    }
    std::memcpy(crcs.data(), hdr.data() + sizeof(Header), crc_bytes);

    std::span<TensorBuffer> checkpoint = state.checkpointed();
    if (checkpoint.size() != layout.buffers.size()) {
        throw std::invalid_argument("state does not match checkpoint layout");
    }
    for (std::size_t b = 0; b < layout.buffers.size(); ++b) {
        if (checkpoint[b].padded().size() < layout.buffers[b].disk_size) {
            throw std::invalid_argument("state buffer smaller than checkpoint buffer");
        }
    }

    Ring ring{kQueueDepth};
    unsigned batch = 0;
    std::vector<std::uint64_t> bytes_read(layout.pieces.size());
    std::vector<unsigned> num_retries(layout.pieces.size());
    std::deque<std::size_t> pending(layout.pieces.size());
    std::iota(pending.begin(), pending.end(), 0);

    auto prep_piece = [&](std::size_t i) {
        const Piece& p = layout.pieces[i];
        TensorBuffer& buf = checkpoint[p.buffer_idx];

        struct io_uring_sqe* sqe = ::io_uring_get_sqe(ring.get());
        if (!sqe) {
            throw std::runtime_error("::io_uring_get_sqe");
        }
        sqe->user_data = i;

        ::io_uring_prep_read(
            sqe, file.get(), buf.padded().data() + p.offset_in_buffer + bytes_read[i],
            static_cast<unsigned>(p.size - bytes_read[i]), p.file_offset + bytes_read[i]);
        ++batch;
    };

    auto requeue_no_progress = [&](std::size_t i) {
        if (++num_retries[i] > kMaxRetries) {
            throw std::runtime_error(
                std::format("failed to read piece {} after {} retries", i, kMaxRetries));
        }
        pending.push_back(i);
    };

    auto submit_and_process = [&] {
        int ret = ::io_uring_submit_and_wait(ring.get(), batch);
        if (ret < 0) {
            throw std::system_error(-ret, std::system_category(), "::io_uring_submit");
        }

        struct io_uring_cqe* cqe;
        unsigned head, seen = 0;
        batch = 0;
        io_uring_for_each_cqe(ring.get(), head, cqe) {
            const auto i = static_cast<std::size_t>(cqe->user_data);
            const int res = cqe->res;
            ++seen;

            if (res == -EINTR) {
                requeue_no_progress(i);
                continue;
            }
            if (res < 0) {
                throw std::system_error(-res, std::system_category(), "io_uring read");
            }

            const std::uint64_t before = bytes_read[i];
            bytes_read[i] += static_cast<std::uint64_t>(res);
            bytes_read[i] -= bytes_read[i] % layout.align;

            if (bytes_read[i] == layout.pieces[i].size) {
                // check crc
                const Piece& p = layout.pieces[i];
                std::uint32_t actual_crc =
                    crc32c(checkpoint[p.buffer_idx].padded().data() + p.offset_in_buffer, p.size);
                if (actual_crc != crcs[i]) {
                    throw std::runtime_error(std::format("crc mismatch in piece {}", i));
                }
                continue;
            }
            if (bytes_read[i] == before) {
                requeue_no_progress(i);
            } else {
                pending.push_back(i);
            }
        }
        ::io_uring_cq_advance(ring.get(), seen);
    };

    while (!pending.empty()) {
        while (!pending.empty() && batch < kQueueDepth) {
            prep_piece(pending.front());
            pending.pop_front();
        }
        submit_and_process();
    }
}

}  // namespace pitstop
