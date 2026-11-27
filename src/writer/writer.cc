#include "writer/writer.h"

#include <fcntl.h>
#include <liburing.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <format>
#include <numeric>
#include <span>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "common/align.h"
#include "common/crc32c.h"
#include "common/direct_file.h"
#include "common/ring.h"
#include "gen/state.h"
#include "layout/layout.h"

namespace cistern {

namespace {

constexpr unsigned kQueueDepth = 16;
constexpr unsigned kMaxRetries = 10;

void try_fdatasync(int fd) {
    int ret;
    while ((ret = ::fdatasync(fd)) == -1 && errno == EINTR);
    if (ret == -1) {
        throw std::system_error(errno, std::generic_category(), "::fdatasync");
    }
}

// Makes renames inside `dir` durable.
void fsync_dir(const std::filesystem::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), "::open dir");
    }
    int ret;
    while ((ret = ::fsync(fd)) == -1 && errno == EINTR);
    const int err = errno;
    ::close(fd);
    if (ret == -1) {
        throw std::system_error(err, std::generic_category(), "::fsync dir");
    }
}

}  // namespace

void write_checkpoint(const State& state, const Layout& layout, const std::filesystem::path& dir) {
    assert(layout.align == state.alignment().unit());
    assert(layout.buffers.size() == state.checkpointed().size());

    const std::filesystem::path path = dir / "ckpt.tmp";
    DirectFile file = DirectFile::create(path);
    Ring ring(kQueueDepth);

    std::span<const TensorBuffer> checkpoint = state.checkpointed();
    const std::size_t n = layout.pieces.size();
    unsigned in_flight = 0;  // submitted (or prepared) writes without a completion yet
    std::deque<std::size_t> pending(n);
    std::iota(pending.begin(), pending.end(), 0);
    std::vector<unsigned> num_retries(n, 0);
    std::vector<std::uint64_t> written(n, 0);
    std::vector<std::uint32_t> checksums(n, 0);

    auto prep_piece = [&](std::size_t i) {
        const Piece& p = layout.pieces[i];
        const TensorBuffer& buf = checkpoint[p.buffer_idx];
        const std::byte* src = buf.padded().data() + p.offset_in_buffer;

        struct io_uring_sqe* sqe = ::io_uring_get_sqe(ring.get());
        if (!sqe) {
            throw std::runtime_error("::io_uring_get_sqe");
        }
        sqe->user_data = i;

        if (checksums[i] == 0) {
            checksums[i] = crc32c(src, p.size);
        }

        ::io_uring_prep_write(sqe, file.get(), src + written[i],
                              static_cast<unsigned>(p.size - written[i]),
                              p.file_offset + written[i]);
        ++in_flight;
    };

    auto requeue_no_progress = [&](std::size_t i) {
        if (++num_retries[i] > kMaxRetries) {
            throw std::runtime_error(
                std::format("failed to write piece {} after {} retries", i, kMaxRetries));
        }
        pending.push_back(i);
    };

    // Waits for one completion, not the whole batch, so the caller refills the queue as soon as
    // a slot frees up and the drive never drains while the next pieces are checksummed.
    auto submit_and_process = [&] {
        int ret = ::io_uring_submit_and_wait(ring.get(), 1);
        if (ret < 0) {
            throw std::system_error(-ret, std::system_category(), "::io_uring_submit");
        }

        struct io_uring_cqe* cqe;
        unsigned head, seen = 0;
        io_uring_for_each_cqe(ring.get(), head, cqe) {
            const auto i = static_cast<std::size_t>(cqe->user_data);
            const int res = cqe->res;
            ++seen;
            --in_flight;

            if (res == -EINTR) {
                requeue_no_progress(i);
                continue;
            }
            if (res < 0) {
                throw std::system_error(-res, std::system_category(), "io_uring write");
            }

            const std::uint64_t before = written[i];
            written[i] += static_cast<std::uint64_t>(res);
            written[i] -= written[i] % layout.align;

            if (written[i] == layout.pieces[i].size) {
                continue;
            }
            if (written[i] == before) {
                requeue_no_progress(i);
            } else {
                pending.push_back(i);
            }
        }
        ::io_uring_cq_advance(ring.get(), seen);
    };

    while (!pending.empty() || in_flight > 0) {
        while (!pending.empty() && in_flight < kQueueDepth) {
            prep_piece(pending.front());
            pending.pop_front();
        }
        submit_and_process();
    }

    try_fdatasync(file.get());

    const std::size_t checksum_bytes = checksums.size() * sizeof(std::uint32_t);
    AlignedBuffer header_buf(layout.header_size, layout.align, true);
    const Header header = make_header(layout);
    std::memcpy(header_buf.data(), &header, sizeof(header));
    std::memcpy(header_buf.data() + sizeof(header), checksums.data(), checksum_bytes);
    const std::uint32_t header_checksum =
        crc32c(header_buf.data(), sizeof(header) + checksum_bytes);
    std::memcpy(header_buf.data() + sizeof(header) + checksum_bytes, &header_checksum,
                sizeof(header_checksum));

    std::uint64_t header_written = 0;
    unsigned header_retries = 0;
    while (header_written < layout.header_size) {
        const ssize_t res =
            ::pwrite(file.get(), header_buf.data() + header_written,
                     layout.header_size - header_written, static_cast<off_t>(header_written));
        if (res == -1) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "::pwrite header");
        }
        // O_DIRECT: resume only from an aligned offset
        const std::uint64_t next =
            (header_written + static_cast<std::uint64_t>(res)) / layout.align * layout.align;
        if (next == header_written && ++header_retries > kMaxRetries) {
            throw std::runtime_error(
                std::format("failed to write header after {} retries", kMaxRetries));
        }
        header_written = next;
    }

    // header durable before the rename publishes it
    try_fdatasync(file.get());
    std::filesystem::rename(path, dir / "ckpt");
    fsync_dir(dir);
}

}  // namespace cistern
