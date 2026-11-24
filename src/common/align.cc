#include "common/align.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace cistern {

namespace {
bool pow2(std::uint32_t v) {
    return v != 0 && std::has_single_bit(v);
}
}  // namespace

DioAlign resolve_dio_align(bool have_dioalign, std::uint32_t mem, std::uint32_t offset) {
    if (!have_dioalign || !pow2(mem) || !pow2(offset)) return DioAlign{};
    return DioAlign{mem, offset, true};
}

DioAlign probe_dio_align(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::path target = path;
    std::string tmp_name;
    int tmp_fd = -1;
    if (std::filesystem::is_directory(path, ec)) {
        tmp_name = (path / ".cistern-probe-XXXXXX").string();
        tmp_fd = ::mkstemp(tmp_name.data());
        if (tmp_fd < 0) {
            std::fprintf(stderr,
                         "warning: cannot create probe file in %s (%s); using 4096 alignment\n",
                         path.c_str(), std::strerror(errno));
            return DioAlign{};
        }
        target = tmp_name;
    }
    struct statx sx{};
    const int rc = ::statx(AT_FDCWD, target.c_str(), 0, STATX_DIOALIGN, &sx);
    const bool have = rc == 0 && (sx.stx_mask & STATX_DIOALIGN) != 0;
    if (tmp_fd >= 0) {
        ::close(tmp_fd);
        ::unlink(tmp_name.c_str());
    }
    const DioAlign a = resolve_dio_align(have, have ? sx.stx_dio_mem_align : 0,
                                         have ? sx.stx_dio_offset_align : 0);
    if (!a.from_statx) {
        std::fprintf(stderr,
                     "warning: statx(STATX_DIOALIGN) unavailable or zero for %s; "
                     "falling back to %zu-byte alignment\n",
                     path.c_str(), kFallbackDioAlign);
    }
    return a;
}

std::size_t page_size() {
    static const std::size_t p = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    return p;
}

AlignedBuffer::AlignedBuffer(std::size_t bytes, std::size_t mem_align, bool prefault) {
    if (mem_align > page_size()) {
        throw std::invalid_argument("memory alignment " + std::to_string(mem_align) +
                                    " exceeds page size " + std::to_string(page_size()));
    }
    if (bytes == 0) return;
    map_bytes_ = round_up(bytes, page_size());
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (prefault) flags |= MAP_POPULATE;
    void* p = ::mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) {
        map_bytes_ = 0;
        throw std::system_error(errno, std::generic_category(), "mmap " + std::to_string(bytes));
    }
    data_ = static_cast<std::byte*>(p);
    size_ = bytes;
    if (prefault) {
        for (std::size_t off = 0; off < map_bytes_; off += page_size()) {
            *reinterpret_cast<volatile std::byte*>(data_ + off) = std::byte{0};
        }
    }
}

AlignedBuffer::~AlignedBuffer() {
    if (data_ != nullptr) ::munmap(data_, map_bytes_);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& o) noexcept
    : data_(std::exchange(o.data_, nullptr)),
      size_(std::exchange(o.size_, 0)),
      map_bytes_(std::exchange(o.map_bytes_, 0)) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& o) noexcept {
    if (this != &o) {
        if (data_ != nullptr) ::munmap(data_, map_bytes_);
        data_ = std::exchange(o.data_, nullptr);
        size_ = std::exchange(o.size_, 0);
        map_bytes_ = std::exchange(o.map_bytes_, 0);
    }
    return *this;
}

}  // namespace cistern
