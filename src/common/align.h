#pragma once
// Direct-I/O alignment discovery and page-backed aligned buffers.
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace cistern {

inline constexpr std::size_t kFallbackDioAlign = 4096;

constexpr std::size_t round_up(std::size_t v, std::size_t a) noexcept {
    return (v + a - 1) / a * a;  // a > 0
}

struct DioAlign {
    std::size_t mem = kFallbackDioAlign;     // required memory alignment
    std::size_t offset = kFallbackDioAlign;  // required file offset / length alignment
    bool from_statx = false;                 // false: fallback was used
    // Padding unit: satisfies both constraints (both are powers of two).
    std::size_t unit() const noexcept { return mem > offset ? mem : offset; }
};

// Pure: turns the statx dio fields into a DioAlign, falling back to 4096 when unsupported,
// zero, or not a power of two. `fell_back` (optional) reports whether the fallback was used.
DioAlign resolve_dio_align(bool have_dioalign, std::uint32_t mem, std::uint32_t offset,
                           bool* fell_back = nullptr);

// statx(STATX_DIOALIGN) on a file, or (for a directory) on a probe file created and removed
// inside it. Logs a warning to stderr when falling back to 4096.
DioAlign probe_dio_align(const std::filesystem::path& path);

std::size_t page_size();

// Anonymous-mmap RAII buffer (page aligned). `prefault` touches every page up front.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(std::size_t bytes, std::size_t mem_align, bool prefault);
    ~AlignedBuffer();
    AlignedBuffer(AlignedBuffer&& o) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    std::byte* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t map_bytes_ = 0;
};

}  // namespace cistern
