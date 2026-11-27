#pragma once

#include <nmmintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pitstop {

inline std::uint32_t crc32c(const std::byte* data, std::size_t length) {
    std::uint64_t crc = 0xFFFFFFFF;
    std::size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        std::uint64_t v;
        std::memcpy(&v, data + i, 8);  // safe unaligned load; compiles to one mov
        crc = _mm_crc32_u64(crc, v);
    }
    auto c = static_cast<std::uint32_t>(crc);
    for (; i < length; ++i) c = _mm_crc32_u8(c, std::to_integer<std::uint8_t>(data[i]));
    return ~c;
}

}  // namespace pitstop
