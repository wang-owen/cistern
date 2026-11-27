#pragma once
// Host/environment checks: free memory.
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace cistern::env {

class InsufficientRam : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// Throws InsufficientRam (both numbers in the message) when required > available.
void check_free_ram(std::uint64_t required, std::uint64_t available);
std::optional<std::uint64_t> parse_mem_available(std::string_view meminfo);  // bytes
std::uint64_t mem_available_bytes();                                         // from /proc/meminfo

}  // namespace cistern::env
