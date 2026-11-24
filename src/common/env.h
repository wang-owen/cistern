#pragma once
// Host/environment checks: kernel, liburing, memory, io_uring, memlock.
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cistern::env {

struct KernelVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
    bool operator==(const KernelVersion&) const = default;
    auto operator<=>(const KernelVersion&) const = default;
};

// "6.1.0-18-arm64" -> {6,1,0}; "7.2.12-orbstack-1" -> {7,2,12}; nullopt if unparseable.
std::optional<KernelVersion> parse_kernel_release(std::string_view release);
void require_kernel(int major, int minor);    // throws std::runtime_error if older
void require_liburing(int major, int minor);  // runtime library version; throws if older

class InsufficientRam : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// Throws InsufficientRam (both numbers in the message) when required > available.
void check_free_ram(std::uint64_t required, std::uint64_t available);
std::optional<std::uint64_t> parse_mem_available(std::string_view meminfo);  // bytes
std::uint64_t mem_available_bytes();                                         // from /proc/meminfo

struct IoUringProbe {
    bool ok = false;
    int err = 0;  // positive errno
    std::string message;
};
IoUringProbe probe_io_uring();
// RLIMIT_MEMLOCK soft limit in bytes (UINT64_MAX when unlimited).
std::uint64_t memlock_limit_bytes();
// Warns on stderr when the limit is below `want` bytes.
void warn_if_memlock_low(std::uint64_t want);

}  // namespace cistern::env
