#include "common/env.h"

#include <liburing.h>
#include <sys/resource.h>
#include <sys/utsname.h>

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>

namespace cistern::env {

std::optional<KernelVersion> parse_kernel_release(std::string_view s) {
    KernelVersion v;
    int* parts[3] = {&v.major, &v.minor, &v.patch};
    int got = 0;
    while (got < 3) {
        std::size_t n = 0;
        while (n < s.size() && s[n] >= '0' && s[n] <= '9') ++n;
        if (n == 0) break;
        std::from_chars(s.data(), s.data() + n, *parts[got]);
        ++got;
        s.remove_prefix(n);
        if (got < 3 && !s.empty() && s.front() == '.') {
            s.remove_prefix(1);
        } else {
            break;
        }
    }
    if (got < 2) return std::nullopt;
    return v;
}

std::string kernel_release() {
    struct utsname u{};
    if (::uname(&u) != 0) return "unknown";
    return u.release;
}

void require_kernel(int major, int minor) {
    const std::string rel = kernel_release();
    const auto v = parse_kernel_release(rel);
    if (!v) throw std::runtime_error("cannot parse kernel release '" + rel + "'");
    if (*v < KernelVersion{major, minor, 0}) {
        throw std::runtime_error("kernel " + rel + " is older than required " +
                                 std::to_string(major) + "." + std::to_string(minor));
    }
}

void require_liburing(int major, int minor) {
    const int have_major = io_uring_major_version();
    const int have_minor = io_uring_minor_version();
    if (have_major < major || (have_major == major && have_minor < minor)) {
        throw std::runtime_error("liburing " + std::to_string(have_major) + "." +
                                 std::to_string(have_minor) + " is older than required " +
                                 std::to_string(major) + "." + std::to_string(minor));
    }
}

void check_free_ram(std::uint64_t required, std::uint64_t available) {
    if (required > available) {
        std::ostringstream os;
        os << "insufficient free RAM: need " << required << " bytes ("
           << static_cast<double>(required) / 1e9 << " GB) but MemAvailable is " << available
           << " bytes (" << static_cast<double>(available) / 1e9 << " GB)";
        throw InsufficientRam(os.str());
    }
}

std::optional<std::uint64_t> parse_mem_available(std::string_view text) {
    const std::string_view key = "MemAvailable:";
    const auto pos = text.find(key);
    if (pos == std::string_view::npos) return std::nullopt;
    auto rest = text.substr(pos + key.size());
    while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1);
    std::uint64_t kb = 0;
    const auto [p, ec] = std::from_chars(rest.data(), rest.data() + rest.size(), kb);
    if (ec != std::errc{}) return std::nullopt;
    (void)p;
    return kb * 1024;
}

std::uint64_t mem_available_bytes() {
    std::ifstream in("/proc/meminfo");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto v = parse_mem_available(text);
    if (!v) throw std::runtime_error("cannot read MemAvailable from /proc/meminfo");
    return *v;
}

IoUringProbe probe_io_uring() {
    io_uring ring;
    const int rc = io_uring_queue_init(8, &ring, 0);
    if (rc < 0) return {false, -rc, std::strerror(-rc)};
    io_uring_queue_exit(&ring);
    return {true, 0, "ok"};
}

std::uint64_t memlock_limit_bytes() {
    rlimit rl{};
    if (::getrlimit(RLIMIT_MEMLOCK, &rl) != 0 || rl.rlim_cur == RLIM_INFINITY) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return rl.rlim_cur;
}

void warn_if_memlock_low(std::uint64_t want) {
    const auto lim = memlock_limit_bytes();
    if (lim < want) {
        std::fprintf(
            stderr,
            "warning: RLIMIT_MEMLOCK is %llu bytes (ulimit -l); registered buffers of %llu "
            "bytes will fail until it is raised\n",
            static_cast<unsigned long long>(lim), static_cast<unsigned long long>(want));
    }
}

}  // namespace cistern::env
