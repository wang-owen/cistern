#include "env.h"

#include <charconv>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace pitstop::env {

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
    const auto res = std::from_chars(rest.data(), rest.data() + rest.size(), kb);
    if (res.ec != std::errc{}) return std::nullopt;
    return kb * 1024;
}

std::uint64_t mem_available_bytes() {
    std::ifstream in("/proc/meminfo");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto v = parse_mem_available(text);
    if (!v) throw std::runtime_error("cannot read MemAvailable from /proc/meminfo");
    return *v;
}

}  // namespace pitstop::env
