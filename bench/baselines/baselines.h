#pragma once
// Data-only baselines. They move the checkpointed buffers, packed back to back at their padded
// lengths (baseline-only packing, not the checkpoint file format), and fsync.
// There is no header, tmp file, rename or directory fsync: the commit protocol is the writer's.
// Every number needs a baseline; these are the baselines (write and read).
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

#include "gen/state.h"

namespace cistern {

struct IoResult {
    std::uint64_t fallocate_ns = 0;
    std::uint64_t io_ns = 0;     // write()/pwrite()/read()/pread() loop
    std::uint64_t fsync_ns = 0;  // writes only
    std::uint64_t bytes = 0;     // padded bytes moved
    std::uint64_t total_ns() const noexcept { return fallocate_ns + io_ns + fsync_ns; }
    double gbps() const noexcept {  // decimal GB/s over the whole operation (incl. fsync)
        return total_ns() == 0 ? 0.0 : static_cast<double>(bytes) / static_cast<double>(total_ns());
    }
};

// O_DIRECT is not supported by the target filesystem (open returned EINVAL).
class DirectIoUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Buffered write() in `chunk`-byte pieces, then fsync.
IoResult write_buffered(const State& state, const std::filesystem::path& path, std::size_t chunk);
// pwrite() + O_DIRECT in `chunk`-byte pieces (multiple of the alignment unit), optional
// fallocate of the whole file first, then fsync.
IoResult write_odirect(const State& state, const std::filesystem::path& path, std::size_t chunk,
                       bool fallocate);
// Buffered read(). Callers drop caches first for a cold read.
IoResult read_buffered(const std::filesystem::path& path, State& into, std::size_t chunk);
IoResult read_odirect(const std::filesystem::path& path, State& into, std::size_t chunk);

}  // namespace cistern
