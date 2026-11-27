#pragma once
// Kill points for the crash test: hooks.cc interposes the calls write_checkpoint makes, so
// the process can SIGKILL itself right before or right after any one of them.
#include <array>
#include <cstddef>

namespace crashtest {

enum class Fn : std::size_t { submit_and_wait, fdatasync, pwrite, rename, fsync };
inline constexpr std::size_t kFnCount = 5;

enum class When { before, after };

struct KillPoint {
    Fn fn;
    unsigned nth;  // 1-based, counted from the last reset() or arm()
    When when;
};

using HitCounts = std::array<unsigned, kFnCount>;

// Zeroes the counters and disarms.
void reset();
// Zeroes the counters and arms `point`. Only call this in a forked child.
void arm(const KillPoint& point);
HitCounts hit_counts();

const char* name(Fn fn);

}  // namespace crashtest
