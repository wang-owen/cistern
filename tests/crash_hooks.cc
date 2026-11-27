#include "crash_hooks.h"

#include <dlfcn.h>
#include <liburing.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <optional>

namespace crashtest {

namespace {

// Unsynchronized: write_checkpoint is single-threaded.
HitCounts counts{};
std::optional<KillPoint> armed;

template <class F>
F next_symbol(const char* symbol) {
    void* p = ::dlsym(RTLD_NEXT, symbol);
    if (p == nullptr) std::abort();
    return reinterpret_cast<F>(p);
}

// SIGKILL: no destructors, atexit handlers or stream flushes run.
void die_if_armed(Fn fn, unsigned hit, When when) {
    if (armed && armed->fn == fn && armed->nth == hit && armed->when == when) {
        ::raise(SIGKILL);
    }
}

template <class Call>
auto hooked(Fn fn, Call&& call) {
    const unsigned hit = ++counts[static_cast<std::size_t>(fn)];
    die_if_armed(fn, hit, When::before);
    const auto ret = call();
    die_if_armed(fn, hit, When::after);  // raise() leaves the call's errno irrelevant
    return ret;
}

}  // namespace

void reset() {
    counts = {};
    armed.reset();
}

void arm(const KillPoint& point) {
    reset();
    armed = point;
}

HitCounts hit_counts() {
    return counts;
}

const char* name(Fn fn) {
    switch (fn) {
        case Fn::submit_and_wait:
            return "io_uring_submit_and_wait";
        case Fn::fdatasync:
            return "fdatasync";
        case Fn::pwrite:
            return "pwrite";
        case Fn::rename:
            return "rename";
        case Fn::fsync:
            return "fsync";
    }
    return "?";
}

}  // namespace crashtest

// --- interposed symbols; signatures must match the system headers ---

using crashtest::Fn;

extern "C" int fdatasync(int fd) {
    static const auto real = crashtest::next_symbol<int (*)(int)>("fdatasync");
    return crashtest::hooked(Fn::fdatasync, [&] { return real(fd); });
}

extern "C" int fsync(int fd) {
    static const auto real = crashtest::next_symbol<int (*)(int)>("fsync");
    return crashtest::hooked(Fn::fsync, [&] { return real(fd); });
}

// With _FILE_OFFSET_BITS=64 the header may redirect this to pwrite64, for the writer too.
extern "C" ssize_t pwrite(int fd, const void* buf, size_t n, off_t offset) {
    static const auto real =
        crashtest::next_symbol<ssize_t (*)(int, const void*, size_t, off_t)>("pwrite64");
    return crashtest::hooked(Fn::pwrite, [&] { return real(fd, buf, n, offset); });
}

extern "C" int rename(const char* from, const char* to) noexcept {
    static const auto real = crashtest::next_symbol<int (*)(const char*, const char*)>("rename");
    return crashtest::hooked(Fn::rename, [&] { return real(from, to); });
}

extern "C" int io_uring_submit_and_wait(struct io_uring* ring, unsigned wait_nr) {
    static const auto real =
        crashtest::next_symbol<int (*)(struct io_uring*, unsigned)>("io_uring_submit_and_wait");
    return crashtest::hooked(Fn::submit_and_wait, [&] { return real(ring, wait_nr); });
}
