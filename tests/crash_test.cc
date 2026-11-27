// Process-crash test for write_checkpoint/restore_checkpoint: a forked child writes
// checkpoints and is SIGKILLed; the parent restores what is left and then writes again.
//
// Scope: SIGKILL keeps everything the kernel already accepted, so this checks the tmp+rename
// protocol and recovery. It cannot tell whether the fdatasync/fsync calls are sufficient
// for power loss.
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "crash_hooks.h"
#include "layout.h"
#include "restore.h"
#include "state.h"
#include "test_util.h"
#include "writer.h"

namespace cistern {

namespace {

using crashtest::Fn;
using crashtest::KillPoint;
using crashtest::When;

constexpr std::uint64_t kAlign = 4096;
constexpr std::uint64_t kSeeds = 6;
// random kills: the child writes seeds 1..kRandomWrites, leaving room for the recovery seed
constexpr std::uint64_t kRandomWrites = kSeeds - 2;
constexpr std::uint64_t kNoCheckpoint = 0;  // seeds start at 1

// --- oracle ---

// Every result restore may legally produce after a crash. The child was writing seed
// `committed + 1`; `committed` is the last seed whose write_checkpoint returned.
// kNoCheckpoint in the list means "there is no ckpt file". `kill` is where the child died,
// or nullopt when it was killed at an unknown time. A corrupt or unrecognized checkpoint is
// always a failure, so it has no entry here.
std::vector<std::uint64_t> allowed_outcomes(std::uint64_t committed,
                                            std::optional<KillPoint> kill) {
    const std::uint64_t inflight = committed + 1;
    if (!kill) return {committed, inflight};
    // rename is what publishes a checkpoint; the directory fsync is the only hooked call
    // after it. A process kill cannot undo a completed rename.
    const bool published =
        kill->fn == Fn::fsync || (kill->fn == Fn::rename && kill->when == When::after);
    return {published ? inflight : committed};
}

// --- states and restore ---

// Generated before any fork: the child inherits them, the parent compares against them.
// states()[s - 1] holds seed s.
const std::vector<State>& states() {
    static const std::vector<State> all = [] {
        std::vector<State> v;
        for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
            StateOptions o;
            o.seed = seed;
            o.threads = 2;
            o.align_override = DioAlign{kAlign, kAlign, false};
            v.push_back(State::generate(testutil::tiny_table(), o));
        }
        return v;
    }();
    return all;
}

// align-sized pieces: the most io_uring batches, so the most places to die mid-write
void write_seed(std::uint64_t seed, const std::filesystem::path& dir) {
    const State& state = states().at(seed - 1);
    write_checkpoint(state, make_layout(state.table(), kAlign, kAlign), dir);
}

// The seed restore yields from `dir`, or kNoCheckpoint. Fails the test if ckpt exists but
// does not restore cleanly to exactly one of the known states.
std::uint64_t recovered_seed(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir / "ckpt")) return kNoCheckpoint;
    State out = State::zeros_like(states().front());
    REQUIRE_NOTHROW(restore_checkpoint(dir, out));
    for (std::uint64_t seed = 1; seed <= kSeeds; ++seed) {
        if (!first_mismatch(states()[seed - 1], out)) return seed;
    }
    FAIL("restored state matches no written seed");
    return kNoCheckpoint;
}

// --- child processes ---

// Child side. Messages are short lines, so each write() to the pipe is atomic.
void send(int fd, const std::string& line) {
    const std::string msg = line + "\n";
    if (::write(fd, msg.data(), msg.size()) != static_cast<ssize_t>(msg.size())) ::_exit(3);
}

struct Child {
    pid_t pid;
    int fd;  // read end of the child's pipe
};

// Runs body(write_fd) in a forked child. The child must not use Catch2 macros.
template <class Body>
Child spawn(Body body) {
    states();  // build in the parent
    int fds[2];
    REQUIRE(::pipe(fds) == 0);
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        ::close(fds[0]);
        try {
            body(fds[1]);
        } catch (const std::exception& e) {
            send(fds[1], std::string("error ") + e.what());
            ::_exit(2);
        }
        ::_exit(0);
    }
    ::close(fds[1]);
    return {pid, fds[0]};
}

struct ChildResult {
    int status = 0;  // from waitpid
    std::string output;

    bool killed() const { return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL; }
    bool succeeded() const { return WIFEXITED(status) && WEXITSTATUS(status) == 0; }
};

// Drains the pipe to EOF, then reaps the child.
ChildResult finish(const Child& child) {
    ChildResult r;
    char buf[256];
    for (;;) {
        const ssize_t n = ::read(child.fd, buf, sizeof(buf));
        if (n > 0) {
            r.output.append(buf, static_cast<std::size_t>(n));
        } else if (n == 0 || errno != EINTR) {
            break;
        }
    }
    ::close(child.fd);
    while (::waitpid(child.pid, &r.status, 0) == -1) {
        REQUIRE(errno == EINTR);
    }
    return r;
}

// A fresh, unkilled process must be able to write over whatever the crash left behind.
void check_recovery(std::uint64_t seed, const std::filesystem::path& dir) {
    const ChildResult r = finish(spawn([&](int) { write_seed(seed, dir); }));
    INFO("recovery child: " << r.output);
    REQUIRE(r.succeeded());
    CHECK(recovered_seed(dir) == seed);
}

void check_after_crash(std::uint64_t committed, std::optional<KillPoint> kill,
                       const std::filesystem::path& dir) {
    const std::uint64_t got = recovered_seed(dir);
    const auto allowed = allowed_outcomes(committed, kill);
    CAPTURE(committed, got, allowed);
    CHECK(std::find(allowed.begin(), allowed.end(), got) != allowed.end());
    check_recovery(committed + 2, dir);
}

// Hit counts of one uninterrupted write_checkpoint.
crashtest::HitCounts count_calls() {
    testutil::TempDir dir;
    const ChildResult r = finish(spawn([&](int fd) {
        crashtest::reset();
        write_seed(1, dir.path());
        std::string line = "counts";
        for (unsigned c : crashtest::hit_counts()) line += std::format(" {}", c);
        send(fd, line);
    }));
    INFO("child: " << r.output);
    REQUIRE(r.succeeded());
    std::istringstream in(r.output);
    std::string tag;
    in >> tag;
    crashtest::HitCounts counts{};
    for (unsigned& c : counts) in >> c;
    REQUIRE(in);
    return counts;
}

unsigned count_of(const crashtest::HitCounts& counts, Fn fn) {
    return counts[static_cast<std::size_t>(fn)];
}

unsigned random_iterations() {
    const char* env = std::getenv("CISTERN_CRASH_ITERS");
    return env != nullptr ? static_cast<unsigned>(std::stoul(env)) : 100u;
}

}  // namespace

TEST_CASE("crash hooks see every call write_checkpoint makes", "[crash][hooks]") {
    const auto counts = count_calls();
    CHECK(count_of(counts, Fn::fdatasync) == 2);
    CHECK(count_of(counts, Fn::pwrite) >= 1);
    CHECK(count_of(counts, Fn::rename) == 1);
    CHECK(count_of(counts, Fn::fsync) == 1);
    CHECK(count_of(counts, Fn::submit_and_wait) >= 3);  // distinct first, middle, last
}

TEST_CASE("checkpoint survives a kill at each step", "[crash][deterministic]") {
    // The writer reaps completions as they arrive, so how many submit_and_wait calls a write
    // makes varies from run to run. Each call reaps at most a full queue, so
    // ceil(pieces / queue depth) calls happen in every run: kill points up to it always fire.
    constexpr unsigned kWriterQueueDepth = 16;  // kQueueDepth in src/writer.cc
    const std::size_t pieces = make_layout(states().front().table(), kAlign, kAlign).pieces.size();
    const auto submits =
        static_cast<unsigned>((pieces + kWriterQueueDepth - 1) / kWriterQueueDepth);
    REQUIRE(submits >= 3);  // distinct first, middle, last
    REQUIRE(count_of(count_calls(), Fn::submit_and_wait) >= submits);
    const std::vector<std::pair<Fn, unsigned>> calls = {
        {Fn::submit_and_wait, 1},
        {Fn::submit_and_wait, (submits + 1) / 2},
        {Fn::submit_and_wait, submits},
        {Fn::fdatasync, 1},
        {Fn::pwrite, 1},
        {Fn::fdatasync, 2},
        {Fn::rename, 1},
        {Fn::fsync, 1},
    };

    for (const auto& [fn, nth] : calls) {
        for (const When when : {When::before, When::after}) {
            for (const std::uint64_t committed : {std::uint64_t{0}, std::uint64_t{1}}) {
                const KillPoint point{fn, nth, when};
                INFO("kill " << (when == When::before ? "before " : "after ") << crashtest::name(fn)
                             << " #" << nth << ", committed " << committed);
                testutil::TempDir dir;
                const ChildResult r = finish(spawn([&](int) {
                    if (committed != 0) write_seed(committed, dir.path());
                    crashtest::arm(point);
                    write_seed(committed + 1, dir.path());
                }));
                INFO("child: " << r.output);
                REQUIRE(r.killed());  // otherwise the kill point never fired
                check_after_crash(committed, point, dir.path());
            }
        }
    }
}

TEST_CASE("checkpoint survives kills at random times", "[crash][random]") {
    using Clock = std::chrono::steady_clock;
    states();

    std::chrono::microseconds one_write{};
    {
        testutil::TempDir dir;
        const auto start = Clock::now();
        REQUIRE(finish(spawn([&](int) { write_seed(1, dir.path()); })).succeeded());
        one_write = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
    }

    const std::uint64_t rng_seed = std::random_device{}();
    std::mt19937_64 rng(rng_seed);
    std::uniform_int_distribution<std::int64_t> delay_us(0, 3 * one_write.count());

    for (unsigned iter = 0, n = random_iterations(); iter < n; ++iter) {
        const std::chrono::microseconds delay{delay_us(rng)};
        INFO("rng seed " << rng_seed << ", iteration " << iter << ", delay " << delay.count()
                         << " us");
        testutil::TempDir dir;
        const Child child = spawn([&](int fd) {
            for (std::uint64_t seed = 1; seed <= kRandomWrites; ++seed) {
                write_seed(seed, dir.path());
                send(fd, std::format("committed {}", seed));
            }
        });
        std::this_thread::sleep_for(delay);
        ::kill(child.pid, SIGKILL);  // harmless if it already exited: not yet reaped
        const ChildResult r = finish(child);
        INFO("child: " << r.output);
        REQUIRE((r.killed() || r.succeeded()));

        std::uint64_t committed = 0;
        std::istringstream in(r.output);
        std::string tag;
        std::uint64_t seed;
        while (in >> tag >> seed) committed = seed;

        if (r.succeeded()) {
            REQUIRE(committed == kRandomWrites);
            CHECK(recovered_seed(dir.path()) == committed);
        } else {
            // A kill between write_checkpoint returning and send() leaves `committed` one
            // behind, with `committed + 1` on disk: the oracle's in-flight case.
            check_after_crash(committed, std::nullopt, dir.path());
        }
    }
}

}  // namespace cistern
