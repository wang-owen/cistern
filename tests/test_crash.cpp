// Seam harness: SIGKILL a child mid-checkpoint; the previous checkpoint must still load.
// CISTERN_CRASH_ITERS sets the iteration count (default 200); CISTERN_CRASH_SEED fixes the RNG.
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <thread>

#include "cistern/async.hpp"
#include "cistern/reader.hpp"
#include "cistern/state.hpp"
#include "cistern/writer.hpp"
#include "test_util.hpp"

using namespace cistern;
using Clock = std::chrono::steady_clock;

namespace {

int iterations() {
    const char* e = std::getenv("CISTERN_CRASH_ITERS");
    return e != nullptr && *e != '\0' ? std::atoi(e) : 200;
}

std::uint64_t seed() {
    const char* e = std::getenv("CISTERN_CRASH_SEED");
    return e != nullptr && *e != '\0' ? std::strtoull(e, nullptr, 10) : std::random_device{}();
}

enum class Loaded { A, B, Neither };

Loaded load_and_classify(const std::filesystem::path& dst, const State& a, const State& b) {
    State got = State::zeros_like(a);
    restore_checkpoint(dst, got, ReaderConfig{});  // throws if the checkpoint is unreadable
    if (!first_mismatch(got, a)) return Loaded::A;
    if (!first_mismatch(got, b)) return Loaded::B;
    return Loaded::Neither;
}

// `child_write(B, dst)` runs in the forked child and must return only after the checkpoint is
// durable.
void crash_loop(
    const char* label,
    const std::function<void(const State&, const std::filesystem::path&)>& child_write) {
    testutil::TempDir dir;
    const auto dst = dir.path() / "ckpt.bin";
    StateOptions o;
    o.align_dir = dir.path();
    o.seed = 1;
    State a = State::generate(testutil::tiny_table(), o);
    o.seed = 2;
    State b = State::generate(testutil::tiny_table(), o);

    // Measure one uninterrupted write to scale the kill delay.
    const auto t0 = Clock::now();
    child_write(b, dst);  // in-process, uninterrupted (the async variant joins in its dtor)
    const double write_s = std::chrono::duration<double>(Clock::now() - t0).count();
    write_checkpoint(a, dst, WriterConfig{});  // A is the committed checkpoint

    const auto s = seed();
    std::printf("[%s] crash seed=%llu iters=%d write_time=%.4fs\n", label,
                static_cast<unsigned long long>(s), iterations(), write_s);
    std::mt19937_64 rng(s);
    std::uniform_real_distribution<double> delay(0.0, 1.2 * write_s);

    int killed_before = 0;  // previous checkpoint still current: kill landed before commit
    int committed = 0;      // new checkpoint was already committed when the kill landed
    Loaded prev = Loaded::A;
    const State* cur = &a;
    const State* nxt = &b;
    for (int i = 0; i < iterations(); ++i) {
        const pid_t pid = ::fork();
        REQUIRE(pid >= 0);
        if (pid == 0) {
            try {
                child_write(*nxt, dst);
            } catch (...) {
                ::_exit(3);
            }
            ::_exit(0);
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(delay(rng)));
        ::kill(pid, SIGKILL);
        int status = 0;
        REQUIRE(::waitpid(pid, &status, 0) == pid);

        const Loaded got = load_and_classify(dst, *cur, *nxt);
        INFO("iteration " << i << ": checkpoint is neither the previous nor the new state");
        REQUIRE(got != Loaded::Neither);
        if (got == Loaded::A) {  // "A" here means "equal to cur"
            ++killed_before;
        } else {
            ++committed;
            std::swap(cur, nxt);  // the new checkpoint becomes the previous one
        }
        (void)prev;
    }
    const int n = iterations();
    std::printf("[%s] kills before commit: %d, after commit: %d (of %d)\n", label, killed_before,
                committed, n);
    if (n > 0 && killed_before * 5 < n) {
        std::printf(
            "[%s] WARNING: under 20%% of kills landed mid-write; this run is mostly vacuous\n",
            label);
    }
}

}  // namespace

TEST_CASE("crash: sync writer leaves the previous checkpoint loadable", "[seam][crash]") {
    crash_loop("sync", [](const State& s, const std::filesystem::path& dst) {
        WriterConfig w;
        w.queue_depth = 8;
        w.chunk_bytes = 256 << 10;
        write_checkpoint(s, dst, w);
    });
}

TEST_CASE("crash: async flush leaves the previous checkpoint loadable", "[seam][crash][async]") {
    crash_loop("async", [](const State& s, const std::filesystem::path& dst) {
        WriterConfig w;
        w.queue_depth = 8;
        w.chunk_bytes = 256 << 10;
        AsyncCheckpointer ck(s, w, -1);
        ck.request(dst);
        ck.wait_durable();
    });
}
