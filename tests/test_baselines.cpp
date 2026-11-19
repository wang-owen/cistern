#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "cistern/baselines.hpp"
#include "test_util.hpp"

using namespace cistern;

namespace {
StateOptions opts(const std::filesystem::path& dir) {
    StateOptions o;
    o.seed = 99;
    o.threads = 2;
    o.align_dir = dir;
    return o;
}
}  // namespace

TEST_CASE("buffered baseline round trip", "[unit][baselines]") {
    testutil::TempDir d;
    const auto s = State::generate(testutil::tiny_table(), opts(d.path()));
    const auto path = d.path() / "ckpt.bin";
    const auto w = write_buffered(s, path, 1 << 16);
    CHECK(w.bytes == s.checkpoint_bytes());
    CHECK(std::filesystem::file_size(path) == s.checkpoint_bytes());
    State back = State::zeros_like(s);
    CHECK(first_mismatch(s, back).has_value());
    (void)read_buffered(path, back, 1 << 16);
    CHECK(!first_mismatch(s, back).has_value());
}

TEST_CASE("O_DIRECT baseline round trip", "[unit][baselines]") {
    testutil::TempDir d;
    const auto s = State::generate(testutil::tiny_table(), opts(d.path()));
    const auto path = d.path() / "ckpt.bin";
    for (bool falloc : {false, true}) {
        try {
            const auto w = write_odirect(s, path, s.alignment().unit() * 4, falloc);
            CHECK(w.bytes == s.checkpoint_bytes());
            State back = State::zeros_like(s);
            (void)read_odirect(path, back, s.alignment().unit() * 16);
            const auto bad = first_mismatch(s, back);
            INFO("first mismatch: " << bad.value_or("none"));
            CHECK(!bad.has_value());
        } catch (const DirectIoUnsupported& e) {
            SKIP("O_DIRECT unsupported here: " << e.what());
        } catch (const std::system_error& e) {
            if (falloc && e.code().value() == EOPNOTSUPP) SKIP("fallocate unsupported here");
            throw;
        }
    }
}

TEST_CASE("buffered and O_DIRECT writes produce identical files", "[unit][baselines]") {
    testutil::TempDir d;
    const auto s = State::generate(testutil::tiny_table(), opts(d.path()));
    try {
        (void)write_odirect(s, d.path() / "a.bin", s.alignment().unit(), false);
    } catch (const DirectIoUnsupported&) {
        SKIP("O_DIRECT unsupported here");
    }
    (void)write_buffered(s, d.path() / "b.bin", 1 << 20);
    State a = State::zeros_like(s), b = State::zeros_like(s);
    (void)read_buffered(d.path() / "a.bin", a, 1 << 20);
    (void)read_buffered(d.path() / "b.bin", b, 1 << 20);
    CHECK(!first_mismatch(a, b).has_value());
}

TEST_CASE("baselines validate chunk size and file size", "[unit][baselines]") {
    testutil::TempDir d;
    const auto s = State::generate(testutil::tiny_table(), opts(d.path()));
    CHECK_THROWS_AS(write_buffered(s, d.path() / "x", 0), std::invalid_argument);
    CHECK_THROWS_AS(write_odirect(s, d.path() / "x", 1000, false), std::invalid_argument);
    (void)write_buffered(s, d.path() / "short.bin", 4096);
    std::filesystem::resize_file(d.path() / "short.bin", 4096);
    State back = State::zeros_like(s);
    CHECK_THROWS(read_buffered(d.path() / "short.bin", back, 4096));
}
