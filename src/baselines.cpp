#include "cistern/baselines.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <string>
#include <system_error>

namespace cistern {

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t ns_since(Clock::time_point t0) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
}

[[noreturn]] void fail(const std::string& what, const std::filesystem::path& p, int err = errno) {
    throw std::system_error(err, std::generic_category(), what + " " + p.string());
}

class Fd {
public:
    Fd(const std::filesystem::path& p, int flags, bool direct) : path_(p) {
        fd_ = ::open(p.c_str(), flags | (direct ? O_DIRECT : 0), 0644);
        if (fd_ < 0) {
            if (direct && errno == EINVAL) {
                throw DirectIoUnsupported("O_DIRECT not supported on " + p.string());
            }
            fail("open", p);
        }
    }
    ~Fd() {
        if (fd_ >= 0) ::close(fd_);
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const noexcept { return fd_; }
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    int fd_ = -1;
    std::filesystem::path path_;
};

void check_chunk(const State& s, std::size_t chunk, bool direct) {
    if (chunk == 0) throw std::invalid_argument("chunk must be > 0");
    if (direct && chunk % s.alignment().unit() != 0) {
        throw std::invalid_argument("O_DIRECT chunk " + std::to_string(chunk) +
                                    " is not a multiple of the alignment unit " +
                                    std::to_string(s.alignment().unit()));
    }
}

// Writes every checkpointed buffer (padded length) at its offset.
void write_all(const State& s, const Fd& fd, std::size_t chunk, bool positional) {
    for (const auto& e : s.checkpointed()) {
        const std::byte* p = e.buf->padded.data();
        const std::size_t total = e.buf->padded.size();
        for (std::size_t done = 0; done < total;) {
            const std::size_t n = std::min(chunk, total - done);
            const off_t off = static_cast<off_t>(e.offset + done);
            const ssize_t w =
                positional ? ::pwrite(fd.get(), p + done, n, off) : ::write(fd.get(), p + done, n);
            if (w < 0) {
                if (errno == EINTR) continue;
                fail("write", fd.path());
            }
            done += static_cast<std::size_t>(w);
        }
    }
}

void read_all(State& s, const Fd& fd, std::size_t chunk, bool positional) {
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) fail("fstat", fd.path());
    if (static_cast<std::uint64_t>(st.st_size) != s.checkpoint_bytes()) {
        throw std::runtime_error("file size " + std::to_string(st.st_size) + " != expected " +
                                 std::to_string(s.checkpoint_bytes()) + " for " +
                                 fd.path().string());
    }
    for (const auto& e : s.checkpointed()) {
        std::byte* p = e.buf->padded.data();
        const std::size_t total = e.buf->padded.size();
        for (std::size_t done = 0; done < total;) {
            const std::size_t n = std::min(chunk, total - done);
            const off_t off = static_cast<off_t>(e.offset + done);
            const ssize_t r =
                positional ? ::pread(fd.get(), p + done, n, off) : ::read(fd.get(), p + done, n);
            if (r < 0) {
                if (errno == EINTR) continue;
                fail("read", fd.path());
            }
            if (r == 0) throw std::runtime_error("unexpected EOF reading " + fd.path().string());
            done += static_cast<std::size_t>(r);
        }
    }
}

IoResult do_write(const State& s, const std::filesystem::path& path, std::size_t chunk, bool direct,
                  bool falloc) {
    check_chunk(s, chunk, direct);
    IoResult r;
    r.bytes = s.checkpoint_bytes();
    Fd fd(path, O_WRONLY | O_CREAT | O_TRUNC, direct);
    if (falloc) {
        const auto t0 = Clock::now();
        const int rc = ::fallocate(fd.get(), 0, 0, static_cast<off_t>(r.bytes));
        if (rc != 0) fail("fallocate", path);
        r.fallocate_ns = ns_since(t0);
    }
    auto t0 = Clock::now();
    write_all(s, fd, chunk, direct);
    r.io_ns = ns_since(t0);
    t0 = Clock::now();
    if (::fsync(fd.get()) != 0) fail("fsync", path);
    r.fsync_ns = ns_since(t0);
    return r;
}

IoResult do_read(const std::filesystem::path& path, State& s, std::size_t chunk, bool direct) {
    check_chunk(s, chunk, direct);
    IoResult r;
    r.bytes = s.checkpoint_bytes();
    Fd fd(path, O_RDONLY, direct);
    const auto t0 = Clock::now();
    read_all(s, fd, chunk, direct);
    r.io_ns = ns_since(t0);
    return r;
}

}  // namespace

IoResult write_buffered(const State& s, const std::filesystem::path& p, std::size_t chunk) {
    return do_write(s, p, chunk, false, false);
}
IoResult write_odirect(const State& s, const std::filesystem::path& p, std::size_t chunk,
                       bool fallocate) {
    return do_write(s, p, chunk, true, fallocate);
}
IoResult read_buffered(const std::filesystem::path& p, State& s, std::size_t chunk) {
    return do_read(p, s, chunk, false);
}
IoResult read_odirect(const std::filesystem::path& p, State& s, std::size_t chunk) {
    return do_read(p, s, chunk, true);
}

}  // namespace cistern
