#include "direct_file.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <system_error>
#include <utility>

namespace pitstop {

namespace {
int open_or_throw(const std::filesystem::path& file, int flags) {
    const int fd = ::open(file.c_str(), flags | O_DIRECT | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), "::open " + file.string());
    }
    return fd;
}
}  // namespace

DirectFile DirectFile::create(const std::filesystem::path& file) {
    return DirectFile(open_or_throw(file, O_CREAT | O_WRONLY | O_TRUNC));
}

DirectFile DirectFile::open_read(const std::filesystem::path& file) {
    return DirectFile(open_or_throw(file, O_RDONLY));
}

DirectFile::~DirectFile() {
    if (fd_ != -1) ::close(fd_);
}

DirectFile::DirectFile(DirectFile&& o) noexcept : fd_(std::exchange(o.fd_, -1)) {}

DirectFile& DirectFile::operator=(DirectFile&& o) noexcept {
    if (this != &o) {
        if (fd_ != -1) ::close(fd_);
        fd_ = std::exchange(o.fd_, -1);
    }
    return *this;
}

}  // namespace pitstop
