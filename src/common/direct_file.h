#pragma once
// O_DIRECT file descriptors with an explicit open mode.
#include <filesystem>

namespace cistern {

class DirectFile {
public:
    // Creates or truncates `file` for writing.
    static DirectFile create(const std::filesystem::path& file);
    // Opens an existing `file` read-only.
    static DirectFile open_read(const std::filesystem::path& file);

    ~DirectFile();
    DirectFile(DirectFile&& o) noexcept;
    DirectFile& operator=(DirectFile&& o) noexcept;
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    int get() const noexcept { return fd_; }

private:
    explicit DirectFile(int fd) noexcept : fd_(fd) {}
    int fd_ = -1;
};

}  // namespace cistern
