#pragma once
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "tensor_table.h"

namespace testutil {

inline std::filesystem::path fixture(const std::string& name) {
    return std::filesystem::path(PITSTOP_FIXTURE_DIR) / name;
}
inline pitstop::TensorTable tiny_table() {
    return pitstop::TensorTable::load(fixture("tiny.tensors.json"));
}

// Temp directory under $PITSTOP_TEST_DIR (default: the system temp dir); removed on destruction.
class TempDir {
public:
    TempDir() {
        const char* base = std::getenv("PITSTOP_TEST_DIR");
        std::string tmpl =
            ((base != nullptr && *base != '\0') ? std::filesystem::path(base)
                                                : std::filesystem::temp_directory_path())
                .string() +
            "/pitstop-test-XXXXXX";
        if (::mkdtemp(tmpl.data()) == nullptr) throw std::runtime_error("mkdtemp failed");
        path_ = tmpl;
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace testutil
