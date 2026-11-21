#pragma once
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "table/tensor_table.h"

namespace testutil {

inline std::filesystem::path fixture(const std::string& name) {
    return std::filesystem::path(CISTERN_FIXTURE_DIR) / name;
}
inline std::filesystem::path real_table() {
    return cistern::default_table_path();
}
inline cistern::TensorTable tiny_table() {
    return cistern::TensorTable::load(fixture("tiny.tensors.json"));
}

// Temp directory under $CISTERN_TEST_DIR (default: the system temp dir); removed on destruction.
class TempDir {
public:
    TempDir() {
        const char* base = std::getenv("CISTERN_TEST_DIR");
        std::string tmpl =
            ((base != nullptr && *base != '\0') ? std::filesystem::path(base)
                                                : std::filesystem::temp_directory_path())
                .string() +
            "/cistern-test-XXXXXX";
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
