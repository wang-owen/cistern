#pragma once
// Tensor table loaded from a pitstop.tensors/1 JSON document; "source" records where it came from.
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pitstop {

class TableError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class DType { bf16, fp16, fp32 };

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> shape;
    DType dtype = DType::bf16;
    std::uint64_t bytes = 0;
};

struct TensorTable {
    std::string name;
    std::vector<TensorInfo> tensors;
    std::uint64_t total_params = 0;
    std::uint64_t total_bytes = 0;
    bool verified = false;
    std::string source_revision;

    // Throws TableError unless bytes, totals, count and name uniqueness are consistent.
    static TensorTable parse(std::string_view json_text);
    static TensorTable load(const std::filesystem::path& path);
};

// <source dir>/src/llama-3.2-1b.tensors.json
std::filesystem::path default_table_path();

}  // namespace pitstop
