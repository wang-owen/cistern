#pragma once
// Tensor table loaded from the JSON written by tools/tensor_table.py.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cistern {

class TableError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class DType { bf16, fp16, fp32 };

std::size_t dtype_size(DType d) noexcept;
const char* dtype_name(DType d) noexcept;
DType parse_dtype(std::string_view s);  // throws TableError

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

    // Parses and validates (bytes = prod(shape) * dtype size, totals, count, unique names).
    static TensorTable parse(std::string_view json_text);
    static TensorTable load(const std::filesystem::path& path);
};

// The committed Llama 3.2 1B table: <source dir>/src/table/llama-3.2-1b.tensors.json.
std::filesystem::path default_table_path();

}  // namespace cistern
