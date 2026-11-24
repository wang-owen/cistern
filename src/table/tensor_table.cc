#include "table/tensor_table.h"

#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <set>

namespace cistern {

namespace {

std::size_t dtype_size(DType d) noexcept {
    switch (d) {
        case DType::bf16:
        case DType::fp16:
            return 2;
        case DType::fp32:
            return 4;
    }
    return 0;
}

DType parse_dtype(std::string_view s) {
    if (s == "bf16") return DType::bf16;
    if (s == "fp16") return DType::fp16;
    if (s == "fp32") return DType::fp32;
    throw TableError("unknown dtype '" + std::string(s) + "'");
}

}  // namespace

TensorTable TensorTable::parse(std::string_view text) {
    using nlohmann::json;
    try {
        const json j = json::parse(text);
        if (j.value("schema", "") != "cistern.tensors/1") {
            throw TableError("unsupported or missing schema (expected cistern.tensors/1)");
        }
        TensorTable t;
        t.name = j.at("name").get<std::string>();
        t.verified = j.at("verified").get<bool>();
        if (j.contains("source") && j["source"].contains("revision")) {
            t.source_revision = j["source"]["revision"].get<std::string>();
        }
        std::set<std::string> seen;
        for (const auto& e : j.at("tensors")) {
            TensorInfo ti;
            ti.name = e.at("name").get<std::string>();
            ti.shape = e.at("shape").get<std::vector<std::uint64_t>>();
            ti.dtype = parse_dtype(e.at("dtype").get<std::string>());
            ti.bytes = e.at("bytes").get<std::uint64_t>();
            if (!seen.insert(ti.name).second) throw TableError("duplicate tensor " + ti.name);
            std::uint64_t elems = 1;
            for (auto d : ti.shape) elems *= d;
            if (ti.shape.empty() || elems == 0) throw TableError("empty shape for " + ti.name);
            if (elems * dtype_size(ti.dtype) != ti.bytes) {
                throw TableError("bytes != prod(shape) * dtype size for " + ti.name);
            }
            t.total_params += elems;
            t.total_bytes += ti.bytes;
            t.tensors.push_back(std::move(ti));
        }
        if (t.tensors.empty()) throw TableError("table has no tensors");
        if (t.total_bytes != j.at("total_bytes").get<std::uint64_t>()) {
            throw TableError("total_bytes does not match the sum of tensor bytes");
        }
        if (t.total_params != j.at("total_params").get<std::uint64_t>()) {
            throw TableError("total_params does not match the sum of tensor shapes");
        }
        if (t.tensors.size() != j.at("tensor_count").get<std::size_t>()) {
            throw TableError("tensor_count does not match the number of tensors");
        }
        return t;
    } catch (const nlohmann::json::exception& e) {
        throw TableError(std::string("malformed tensor table: ") + e.what());
    }
}

TensorTable TensorTable::load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw TableError("cannot open tensor table " + path.string());
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(text);
}

std::filesystem::path default_table_path() {
    return std::filesystem::path(CISTERN_SOURCE_DIR) / "src" / "table" /
           "llama-3.2-1b.tensors.json";
}

}  // namespace cistern
