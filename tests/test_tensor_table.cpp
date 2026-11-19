#include <catch2/catch_test_macros.hpp>

#include "cistern/tensor_table.hpp"
#include "test_util.hpp"

using cistern::TableError;
using cistern::TensorTable;

namespace {
const char* kGood = R"({"schema":"cistern.tensors/1","name":"t","tensor_count":2,"total_params":10,
 "total_bytes":20,"source":{"revision":"abc"},"verified":false,
 "tensors":[{"name":"a","shape":[2,3],"dtype":"bf16","bytes":12},{"name":"b","shape":[4],"dtype":"bf16","bytes":8}]})";
}

TEST_CASE("table parses a valid document", "[unit][table]") {
    const auto t = TensorTable::parse(kGood);
    CHECK(t.tensors.size() == 2);
    CHECK(t.total_bytes == 20);
    CHECK(t.total_params == 10);
    CHECK(t.source_revision == "abc");
}

TEST_CASE("table rejects inconsistent documents", "[unit][table]") {
    auto bad = [](std::string from, std::string to) {
        std::string s = kGood;
        const auto p = s.find(from);
        REQUIRE(p != std::string::npos);
        s.replace(p, from.size(), to);
        return s;
    };
    CHECK_THROWS_AS(TensorTable::parse(bad("\"bytes\":12", "\"bytes\":14")), TableError);
    CHECK_THROWS_AS(TensorTable::parse(bad("\"total_bytes\":20", "\"total_bytes\":21")),
                    TableError);
    CHECK_THROWS_AS(TensorTable::parse(bad("\"total_params\":10", "\"total_params\":9")),
                    TableError);
    CHECK_THROWS_AS(TensorTable::parse(bad("\"tensor_count\":2", "\"tensor_count\":3")),
                    TableError);
    CHECK_THROWS_AS(TensorTable::parse(bad("\"name\":\"b\"", "\"name\":\"a\"")), TableError);
    CHECK_THROWS_AS(TensorTable::parse(bad("\"bf16\"", "\"int3\"")), TableError);
    CHECK_THROWS_AS(TensorTable::parse(bad("cistern.tensors/1", "other/9")), TableError);
    CHECK_THROWS_AS(TensorTable::parse("{not json"), TableError);
    CHECK_THROWS_AS(TensorTable::load("/nonexistent/table.json"), TableError);
}

TEST_CASE("committed Llama 3.2 1B table has verified totals", "[unit][table]") {
    const auto t = TensorTable::load(testutil::data_file("llama-3.2-1b.tensors.json"));
    CHECK(t.verified);
    CHECK(t.tensors.size() == 146);
    CHECK(t.total_params == 1235814400ull);
    CHECK(t.total_bytes == 2471628800ull);
    for (const auto& x : t.tensors) CHECK(x.name != "lm_head.weight");  // tied embeddings
}

TEST_CASE("tiny fixture loads and has odd sizes", "[unit][table]") {
    const auto t = testutil::tiny_table();
    bool odd = false;
    for (const auto& x : t.tensors) odd = odd || (x.bytes % 4096 != 0);
    CHECK(odd);
    CHECK(t.total_bytes < (4u << 20));
}
