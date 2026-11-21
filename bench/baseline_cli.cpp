// One pwrite + O_DIRECT baseline write of the Llama-3.2-1B checkpoint (~7.4 GB: bf16 weights +
// two Adam moments), 1 MiB chunks, fallocate first, then fsync. Prints decimal GB/s and nothing
// else. No loop, no trim, no sleep: the drift loop lives in the shell.
// usage: cistern-baseline <output file>
#include <cstdio>
#include <exception>
#include <filesystem>

#include "cistern/baselines.hpp"
#include "cistern/state.hpp"
#include "cistern/tensor_table.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output file>\n", argv[0]);
        return 2;
    }
    try {
        const std::filesystem::path out = argv[1];
        const auto table =
            cistern::TensorTable::load(std::filesystem::path(CISTERN_SOURCE_DIR) / "data" /
                                       "llama-3.2-1b.tensors.json");
        cistern::StateOptions opts;
        opts.align_dir = out.parent_path().empty() ? "." : out.parent_path();
        const auto state = cistern::State::generate(table, opts);
        const auto r = cistern::write_odirect(state, out, std::size_t{1} << 20, true);
        std::printf("%.3f\n", r.gbps());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
