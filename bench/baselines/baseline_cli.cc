// Baseline write of the checkpoint state (default: the Llama-3.2-1B table, ~7.4 GB of bf16
// weights + two Adam moments). Prints decimal GB/s over the whole operation (incl. fsync) and
// nothing else. No loop, no trim, no sleep: the drift loop lives in the shell.
//
// usage: cistern-baseline [--mode odirect|buffered] [--table PATH] <output file>
//   odirect  (default) pwrite + O_DIRECT, 1 MiB chunks, fallocate first, then fsync
//   buffered           write() + fsync, 1 MiB chunks, no fallocate
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "baselines/baselines.h"
#include "gen/state.h"
#include "table/tensor_table.h"

namespace {

constexpr std::size_t kChunk = std::size_t{1} << 20;

void usage(const char* argv0, std::FILE* to) {
    std::fprintf(to,
                 "usage: %s [--mode odirect|buffered] [--table PATH] <output file>\n"
                 "  --mode   odirect (default): pwrite + O_DIRECT + fallocate + fsync\n"
                 "           buffered: write() + fsync\n"
                 "  --table  tensor table JSON (default: committed Llama 3.2 1B)\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = "odirect";
    std::filesystem::path table_path = cistern::default_table_path();
    std::filesystem::path out;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage(argv[0], stdout);
            return 0;
        }
        if (a == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (a == "--table" && i + 1 < argc) {
            table_path = argv[++i];
        } else if (!a.empty() && a[0] == '-') {
            usage(argv[0], stderr);
            return 2;
        } else if (out.empty()) {
            out = a;
        } else {
            usage(argv[0], stderr);
            return 2;
        }
    }
    if (out.empty() || (mode != "odirect" && mode != "buffered")) {
        usage(argv[0], stderr);
        return 2;
    }
    try {
        const auto table = cistern::TensorTable::load(table_path);
        cistern::StateOptions opts;
        opts.align_dir = out.parent_path().empty() ? "." : out.parent_path();
        const auto state = cistern::State::generate(table, opts);
        const auto r = mode == "odirect" ? cistern::write_odirect(state, out, kChunk, true)
                                         : cistern::write_buffered(state, out, kChunk);
        std::printf("%.3f\n", r.gbps());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
