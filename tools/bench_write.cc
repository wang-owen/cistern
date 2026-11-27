// cistern-bench-write <output dir>: times one write_checkpoint of the Llama 3.2 1B state
// (~7.4 GB, 1 MiB pieces like the fio job) and prints decimal GB/s over the whole call: data
// writes, fdatasync, header, fdatasync, rename, dir fsync. State generation and buffer
// registration (Writer construction, a one-time cost per State) are not timed.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>

#include "layout.h"
#include "state.h"
#include "tensor_table.h"
#include "writer.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output dir>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    try {
        cistern::StateOptions opts;
        opts.align_dir = dir;
        const auto state = cistern::State::generate(
            cistern::TensorTable::load(cistern::default_table_path()), opts);
        const auto layout =
            cistern::make_layout(state.table(), state.alignment().unit(), std::uint64_t{1} << 20);

        cistern::Writer writer(state);
        if (!writer.fixed_buffers()) {
            std::fprintf(stderr, "warning: buffer registration failed; using plain writes\n");
        }

        const auto t0 = std::chrono::steady_clock::now();
        writer.write(layout, dir);
        const std::chrono::duration<double, std::nano> ns = std::chrono::steady_clock::now() - t0;

        std::printf("%.3f\n", static_cast<double>(layout.file_size) / ns.count());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
