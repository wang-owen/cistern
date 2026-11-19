#pragma once
// Seam header for the owner-implemented async checkpointer. Declarations only; reshape freely.
#include <filesystem>
#include <memory>

#include "cistern/state.hpp"
#include "cistern/writer.hpp"

namespace cistern {

class AsyncCheckpointer {
public:
    // flush_cpu < 0 means "do not pin the flush thread".
    AsyncCheckpointer(const State& state, WriterConfig cfg, int flush_cpu);
    ~AsyncCheckpointer();
    AsyncCheckpointer(const AsyncCheckpointer&) = delete;
    AsyncCheckpointer& operator=(const AsyncCheckpointer&) = delete;

    // Copies the state to the staging buffer and returns once staged. A request made
    // mid-flush blocks until that flush ends.
    void request(const std::filesystem::path& dst);
    // Blocks until the last requested checkpoint is durable. The bench harness calls this
    // from a helper thread to timestamp "durable", so it must be callable from any thread.
    void wait_durable();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cistern
