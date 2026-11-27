#pragma once
// Owning wrapper around an io_uring instance.
#include <liburing.h>

namespace pitstop {

class Ring {
public:
    explicit Ring(unsigned entries);
    ~Ring();
    Ring(const Ring&) = delete;
    Ring& operator=(const Ring&) = delete;

    struct io_uring* get() noexcept { return &ring_; }

private:
    struct io_uring ring_;
};

}  // namespace pitstop
