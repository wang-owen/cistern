#include "ring.h"

#include <system_error>

namespace cistern {

Ring::Ring(unsigned entries) {
    const int ret = ::io_uring_queue_init(entries, &ring_, 0);
    if (ret < 0) {
        throw std::system_error(-ret, std::system_category(), "io_uring_queue_init");
    }
}

Ring::~Ring() {
    ::io_uring_queue_exit(&ring_);
}

}  // namespace cistern
