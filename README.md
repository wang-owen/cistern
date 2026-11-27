# cistern

Checkpoint writer for ML training state, using io_uring and O_DIRECT on NVMe.

A checkpoint holds each tensor's weights and Adam moments in a single file: a header, a CRC32C for each piece, then the tensor data aligned for direct I/O. The writer publishes a checkpoint only once it is durable, so a crash mid-write never replaces the previous one. Restore checks the file and rejects anything truncated, corrupt or mismatched.

The writer is benchmarked against an fio sequential-write ceiling on the same drive, using a ~7.4 GB Llama 3.2 1B checkpoint. The latest results are in `results/`.
