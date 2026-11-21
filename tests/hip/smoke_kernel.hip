// Toolchain wiring only (hipcc -> object -> link): adds one to every element. Not the Adam kernel.
#include <hip/hip_runtime.h>

__global__ void add_one_kernel(int* data, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) data[i] += 1;
}

extern "C" int cistern_hip_add_one(int* host, int n) {
    int* dev = nullptr;
    if (hipMalloc(&dev, static_cast<size_t>(n) * sizeof(int)) != hipSuccess) return 1;
    if (hipMemcpy(dev, host, static_cast<size_t>(n) * sizeof(int), hipMemcpyHostToDevice) !=
        hipSuccess) return 2;
    add_one_kernel<<<dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0, 0>>>(dev, n);
    if (hipDeviceSynchronize() != hipSuccess) return 3;
    if (hipMemcpy(host, dev, static_cast<size_t>(n) * sizeof(int), hipMemcpyDeviceToHost) !=
        hipSuccess) return 4;
    (void)hipFree(dev);
    return 0;
}
