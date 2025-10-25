#include "binstatcuda/device_info.cuh"

#include <cuda_runtime.h>

namespace binstatcuda {

int cuda_device_count() noexcept {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return -1;
    }
    return count;
}

}  // namespace binstatcuda
