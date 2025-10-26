#include "binstatcuda/histogram.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace binstatcuda {
namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr std::size_t MAX_SHARED_BINS = 4096;

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() noexcept {
        reset();
    }

    cudaError_t allocate(std::size_t count) noexcept {
        reset();
        if (count == 0) {
            return cudaSuccess;
        }
        T* ptr = nullptr;
        const cudaError_t status = cudaMalloc(
            reinterpret_cast<void**>(&ptr), count * sizeof(T)
        );
        if (status == cudaSuccess) {
            ptr_ = ptr;
            size_ = count;
        }
        return status;
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    [[nodiscard]] T* get() noexcept {
        return ptr_;
    }

    [[nodiscard]] const T* get() const noexcept {
        return ptr_;
    }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

__device__ __forceinline__ int find_bin(
    float value,
    const float* edges,
    int edge_count
) noexcept {
    if (value < edges[0] || value > edges[edge_count - 1]) {
        return -1;
    }

    int left = 0;
    int right = edge_count - 1;

    while (left < right) {
        const int mid = (left + right) / 2;
        if (value >= edges[mid]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    int bin = left - 1;
    if (bin < 0) {
        return -1;
    }

    if (value == edges[edge_count - 1]) {
        bin = edge_count - 2;
    }

    return bin;
}

__global__ void histogram_1d_global_kernel(
    const float* samples,
    std::size_t sample_count,
    const float* edges,
    int edge_count,
    unsigned long long* counts
) {
    const std::size_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;

    for (std::size_t idx = global_id; idx < sample_count; idx += stride) {
        const float value = samples[idx];
        const int bin = find_bin(value, edges, edge_count);
        if (bin >= 0) {
            atomicAdd(&counts[bin], 1ULL);
        }
    }
}

__global__ void histogram_1d_shared_kernel(
    const float* samples,
    std::size_t sample_count,
    const float* edges,
    int edge_count,
    unsigned long long* counts
) {
    extern __shared__ unsigned long long shared_counts[];

    const int bin_count = edge_count - 1;

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        shared_counts[idx] = 0ULL;
    }
    __syncthreads();

    const std::size_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;

    for (std::size_t sample_idx = global_id;
         sample_idx < sample_count;
         sample_idx += stride) {
        const float value = samples[sample_idx];
        const int bin = find_bin(value, edges, edge_count);
        if (bin >= 0) {
            atomicAdd(&shared_counts[bin], 1ULL);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        const unsigned long long value = shared_counts[idx];
        if (value != 0ULL) {
            atomicAdd(&counts[idx], value);
        }
    }
}

__global__ void histogram_2d_global_kernel(
    const float* xs,
    const float* ys,
    std::size_t sample_count,
    const float* x_edges,
    int x_edge_count,
    const float* y_edges,
    int y_edge_count,
    unsigned long long* counts
) {
    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    const int y_bins = y_edge_count - 1;

    for (std::size_t idx = global_id; idx < sample_count; idx += stride) {
        const int bin_x = find_bin(xs[idx], x_edges, x_edge_count);
        if (bin_x < 0) {
            continue;
        }
        const int bin_y = find_bin(ys[idx], y_edges, y_edge_count);
        if (bin_y < 0) {
            continue;
        }
        const std::size_t offset =
            static_cast<std::size_t>(bin_x)
            * static_cast<std::size_t>(y_bins)
            + static_cast<std::size_t>(bin_y);
        atomicAdd(&counts[offset], 1ULL);
    }
}

__global__ void histogram_2d_shared_kernel(
    const float* xs,
    const float* ys,
    std::size_t sample_count,
    const float* x_edges,
    int x_edge_count,
    const float* y_edges,
    int y_edge_count,
    unsigned long long* counts
) {
    const int x_bins = x_edge_count - 1;
    const int y_bins = y_edge_count - 1;
    const int bin_count = x_bins * y_bins;

    extern __shared__ unsigned long long shared_counts[];

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        shared_counts[idx] = 0ULL;
    }
    __syncthreads();

    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t sample_idx = global_id;
         sample_idx < sample_count;
         sample_idx += stride) {
        const int bin_x = find_bin(xs[sample_idx], x_edges, x_edge_count);
        if (bin_x < 0) {
            continue;
        }
        const int bin_y = find_bin(ys[sample_idx], y_edges, y_edge_count);
        if (bin_y < 0) {
            continue;
        }
        const std::size_t offset =
            static_cast<std::size_t>(bin_x)
            * static_cast<std::size_t>(y_bins)
            + static_cast<std::size_t>(bin_y);
        atomicAdd(&shared_counts[offset], 1ULL);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        const unsigned long long value = shared_counts[idx];
        if (value != 0ULL) {
            atomicAdd(&counts[idx], value);
        }
    }
}

__global__ void binned_statistic_1d_global_kernel(
    const float* samples,
    const float* values,
    std::size_t sample_count,
    const float* edges,
    int edge_count,
    unsigned long long* counts,
    float* sums,
    unsigned int* bin_numbers
) {
    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t idx = global_id; idx < sample_count; idx += stride) {
        const int bin = find_bin(samples[idx], edges, edge_count);
        const unsigned int bin_number =
            bin >= 0 ? static_cast<unsigned int>(bin + 1) : 0U;
        bin_numbers[idx] = bin_number;
        if (bin < 0) {
            continue;
        }

        atomicAdd(&counts[bin], 1ULL);
        atomicAdd(&sums[bin], values[idx]);
    }
}

__global__ void binned_statistic_1d_shared_kernel(
    const float* samples,
    const float* values,
    std::size_t sample_count,
    const float* edges,
    int edge_count,
    unsigned long long* counts,
    float* sums,
    unsigned int* bin_numbers
) {
    const int bin_count = edge_count - 1;
    extern __shared__ unsigned char shared_storage[];
    auto* shared_counts =
        reinterpret_cast<unsigned long long*>(shared_storage);
    float* shared_sums =
        reinterpret_cast<float*>(shared_counts + bin_count);

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        shared_counts[idx] = 0ULL;
        shared_sums[idx] = 0.0F;
    }
    __syncthreads();

    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t sample_idx = global_id;
         sample_idx < sample_count;
         sample_idx += stride) {
        const int bin = find_bin(samples[sample_idx], edges, edge_count);
        const unsigned int bin_number =
            bin >= 0 ? static_cast<unsigned int>(bin + 1) : 0U;
        bin_numbers[sample_idx] = bin_number;
        if (bin < 0) {
            continue;
        }

        atomicAdd(&shared_counts[bin], 1ULL);
        atomicAdd(&shared_sums[bin], values[sample_idx]);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        atomicAdd(&counts[idx], shared_counts[idx]);
        atomicAdd(&sums[idx], shared_sums[idx]);
    }
}

__global__ void binned_statistic_2d_global_kernel(
    const float* xs,
    const float* ys,
    const float* values,
    std::size_t sample_count,
    const float* x_edges,
    int x_edge_count,
    const float* y_edges,
    int y_edge_count,
    unsigned long long* counts,
    float* sums,
    unsigned int* bin_numbers_x,
    unsigned int* bin_numbers_y
) {
    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    const int y_bins = y_edge_count - 1;

    for (std::size_t idx = global_id; idx < sample_count; idx += stride) {
        const int bin_x = find_bin(xs[idx], x_edges, x_edge_count);
        const int bin_y = find_bin(ys[idx], y_edges, y_edge_count);

        const unsigned int bin_number_x =
            bin_x >= 0 ? static_cast<unsigned int>(bin_x + 1) : 0U;
        const unsigned int bin_number_y =
            bin_y >= 0 ? static_cast<unsigned int>(bin_y + 1) : 0U;
        bin_numbers_x[idx] = bin_number_x;
        bin_numbers_y[idx] = bin_number_y;

        if (bin_x < 0 || bin_y < 0) {
            continue;
        }

        const std::size_t offset =
            static_cast<std::size_t>(bin_x)
            * static_cast<std::size_t>(y_bins)
            + static_cast<std::size_t>(bin_y);

        atomicAdd(&counts[offset], 1ULL);
        atomicAdd(&sums[offset], values[idx]);
    }
}

__global__ void binned_statistic_2d_shared_kernel(
    const float* xs,
    const float* ys,
    const float* values,
    std::size_t sample_count,
    const float* x_edges,
    int x_edge_count,
    const float* y_edges,
    int y_edge_count,
    unsigned long long* counts,
    float* sums,
    unsigned int* bin_numbers_x,
    unsigned int* bin_numbers_y
) {
    const int x_bins = x_edge_count - 1;
    const int y_bins = y_edge_count - 1;
    const int bin_count = x_bins * y_bins;

    extern __shared__ unsigned char shared_storage[];
    auto* shared_counts =
        reinterpret_cast<unsigned long long*>(shared_storage);
    float* shared_sums =
        reinterpret_cast<float*>(shared_counts + bin_count);

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        shared_counts[idx] = 0ULL;
        shared_sums[idx] = 0.0F;
    }
    __syncthreads();

    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t sample_idx = global_id;
         sample_idx < sample_count;
         sample_idx += stride) {
        const int bin_x = find_bin(xs[sample_idx], x_edges, x_edge_count);
        const int bin_y = find_bin(ys[sample_idx], y_edges, y_edge_count);

        const unsigned int bin_number_x =
            bin_x >= 0 ? static_cast<unsigned int>(bin_x + 1) : 0U;
        const unsigned int bin_number_y =
            bin_y >= 0 ? static_cast<unsigned int>(bin_y + 1) : 0U;
        bin_numbers_x[sample_idx] = bin_number_x;
        bin_numbers_y[sample_idx] = bin_number_y;

        if (bin_x < 0 || bin_y < 0) {
            continue;
        }

        const std::size_t offset =
            static_cast<std::size_t>(bin_x)
            * static_cast<std::size_t>(y_bins)
            + static_cast<std::size_t>(bin_y);

        atomicAdd(&shared_counts[offset], 1ULL);
        atomicAdd(&shared_sums[offset], values[sample_idx]);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < bin_count; idx += blockDim.x) {
        atomicAdd(&counts[idx], shared_counts[idx]);
        atomicAdd(&sums[idx], shared_sums[idx]);
    }
}

[[nodiscard]] int compute_grid_size(std::size_t sample_count) noexcept {
    if (sample_count == 0) {
        return 1;
    }
    const std::size_t blocks =
        (sample_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<int>(std::min<std::size_t>(blocks, 65535));
}

[[nodiscard]] cudaError_t launch_histogram_1d(
    const float* d_samples,
    std::size_t sample_count,
    const float* d_edges,
    int edge_count,
    unsigned long long* d_counts
) noexcept {
    const int grid_size = compute_grid_size(sample_count);

    const int bin_count = edge_count - 1;
    const bool use_shared =
        static_cast<std::size_t>(bin_count) <= MAX_SHARED_BINS;

    if (use_shared) {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(bin_count)
            * sizeof(unsigned long long);
        histogram_1d_shared_kernel<<<grid_size, THREADS_PER_BLOCK, shared_bytes>>>(
            d_samples,
            sample_count,
            d_edges,
            edge_count,
            d_counts
        );
    } else {
        histogram_1d_global_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
            d_samples,
            sample_count,
            d_edges,
            edge_count,
            d_counts
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

[[nodiscard]] cudaError_t launch_histogram_2d(
    const float* d_xs,
    const float* d_ys,
    std::size_t sample_count,
    const float* d_x_edges,
    int x_edge_count,
    const float* d_y_edges,
    int y_edge_count,
    unsigned long long* d_counts
) noexcept {
    const int grid_size = compute_grid_size(sample_count);

    const int x_bins = x_edge_count - 1;
    const int y_bins = y_edge_count - 1;
    const std::size_t bin_count =
        static_cast<std::size_t>(x_bins) * static_cast<std::size_t>(y_bins);
    const bool use_shared = bin_count <= MAX_SHARED_BINS;

    if (use_shared) {
        const std::size_t shared_bytes =
            bin_count * sizeof(unsigned long long);
        histogram_2d_shared_kernel<<<grid_size, THREADS_PER_BLOCK, shared_bytes>>>(
            d_xs,
            d_ys,
            sample_count,
            d_x_edges,
            x_edge_count,
            d_y_edges,
            y_edge_count,
            d_counts
        );
    } else {
        histogram_2d_global_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
            d_xs,
            d_ys,
            sample_count,
            d_x_edges,
            x_edge_count,
            d_y_edges,
            y_edge_count,
            d_counts
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

[[nodiscard]] cudaError_t launch_binned_statistic_1d(
    const float* d_samples,
    const float* d_values,
    std::size_t sample_count,
    const float* d_edges,
    int edge_count,
    unsigned long long* d_counts,
    float* d_sums,
    unsigned int* d_bin_numbers
) noexcept {
    const int grid_size = compute_grid_size(sample_count);
    const int bin_count = edge_count - 1;
    const bool use_shared =
        static_cast<std::size_t>(bin_count) <= MAX_SHARED_BINS;

    if (use_shared) {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(bin_count)
            * (sizeof(unsigned long long) + sizeof(float));
        binned_statistic_1d_shared_kernel<<<grid_size, THREADS_PER_BLOCK, shared_bytes>>>(
            d_samples,
            d_values,
            sample_count,
            d_edges,
            edge_count,
            d_counts,
            d_sums,
            d_bin_numbers
        );
    } else {
        binned_statistic_1d_global_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
            d_samples,
            d_values,
            sample_count,
            d_edges,
            edge_count,
            d_counts,
            d_sums,
            d_bin_numbers
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

[[nodiscard]] cudaError_t launch_binned_statistic_2d(
    const float* d_xs,
    const float* d_ys,
    const float* d_values,
    std::size_t sample_count,
    const float* d_x_edges,
    int x_edge_count,
    const float* d_y_edges,
    int y_edge_count,
    unsigned long long* d_counts,
    float* d_sums,
    unsigned int* d_bin_numbers_x,
    unsigned int* d_bin_numbers_y
) noexcept {
    const int grid_size = compute_grid_size(sample_count);

    const int x_bins = x_edge_count - 1;
    const int y_bins = y_edge_count - 1;
    const std::size_t bin_count =
        static_cast<std::size_t>(x_bins) * static_cast<std::size_t>(y_bins);
    const bool use_shared = bin_count <= MAX_SHARED_BINS;

    if (use_shared) {
        const std::size_t shared_bytes =
            bin_count * (sizeof(unsigned long long) + sizeof(float));
        binned_statistic_2d_shared_kernel<<<grid_size, THREADS_PER_BLOCK, shared_bytes>>>(
            d_xs,
            d_ys,
            d_values,
            sample_count,
            d_x_edges,
            x_edge_count,
            d_y_edges,
            y_edge_count,
            d_counts,
            d_sums,
            d_bin_numbers_x,
            d_bin_numbers_y
        );
    } else {
        binned_statistic_2d_global_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
            d_xs,
            d_ys,
            d_values,
            sample_count,
            d_x_edges,
            x_edge_count,
            d_y_edges,
            y_edge_count,
            d_counts,
            d_sums,
            d_bin_numbers_x,
            d_bin_numbers_y
        );
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

}  // namespace

cudaError_t histogram_1d(
    const float* host_samples,
    std::size_t sample_count,
    const float* host_edges,
    int edge_count,
    unsigned long long* host_counts
) noexcept {
    if (edge_count < 2) {
        return cudaErrorInvalidValue;
    }

    const int bin_count = edge_count - 1;
    std::fill(host_counts, host_counts + bin_count, 0ULL);

    if (sample_count == 0) {
        return cudaSuccess;
    }

    DeviceBuffer<float> d_samples;
    DeviceBuffer<float> d_edges;
    DeviceBuffer<unsigned long long> d_counts;

    const cudaError_t alloc_samples = d_samples.allocate(sample_count);
    if (alloc_samples != cudaSuccess) {
        return alloc_samples;
    }

    const cudaError_t alloc_edges = d_edges.allocate(edge_count);
    if (alloc_edges != cudaSuccess) {
        return alloc_edges;
    }

    const cudaError_t alloc_counts = d_counts.allocate(bin_count);
    if (alloc_counts != cudaSuccess) {
        return alloc_counts;
    }

    cudaError_t status = cudaMemcpy(
        d_samples.get(),
        host_samples,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_edges.get(),
        host_edges,
        edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemset(
        d_counts.get(),
        0,
        static_cast<std::size_t>(bin_count) * sizeof(unsigned long long)
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = launch_histogram_1d(
        d_samples.get(),
        sample_count,
        d_edges.get(),
        edge_count,
        d_counts.get()
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_counts,
        d_counts.get(),
        static_cast<std::size_t>(bin_count) * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

cudaError_t histogram_2d(
    const float* host_x,
    const float* host_y,
    std::size_t sample_count,
    const float* host_x_edges,
    int x_edge_count,
    const float* host_y_edges,
    int y_edge_count,
    unsigned long long* host_counts
) noexcept {
    if (x_edge_count < 2 || y_edge_count < 2) {
        return cudaErrorInvalidValue;
    }

    const int x_bin_count = x_edge_count - 1;
    const int y_bin_count = y_edge_count - 1;
    const std::size_t total_bins =
        static_cast<std::size_t>(x_bin_count)
        * static_cast<std::size_t>(y_bin_count);
    std::fill(host_counts, host_counts + total_bins, 0ULL);

    if (sample_count == 0) {
        return cudaSuccess;
    }

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_y;
    DeviceBuffer<float> d_x_edges;
    DeviceBuffer<float> d_y_edges;
    DeviceBuffer<unsigned long long> d_counts;

    cudaError_t status = d_x.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_y.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_x_edges.allocate(x_edge_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_y_edges.allocate(y_edge_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_counts.allocate(total_bins);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_x.get(),
        host_x,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_y.get(),
        host_y,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_x_edges.get(),
        host_x_edges,
        x_edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_y_edges.get(),
        host_y_edges,
        y_edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemset(
        d_counts.get(),
        0,
        total_bins * sizeof(unsigned long long)
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = launch_histogram_2d(
        d_x.get(),
        d_y.get(),
        sample_count,
        d_x_edges.get(),
        x_edge_count,
        d_y_edges.get(),
        y_edge_count,
        d_counts.get()
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_counts,
        d_counts.get(),
        total_bins * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

cudaError_t binned_statistic_1d(
    const float* host_samples,
    const float* host_values,
    std::size_t sample_count,
    const float* host_edges,
    int edge_count,
    unsigned long long* host_counts,
    float* host_sums,
    unsigned int* host_bin_numbers
) noexcept {
    if (edge_count < 2) {
        return cudaErrorInvalidValue;
    }

    const int bin_count = edge_count - 1;
    std::fill(host_counts, host_counts + bin_count, 0ULL);
    std::fill(host_sums, host_sums + bin_count, 0.0F);
    if (sample_count > 0U && host_bin_numbers != nullptr) {
        std::fill(host_bin_numbers, host_bin_numbers + sample_count, 0U);
    }

    if (sample_count == 0) {
        return cudaSuccess;
    }

    DeviceBuffer<float> d_samples;
    DeviceBuffer<float> d_values;
    DeviceBuffer<float> d_edges;
    DeviceBuffer<unsigned long long> d_counts;
    DeviceBuffer<float> d_sums;
    DeviceBuffer<unsigned int> d_bin_numbers;

    cudaError_t status = d_samples.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_values.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_edges.allocate(edge_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_counts.allocate(bin_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_sums.allocate(bin_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_bin_numbers.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_samples.get(),
        host_samples,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_values.get(),
        host_values,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_edges.get(),
        host_edges,
        edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemset(
        d_counts.get(),
        0,
        static_cast<std::size_t>(bin_count) * sizeof(unsigned long long)
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemset(
        d_sums.get(),
        0,
        static_cast<std::size_t>(bin_count) * sizeof(float)
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = launch_binned_statistic_1d(
        d_samples.get(),
        d_values.get(),
        sample_count,
        d_edges.get(),
        edge_count,
        d_counts.get(),
        d_sums.get(),
        d_bin_numbers.get()
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_counts,
        d_counts.get(),
        static_cast<std::size_t>(bin_count) * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_sums,
        d_sums.get(),
        static_cast<std::size_t>(bin_count) * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_bin_numbers,
        d_bin_numbers.get(),
        sample_count * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

cudaError_t binned_statistic_2d(
    const float* host_x,
    const float* host_y,
    const float* host_values,
    std::size_t sample_count,
    const float* host_x_edges,
    int x_edge_count,
    const float* host_y_edges,
    int y_edge_count,
    unsigned long long* host_counts,
    float* host_sums,
    unsigned int* host_bin_numbers_x,
    unsigned int* host_bin_numbers_y
) noexcept {
    if (x_edge_count < 2 || y_edge_count < 2) {
        return cudaErrorInvalidValue;
    }

    const int x_bin_count = x_edge_count - 1;
    const int y_bin_count = y_edge_count - 1;
    const std::size_t total_bins =
        static_cast<std::size_t>(x_bin_count)
        * static_cast<std::size_t>(y_bin_count);

    std::fill(host_counts, host_counts + total_bins, 0ULL);
    std::fill(host_sums, host_sums + total_bins, 0.0F);
    if (sample_count > 0U) {
        if (host_bin_numbers_x != nullptr) {
            std::fill(host_bin_numbers_x, host_bin_numbers_x + sample_count, 0U);
        }
        if (host_bin_numbers_y != nullptr) {
            std::fill(host_bin_numbers_y, host_bin_numbers_y + sample_count, 0U);
        }
    }

    if (sample_count == 0) {
        return cudaSuccess;
    }

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_y;
    DeviceBuffer<float> d_values;
    DeviceBuffer<float> d_x_edges;
    DeviceBuffer<float> d_y_edges;
    DeviceBuffer<unsigned long long> d_counts;
    DeviceBuffer<float> d_sums;
    DeviceBuffer<unsigned int> d_bin_numbers_x;
    DeviceBuffer<unsigned int> d_bin_numbers_y;

    cudaError_t status = d_x.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_y.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_values.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_x_edges.allocate(x_edge_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_y_edges.allocate(y_edge_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_counts.allocate(total_bins);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_sums.allocate(total_bins);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_bin_numbers_x.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = d_bin_numbers_y.allocate(sample_count);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_x.get(),
        host_x,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_y.get(),
        host_y,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_values.get(),
        host_values,
        sample_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_x_edges.get(),
        host_x_edges,
        x_edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        d_y_edges.get(),
        host_y_edges,
        y_edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemset(
        d_counts.get(),
        0,
        total_bins * sizeof(unsigned long long)
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemset(
        d_sums.get(),
        0,
        total_bins * sizeof(float)
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = launch_binned_statistic_2d(
        d_x.get(),
        d_y.get(),
        d_values.get(),
        sample_count,
        d_x_edges.get(),
        x_edge_count,
        d_y_edges.get(),
        y_edge_count,
        d_counts.get(),
        d_sums.get(),
        d_bin_numbers_x.get(),
        d_bin_numbers_y.get()
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_counts,
        d_counts.get(),
        total_bins * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_sums,
        d_sums.get(),
        total_bins * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_bin_numbers_x,
        d_bin_numbers_x.get(),
        sample_count * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaMemcpy(
        host_bin_numbers_y,
        d_bin_numbers_y.get(),
        sample_count * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    return cudaSuccess;
}

}  // namespace binstatcuda
