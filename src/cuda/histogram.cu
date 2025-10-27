#include "binstatcuda/histogram.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>

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

__global__ void binned_statistic_1d_kernel(
    const float* samples,
    const float* values,
    std::size_t sample_count,
    const float* edges,
    int edge_count,
    unsigned long long* counts,
    float* sums,
    float* sum_squares,
    int* bin_numbers
) {
    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t idx = global_id; idx < sample_count; idx += stride) {
        const float sample = samples[idx];
        const int bin = find_bin(sample, edges, edge_count);

        if (bin_numbers != nullptr) {
            bin_numbers[idx] = bin;
        }
        if (bin < 0) {
            continue;
        }

        atomicAdd(&counts[bin], 1ULL);

        if (sums != nullptr || sum_squares != nullptr) {
            const float value = values[idx];
            if (sums != nullptr) {
                atomicAdd(&sums[bin], value);
            }
            if (sum_squares != nullptr) {
                atomicAdd(&sum_squares[bin], value * value);
            }
        }
    }
}

__global__ void binned_statistic_2d_kernel(
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
    float* sum_squares,
    int* bin_numbers
) {
    const std::size_t global_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    const int y_bins = y_edge_count - 1;

    for (std::size_t idx = global_id; idx < sample_count; idx += stride) {
        const int bin_x = find_bin(xs[idx], x_edges, x_edge_count);
        const int bin_y = find_bin(ys[idx], y_edges, y_edge_count);
        const bool valid = (bin_x >= 0) && (bin_y >= 0);

        if (bin_numbers != nullptr) {
            const int flattened = valid
                ? bin_x * y_bins + bin_y
                : -1;
            bin_numbers[idx] = flattened;
        }

        if (!valid) {
            continue;
        }

        const std::size_t offset =
            static_cast<std::size_t>(bin_x)
            * static_cast<std::size_t>(y_bins)
            + static_cast<std::size_t>(bin_y);

        atomicAdd(&counts[offset], 1ULL);

        if (sums != nullptr || sum_squares != nullptr) {
            const float value = values[idx];
            if (sums != nullptr) {
                atomicAdd(&sums[offset], value);
            }
            if (sum_squares != nullptr) {
                atomicAdd(&sum_squares[offset], value * value);
            }
        }
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
    float* d_sumsq,
    int* d_bin_numbers
) noexcept {
    const int grid_size = compute_grid_size(sample_count);

    binned_statistic_1d_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_samples,
        d_values,
        sample_count,
        d_edges,
        edge_count,
        d_counts,
        d_sums,
        d_sumsq,
        d_bin_numbers
    );

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
    float* d_sumsq,
    int* d_bin_numbers
) noexcept {
    const int grid_size = compute_grid_size(sample_count);

    binned_statistic_2d_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
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
        d_sumsq,
        d_bin_numbers
    );

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
    StatisticKind statistic,
    float* host_result
) noexcept {
    if (host_result == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (edge_count < 2) {
        return cudaErrorInvalidValue;
    }

    const int bin_count = edge_count - 1;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    if (statistic == StatisticKind::kCount
        || statistic == StatisticKind::kSum) {
        std::fill(host_result, host_result + bin_count, 0.0F);
    } else {
        std::fill(host_result, host_result + bin_count, nan);
    }

    if (bin_count == 0 || sample_count == 0) {
        return cudaSuccess;
    }

    const bool needs_sum = statistic == StatisticKind::kSum
        || statistic == StatisticKind::kMean
        || statistic == StatisticKind::kStd;
    const bool needs_sumsq = statistic == StatisticKind::kStd;
    const bool needs_values = needs_sum || needs_sumsq;

    DeviceBuffer<float> d_samples;
    DeviceBuffer<float> d_edges;
    DeviceBuffer<unsigned long long> d_counts;
    DeviceBuffer<float> d_values;
    DeviceBuffer<float> d_sums;
    DeviceBuffer<float> d_sumsq;

    cudaError_t status = d_samples.allocate(sample_count);
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

    if (needs_values) {
        status = d_values.allocate(sample_count);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (needs_sum) {
        status = d_sums.allocate(bin_count);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (needs_sumsq) {
        status = d_sumsq.allocate(bin_count);
        if (status != cudaSuccess) {
            return status;
        }
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
        d_edges.get(),
        host_edges,
        edge_count * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        return status;
    }

    if (needs_values) {
        status = cudaMemcpy(
            d_values.get(),
            host_values,
            sample_count * sizeof(float),
            cudaMemcpyHostToDevice
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    status = cudaMemset(
        d_counts.get(),
        0,
        static_cast<std::size_t>(bin_count) * sizeof(unsigned long long)
    );
    if (status != cudaSuccess) {
        return status;
    }

    if (needs_sum) {
        status = cudaMemset(
            d_sums.get(),
            0,
            static_cast<std::size_t>(bin_count) * sizeof(float)
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    if (needs_sumsq) {
        status = cudaMemset(
            d_sumsq.get(),
            0,
            static_cast<std::size_t>(bin_count) * sizeof(float)
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    status = launch_binned_statistic_1d(
        d_samples.get(),
        needs_values ? d_values.get() : nullptr,
        sample_count,
        d_edges.get(),
        edge_count,
        d_counts.get(),
        needs_sum ? d_sums.get() : nullptr,
        needs_sumsq ? d_sumsq.get() : nullptr,
        nullptr
    );
    if (status != cudaSuccess) {
        return status;
    }

    std::vector<unsigned long long> host_counts(
        static_cast<std::size_t>(bin_count),
        0ULL
    );
    status = cudaMemcpy(
        host_counts.data(),
        d_counts.get(),
        static_cast<std::size_t>(bin_count) * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    std::vector<float> host_sums;
    if (needs_sum) {
        host_sums.resize(bin_count);
        status = cudaMemcpy(
            host_sums.data(),
            d_sums.get(),
            static_cast<std::size_t>(bin_count) * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    std::vector<float> host_sumsq;
    if (needs_sumsq) {
        host_sumsq.resize(bin_count);
        status = cudaMemcpy(
            host_sumsq.data(),
            d_sumsq.get(),
            static_cast<std::size_t>(bin_count) * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    switch (statistic) {
    case StatisticKind::kCount:
        for (int bin = 0; bin < bin_count; ++bin) {
            host_result[bin] = static_cast<float>(host_counts[bin]);
        }
        break;
    case StatisticKind::kSum:
        std::copy(host_sums.begin(), host_sums.end(), host_result);
        break;
    case StatisticKind::kMean:
        for (int bin = 0; bin < bin_count; ++bin) {
            const unsigned long long count = host_counts[bin];
            if (count == 0ULL) {
                host_result[bin] = nan;
            } else {
                host_result[bin] =
                    host_sums[bin] / static_cast<float>(count);
            }
        }
        break;
    case StatisticKind::kStd:
        for (int bin = 0; bin < bin_count; ++bin) {
            const unsigned long long count = host_counts[bin];
            if (count == 0ULL) {
                host_result[bin] = nan;
                continue;
            }
            if (count == 1ULL) {
                host_result[bin] = 0.0F;
                continue;
            }
            const double count_d = static_cast<double>(count);
            const double sum_d = static_cast<double>(host_sums[bin]);
            const double sumsq_d = static_cast<double>(host_sumsq[bin]);
            const double mean = sum_d / count_d;
            double variance = sumsq_d / count_d - mean * mean;
            variance = variance < 0.0 ? 0.0 : variance;
            host_result[bin] = static_cast<float>(std::sqrt(variance));
        }
        break;
    default:
        return cudaErrorInvalidValue;
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
    StatisticKind statistic,
    float* host_result
) noexcept {
    if (host_result == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (x_edge_count < 2 || y_edge_count < 2) {
        return cudaErrorInvalidValue;
    }

    const int x_bin_count = x_edge_count - 1;
    const int y_bin_count = y_edge_count - 1;
    const std::size_t total_bins =
        static_cast<std::size_t>(x_bin_count)
        * static_cast<std::size_t>(y_bin_count);
    if (total_bins == 0) {
        return cudaSuccess;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (statistic == StatisticKind::kCount
        || statistic == StatisticKind::kSum) {
        std::fill(host_result, host_result + total_bins, 0.0F);
    } else {
        std::fill(host_result, host_result + total_bins, nan);
    }

    if (sample_count == 0) {
        return cudaSuccess;
    }

    const bool needs_sum = statistic == StatisticKind::kSum
        || statistic == StatisticKind::kMean
        || statistic == StatisticKind::kStd;
    const bool needs_sumsq = statistic == StatisticKind::kStd;
    const bool needs_values = needs_sum || needs_sumsq;

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_y;
    DeviceBuffer<float> d_x_edges;
    DeviceBuffer<float> d_y_edges;
    DeviceBuffer<unsigned long long> d_counts;
    DeviceBuffer<float> d_values;
    DeviceBuffer<float> d_sums;
    DeviceBuffer<float> d_sumsq;

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
    if (needs_values) {
        status = d_values.allocate(sample_count);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (needs_sum) {
        status = d_sums.allocate(total_bins);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if (needs_sumsq) {
        status = d_sumsq.allocate(total_bins);
        if (status != cudaSuccess) {
            return status;
        }
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

    if (needs_values) {
        status = cudaMemcpy(
            d_values.get(),
            host_values,
            sample_count * sizeof(float),
            cudaMemcpyHostToDevice
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    status = cudaMemset(
        d_counts.get(),
        0,
        total_bins * sizeof(unsigned long long)
    );
    if (status != cudaSuccess) {
        return status;
    }

    if (needs_sum) {
        status = cudaMemset(
            d_sums.get(),
            0,
            total_bins * sizeof(float)
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    if (needs_sumsq) {
        status = cudaMemset(
            d_sumsq.get(),
            0,
            total_bins * sizeof(float)
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    status = launch_binned_statistic_2d(
        d_x.get(),
        d_y.get(),
        needs_values ? d_values.get() : nullptr,
        sample_count,
        d_x_edges.get(),
        x_edge_count,
        d_y_edges.get(),
        y_edge_count,
        d_counts.get(),
        needs_sum ? d_sums.get() : nullptr,
        needs_sumsq ? d_sumsq.get() : nullptr,
        nullptr
    );
    if (status != cudaSuccess) {
        return status;
    }

    std::vector<unsigned long long> host_counts(total_bins, 0ULL);
    status = cudaMemcpy(
        host_counts.data(),
        d_counts.get(),
        total_bins * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        return status;
    }

    std::vector<float> host_sums;
    if (needs_sum) {
        host_sums.resize(total_bins);
        status = cudaMemcpy(
            host_sums.data(),
            d_sums.get(),
            total_bins * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    std::vector<float> host_sumsq;
    if (needs_sumsq) {
        host_sumsq.resize(total_bins);
        status = cudaMemcpy(
            host_sumsq.data(),
            d_sumsq.get(),
            total_bins * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) {
            return status;
        }
    }

    switch (statistic) {
    case StatisticKind::kCount:
        for (std::size_t idx = 0; idx < total_bins; ++idx) {
            host_result[idx] = static_cast<float>(host_counts[idx]);
        }
        break;
    case StatisticKind::kSum:
        std::copy(host_sums.begin(), host_sums.end(), host_result);
        break;
    case StatisticKind::kMean:
        for (std::size_t idx = 0; idx < total_bins; ++idx) {
            const unsigned long long count = host_counts[idx];
            if (count == 0ULL) {
                host_result[idx] = nan;
            } else {
                host_result[idx] =
                    host_sums[idx] / static_cast<float>(count);
            }
        }
        break;
    case StatisticKind::kStd:
        for (std::size_t idx = 0; idx < total_bins; ++idx) {
            const unsigned long long count = host_counts[idx];
            if (count == 0ULL) {
                host_result[idx] = nan;
                continue;
            }
            if (count == 1ULL) {
                host_result[idx] = 0.0F;
                continue;
            }
            const double count_d = static_cast<double>(count);
            const double sum_d = static_cast<double>(host_sums[idx]);
            const double sumsq_d = static_cast<double>(host_sumsq[idx]);
            const double mean = sum_d / count_d;
            double variance = sumsq_d / count_d - mean * mean;
            variance = variance < 0.0 ? 0.0 : variance;
            host_result[idx] = static_cast<float>(std::sqrt(variance));
        }
        break;
    default:
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}


}  // namespace binstatcuda
