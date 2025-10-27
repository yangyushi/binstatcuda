#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace binstatcuda {

enum class StatisticKind : int {
    kCount = 0,
    kSum = 1,
    kMean = 2,
    kStd = 3,
};

cudaError_t histogram_1d(
    const float* host_samples,
    std::size_t sample_count,
    const float* host_edges,
    int edge_count,
    unsigned long long* host_counts
) noexcept;

cudaError_t histogram_2d(
    const float* host_x,
    const float* host_y,
    std::size_t sample_count,
    const float* host_x_edges,
    int x_edge_count,
    const float* host_y_edges,
    int y_edge_count,
    unsigned long long* host_counts
) noexcept;

cudaError_t binned_statistic_1d(
    const float* host_samples,
    const float* host_values,
    std::size_t sample_count,
    const float* host_edges,
    int edge_count,
    StatisticKind statistic,
    float* host_result
) noexcept;

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
) noexcept;

}  // namespace binstatcuda
