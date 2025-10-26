#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace binstatcuda {

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
    unsigned long long* host_counts,
    float* host_sums,
    unsigned int* host_bin_numbers
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
    unsigned long long* host_counts,
    float* host_sums,
    unsigned int* host_bin_numbers_x,
    unsigned int* host_bin_numbers_y
) noexcept;

}  // namespace binstatcuda
