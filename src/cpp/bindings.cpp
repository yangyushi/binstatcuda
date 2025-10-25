#include <string>

#include <cuda_runtime_api.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "binstatcuda/device_info.cuh"
#include "binstatcuda/histogram.cuh"

namespace py = pybind11;

namespace {

constexpr const char* MODULE_DOC =
    "Core CUDA bindings for binstatcuda. "
    "Currently provides device information utilities.";

using ContigFloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = MODULE_DOC;

    m.def(
        "histogram",
        [](ContigFloatArray samples, ContigFloatArray edges) {
            const py::buffer_info sample_info = samples.request();
            if (sample_info.ndim != 1) {
                throw py::value_error(
                    "Samples must be a one-dimensional array."
                );
            }

            const py::buffer_info edge_info = edges.request();
            if (edge_info.ndim != 1) {
                throw py::value_error("Edges must be a one-dimensional array.");
            }

            const auto sample_count =
                static_cast<std::size_t>(sample_info.shape[0]);
            const int edge_count = static_cast<int>(edge_info.shape[0]);
            if (edge_count < 2) {
                throw py::value_error("At least two edges are required.");
            }

            const float* edge_ptr = static_cast<const float*>(edge_info.ptr);
            for (int idx = 1; idx < edge_count; ++idx) {
                if (edge_ptr[idx] <= edge_ptr[idx - 1]) {
                    throw py::value_error("Edges must be strictly increasing.");
                }
            }

            py::array_t<unsigned long long> counts(edge_count - 1);
            auto counts_info = counts.request();

            const float* sample_ptr =
                static_cast<const float*>(sample_info.ptr);
            auto* count_ptr =
                static_cast<unsigned long long*>(counts_info.ptr);

            {
                py::gil_scoped_release release;
                const cudaError_t status = binstatcuda::histogram_1d(
                    sample_ptr,
                    sample_count,
                    edge_ptr,
                    edge_count,
                    count_ptr
                );
                if (status != cudaSuccess) {
                    throw py::value_error(
                        std::string("CUDA histogram failed: ")
                        + cudaGetErrorString(status)
                    );
                }
            }

            return counts;
        },
        py::arg("samples"),
        py::arg("edges"),
        "Compute a 1D histogram using CUDA."
    );

    m.def(
        "histogram_2d",
        [](
            ContigFloatArray xs,
            ContigFloatArray ys,
            ContigFloatArray x_edges,
            ContigFloatArray y_edges
        ) {
            const py::buffer_info x_info = xs.request();
            if (x_info.ndim != 1) {
                throw py::value_error("x must be a one-dimensional array.");
            }
            const py::buffer_info y_info = ys.request();
            if (y_info.ndim != 1) {
                throw py::value_error("y must be a one-dimensional array.");
            }
            if (x_info.shape[0] != y_info.shape[0]) {
                throw py::value_error("x and y must have the same length.");
            }

            const py::buffer_info x_edges_info = x_edges.request();
            const py::buffer_info y_edges_info = y_edges.request();
            if (x_edges_info.ndim != 1 || y_edges_info.ndim != 1) {
                throw py::value_error("Edge arrays must be one-dimensional.");
            }

            const int x_edge_count = static_cast<int>(x_edges_info.shape[0]);
            const int y_edge_count = static_cast<int>(y_edges_info.shape[0]);
            if (x_edge_count < 2 || y_edge_count < 2) {
                throw py::value_error("Each axis requires at least two edges.");
            }

            const float* x_edge_ptr =
                static_cast<const float*>(x_edges_info.ptr);
            const float* y_edge_ptr =
                static_cast<const float*>(y_edges_info.ptr);

            for (int idx = 1; idx < x_edge_count; ++idx) {
                if (x_edge_ptr[idx] <= x_edge_ptr[idx - 1]) {
                    throw py::value_error(
                        "x_edges must be strictly increasing."
                    );
                }
            }
            for (int idx = 1; idx < y_edge_count; ++idx) {
                if (y_edge_ptr[idx] <= y_edge_ptr[idx - 1]) {
                    throw py::value_error(
                        "y_edges must be strictly increasing."
                    );
                }
            }

            const std::size_t sample_count =
                static_cast<std::size_t>(x_info.shape[0]);

            py::array_t<unsigned long long> counts(
                {x_edge_count - 1, y_edge_count - 1}
            );
            auto counts_info = counts.request();

            const float* x_ptr = static_cast<const float*>(x_info.ptr);
            const float* y_ptr = static_cast<const float*>(y_info.ptr);
            auto* count_ptr =
                static_cast<unsigned long long*>(counts_info.ptr);

            {
                py::gil_scoped_release release;
                const cudaError_t status = binstatcuda::histogram_2d(
                    x_ptr,
                    y_ptr,
                    sample_count,
                    x_edge_ptr,
                    x_edge_count,
                    y_edge_ptr,
                    y_edge_count,
                    count_ptr
                );
                if (status != cudaSuccess) {
                    throw py::value_error(
                        std::string("CUDA histogram_2d failed: ")
                        + cudaGetErrorString(status)
                    );
                }
            }

            counts.resize({x_edge_count - 1, y_edge_count - 1});
            return counts;
        },
        py::arg("x"),
        py::arg("y"),
        py::arg("x_edges"),
        py::arg("y_edges"),
        "Compute a 2D histogram using CUDA."
    );

    m.def(
        "device_count",
        []() { return binstatcuda::cuda_device_count(); },
        "Return the number of CUDA devices visible to the runtime.\n\n"
        "Returns:\n"
        "    int: GPU count, or -1 if CUDA reports an error."
    );
}
