#include <pybind11/pybind11.h>

#include "binstatcuda/device_info.cuh"

namespace py = pybind11;

namespace {

constexpr const char* MODULE_DOC =
    "Core CUDA bindings for binstatcuda. "
    "Currently provides device information utilities.";

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = MODULE_DOC;

    m.def(
        "device_count",
        []() { return binstatcuda::cuda_device_count(); },
        "Return the number of CUDA devices visible to the runtime.\n\n"
        "Returns:\n"
        "    int: GPU count, or -1 if CUDA reports an error."
    );
}
