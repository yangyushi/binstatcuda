"""
CUDA-accelerated binned statistics.

The current implementation bootstraps the native extension and exposes
device discovery helpers that validate the CUDA toolchain.
"""

from importlib import metadata
import numpy as np
import numpy.typing as npt

# NOTE: Importing the native module is expected to fail loudly if CUDA is
# unavailable so callers can detect misconfiguration early.
try:
    from . import _core
except ImportError as exc:  # pragma: no cover - surface import errors fast.
    msg = (
        "Failed to import the binstatcuda native extension. "
        "Ensure the package is built with a CUDA-enabled toolchain."
    )
    raise ImportError(msg) from exc

__all__ = [
    "__version__",
    "binned_statistic",
    "binned_statistic_2d",
    "device_count",
    "histogram",
    "histogram2d",
]

ArrayLike = npt.ArrayLike
UIntArray = npt.NDArray[np.uint64]
FloatArray = npt.NDArray[np.float32]
UInt32Array = npt.NDArray[np.uint32]


def _require_float32_c_array(value: ArrayLike, name: str) -> np.ndarray:
    """
    Validate that ``value`` is a C-contiguous float32 ndarray.

    Args:
        value (ArrayLike): Array to validate.
        name (str): Argument name for error messaging.

    Returns:
        numpy.ndarray: The validated array (no copy performed).

    Raises:
        TypeError: If the value is not a NumPy array of dtype float32.
        ValueError: If the array is not C-contiguous.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray with dtype float32.")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must have dtype numpy.float32.")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return value


def _ensure_float32_c_array(value: ArrayLike, name: str) -> np.ndarray:
    """
    Convert ``value`` into a float32 C-contiguous ndarray.

    Args:
        value (ArrayLike): Array to coerce.
        name (str): Argument name for error messaging.

    Returns:
        numpy.ndarray: C-contiguous float32 array (copy allowed).

    Raises:
        TypeError: If ``value`` cannot be converted to float32.
    """
    try:
        array = np.ascontiguousarray(value, dtype=np.float32)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be convertible to numpy.float32."
        ) from exc
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype numpy.float32.")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return array


def device_count() -> int:
    """
    Report the number of CUDA devices visible to the runtime.

    Returns:
        int: GPU count, or -1 if CUDA reports an error.

    Examples:
        >>> isinstance(device_count(), int)
        True
    """
    return _core.device_count()


def histogram(samples: ArrayLike, edges: ArrayLike) -> UIntArray:
    """
    Bin samples into 1D histogram bins on the GPU.

    Args:
        samples (ArrayLike): One-dimensional sample values.
        edges (ArrayLike): One-dimensional, strictly increasing bin edges.

    Returns:
        numpy.ndarray: Bin counts as uint64 with shape (len(edges) - 1,).

    Raises:
        TypeError: If samples are not float32 NumPy arrays or edges cannot be
            converted to float32.
        ValueError: If the provided arrays are not C-contiguous.

    Examples:
        >>> import numpy as np
        >>> sample = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> histogram(sample, edges).tolist()
        [2, 1]
    """
    samples_arr = _require_float32_c_array(samples, "samples")
    edges_arr = _ensure_float32_c_array(edges, "edges")
    return _core.histogram(samples_arr, edges_arr)


def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
) -> UIntArray:
    """
    Bin paired samples into 2D histogram bins on the GPU.

    Args:
        x (ArrayLike): One-dimensional x-coordinates.
        y (ArrayLike): One-dimensional y-coordinates (same length as ``x``).
        x_edges (ArrayLike): Bin edges for the x-axis.
        y_edges (ArrayLike): Bin edges for the y-axis.

    Returns:
        numpy.ndarray: Bin counts as uint64 with shape
        ``(len(x_edges) - 1, len(y_edges) - 1)``.

    Raises:
        TypeError: If samples are not float32 NumPy arrays or edges cannot be
            converted to float32.
        ValueError: If the provided arrays are not C-contiguous.

    Examples:
        >>> import numpy as np
        >>> xs = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> ys = np.array([0.2, 0.6, 0.8], dtype=np.float32)
        >>> x_edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> y_edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> histogram2d(xs, ys, x_edges, y_edges).tolist()
        [[1, 0], [1, 1]]
    """
    x_arr = _require_float32_c_array(x, "x")
    y_arr = _require_float32_c_array(y, "y")
    x_edges_arr = _ensure_float32_c_array(x_edges, "x_edges")
    y_edges_arr = _ensure_float32_c_array(y_edges, "y_edges")
    return _core.histogram2d(x_arr, y_arr, x_edges_arr, y_edges_arr)


def binned_statistic(
    samples: ArrayLike,
    values: ArrayLike,
    edges: ArrayLike,
) -> tuple[UIntArray, FloatArray, UInt32Array]:
    """
    Compute 1D bin counts and sums on the GPU.

    Args:
        samples (ArrayLike): Sample coordinates.
        values (ArrayLike): Values associated with each sample.
        edges (ArrayLike): Histogram bin edges.

    Returns:
        tuple: ``(counts, sums, bin_numbers)`` where counts is uint64 with
        shape ``(len(edges) - 1,)``, sums is float32 with the same shape, and
        bin_numbers is uint32 with shape ``(len(samples),)``.

    Raises:
        TypeError: If inputs are not C-contiguous float32 arrays.
        ValueError: If array dimensionalities or sizes do not match.

    Examples:
        >>> import numpy as np
        >>> coords = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> counts, sums, bins = binned_statistic(coords, vals, edges)
        >>> counts.tolist()
        [2, 1]
        >>> sums.tolist()
        [3.0, 3.0]
        >>> bins.tolist()
        [0, 0, 1]
    """
    samples_arr = _require_float32_c_array(samples, "samples")
    values_arr = _require_float32_c_array(values, "values")
    edges_arr = _ensure_float32_c_array(edges, "edges")

    if samples_arr.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if values_arr.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if samples_arr.shape[0] != values_arr.shape[0]:
        raise ValueError("samples and values must have the same length.")

    counts, sums, bin_numbers = _core.binned_statistic(
        samples_arr,
        values_arr,
        edges_arr,
    )
    return counts, sums, bin_numbers


def binned_statistic_2d(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
) -> tuple[UIntArray, FloatArray, UInt32Array, UInt32Array]:
    """
    Compute 2D bin counts and sums on the GPU.

    Args:
        x (ArrayLike): Sample x-coordinates.
        y (ArrayLike): Sample y-coordinates.
        values (ArrayLike): Values associated with each sample.
        x_edges (ArrayLike): Histogram bin edges along the x-axis.
        y_edges (ArrayLike): Histogram bin edges along the y-axis.

    Returns:
        tuple: ``(counts, sums, bin_numbers_x, bin_numbers_y)`` where counts
        and sums have shape ``(len(x_edges) - 1, len(y_edges) - 1)``, and the
        bin number arrays have shape ``(len(x),)``.

    Raises:
        TypeError: If inputs are not C-contiguous float32 arrays.
        ValueError: If dimensionalities or lengths do not match.

    Examples:
        >>> import numpy as np
        >>> xs = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> ys = np.array([0.2, 0.6, 0.8], dtype=np.float32)
        >>> vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> x_edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> y_edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> out = binned_statistic_2d(xs, ys, vals, x_edges, y_edges)
        >>> out[0].tolist()
        [[1, 0], [1, 1]]
        >>> out[1].tolist()
        [[1.0, 0.0], [2.0, 3.0]]
    """
    x_arr = _require_float32_c_array(x, "x")
    y_arr = _require_float32_c_array(y, "y")
    values_arr = _require_float32_c_array(values, "values")
    x_edges_arr = _ensure_float32_c_array(x_edges, "x_edges")
    y_edges_arr = _ensure_float32_c_array(y_edges, "y_edges")

    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if values_arr.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")
    if values_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("values must match the length of x and y.")

    counts, sums, bin_numbers_x, bin_numbers_y = _core.binned_statistic_2d(
        x_arr,
        y_arr,
        values_arr,
        x_edges_arr,
        y_edges_arr,
    )
    return counts, sums, bin_numbers_x, bin_numbers_y


def __getattr__(name: str) -> str:
    if name == "__version__":
        return metadata.version("binstatcuda")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
