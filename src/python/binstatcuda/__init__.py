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

_VALID_STATISTICS: frozenset[str] = frozenset(
    {
        "count",
        "sum",
        "mean",
        "std",
    }
)


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


def _normalize_statistic(statistic: str) -> str:
    """
    Normalize and validate the statistic name.

    Args:
        statistic (str): Name of the statistic to compute.

    Returns:
        str: Lower-case statistic name.

    Raises:
        TypeError: If ``statistic`` is not a string.
        ValueError: If the statistic is unsupported.
    """
    if not isinstance(statistic, str):
        raise TypeError("statistic must be a string.")
    normalized = statistic.lower()
    if normalized not in _VALID_STATISTICS:
        raise ValueError(
            f"Unsupported statistic {statistic!r}. "
            f"Valid options are: {sorted(_VALID_STATISTICS)}."
        )
    return normalized


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


def histogram(a: ArrayLike, bins: ArrayLike) -> UIntArray:
    """
    Bin samples into 1D histogram bins on the GPU.

    Args:
        a (ArrayLike): One-dimensional sample values.
        bins (ArrayLike): One-dimensional NumPy array of bin edges.

    Returns:
        numpy.ndarray: Bin counts as uint64 with shape (len(bins) - 1,).

    Raises:
        TypeError: If samples are not float32 NumPy arrays or bins are not
            NumPy arrays of edges convertible to float32.
        ValueError: If the provided arrays are not C-contiguous.

    Examples:
        >>> import numpy as np
        >>> sample = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> histogram(sample, edges).tolist()
        [2, 1]
    """
    samples_arr = _require_float32_c_array(a, "a")
    if not isinstance(bins, np.ndarray):
        raise TypeError("bins must be a numpy.ndarray of bin edges.")
    edges_arr = _ensure_float32_c_array(bins, "bins")
    return _core.histogram(samples_arr, edges_arr)


def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: tuple[ArrayLike, ArrayLike],
) -> UIntArray:
    """
    Bin paired samples into 2D histogram bins on the GPU.

    Args:
        x (ArrayLike): One-dimensional x-coordinates.
        y (ArrayLike): One-dimensional y-coordinates (same length as ``x``).
        bins (tuple[ArrayLike, ArrayLike]): Tuple of NumPy arrays containing
            the x- and y-axis bin edges.

    Returns:
        numpy.ndarray: Bin counts as uint64 with shape
        ``(len(bins[0]) - 1, len(bins[1]) - 1)``.

    Raises:
        TypeError: If samples are not float32 NumPy arrays or bins are not
            NumPy arrays of edges convertible to float32.
        ValueError: If the provided arrays are not C-contiguous.

    Examples:
        >>> import numpy as np
        >>> xs = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> ys = np.array([0.2, 0.6, 0.8], dtype=np.float32)
        >>> x_edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> y_edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> histogram2d(xs, ys, (x_edges, y_edges)).tolist()
        [[1, 0], [1, 1]]
    """
    x_arr = _require_float32_c_array(x, "x")
    y_arr = _require_float32_c_array(y, "y")
    if not isinstance(bins, tuple) or len(bins) != 2:
        raise TypeError(
            "bins must be a tuple of two numpy.ndarray bin edge arrays."
        )
    x_bins, y_bins = bins
    if not isinstance(x_bins, np.ndarray) or not isinstance(y_bins, np.ndarray):
        raise TypeError(
            "bins entries must each be numpy.ndarray of bin edges."
        )
    x_edges_arr = _ensure_float32_c_array(x_bins, "bins[0]")
    y_edges_arr = _ensure_float32_c_array(y_bins, "bins[1]")
    return _core.histogram2d(x_arr, y_arr, x_edges_arr, y_edges_arr)


def binned_statistic(
    x: ArrayLike,
    values: ArrayLike,
    statistic: str = "mean",
    bins: ArrayLike | None = None,
) -> FloatArray:
    """
    Compute a 1D binned statistic on the GPU.

    Args:
        x (ArrayLike): Sample coordinates.
        values (ArrayLike): Values associated with each coordinate.
        statistic (str): Statistic to compute (``count``, ``sum``, ``mean``,
            or ``std``).
        bins (ArrayLike | None): Histogram bin edges. ``None`` is unsupported
            and will raise ``ValueError``.

    Returns:
        numpy.ndarray: The per-bin statistic as float32 with shape
        ``(len(bins) - 1,)``.

    Raises:
        TypeError: If inputs are not C-contiguous float32 arrays or statistic
            is not a string.
        ValueError: If array dimensionalities or sizes do not match, if bins
            are missing, or if the statistic is unsupported.

    Examples:
        >>> import numpy as np
        >>> coords = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> edges = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> binned_statistic(coords, vals, statistic="mean", bins=edges).tolist()
        [1.5, 3.0]
    """
    if bins is None:
        raise ValueError("bins must be provided.")

    x_arr = _require_float32_c_array(x, "x")
    values_arr = _require_float32_c_array(values, "values")
    if not isinstance(bins, np.ndarray):
        raise TypeError("bins must be a numpy.ndarray of bin edges.")
    bins_arr = _ensure_float32_c_array(bins, "bins")
    name = _normalize_statistic(statistic)

    if x_arr.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if values_arr.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if x_arr.shape[0] != values_arr.shape[0]:
        raise ValueError("x and values must have the same length.")

    return _core.binned_statistic(
        x_arr,
        values_arr,
        bins_arr,
        name,
    )


def binned_statistic_2d(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    statistic: str = "mean",
    bins: tuple[ArrayLike, ArrayLike] | None = None,
) -> FloatArray:
    """
    Compute a 2D binned statistic on the GPU.

    Args:
        x (ArrayLike): Sample x-coordinates.
        y (ArrayLike): Sample y-coordinates (must match ``x`` length).
        values (ArrayLike): Values associated with each coordinate pair.
        statistic (str): Statistic to compute (``count``, ``sum``, ``mean``,
            ``std``, ``median``, ``min``, or ``max``).
        bins (tuple[ArrayLike, ArrayLike] | None): Tuple containing the x- and
            y-axis bin edges. ``None`` is unsupported and will raise
            ``ValueError``.

    Returns:
        numpy.ndarray: The per-bin statistic as float32 with shape
        ``(len(bins[0]) - 1, len(bins[1]) - 1)``.

    Raises:
        TypeError: If inputs are not C-contiguous float32 arrays or statistic
            is not a string.
        ValueError: If dimensionalities, lengths, or statistic name are
            invalid, or if bins are not supplied.

    Examples:
        >>> import numpy as np
        >>> xs = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        >>> ys = np.array([0.2, 0.6, 0.8], dtype=np.float32)
        >>> vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> xb = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> yb = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        >>> binned_statistic_2d(xs, ys, vals, statistic="sum", bins=(xb, yb)).tolist()
        [[1.0, 0.0], [2.0, 3.0]]
    """
    if bins is None:
        raise ValueError("bins must be provided.")
    if not isinstance(bins, tuple) or len(bins) != 2:
        raise TypeError(
            "bins must be a tuple of two numpy.ndarray bin edge arrays."
        )

    x_arr = _require_float32_c_array(x, "x")
    y_arr = _require_float32_c_array(y, "y")
    values_arr = _require_float32_c_array(values, "values")
    name = _normalize_statistic(statistic)

    x_bins, y_bins = bins
    if not isinstance(x_bins, np.ndarray) or not isinstance(y_bins, np.ndarray):
        raise TypeError(
            "bins entries must each be numpy.ndarray of bin edges."
        )
    x_edges_arr = _ensure_float32_c_array(x_bins, "bins[0]")
    y_edges_arr = _ensure_float32_c_array(y_bins, "bins[1]")

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

    return _core.binned_statistic_2d(
        x_arr,
        y_arr,
        values_arr,
        x_edges_arr,
        y_edges_arr,
        name,
    )


def __getattr__(name: str) -> str:
    if name == "__version__":
        return metadata.version("binstatcuda")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
