"""
CUDA-accelerated binned statistics.

The current implementation bootstraps the native extension and exposes
device discovery helpers that validate the CUDA toolchain.
"""

from importlib import metadata

try:
    from . import _core
except ImportError as exc:  # pragma: no cover - import error should surface quickly.
    msg = (
        "Failed to import the binstatcuda native extension. "
        "Ensure the package is built with a CUDA-enabled toolchain."
    )
    raise ImportError(msg) from exc

__all__ = [
    "__version__",
    "device_count",
]


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


def __getattr__(name: str) -> str:
    if name == "__version__":
        return metadata.version("binstatcuda")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
