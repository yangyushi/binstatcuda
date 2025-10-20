# Agents.md

## Environment

- The Python that you have access to via terminal DO contain all necessary packages. Use `python` directly.
- The compilers that you have access to are configured correctly. Use `gcc/g++/nvcc` directly.
- NEVER install things on the computer.
- NEVER download things to the computer.
- You are using Linux, but try to make your code cross-platform.

## Code Style

### General

- DO follow the existing coding style in the codebase.
- DO NOT hardcode credentials or tokens.
- DO keep dependencies to a minimum.
- DO follow the latest developments of the programming language.

### Python

#### Style

- DO follow PEP8.
    - with maximum line width of 80 characters.
    - DO NOT include trailing whitespaces.
- DO follow PEP8 naming
    - `snake_case` for functions and variables
    - `CapWords` for classes
    - `UPPER_SNAKE` for constants.
- DO use 4 spaces per indentation level.
- DO use f-strings for formatting, unless compatibility with Python < 3.6 is required.
- DO use linting tools (flake8).
- DO use prefix `is_`, `has_`, `can_` for boolean variables and functions.

Example

```python
DEFAULT_PATCH_SIZE = 16


class Image:
    ...


def get_image_patches(
    image: Image,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> Sequence[Image]:
    ...
```

#### Comment

- DO NOT abuse inline comments.
- DO use `# TODO:` to mark pending tasks.
- DO comment critical information like array shapes.
- DO prefer comments that explain *why*, rather than *what*.

Example

```python
x = x.transpose(0, 2, 1)  # (B, C, L) -> (B, L, C)
z = x.reshape(B, L // P, P, C)  # reshape to patches for averaging
y = z.mean(axis=2)
```

#### Docstring

- DO write good documentation.
- DO use the Google-style docstring, DO NOT mix Google/Numpy/ReST styles.
- DO use the same section names everywhere (`Args`, `Returns`, `Raises`, `Examples`)
- DO include concise and self contained examples that,
    - can be executed by `doctest`,
    - do not depend on extra library,
    - do not incur side effects (file io or network download).

Example

```python
def add_int(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: the sum of two integers

    Raises:
        ValueError: if either `a` or `b` is not an integer

    Examples:
        >>> add_int(2, 3)
        5
        >>> add_int(-1, 1)
        0
        >>> add_int(2.5, 3)
        Traceback (most recent call last):
            ...
        ValueError: Both a and b must be integers.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Both a and b must be integers.")

    return a + b
```

#### Dependencies

- DO follow import order: stdlib ➜ third-party ➜ local.
- DO NOT use wildcard imports (`from module import *`).
- DO NOT import unused modules or objects.
- DO prefer smaller, well‑maintained libs; justify heavy deps.

Example

```python
import os
from pathlib import Path

import numpy as np
from scipy import stats

from custom_model import Model
```

#### Exceptions

- DO NOT use bare `except` clauses.
- DO NOT use exceptions for flow control.
- DO NOT suppress exceptions without handling them.
- DO use informative messages to make error actionable.
- DO prefer specific exceptions (`NPUMemError_BadAlloc`) over general ones (`RuntimeError`).

Example

```python
class ImageError(ValueError): ...
class ImageError_InvalidShape(ImageError): ...

def process_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ImageError_InvalidShape(
            f"Expected 3D image, got shape {image.shape}"
        )
    ...
    return processed_image


def main():
    try:
        ...
    except ImageError as e:
        print(f"Image processing failed: {e}")
```

#### Typing

- DO use the type hints (PEP 484) consistently.
- DO use abstract types (`Mapping`, `Sequence`) to express intent.
- DO use `Optional[T]` or `T | None` instead of `Union[T, None]`.
- DO use tools like `mypy` for static check.
- DO NOT use `Any`.
- DO NOT suppress type check warnings.

Example

```python
from typing import Mapping, Sequence, Optional

def get_image_metadata(
    image_id: str,
    metadata_db: Mapping[str, Mapping[str, str]],
) -> Optional[Mapping[str, str]]:
    ...
```

#### OOP

- DO prefer composition over inheritance.
- DO write single-purpose and orthogonal classes
- DO prefer interfaces via `Protocol` over of deep hierarchies.
- DO use `_` prefix hint a method being non-public.
- DO use `__` prefix to hide a method if name mangling is absolutely necessary.

Example

```python
from typing import Protocol


class ImageLoader(Protocol):
    def load(self, path: Path) -> Image:
        ...


class PNGLoader:
    def load(self, path: Path) -> Image:
        ...

    def _validate(self, image: Image) -> bool:
        ...


class TIFFLoader:
    def load(self, path: Path) -> Image:
        ...

    def _validate(self, image: Image) -> bool:
        ...


def load_image(path: Path, loader: ImageLoader) -> Image:
    return loader.load(path)
```


#### Data Structures

- DO prefer built-in data structures (`list`, `dict`, `set`, `tuple`).
- DO use `Enum` to represent a fixed set of constants.
- DO use `dataclass` to hold tightly coupled items together, use `slots=True` and `frozen=True` where applicable.

Example

```python
from dataclasses import dataclass
from enum import Enum

class ImageFormat(Enum):
    PNG = "png"
    JPEG = "jpeg"
    BMP = "bmp"

@dataclass(slots=True, frozen=True)
class ImageMetadata:
    width: int
    height: int
    format: ImageFormat
```

#### Test

- DO measure the coverage of the code.
- DO use `pytest` as primary testing tools.
- DO use `doctests` for illustrative examples in docstrings.
- DO use `tests/unit/` and `tests/integration/` in the project.

### C++

#### Style

* DO use C++17 as the baseline (C++11 allowed in legacy modules; avoid mixing in one TU).
* DO use 4 spaces per indentation level; DO NOT use tabs.
* DO use K&R braces (`int f() { ... }`).
* DO place one declaration per line.
* DO use `snake_case` for functions and variables.
* DO use `CapWords` for classes, structs, and enum types.
* DO use `UPPER_SNAKE` for constants and macros.
* DO use `lower_snake` for namespaces.
* DO mark single-argument constructors `explicit`.
* DO mark overrides with `override` and non-throwing functions `noexcept` where correct.
* DO prefer `using` over `typedef`.
* DO use `auto` when it **improves** readability (obvious initializer), NOT to hide types.
* DO use `const` and `constexpr` aggressively.
* DO use RAII and smart pointers (`std::unique_ptr` by default).

#### Comments

* DO NOT abuse inline comments.
* DO use `// TODO:` and `// FIXME:` for actionable items.
* DO prefer comments that explain *why*, not *what*.
* DO document tricky invariants, units, shapes, and complexity.

Example

```cpp
transpose_bcl_to_blc(tensor);  // to be cache-friendly

auto patches = reshape_to_patches(x, P);  // (B, L, C) -> (B, L/P, P, C)
```

#### Documentation

* DO write minimal, precise **Doxygen** comments on public APIs.
* DO keep types out of prose when they are obvious from the signature.
* DO use the same section names everywhere (`@param`, `@return`, `@retval`, `@throws`, `@example`).
* DO include concise, self-contained examples (no I/O, no network).

Example

```cpp
/**
 * Compute the mean value of fixed-length patches.
 *
 * @param x        Input flattened buffer (length L * C).
 * @param L        Sequence length (must be divisible by patch_len).
 * @param C        Channel count.
 * @param patch_len Patch length.
 * @return         Mean per patch (size (L/patch_len) * C).
 * @retval Err::InvalidShape if L % patch_len != 0.
 *
 * @example
 *   auto r = mean_patches(x, 8, 2, 4);
 *   if (!r) { /* handle error */ }
 */
```

#### Dependencies & Includes

* DO follow include order: **standard library ➜ third-party ➜ local**.
* DO use `#pragma once` in headers (or traditional include guards).
* DO NOT include what you don’t use; keep headers minimal.
* DO prefer small, well-maintained libs; justify heavy deps in the PR.

Example

```cpp
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "project/image.h"
#include "project/math/mean.h"
```

#### Error Handling

* DO prefer handle error explicitly over throwing exceptions.
* DO ensure every error path carries actionable context (what failed, expected vs. got).
* DO NOT use naked `catch(...)` unless rethrowing with context.
* DO NOT ignore return values.

#### Data Structures

- DO prefer STL containers
    - `std::vector`
    - `std::array`
    - `std::string`
    - `std::unordered_map`
- DO prefer `std::optional` (C++ 17) over sentinel values (`-1`, `nullptr`, ...).

#### Build

- DO use automatic building tools
    - use Makefile for small tools;
    - use CMake for libraries and larger builds.
- DO set the standard and strict warnings.
- DO keep one `CMakeLists.txt` per directory.


#### Testing

* DO use a test tool ([Catch2](https://github.com/catchorg/Catch2)) for unit and integration tests.
* DO keep tests deterministic; set seeds; avoid I/O and network in unit tests.
* DO structure tests under `tests/unit/` and `tests/integration/`.
* DO test both **Ok** and **Err** paths for `Result`.


#### Performance & Safety

* DO avoid dynamic allocation in hot paths; preallocate and reuse buffers.
* DO use `reserve`, `shrink_to_fit` judiciously; DO check iterator invalidation rules.
* DO use `[[nodiscard]]` on functions returning important status/values.
* DO measure with benchmarks; DO NOT micro-optimize without data.

Example

```cpp
[[nodiscard]] core::Result<std::vector<float>>
normalize_inplace(std::vector<float>& x) noexcept;
```

#### CUDA Interop

* DO isolate CUDA code in `.cu/.cuh` with thin C++ wrappers in `.cc/.h`.
* DO use RAII for device memory and streams; DO convert CUDA errors to your `Result`.
* DO prefer asynchronous kernels and avoid unnecessary host–device syncs.
* DO annotate with `__host__`, `__device__` only when needed; avoid leaking CUDA into non-CUDA headers.
* DO provide CPU fallbacks when feasible (for tests and portability).

Example

```cpp
#define CHECK_CUDA(expr)                                   \
  do {                                                     \
    cudaError_t _e = (expr);                               \
    if (_e != cudaSuccess)                                 \
      return core::Result<void>::err(core::Err::IoError,   \
          std::string("CUDA: ") + cudaGetErrorString(_e)); \
  } while (0)

core::Result<void> fill_device(float* dptr, size_t n, float v) {
  CHECK_CUDA(cudaMemset(dptr, 0, n * sizeof(float)));
  return core::Result<void>::ok({});
}
```
