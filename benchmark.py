import time

import scipy
import numpy as np

import binstatcuda as bsc


N = int(1e7)
REPEAT = 5


def benchmark(
    label: str,
    func,
    *args,
    repeat: int = REPEAT,
) -> None:
    func(*args)
    durations: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        durations.append(1000.0 * (end - start))
    mean = float(np.mean(durations))
    std = float(np.std(durations))
    print(f"{label:>24}: {mean:6.2f} +/- {std:6.2f} ms")


def cpu_binned_statistic_1d(
    samples: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, _ = np.histogram(samples, edges)
    weighted, _ = np.histogram(samples, edges, weights=values)

    bin_idx = np.searchsorted(edges, samples, side="right") - 1
    last_edge = edges.size - 1
    bin_idx[bin_idx == last_edge] = last_edge - 1
    invalid = (samples < edges[0]) | (samples > edges[-1])
    bin_idx[invalid] = -1

    bin_numbers = np.where(bin_idx >= 0, bin_idx + 1, 0).astype(np.uint32)
    return (
        counts.astype(np.uint64),
        weighted.astype(np.float32),
        bin_numbers,
    )


def cpu_binned_statistic_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    weighted, _, _ = np.histogram2d(
        xs,
        ys,
        bins=[x_edges, y_edges],
        weights=values,
    )

    bin_x = np.searchsorted(x_edges, xs, side="right") - 1
    last_x = x_edges.size - 1
    bin_x[bin_x == last_x] = last_x - 1
    invalid_x = (xs < x_edges[0]) | (xs > x_edges[-1])
    bin_x[invalid_x] = -1

    bin_y = np.searchsorted(y_edges, ys, side="right") - 1
    last_y = y_edges.size - 1
    bin_y[bin_y == last_y] = last_y - 1
    invalid_y = (ys < y_edges[0]) | (ys > y_edges[-1])
    bin_y[invalid_y] = -1

    bin_numbers_x = np.where(bin_x >= 0, bin_x + 1, 0).astype(np.uint32)
    bin_numbers_y = np.where(bin_y >= 0, bin_y + 1, 0).astype(np.uint32)

    return (
        counts.astype(np.uint64),
        weighted.astype(np.float32),
        bin_numbers_x,
        bin_numbers_y,
    )


def main() -> None:
    rng = np.random.default_rng(42)

    x_1d = rng.normal(size=N).astype(np.float32)
    values_1d = rng.normal(size=N).astype(np.float32)
    bin_edge_1d = np.linspace(-5.0, 5.0, 200, dtype=np.float32)

    x_2d = rng.normal(size=(2, N)).astype(np.float32)
    values_2d = rng.normal(size=N).astype(np.float32)
    x_edges_2d = np.linspace(-5.0, 5.0, 100, dtype=np.float32)
    y_edges_2d = np.linspace(-5.0, 5.0, 100, dtype=np.float32)

    benchmark("np.histogram", np.histogram, x_1d, bin_edge_1d)
    benchmark("bsc.histogram", bsc.histogram, x_1d, bin_edge_1d)

    benchmark(
        "np.histogram2d",
        np.histogram2d,
        x_2d[0],
        x_2d[1],
        [x_edges_2d, y_edges_2d],
    )
    benchmark(
        "bsc.histogram2d",
        bsc.histogram2d,
        x_2d[0],
        x_2d[1],
        x_edges_2d,
        y_edges_2d,
    )

    benchmark(
        "cpu.binned_statistic",
        cpu_binned_statistic_1d,
        x_1d,
        values_1d,
        bin_edge_1d,
    )
    benchmark(
        "bsc.binned_statistic",
        bsc.binned_statistic,
        x_1d,
        values_1d,
        bin_edge_1d,
    )

    benchmark(
        "cpu.binned_statistic2d",
        cpu_binned_statistic_2d,
        x_2d[0],
        x_2d[1],
        values_2d,
        x_edges_2d,
        y_edges_2d,
    )
    benchmark(
        "bsc.binned_statistic2d",
        bsc.binned_statistic_2d,
        x_2d[0],
        x_2d[1],
        values_2d,
        x_edges_2d,
        y_edges_2d,
    )


if __name__ == "__main__":
    main()
