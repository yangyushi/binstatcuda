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
    statistic: str,
) -> np.ndarray:
    result, _, _ = scipy.stats.binned_statistic(
        samples,
        values,
        statistic=statistic,
        bins=edges,
    )
    return result


def cpu_binned_statistic_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    statistic: str,
) -> np.ndarray:
    result, _, _, _ = scipy.stats.binned_statistic_2d(
        xs,
        ys,
        values,
        statistic=statistic,
        bins=[x_edges, y_edges],
    )
    return result


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
        (x_edges_2d, y_edges_2d),
    )

    benchmark(
        "cpu.binned_statistic",
        cpu_binned_statistic_1d,
        x_1d,
        values_1d,
        bin_edge_1d,
        "mean",
    )
    benchmark(
        "bsc.binned_statistic",
        bsc.binned_statistic,
        x_1d,
        values_1d,
        "mean",
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
        "mean",
    )
    benchmark(
        "bsc.binned_statistic2d",
        bsc.binned_statistic_2d,
        x_2d[0],
        x_2d[1],
        values_2d,
        "mean",
        (x_edges_2d, y_edges_2d),
    )


if __name__ == "__main__":
    main()
