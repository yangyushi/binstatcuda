import numpy as np
import numpy.testing as npt

import binstatcuda as bsc


def _reference_binned_statistic(
    samples: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    edges = np.asarray(edges, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    counts, _ = np.histogram(samples, edges)
    sums, _ = np.histogram(samples, edges, weights=values)
    squares, _ = np.histogram(samples, edges, weights=values * values)

    bin_idx = np.searchsorted(edges, samples, side="right") - 1
    last_edge = edges.size - 1
    bin_idx[bin_idx == last_edge] = last_edge - 1
    invalid = (samples < edges[0]) | (samples > edges[-1])
    bin_idx[invalid] = -1

    bin_count = edges.size - 1
    result: dict[str, np.ndarray] = {}

    counts_f = counts.astype(np.float32)
    sums_f = sums.astype(np.float32)
    result["count"] = counts_f
    result["sum"] = sums_f

    mean = np.full(bin_count, np.nan, dtype=np.float64)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]
    mean = mean.astype(np.float32)
    result["mean"] = mean

    std = np.full(bin_count, np.nan, dtype=np.float64)
    if nonzero.any():
        variance = np.zeros(bin_count, dtype=np.float64)
        variance[nonzero] = squares[nonzero] / counts[nonzero] - (
            sums[nonzero] / counts[nonzero]
        ) ** 2
        variance[variance < 0.0] = 0.0
        std[nonzero] = np.sqrt(variance[nonzero])
    result["std"] = std.astype(np.float32)

    return result


def _reference_binned_statistic_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    x_edges = np.asarray(x_edges, dtype=np.float64)
    y_edges = np.asarray(y_edges, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    sums, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=values)
    squares, _, _ = np.histogram2d(
        xs,
        ys,
        bins=[x_edges, y_edges],
        weights=values * values,
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

    y_bins = y_edges.size - 1
    bin_flat = bin_x * y_bins + bin_y

    x_bins = x_edges.size - 1
    total_bins = x_bins * y_bins

    counts_flat = counts.reshape(-1).astype(np.float32)
    sums_flat = sums.reshape(-1).astype(np.float32)

    result: dict[str, np.ndarray] = {}
    result["count"] = counts_flat.reshape(x_bins, y_bins)
    result["sum"] = sums_flat.reshape(x_bins, y_bins)

    mean = np.full(total_bins, np.nan, dtype=np.float64)
    nonzero = counts_flat > 0
    mean[nonzero] = sums_flat[nonzero] / counts_flat[nonzero]

    std = np.full(total_bins, np.nan, dtype=np.float64)
    if nonzero.any():
        variance = squares.reshape(-1)
        variance[nonzero] = (
            variance[nonzero] / counts_flat[nonzero] - mean[nonzero] ** 2
        )
        variance[variance < 0.0] = 0.0
        std[nonzero] = np.sqrt(variance[nonzero])

    result["mean"] = mean.reshape(x_bins, y_bins).astype(np.float32)
    result["std"] = std.reshape(x_bins, y_bins).astype(np.float32)
    return result


def test_histogram_matches_numpy() -> None:
    rng = np.random.default_rng(1234)
    samples = rng.random(4096, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, num=33, dtype=np.float32)

    expected, _ = np.histogram(samples, edges)
    result = bsc.histogram(samples, edges)

    assert result.dtype == np.uint64
    npt.assert_array_equal(result, expected.astype(np.uint64))


def test_histogram2d_matches_numpy() -> None:
    rng = np.random.default_rng(5678)
    xs = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
    ys = rng.uniform(-2.0, 2.0, size=2048).astype(np.float32)

    x_edges = np.linspace(-3.0, 3.0, num=25, dtype=np.float32)
    y_edges = np.linspace(-2.0, 2.0, num=17, dtype=np.float32)

    expected, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    result = bsc.histogram2d(xs, ys, (x_edges, y_edges))

    assert result.dtype == np.uint64
    assert result.shape == expected.shape
    npt.assert_array_equal(result, expected.astype(np.uint64))


def test_histogram_rejects_wrong_dtype() -> None:
    samples = np.arange(4, dtype=np.float64)
    edges = np.linspace(0.0, 4.0, num=5, dtype=np.float32)

    with np.testing.assert_raises(TypeError):
        bsc.histogram(samples, edges)


def test_histogram_rejects_non_contiguous() -> None:
    samples = np.arange(8, dtype=np.float32)[::2]
    edges = np.linspace(0.0, 1.0, num=3, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.histogram(samples, edges)


def test_histogram_requires_array_bins() -> None:
    samples = np.linspace(0.0, 1.0, num=4, dtype=np.float32)
    with np.testing.assert_raises(TypeError):
        bsc.histogram(samples, [0.0, 0.5, 1.0])


def test_histogram_accepts_float64_edges() -> None:
    samples = np.linspace(0.0, 1.0, num=10, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, num=5, dtype=np.float64)

    result = bsc.histogram(samples, edges)
    assert result.sum() == samples.size


def test_histogram2d_accepts_float64_edges() -> None:
    xs = np.linspace(-1.0, 1.0, num=10, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=10, dtype=np.float32)
    x_edges = np.linspace(-1.0, 1.0, num=4, dtype=np.float64)
    y_edges = np.linspace(-1.0, 1.0, num=5, dtype=np.float64)

    result = bsc.histogram2d(xs, ys, (x_edges, y_edges))
    assert result.sum() == xs.size


def test_histogram2d_requires_tuple_bins() -> None:
    xs = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    edges = np.linspace(-1.0, 1.0, num=3, dtype=np.float32)

    with np.testing.assert_raises(TypeError):
        bsc.histogram2d(xs, ys, edges)  # type: ignore[arg-type]

    with np.testing.assert_raises(TypeError):
        bsc.histogram2d(
            xs,
            ys,
            (edges, [-1.0, 0.0, 1.0]),  # type: ignore[arg-type]
        )


def test_binned_statistic_matches_reference() -> None:
    rng = np.random.default_rng(91011)
    samples = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
    values = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
    edges = np.linspace(-3.0, 3.0, num=65, dtype=np.float32)

    references = _reference_binned_statistic(
        samples,
        values,
        edges,
    )

    for statistic, expected in references.items():
        result = bsc.binned_statistic(
            samples,
            values,
            statistic=statistic,
            bins=edges,
        )
        assert result.dtype == np.float32
        npt.assert_allclose(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            equal_nan=True,
        )

    upper = bsc.binned_statistic(
        samples,
        values,
        statistic="MEAN",
        bins=edges,
    )
    npt.assert_allclose(
        upper,
        references["mean"],
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )


def test_binned_statistic_length_mismatch() -> None:
    samples = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
    values = np.linspace(1.0, 2.0, num=7, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, num=4, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.binned_statistic(
            samples,
            values,
            statistic="mean",
            bins=edges,
        )


def test_binned_statistic_requires_numpy_bins() -> None:
    samples = np.linspace(0.0, 1.0, num=4, dtype=np.float32)
    values = np.linspace(0.0, 1.0, num=4, dtype=np.float32)

    with np.testing.assert_raises(TypeError):
        bsc.binned_statistic(
            samples,
            values,
            statistic="mean",
            bins=[0.0, 0.5, 1.0],  # type: ignore[arg-type]
        )


def test_binned_statistic_invalid_statistic() -> None:
    samples = np.linspace(0.0, 1.0, num=4, dtype=np.float32)
    values = np.linspace(0.0, 1.0, num=4, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, num=3, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.binned_statistic(samples, values, statistic="mode", bins=edges)


def test_binned_statistic2d_matches_reference() -> None:
    rng = np.random.default_rng(121314)
    xs = rng.normal(0.0, 1.0, size=1024).astype(np.float32)
    ys = rng.uniform(-2.0, 2.0, size=1024).astype(np.float32)
    values = rng.normal(0.0, 1.0, size=1024).astype(np.float32)

    x_edges = np.linspace(-3.0, 3.0, num=33, dtype=np.float32)
    y_edges = np.linspace(-2.0, 2.0, num=25, dtype=np.float32)

    references = _reference_binned_statistic_2d(
        xs,
        ys,
        values,
        x_edges,
        y_edges,
    )

    for statistic, expected in references.items():
        result = bsc.binned_statistic_2d(
            xs,
            ys,
            values,
            statistic=statistic,
            bins=(x_edges, y_edges),
        )
        assert result.dtype == np.float32
        npt.assert_allclose(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            equal_nan=True,
        )


def test_binned_statistic2d_length_mismatch() -> None:
    xs = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    values = np.linspace(-1.0, 1.0, num=7, dtype=np.float32)
    x_edges = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    y_edges = np.linspace(-1.0, 1.0, num=5, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.binned_statistic_2d(
            xs,
            ys,
            values,
            statistic="mean",
            bins=(x_edges, y_edges),
        )


def test_binned_statistic2d_requires_numpy_bins() -> None:
    xs = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    values = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    edges = np.linspace(-1.0, 1.0, num=3, dtype=np.float32)

    with np.testing.assert_raises(TypeError):
        bsc.binned_statistic_2d(
            xs,
            ys,
            values,
            statistic="mean",
            bins=(edges, [-1.0, 0.0, 1.0]),  # type: ignore[arg-type]
        )


def test_binned_statistic2d_invalid_statistic() -> None:
    xs = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    values = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    edges = np.linspace(-1.0, 1.0, num=3, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.binned_statistic_2d(
            xs,
            ys,
            values,
            statistic="mode",
            bins=(edges, edges),
        )
