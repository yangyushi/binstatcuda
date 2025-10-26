import numpy as np
import numpy.testing as npt

import binstatcuda as bsc


def _reference_binned_statistic(
    samples: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

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


def _reference_binned_statistic_2d(
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
    x_edges = np.asarray(x_edges, dtype=np.float32)
    y_edges = np.asarray(y_edges, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

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
    result = bsc.histogram2d(xs, ys, x_edges, y_edges)

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

    result = bsc.histogram2d(xs, ys, x_edges, y_edges)
    assert result.sum() == xs.size


def test_binned_statistic_matches_reference() -> None:
    rng = np.random.default_rng(91011)
    samples = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
    values = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
    edges = np.linspace(-3.0, 3.0, num=65, dtype=np.float32)

    counts, sums, bin_numbers = bsc.binned_statistic(samples, values, edges)
    ref_counts, ref_sums, ref_bins = _reference_binned_statistic(
        samples,
        values,
        edges,
    )

    assert counts.dtype == np.uint64
    assert sums.dtype == np.float32
    assert bin_numbers.dtype == np.uint32

    npt.assert_array_equal(counts, ref_counts)
    npt.assert_allclose(sums, ref_sums, rtol=1e-5, atol=1e-5)
    npt.assert_array_equal(bin_numbers, ref_bins)


def test_binned_statistic_length_mismatch() -> None:
    samples = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
    values = np.linspace(1.0, 2.0, num=7, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, num=4, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.binned_statistic(samples, values, edges)


def test_binned_statistic2d_matches_reference() -> None:
    rng = np.random.default_rng(121314)
    xs = rng.normal(0.0, 1.0, size=1024).astype(np.float32)
    ys = rng.uniform(-2.0, 2.0, size=1024).astype(np.float32)
    values = rng.normal(0.0, 1.0, size=1024).astype(np.float32)

    x_edges = np.linspace(-3.0, 3.0, num=33, dtype=np.float32)
    y_edges = np.linspace(-2.0, 2.0, num=25, dtype=np.float32)

    counts, sums, bins_x, bins_y = bsc.binned_statistic_2d(
        xs,
        ys,
        values,
        x_edges,
        y_edges,
    )
    ref_counts, ref_sums, ref_bins_x, ref_bins_y = (
        _reference_binned_statistic_2d(
            xs,
            ys,
            values,
            x_edges,
            y_edges,
        )
    )

    assert counts.dtype == np.uint64
    assert sums.dtype == np.float32
    assert bins_x.dtype == np.uint32
    assert bins_y.dtype == np.uint32

    npt.assert_array_equal(counts, ref_counts)
    npt.assert_allclose(sums, ref_sums, rtol=1e-5, atol=1e-5)
    npt.assert_array_equal(bins_x, ref_bins_x)
    npt.assert_array_equal(bins_y, ref_bins_y)


def test_binned_statistic2d_length_mismatch() -> None:
    xs = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    values = np.linspace(-1.0, 1.0, num=7, dtype=np.float32)
    x_edges = np.linspace(-1.0, 1.0, num=4, dtype=np.float32)
    y_edges = np.linspace(-1.0, 1.0, num=5, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        bsc.binned_statistic_2d(xs, ys, values, x_edges, y_edges)
