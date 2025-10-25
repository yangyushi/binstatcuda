import numpy as np
import numpy.testing as npt

import binstatcuda as bsc


def test_histogram_matches_numpy() -> None:
    rng = np.random.default_rng(1234)
    samples = rng.random(4096, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, num=33, dtype=np.float32)

    expected, _ = np.histogram(samples, edges)
    result = bsc.histogram(samples, edges)

    assert result.dtype == np.uint64
    npt.assert_array_equal(result, expected.astype(np.uint64))


def test_histogram_2d_matches_numpy() -> None:
    rng = np.random.default_rng(5678)
    xs = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
    ys = rng.uniform(-2.0, 2.0, size=2048).astype(np.float32)

    x_edges = np.linspace(-3.0, 3.0, num=25, dtype=np.float32)
    y_edges = np.linspace(-2.0, 2.0, num=17, dtype=np.float32)

    expected, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    result = bsc.histogram_2d(xs, ys, x_edges, y_edges)

    assert result.dtype == np.uint64
    assert result.shape == expected.shape
    npt.assert_array_equal(result, expected.astype(np.uint64))
