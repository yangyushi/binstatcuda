import time

import numpy as np

import binstatcuda as bsc


N = int(1e8)
REPEAT = 5


x_1d = np.random.randn(N)
bin_edge_1d = np.linspace(-5, 5, 200)

x_2d = np.random.randn(2, N)
bin_edge_2d = (np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))


costs_np_1d = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    np.histogram(x_1d, bin_edge_1d)
    t1 = time.perf_counter()
    costs_np_1d.append(1000 * (t1 - t0))

name = "np.histogram"
data = costs_np_1d
print(f"{name: >14}: {np.mean(data): <4.2f} +/- {np.std(data):.2f} (ms)")


costs_bsc_1d = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    bsc.histogram(x_1d, bin_edge_1d)
    t1 = time.perf_counter()
    costs_bsc_1d.append(1000 * (t1 - t0))

name = "bsc.histogram"
data = costs_bsc_1d
print(f"{name: >14}: {np.mean(data): <4.2f} +/- {np.std(data):.2f} (ms)")


costs_np_2d = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    np.histogram2d(
        x_2d[0], x_2d[1], bins=bin_edge_2d
    )
    t1 = time.perf_counter()
    costs_np_2d.append(1000 * (t1 - t0))

name = "np.histogram2d"
data = costs_np_2d
print(f"{name: >14}: {np.mean(data): <4.2f} +/- {np.std(data):.2f} (ms)")


costs_bsc_1d = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    bsc.histogram_2d(
        x_2d[0], x_2d[1], bin_edge_2d[0], bin_edge_2d[1]
    )
    t1 = time.perf_counter()
    costs_bsc_1d.append(1000 * (t1 - t0))

name = "bsc.histogram2d"
data = costs_bsc_1d
print(f"{name: >14}: {np.mean(data): <4.2f} +/- {np.std(data):.2f} (ms)")
