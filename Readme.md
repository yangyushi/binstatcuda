# Bin Stat with CUDA Acceleration

## Introduction

Implementing statistical tools that I commonly use in Python

- `np.histogram`
- `np.histogram2d`
- `scipy.stats.binned_statistic`
- `scipy.stats.binned_statistic2d`

The code is written in CUDA/C++ and ported to Python with pybind11.

## Install


```sh
pip install -e . 
```

## Usage


```python
import numpy as np
import binstatcuda as bsc

n_sample = int(1e7)
x_edges = np.linspace(0.0, 1.0, 128, dtype=np.float32)
y_edges = np.linspace(0.0, 1.0, 64, dtype=np.float32)

x = np.random.random(n_sample).astype(np.float32)
y = np.random.random(n_sample).astype(np.float32)

hist_1d = bsc.histogram(x, bins=x_edges)
hist_2d = bsc.histogram2d(x, y, bins=(x_edges, y_edges))

values = np.random.random(n_sample).astype(np.float32)

binned_mean = bsc.binned_statistic(
    x=x,
    values=values,
    statistic='mean',
    bins=x_edges,
)

binned_mean_2d = bsc.binned_statistic(
    x=x,
    y=y,
    values=values,
    statistic='mean',
    bins=(x_edges, y_edges)
)
```

## Note


- all float point arithematic are performed in `float`, inplicit casting happens
- all counting are performed in `unsigned long long`


## TODO


Implement more functions

- `np.histogramdd`
- `scipy.stats.binned_statisticdd`
