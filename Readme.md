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
bin_edges = 

x = np.random.random(n_sample)
y = np.random.random(n_sample)

hist_1d = bsc.histogram(x, bin_edges)
hist_2d = bsc.histogram_2d(x, y, bin_edges)
```


## TODO


Implement more functions

- `np.histogramdd`
- `scipy.stats.binned_statisticdd`
