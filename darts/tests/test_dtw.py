#TODO follow test format

from darts.timeseries import TimeSeries
from darts.metrics import mape

import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import collections as mpc

from darts.utils.timeseries_generation import sine_timeseries
import timeit

import darts.dataprocessing.dtw as dtw

def test_periodic():
    input1 = [1, 1, 1, 1, 1.2, 1.4, 1.2, 1, 1, 1, 1, 1, 1, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1, 1, 1, 1]
    input2 = [0, 0, 0, 0, 0, 0.2, 0.4, 0.2, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0]
    input2 = [x+1 for x in input2]

    phase_shift = np.pi / 4

    size = 2**6
    freq = 1/size

    input1 = np.cos(np.arange(size) * 2*np.pi*freq)
    input2 = np.sin(np.arange(size) * 2*np.pi*freq) + 0.1*np.random.random(size=size)

    start = "20021001"
    end = "20021005"
    series1 = pd.Series(input1, index=pd.date_range(start, end, periods=len(input1)), name="Cos")
    series2 = pd.Series(input2, index=pd.date_range(start, end, periods=len(input2)), name="Noisy Sin")

    time_series1 = TimeSeries.from_series(series1)
    time_series2 = TimeSeries.from_series(series2)

    print("CREATING TIME SERIES")
    freq = 4

    alignment = dtw.dtw(time_series1, time_series2, window=dtw.NoWindow(), multigrid_radius=-1)
    alignment.plot()

    plt.show()

test_periodic()
