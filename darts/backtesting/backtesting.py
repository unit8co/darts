"""
Backtesting Functions
---------------------
"""

from typing import Iterable, Optional, Callable, List
from itertools import product
import math
import time
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from ..timeseries import TimeSeries
from ..models.forecasting_model import ForecastingModel, UnivariateForecastingModel, MultivariateForecastingModel
from ..models.torch_forecasting_model import TorchForecastingModel
from ..models.regression_model import RegressionModel
from ..models import NaiveSeasonal, AutoARIMA, ExponentialSmoothing, FFT, Prophet, Theta
from .. import metrics
from ..utils import _build_tqdm_iterator
from ..utils.statistics import plot_acf
from ..utils.missing_values import auto_fillna
from ..logging import raise_if_not, get_logger


logger = get_logger(__name__)


def plot_residuals_analysis(residuals: TimeSeries,
                            num_bins: int = 20,
                            fill_nan: bool = True):
    """ Plots data relevant to residuals.

    This function takes a univariate TimeSeries instance of residuals and plots their values,
    their distribution and their ACF.
    Please note that if the residual TimeSeries instance contains NaN values while, the plots
    might be displayed incorrectly. If `fill_nan` is set to True, the missing values will
    be interpolated.

    Parameters
    ----------
    residuals
        Univariate TimeSeries instance representing residuals.
    num_bins
        Optionally, an integer value determining the number of bins in the histogram.
    fill_nan:
        A boolean value indicating whether NaN values should be filled in the residuals.
    """

    residuals._assert_univariate()

    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    gs = fig.add_gridspec(2, 2)

    if fill_nan:
        residuals = auto_fillna(residuals)

    # plot values
    ax1 = fig.add_subplot(gs[:1, :])
    residuals.plot(ax=ax1)
    ax1.set_ylabel('value')
    ax1.set_title('Residual values')

    # plot distribution
    res_mean, res_std = np.mean(residuals.univariate_values()), np.std(residuals.univariate_values())
    res_min, res_max = min(residuals.univariate_values()), max(residuals.univariate_values())
    x = np.linspace(res_min, res_max, 100)
    ax2 = fig.add_subplot(gs[1:, 1:])
    ax2.hist(residuals.univariate_values(), bins=num_bins)
    ax2.plot(x, norm(res_mean, res_std).pdf(x) * len(residuals) * (res_max - res_min) / num_bins)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_title('Distribution')
    ax2.set_ylabel('count')
    ax2.set_xlabel('value')

    # plot ACF
    ax3 = fig.add_subplot(gs[1:, :1])
    plot_acf(residuals, axis=ax3)
    ax3.set_ylabel('ACF value')
    ax3.set_xlabel('lag')
    ax3.set_title('ACF')

def explore_models(train_series: TimeSeries,
                   val_series: TimeSeries,
                   test_series: TimeSeries,
                   metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                   model_parameter_tuples: Optional[list] = None,
                   plot_width: int = 3,
                   verbose: bool = False):
    """ A function for exploring the suitability of multiple models on a given train/validation/test split
    of a univariate series.

    This funtion iterates through a list of models, training each on `train_series` and `val_series`
    and evaluating them on `test_series`. Models with free hyperparameters are first
    tuned by trying out a 'reasonable' set of hyperparameter combinations. A model for
    each combination is trained on `train_series` and evaluated on `val_series`.
    The best-performing combination of hyperparameters (the one resulting in the lowest output of the `metric`
    function) is then chosen for that type of model. This way, none of the models have 'looked at' the test set
    before the final evaluation.

    The performance of every model type is then plotted against the series of actual observations.
    Additionally, the `metric` values and the runtimes of every model type are plotted in bar charts.

    Parameters
    ----------
    train_series
        A univariate TimeSeries instance used for training during model tuning and evaluation.
    val_series
        A univariate TimeSeries instance used for validation during model tuning and for training during evaluation.
    test_series
        A univariate TimeSeries instance used for validation when evaluating a model.
    metric:
        A function that takes two TimeSeries instances as inputs and returns a float error value.
    model_parameter_tuples:
        Optionally, a custom list of (model class, hyperparameter dictionary) tuples that will be
        explored instead of the default selection.
    plot_width
        An integer indicating the number of plots that are displayed in one row.
    verbose:
        Whether to print progress.
    """

    train_series._assert_univariate()
    val_series._assert_univariate()
    test_series._assert_univariate()

    raise_if_not(plot_width > 1, "Please choose an integer 'plot_width' value larger than 1", logger)

    # list of tuples containing model classes and hyperparameter selection, if required
    if model_parameter_tuples is None:
        model_parameter_tuples = [
            (ExponentialSmoothing, {
                'trend': ['additive', 'multiplicative'],
                'seasonal': ['additive', 'multiplicative'],
                'seasonal_periods': [7, 12, 30]
            }),
            (NaiveSeasonal, {
                'K': list(range(1, 31))
            }),
            (FFT, {
                'nr_freqs_to_keep': [2, 3, 5, 10, 25, 50, 100, 150, 200],
                'trend': [None, 'poly', 'exp']
            }),
            (Theta, {
                'theta': np.delete(np.linspace(-10, 10, 51), 25)
            }),
            (Prophet, {}),
            (AutoARIMA, {})
        ]

    # set up plot grid
    plot_height = math.ceil(len(model_parameter_tuples) / plot_width) + 1
    fig = plt.figure(constrained_layout=True, figsize=(plot_width * 4, plot_height * 4))
    gs = fig.add_gridspec(plot_height, plot_width)

    # arrays to store data for bar charts
    metric_vals = []
    times = []
    index = []

    # iterate through model type selection
    iterator = _build_tqdm_iterator(model_parameter_tuples, verbose)
    for i, (model_class, params) in enumerate(iterator):

        # if necessary, tune hyperparameters using train_series and val_series
        if (len(params.keys()) > 0):
            model = backtest_gridsearch(model_class, params, train_series, val_series=val_series, metric=metric)
        else:
            model = model_class()

        # fit on train and val series, predict test series
        train_and_val_series = train_series.append(val_series)
        start_time = time.time()
        model.fit(train_and_val_series)
        end_time = time.time()
        runtime = end_time - start_time
        predictions = model.predict(len(test_series))

        # plot predictions against observations
        y = int(i / plot_width)
        x = i - y * plot_width
        ax = fig.add_subplot(gs[y, x])
        metric_val = metric(predictions, test_series)
        ax.set_title("{}\n{}: {}, runtime: {}s".format(model, metric.__name__, round(metric_val, 2),
                                                       round(runtime, 2)))

        train_and_val_series.plot(label='train and val', ax=ax)
        test_series.plot(label='test', ax=ax)
        predictions.plot(label='pred', ax=ax)
        plt.legend()

        # record data for bar charts
        metric_vals.append(metric_val)
        times.append(runtime)
        index.append(str(model)[:3] + "..")

    # plot bar charts
    ax_metrics = fig.add_subplot(gs[-1, 0])
    ax_metrics.set_title("Performance")
    ax_metrics.set_ylabel(metric.__name__ + ' value')
    pd.Series(metric_vals, index=index).plot.bar(ax=ax_metrics, rot=0)
    ax_times = fig.add_subplot(gs[-1, 1])
    ax_times.set_title("Runtime")
    ax_times.set_ylabel('seconds')
    pd.Series(times, index=index).plot.bar(ax=ax_times, rot=0)

    plt.show()
