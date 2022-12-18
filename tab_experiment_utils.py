import cProfile
import pstats
import time
from functools import reduce
from itertools import product
from pstats import SortKey
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.utils.data.tabularization import _create_lagged_data, create_lagged_data


def create_index_series(
    num_timesteps: int,
    num_components: int,
    offset: int = 0,
    freq: Union[int, str] = 1,
    start: Optional[Union[int, str]] = None,
):
    """
    Creates a time series whose values incrementally increase along the `timestep` and
    `components` axes, starting from an `offset` value.

    Parameters
    ----------
    num_timesteps
        Number of timesteps in the created series
    num_components
        Number of components in created series
    offset
        Optionally, the smallest value in the created series
    freq
        Optionally, the frequency of the series. If an `int`, the returned series will have a
        `pd.RangeIndex`-typed `time_index`; if a `str`, the returned series will have
        a `pd.DatetimeIndex`-typed `time_index`.
    start
        Optionally, the start date of the series. If not specified, is set to `0` if
        `freq` is an `int, or to `1/1/1900` if `freq` is a `str`.
    """
    vals = np.arange(0, num_timesteps * num_components, 1) + offset
    vals = vals.reshape(num_timesteps, num_components, 1)
    if not isinstance(freq, str):
        if start is None:
            # Default range index start:
            start = 0
        dates = freq * pd.RangeIndex(num_timesteps) + start
    else:
        if start is None:
            # Default datetime index start:
            start = "1/1/1900"
        dates = pd.date_range(start=start, periods=num_timesteps, freq=freq)
    return TimeSeries.from_times_and_values(values=vals, times=dates)


def create_test_series(num_timesteps: int, equal_freq: bool, use_range_idx: bool):
    """
    Convenience function that creates 'standardised' `target_series`,
    `past_covariates`, and `future_covariates` series. More specifically,
    each of these series have differing number of components, different number
    of timesteps, different start times, and (potentially) different frequencies.

    Parameters
    ----------
    num_timesteps
        The number of timesteps to place in the `target_series`; the `past_covariates`
        and `future_covariates` will have 13 less and 5 more timepoints respectively.
    equal_freq
        Specifies that all of the generated series should be of equal frequency.
    use_range_idx
        Specifies that the generated series should have a `pd.RangeIndex`-type `time_index.
    """

    if num_timesteps <= 13:
        raise ValueError("Must have at least 14 timesteps.")

    if equal_freq and use_range_idx:
        freqs = 3 * [1]
    elif equal_freq and not use_range_idx:
        freqs = 3 * ["d"]
    elif not equal_freq and use_range_idx:
        freqs = [1, 2, 3]
    else:
        freqs = ["d", "2d", "3d"]

    start = 5 if use_range_idx else "1/5/1900"
    target_series = create_index_series(
        num_timesteps, num_components=2, offset=10, start=start, freq=freqs[0]
    )
    past_covariates = create_index_series(
        num_timesteps - 13, num_components=3, offset=30, freq=freqs[1]
    )
    future_covariates = create_index_series(
        num_timesteps + 5, num_components=5, offset=60, freq=freqs[2]
    )
    return target_series, past_covariates, future_covariates


def test_correctness(
    equal_freq: bool,
    use_range_idx: bool,
    num_timesteps: int = 1000,
    print_freq: int = 1000,
):

    """
    Checks the correctness of `create_lagged_data` against what's produced by `_create_lagged_data` over
    a very large number of input paramter combinations. An error is thrown if the `X`, `y`, or `times`
    produced by `create_lagged_data` doesn't exactly match what's produced by `_create_lagged_data`.
    """

    target_series, past_covariates, future_covariates = create_test_series(
        num_timesteps, equal_freq, use_range_idx
    )

    # Check case where `past_covariates` is and is not specified:
    past_covariates_combos = [past_covariates, None]
    # Check case where `future_covariates` is and is not specified:
    future_covariates_combos = [future_covariates, None]
    # Single lags, multiple lags, multiple + noncontiguous lags:
    lags_combos = lags_past_covariates_combos = lags_future_covariates_combos = (
        [-1],
        [-2, -1],
        [-5, -2],
        [-6, -4, -2],
    )
    # Try out variety of `output_chunk_length`s:
    output_chunk_length_combos = [1, 5, 10, 20]
    # With and without multiple model predictions:
    multi_models_combos = [False, True]
    # Differing max number of `max_sample_per_ts` (`None` means no limit on number of samples)""
    max_sample_per_ts_combos = [1, 5, 10, 20, None]
    # Take Cartersian product of all input combinations:
    all_combos = [
        past_covariates_combos,
        future_covariates_combos,
        lags_combos,
        lags_past_covariates_combos,
        lags_future_covariates_combos,
        output_chunk_length_combos,
        multi_models_combos,
        max_sample_per_ts_combos,
    ]
    num_param_combos = reduce(lambda a, b: a * b, [len(combo) for combo in all_combos])
    for i, (
        past_covariates,
        future_covariates,
        lags,
        lags_past_covariates,
        lags_future_covariates,
        output_chunk_length,
        multi_models,
        max_samples_per_ts,
    ) in enumerate(product(*all_combos)):

        # If `*_covariates` is not specified, set `lags_*_covariates` to `None`,
        # otherwise `_create_lagged_data` throws an error:
        if past_covariates is None:
            lags_past_covariates = None
        if future_covariates is None:
            lags_future_covariates = None

        # Current implmentation:
        (X, y, Ts) = _create_lagged_data(
            target_series,
            lags=lags,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            multi_models=multi_models,
            is_training=True,
            max_samples_per_ts=max_samples_per_ts,
        )
        # Refactored implementation:
        my_X, my_y, my_Ts = create_lagged_data(
            target_series,
            lags=lags,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            multi_models=multi_models,
            max_samples_per_ts=max_samples_per_ts,
            use_moving_windows=equal_freq,
        )

        try:
            assert list(my_Ts) == list(Ts[0])
        except AssertionError:
            raise ValueError("Ts incorrect")

        try:
            assert np.allclose(my_X.squeeze(), X.squeeze())
        except AssertionError:
            raise ValueError("X incorrect")

        try:
            assert np.allclose(my_y.squeeze(), y.squeeze())
        except AssertionError:
            raise ValueError("y incorrect")

        if (i + 1) % print_freq == 0:
            print(f"Passed {i+1}/{num_param_combos}")
    return None


def perform_profiling(
    num_repeats: int,
    lags: Sequence[int],
    lags_past_covariates: Sequence[int],
    lags_future_covariates: Sequence[int],
    output_chunk_length: int,
    max_samples_per_ts: int,
    multi_models: bool,
    use_moving_windows: bool,
    equal_freq: bool,
    num_timesteps: int = 10000,
):
    """
    Returns the profiling statistics of `create_lagged_data` over `num_repeats` repititions of calling
    the function. The printed statistics are limited to those functions directly running inside of
    `tabularization.py` and are ranked according to cumulative runtime.
    """

    target_series, past_covariates, future_covariates = create_test_series(
        num_timesteps, equal_freq
    )

    with cProfile.Profile() as pr:
        for _ in range(num_repeats):
            my_X, my_y, my_Ts = create_lagged_data(
                target_series,
                lags=lags,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                lags_past_covariates=lags_past_covariates,
                lags_future_covariates=lags_future_covariates,
                output_chunk_length=output_chunk_length,
                multi_models=multi_models,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=use_moving_windows,
            )
    sortby = SortKey.CUMULATIVE
    pstats.Stats(pr).sort_stats(sortby).print_stats("tabularization.py")
    return None


def perform_benchmarks(
    num_repeats: int,
    use_range_idx: bool,
    lags: Sequence[int],
    lags_past_covariates: Sequence[int],
    lags_future_covariates: Sequence[int],
    output_chunk_length: int,
    max_samples_per_ts: int,
    multi_models: bool,
    num_timesteps: int = 10000,
):
    """
    Benchmarks the performance of `create_lagged_data` against `_create_lagged_data` across three scenarios:
        1. When `target_series`, `past_covariates`, and `future_covariates` are all of different frequencies.
        2. When `target_series`, `past_covariates`, and `future_covariates` are all of the same frequency, and
        the 'moving window' method is used in `create_lagged_data` (i.e. `use_moving_windows = True`).
        3. When `target_series`, `past_covariates`, and `future_covariates` are all of the same frequency, and
        the 'time intersection' method is used in `create_lagged_data` (i.e. `use_moving_windows = False`).
    The returned benchmarks are computed using `num_repeats` repititions of each function.
    """

    benchmarks = {
        "With Unequal Frequency Timeseries": {
            "equal_freq": False,
            "use_moving_windows": True,
        },
        "With Equal Frequency Timeseries, Using Moving Windows": {
            "equal_freq": True,
            "use_moving_windows": True,
        },
        "With Equal Frequency Timeseries, Using Time Intersections": {
            "equal_freq": True,
            "use_moving_windows": False,
        },
    }
    for name, params in benchmarks.items():
        print(name + ":")
        _perform_single_benchmark(
            num_repeats,
            use_range_idx,
            lags,
            lags_past_covariates,
            lags_future_covariates,
            output_chunk_length,
            max_samples_per_ts,
            multi_models,
            params["use_moving_windows"],
            params["equal_freq"],
            num_timesteps,
        )
        print()


def _perform_single_benchmark(
    num_repeats: int,
    use_range_idx: bool,
    lags: Sequence[int],
    lags_past_covariates: Sequence[int],
    lags_future_covariates: Sequence[int],
    output_chunk_length: int,
    max_samples_per_ts: int,
    multi_models: bool,
    use_moving_windows: bool,
    equal_freq: bool,
    num_timesteps: int,
):

    """
    Convenience function called by `perform_benchmarks`.
    """

    target_series, past_covariates, future_covariates = create_test_series(
        num_timesteps, equal_freq, use_range_idx
    )

    start_time = time.time()
    for _ in range(num_repeats):
        (X, y, Ts) = _create_lagged_data(
            target_series,
            lags=lags,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            multi_models=multi_models,
            is_training=True,
            max_samples_per_ts=max_samples_per_ts,
        )
    current_implem_time = time.time() - start_time

    start_time = time.time()
    for _ in range(num_repeats):
        my_X, my_y, my_Ts = create_lagged_data(
            target_series,
            lags=lags,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            multi_models=multi_models,
            max_samples_per_ts=max_samples_per_ts,
            use_moving_windows=use_moving_windows,
        )
    refact_implem_time = time.time() - start_time

    # Ensure reimplemented function is correct:
    assert np.allclose(my_X.squeeze(), X.squeeze())
    assert np.allclose(my_y.squeeze(), y.squeeze())
    assert list(my_Ts) == list(Ts[0])

    print(
        f"Current implementation: {current_implem_time} secs for {num_repeats} repetitions"
    )
    print(
        f"New implementation: {refact_implem_time} secs for {num_repeats} repetitions"
    )
    print(f"Speed up = {current_implem_time/refact_implem_time} fold")

    return None
