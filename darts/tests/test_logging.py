import logging
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from testfixtures import LogCapture

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not, raise_log, time_log


@pytest.fixture(scope="module", autouse=True)
def setup_logging():
    logging.disable(logging.NOTSET)


def test_raise_log():
    exception_was_raised = False
    with LogCapture() as lc:
        logger = get_logger(__name__)
        logger.handlers = []
        try:
            raise_log(Exception("test"), logger)
        except Exception:
            exception_was_raised = True

    # testing correct log message
    lc.check((__name__, "ERROR", "Exception: test"))

    # checking whether exception was properly raised
    assert exception_was_raised


def test_raise_if_not():
    exception_was_raised = False
    with LogCapture() as lc:
        logger = get_logger(__name__)
        logger.handlers = []
        try:
            raise_if_not(True, "test", logger)
            raise_if_not(False, "test", logger)
        except Exception:
            exception_was_raised = True

    # testing correct log message
    lc.check((__name__, "ERROR", "ValueError: test"))

    # checking whether exception was properly raised
    assert exception_was_raised


def test_timeseries_constructor_error_log():
    # test assert error log when trying to construct a TimeSeries that is too short
    empty_series = pd.DataFrame()
    with LogCapture() as lc:
        get_logger("darts.timeseries").handlers = []
        try:
            TimeSeries(xr.DataArray(empty_series))
        except Exception:
            pass

    lc.check((
        "darts.timeseries",
        "ERROR",
        "ValueError: TimeSeries require DataArray of dimensionality 3 (('time', 'component', 'sample')).",
    ))


def test_timeseries_split_error_log():
    # test raised error log that occurs when trying to split TimeSeries at a point outside of the time index range
    times = pd.date_range(start="2000-01-01", periods=3, freq="D")
    values = np.array(range(3))
    ts = TimeSeries.from_times_and_values(times, values)
    with LogCapture() as lc:
        get_logger("darts.timeseries").handlers = []
        try:
            ts.split_after(pd.Timestamp("2020-02-01"))
        except Exception:
            pass

    lc.check((
        "darts.timeseries",
        "ERROR",
        "ValueError: Timestamp must be between 2000-01-01 00:00:00 and 2000-01-03 00:00:00",
    ))


def test_time_log():
    logger = get_logger(__name__)
    logger.handlers = []

    @time_log(logger)
    def _my_timed_fn():
        # do something for some time
        for _ in range(2):
            pass

    with LogCapture() as lc:
        _my_timed_fn()

    logged_message = lc.records[-1].getMessage()
    assert (
        re.fullmatch(
            "_my_timed_fn function ran for [0-9]+ milliseconds", logged_message
        )
        is not None
    )
