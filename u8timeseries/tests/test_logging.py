import unittest
import pandas as pd
import numpy as np
import re
from testfixtures import LogCapture

from ..timeseries import TimeSeries
from u8timeseries.utils.timeseries_generation import linear_timeseries, constant_timeseries
from u8timeseries.models.theta import Theta
from ..custom_logging import raise_log, check_value_log, time_log, get_logger


class LoggingTestCase(unittest.TestCase):

    def test_raise_log(self):
        exception_was_raised = False
        with LogCapture() as lc:
            logger = get_logger(__name__)
            try:
                raise_log(Exception('test'), logger)
            except:
                exception_was_raised = True

        # testing correct log message
        lc.check(
            (__name__, 'ERROR', 'Exception: test')
        )

        # checking whether exception was properly raised
        self.assertTrue(exception_was_raised)

    def test_check_value_log(self):
        exception_was_raised = False
        with LogCapture() as lc:
            logger = get_logger(__name__)
            try:
                check_value_log(True, "test", logger)
                check_value_log(False, "test", logger)
            except:
                exception_was_raised = True

        # testing correct log message
        lc.check(
            (__name__, 'ERROR', 'ValueError: test')
        )

        # checking whether exception was properly raised
        self.assertTrue(exception_was_raised)
    
    def test_timeseries_constructor_error_log(self):
        # test assert error log when trying to construct a TimeSeries that is too short
        times = pd.date_range(start='2000-01-01', periods=2, freq='D')
        values = np.array([1, 2])
        with LogCapture() as lc:
            try:
                ts = TimeSeries.from_times_and_values(times, values)
            except:
                pass
            
        lc.check(
            ('u8timeseries.timeseries', 'ERROR', 'ValueError: Series must have at least three values.')
        )

    def test_timeseries_split_error_log(self):
        # test raised error log that occurs when trying to split TimeSeries at a point outside of the time index range
        times = pd.date_range(start='2000-01-01', periods=3, freq='D')
        values = np.array(range(3))
        ts = TimeSeries.from_times_and_values(times, values)
        with LogCapture() as lc:
            try:
                ts.split_after(pd.Timestamp('2020-02-01'))
            except:
                pass
            
        lc.check(
            ('u8timeseries.timeseries', 'ERROR', 'ValueError: Timestamp must be between 2000-01-01 00:00:00 and 2000-01-03 00:00:00')
        )

    def test_time_log(self):
        # test time log decorator log message when running theta model
        ts = constant_timeseries(length=100)
        model = Theta()
        with LogCapture() as lc:
            model.fit(ts)
        
        logged_message = lc.records[-1].getMessage()
        self.assertTrue(re.match("fit function ran for [0-9]+ milliseconds", logged_message))