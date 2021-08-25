"""
Linear Regression model
-----------------------

A forecasting model using a linear regression of some of the target series' lags, as well as optionally some
covariate series' lags in order to obtain a forecast.
"""
from typing import Union, Tuple, List
from ..logging import get_logger
from .regression_model import RegressionModel
from sklearn.linear_model import LinearRegression

logger = get_logger(__name__)


class LinearRegressionModel(RegressionModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_past_covariates: Union[int, List[int]] = None,
                 lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
                 **kwargs):
        """
        Simple wrapper for the linear regression model in scikit-learn, LinearRegression().

        Parameters
        ----------
        lags
            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags
            are used (from -1 backward). Otherwise a list of integers with lags is required (each lag must be < 0).
        lags_past_covariates
            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last
            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers
            with lags < 0 is required.
        lags_future_covariates
            Number of lagged future_covariates values used to predict the next time step. If an tuple (past, future) is
            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first
            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list
            of integers with lags is required.
        **kwargs
            Additional keyword arguments passed to `sklearn.linear_model.LinearRegression`.
        """
        self.kwargs = kwargs
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            model=LinearRegression(**kwargs))

    def __str__(self):
        return (f"LinearRegression(lags={self.lags}, lags_past_covariates={self.lags_past_covariates}, "
                f"lags_historical_covariates={self.lags_historical_covariates}, "
                f"lags_future_covariates={self.lags_future_covariates}")
