"""
Linear Regression model
-----------------------

A forecasting model using a linear regression of some of the target series' lags, as well as optionally some
covariate series' lags in order to obtain a forecast.
"""
from typing import List, Tuple, Union

from sklearn.linear_model import LinearRegression

from darts.logging import get_logger
from darts.models.forecasting.regression_model import RegressionModel

logger = get_logger(__name__)


class LinearRegressionModel(RegressionModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        **kwargs,
    ):
        """Linear regression model.

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
        output_chunk_length
            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast
            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may
            be useful if the covariates don't extend far enough into the future.
        **kwargs
            Additional keyword arguments passed to `sklearn.linear_model.LinearRegression`.
        """
        self.kwargs = kwargs
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            model=LinearRegression(**kwargs),
        )

    def __str__(self):
        return f"LinearRegression(lags={self.lags})"
