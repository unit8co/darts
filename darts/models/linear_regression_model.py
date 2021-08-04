"""
Standard Regression model
-------------------------
"""
from typing import Union, Tuple
from ..logging import get_logger
from .regression_model import RegressionModel
from sklearn.linear_model import LinearRegression

logger = get_logger(__name__)


class LinearRegressionModel(RegressionModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_past_covariates: Union[int, list] = None,
                 lags_future_covariates: Union[Tuple[int, int], list] = None,
                 **kwargs):
        """
        Simple wrapper for the linear regression model in scikit-learn, LinearRegression().

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
        lags_covariates : Union[int, list, bool] # TODO fix doc
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive). Otherwise a list of integers with lags is required.
            If True `lags` will be used to determine `lags_covariates`. If False, the values of all covariates at the
            current time `t`. This might lead to leakage if for predictions the values of the covariates at time `t`
            are not known.
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
