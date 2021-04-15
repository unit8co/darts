"""
Standard Regression model
-------------------------
"""
import numpy as np
import pandas as pd

from typing import Union
from ..logging import get_logger
from .regression_model import RegressionModel
from sklearn.linear_model import LinearRegression

logger = get_logger(__name__)


class StandardRegressionModel(RegressionModel):

    def __init__(self,
                 lags: Union[int, list],
                 lags_exog: Union[int, list, bool] = True,
                 **kwargs):
        """
        Simple wrapper for the linear regression model in scikit-learn, LinearRegression().

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags.
        lags_exog : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_exog` lags are used (inclusive). Otherwise a list of integers with lags. If False,
            the value at time `t` is used which might lead to leakage. If True the same lags as for the
            target variable are used.
        """
        self.kwargs = kwargs
        super().__init__(
            lags=lags,
            lags_exog=lags_exog,
            model=LinearRegression(n_jobs=-1, fit_intercept=False, **kwargs)
        )

    def __str__(self):
        return 'LinearRegression(lags={}, lags_exog={})'.format(
            self.lags, self.lags_exog
        )