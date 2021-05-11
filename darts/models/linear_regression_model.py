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


class LinearRegressionModel(RegressionModel):

    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_exog: Union[int, list, bool] = None,
                 **kwargs):
        """
        Simple wrapper for the linear regression model in scikit-learn, LinearRegression().

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
        lags_exog : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_exog` lags are used (inclusive). Otherwise a list of integers with lags is required.
            If True `lags` will be used to determine lags_exog. If False, the values of all exogenous variables
            at the current time `t`. This might lead to leakage if for predictions the values of the exogenous
            variables at time `t` are not known.
        **kwargs
            Additional keyword arguments passed to `sklearn.linear_model.LinearRegression`.
        """
        self.kwargs = kwargs
        super().__init__(
            lags=lags,
            lags_exog=lags_exog,
            model=LinearRegression(**kwargs)
        )

    def __str__(self):
        return 'LinearRegression(lags={}, lags_exog={})'.format(self.lags, self.lags_exog)