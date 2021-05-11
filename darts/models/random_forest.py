"""
Random Forest
-------------

Model for the Random Forest Regressor[1].
The implementations is wrapped around `RandomForestRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Random_forest
"""
import numpy as np
import pandas as pd

from ..logging import get_logger
from typing import Optional, Union
from .regression_model import RegressionModel
from sklearn.ensemble import RandomForestRegressor

logger = get_logger(__name__)


class RandomForest(RegressionModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_exog: Union[int, list, bool] = None,
                 n_estimators: Optional[int] = 100,
                 max_depth: Optional[int] = None,
                 **kwargs):
        """ Random Forest

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
        lags_exog : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_exog` lags are used (inclusive). Otherwise a list of integers with lags is required.
            If True `lags` will be used to determine lags_exog. If False, the values of all exogenous variables
            at the current time `t`. This might lead to leakage if for **predictions** the values of the exogenous
            variables at time `t` are not known.
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        **kwargs
            Additional keyword arguments passed to `sklearn.ensemble.RandomForest`.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.kwargs["n_estimators"] = self.n_estimators
        self.kwargs["max_depth"] = self.max_depth

        super().__init__(
            lags=lags,
            lags_exog=lags_exog,
            model=RandomForestRegressor(
                **kwargs
            )
        )

    def __str__(self):
        return 'RandomForest(lags={}, lags_exog={}, n_estimators={}, max_depth={})'.format(
            self.lags, self.lags_exog, self.n_estimators, self.max_depth
        )

