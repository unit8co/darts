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
from ..logging import get_logger
from typing import Optional, Union, Tuple
from .regression_model import RegressionModel
from sklearn.ensemble import RandomForestRegressor

logger = get_logger(__name__)


class RandomForest(RegressionModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_past_covariates: Union[int, list] = None,
                 lags_future_covariates: Union[Tuple[int, int], list] = None,
                 n_estimators: Optional[int] = 100,
                 max_depth: Optional[int] = None,
                 **kwargs):
        """Random Forest

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
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.
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
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            model=RandomForestRegressor(**kwargs)
        )

    def __str__(self):
        return (f"RandomForest(lags={self.lags}, lags_past_covariates={self.lags_past_covariates}, "
                f"lags_historical_covariates={self.lags_historical_covariates}, "
                f"lags_future_covariates={self.lags_future_covariates}, "
                f"n_estimators={self.n_estimators}, max_depth={self.max_depth}")
