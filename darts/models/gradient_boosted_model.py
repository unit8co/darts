"""
Gradient Boosted Model
----------------------

This is a LightGBM implementation of Gradient Boosted Trees algorightm.

Note: to use LightGBM on your Mac, you need to have `openmp` installed. Please refer to the installation
documentation[1] for your OS from LightGBM website[2].

Warning: as of July 2021 there is an issue with ``libomp`` version 12.0 that results in segmentation fault[3]
on Mac OS Big Sur. Please refer[4] to the github issue for details on how to downgrade the ``libomp`` library.

References
----------
.. [1] https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
.. [2] https://lightgbm.readthedocs.io/en/latest/index.html
.. [3] https://github.com/microsoft/LightGBM/issues/4229
.. [4] https://github.com/microsoft/LightGBM/issues/4229#issue-867528353
"""

from ..logging import get_logger
from typing import Union, Optional, Tuple
from .regression_model import RegressionModel
from ..timeseries import TimeSeries
import lightgbm as lgb

logger = get_logger(__name__)


class GradientBoostedModel(RegressionModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_exog: Union[int, list, bool] = None,
                 **kwargs):
        """ Light Gradient Boosted Model

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
        **kwargs
            Additional keyword arguments passed to `lightgbm.LGBRegressor`.
        """
        self.kwargs = kwargs

        super().__init__(
            lags=lags,
            lags_exog=lags_exog,
            model=lgb.LGBMRegressor(
                **kwargs
            )
        )

    def __str__(self):
        return 'LGBModel(lags={}, lags_exog={})'.format(
            self.lags, self.lags_exog
        )

    def fit(self,
            series: TimeSeries, exog: Optional[TimeSeries] = None,
            eval_set: Tuple[TimeSeries, Optional[TimeSeries]] = None,
            **kwargs) -> None:

        if eval_set is not None:  # TODO: clean this up
            X_eval = eval_set[0]
            y_eval = eval_set[1]
            X_eval_lagged = self._create_training_data(X_eval)
            kwargs['eval_set'] = (X_eval_lagged, y_eval)

        super().fit(series, exog, **kwargs)

