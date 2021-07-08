"""
LGBM
----

Note: to use LightGBM on your Mac, you need to have `openmp` installed. Please refer to the installation
documentation[1] for your OS from LightGBM website[2].

References
----------
.. [1] https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
.. [2] https://lightgbm.readthedocs.io/en/latest/index.html
"""

from ..logging import get_logger
from typing import Union
from .regression_model import RegressionModel
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

