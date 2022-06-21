"""
Croston method
--------------
"""

import numpy as np
from numba.core import errors
from statsforecast.models import croston_classic, croston_optimized, croston_sba
from statsforecast.models import tsb as croston_tsb

from darts.logging import raise_if, raise_if_not
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class Croston(ForecastingModel):
    def __init__(
        self, version: str = "classic", alpha_d: float = None, alpha_p: float = None
    ):
        """An implementation of the `Croston method
        <https://otexts.com/fpp3/counts.html>`_ for intermittent
        count series.

        Relying on the implementation of `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Parameters
        ----------
        version
            - "classic" corresponds to classic Croston.
            - "optimized" corresponds to optimized classic Croston, which searches
              for the optimal ``alpha`` smoothing parameter and can take longer
              to run. Otherwise, a fixed value of ``alpha=0.1`` is used.
            - "sba" corresponds to the adjustment of the Croston method known as
              the Syntetos-Boylan Approximation [1]_.
            - "tsb" corresponds to the adjustment of the Croston method proposed by
              Teunter, Syntetos and Babai [2]_. In this case, `alpha_d` and `alpha_p` must
              be set.
        alpha_d
            For the "tsb" version, the alpha smoothing parameter to apply on demand.
        alpha_p
            For the "tsb" version, the alpha smoothing parameter to apply on probability.

        References
        ----------
        .. [1] Aris A. Syntetos and John E. Boylan. The accuracy of intermittent demand estimates.
               International Journal of Forecasting, 21(2):303 – 314, 2005.
        .. [2] Ruud H. Teunter, Aris A. Syntetos, and M. Zied Babai.
               Intermittent demand: Linking forecasting to inventory obsolescence.
               European Journal of Operational Research, 214(3):606 – 615, 2011.
        """
        super().__init__()
        raise_if_not(
            version.lower() in ["classic", "optimized", "sba", "tsb"],
            'The provided "version" parameter must be set to "classic", "optimized", "sba" or "tsb".',
        )

        if version == "classic":
            self.method = croston_classic
        elif version == "optimized":
            self.method = croston_optimized
        elif version == "sba":
            self.method = croston_sba
        else:
            raise_if(
                alpha_d is None or alpha_p is None,
                'alpha_d and alpha_p must be specified when using "tsb".',
            )
            self.method = croston_tsb
            self.alpha_d = alpha_d
            self.alpha_p = alpha_p

        self.version = version

    def __str__(self):
        return "Croston"

    def fit(self, series: TimeSeries):
        super().fit(series)
        series._assert_univariate()
        series = self.training_series

        if self.version == "tsb":
            self.forecast_val = self.method(
                series.values(copy=False),
                h=1,
                future_xreg=None,
                alpha_d=self.alpha_d,
                alpha_p=self.alpha_p,
            )
        elif self.version == "sba":
            try:
                self.forecast_val = self.method(
                    series.values(copy=False), h=1, future_xreg=None
                )
            except errors.TypingError:
                raise_if(
                    True,
                    '"sba" version is not supported with this version of statsforecast.',
                )

        else:
            self.forecast_val = self.method(
                series.values(copy=False), h=1, future_xreg=None
            )
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
    ):
        super().predict(n, num_samples)
        values = np.tile(self.forecast_val, n)
        return self._build_forecast_series(values)

    @property
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        return True

    def _is_probabilistic(self) -> bool:
        return False
