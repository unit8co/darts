"""
StatsForecastETS
-----------
"""

from typing import Optional

from statsforecast.models import ETS

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel


class StatsForecastETS(DualCovariatesForecastingModel):
    def __init__(self, *ets_args, **ets_kwargs):
        """ETS based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the :class:`ExponentialSmoothing` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        This model accepts the same arguments as the `statsforecast ETS
        <https://nixtla.github.io/statsforecast/models.html#ets>`_. package.

        Parameters
        ----------
        season_length
            Number of observations per cycle. Default: 1.
        model
            Three-character string identifying method using the framework
            terminology of Hyndman et al. (2002). Possible values are:

            * "A" or "M" for error state,
            * "N", "A" or "Ad" for trend state,
            * "N", "A" or "M" for season state.

            For instance, "ANN" means additive error, no trend and no seasonality.
            Furthermore, the character "Z" is a placeholder telling statsforecast
            to search for the best model using AICs. Default: "ZZZ".

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastETS
        >>> series = AirPassengersDataset().load()
        >>> model = StatsForecastETS(season_length=12, model="AZZ")
        >>> model.fit(series[:-36])
        >>> pred = model.predict(36)
        """
        super().__init__()
        self.model = ETS(*ets_args, **ets_kwargs)

    def __str__(self):
        return "ETS-Statsforecasts"

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        series._assert_univariate()
        series = self.training_series
        self.model.fit(
            series.values(copy=False).flatten(),
            X=future_covariates.values(copy=False) if future_covariates else None,
        )
        return self

    def _predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
    ):
        super()._predict(n, future_covariates, num_samples)
        forecast_df = self.model.predict(
            h=n,
            X=future_covariates.values(copy=False) if future_covariates else None,
        )

        return self._build_forecast_series(forecast_df["mean"])

    @property
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        return True

    def _is_probabilistic(self) -> bool:
        return False
