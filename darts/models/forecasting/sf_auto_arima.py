"""
StatsForecastAutoARIMA
-----------
"""

from typing import Optional

from statsforecast.models import AutoARIMA as SFAutoARIMA

from darts import TimeSeries
from darts.models.components.statsforecast_utils import (
    create_normal_samples,
    one_sigma_rule,
    unpack_sf_dict,
)
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)


class StatsForecastAutoARIMA(FutureCovariatesLocalForecastingModel):
    def __init__(
        self, *autoarima_args, add_encoders: Optional[dict] = None, **autoarima_kwargs
    ):
        """Auto-ARIMA based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the :class:`AutoARIMA` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        It is probabilistic, whereas :class:`AutoARIMA` is not.

        We refer to the `statsforecast AutoARIMA documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        autoarima_args
            Positional arguments for ``statsforecasts.models.AutoARIMA``.
        add_encoders
            A large number of future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        autoarima_kwargs
            Keyword arguments for ``statsforecasts.models.AutoARIMA``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastAutoARIMA
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define StatsForecastAutoARIMA parameters
        >>> model = StatsForecastAutoARIMA(season_length=12)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[450.55179949],
               [415.00597806],
               [454.61353249],
               [486.51218795],
               [504.09229632],
               [555.06463942]])
        """
        super().__init__(add_encoders=add_encoders)
        self.model = SFAutoARIMA(*autoarima_args, **autoarima_kwargs)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
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
        verbose: bool = False,
    ):
        super()._predict(n, future_covariates, num_samples)
        forecast_dict = self.model.predict(
            h=n,
            X=future_covariates.values(copy=False) if future_covariates else None,
            level=(one_sigma_rule,),  # ask one std for the confidence interval.
        )

        mu, std = unpack_sf_dict(forecast_dict)
        if num_samples > 1:
            samples = create_normal_samples(mu, std, num_samples, n)
        else:
            samples = mu

        return self._build_forecast_series(samples)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def min_train_series_length(self) -> int:
        return 10

    @property
    def _supports_range_index(self) -> bool:
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True
