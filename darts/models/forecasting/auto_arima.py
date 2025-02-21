"""
AutoARIMA
---------
"""

from typing import Optional

from pmdarima import AutoARIMA as PmdAutoARIMA

from darts.logging import get_logger, raise_if
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class AutoARIMA(FutureCovariatesLocalForecastingModel):
    def __init__(
        self, *autoarima_args, add_encoders: Optional[dict] = None, **autoarima_kwargs
    ):
        """Auto-ARIMA

        This implementation is a thin wrapper around `pmdarima AutoARIMA model
        <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_,
        which provides functionality similar to R's `auto.arima
        <https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima>`_.

        This model supports the same parameters as the pmdarima AutoARIMA model.
        See `pmdarima documentation
        <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_
        for an extensive documentation and a list of supported parameters.

        .. note::
            For a faster and probabilistic version of AutoARIMA, checkout
            the :class:`StatsForecastAutoARIMA` model.

        Parameters
        ----------
        autoarima_args
            Positional arguments for the pmdarima.AutoARIMA model
        autoarima_kwargs
            Keyword arguments for the pmdarima.AutoARIMA model
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

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoARIMA
        >>> from darts.utils.timeseries_generation import holidays_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define some boundaries for the parameters
        >>> model = AutoARIMA(start_p=8, max_p=12, start_q=1)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[449.79716178],
               [416.31180633],
               [445.28005229],
               [485.27121314],
               [507.61787454],
               [561.26993332]])
        """
        super().__init__(add_encoders=add_encoders)
        self.model = PmdAutoARIMA(*autoarima_args, **autoarima_kwargs)
        self.trend = self.model.trend

    @property
    def supports_multivariate(self) -> bool:
        return False

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(
            series.values(), X=future_covariates.values() if future_covariates else None
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
        forecast = self.model.predict(
            n_periods=n, X=future_covariates.values() if future_covariates else None
        )
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 10

    @property
    def _supports_range_index(self) -> bool:
        raise_if(
            self.trend and self.trend != "c",
            "'trend' is not None. Range indexing is not supported in that case.",
            logger,
        )
        return True
