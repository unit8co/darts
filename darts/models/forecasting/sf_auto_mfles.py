"""
AutoMFLES
-----------
"""

from typing import Optional

from statsforecast.models import AutoMFLES as SFAutoMFLES

from darts import TimeSeries
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)

logger = get_logger(__name__)


class AutoMFLES(FutureCovariatesLocalForecastingModel):
    def __init__(
        self, *autoMFLES_args, add_encoders: Optional[dict] = None, **autoMFLES_kwargs
    ):
        """Auto-MFLES based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best MFLES model from all feasible combinations of the parameters
        `seasonality_weights`, `smoother`, `ma`, and `seasonal_period`. Selection is made using the sMAPE by default.

        We refer to the `statsforecast AutoMFLES documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#mfles>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        autoMFLES_args
            Positional arguments for ``statsforecasts.models.AutoMFLES``.
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
        autoMFLES_kwargs
            Keyword arguments for ``statsforecasts.models.AutoMFLES``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoMFLES
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define AutoMFLES parameters
        >>> model = AutoMFLES(season_length=12, test_size=12)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[466.03298745],
               [450.76192105],
               [517.6342497 ],
               [511.62988828],
               [520.15305998],
               [593.38690019]])
        """
        if "prediction_intervals" in autoMFLES_kwargs:
            logger.warning(
                "AutoMFLES does not support probabilistic forecasting. "
                "`prediction_intervals` will be ignored."
            )

        super().__init__(add_encoders=add_encoders)
        self.model = SFAutoMFLES(*autoMFLES_args, **autoMFLES_kwargs)

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
            level=None,
        )

        return self._build_forecast_series(forecast_dict["mean"])

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
        return False
