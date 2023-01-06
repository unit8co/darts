"""
StatsForecastETS
-----------
"""

from typing import Optional

import numpy as np
from statsforecast.models import ETS

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)


class StatsForecastETS(FutureCovariatesLocalForecastingModel):
    def __init__(self, *ets_args, add_encoders: Optional[dict] = None, **ets_kwargs):
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

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastETS
        >>> series = AirPassengersDataset().load()
        >>> model = StatsForecastETS(season_length=12, model="AZZ")
        >>> model.fit(series[:-36])
        >>> pred = model.predict(36)
        """
        super().__init__(add_encoders=add_encoders)
        self.model = ETS(*ets_args, **ets_kwargs)

    def __str__(self):
        return "ETS-Statsforecasts"

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series

        if future_covariates is not None:
            linreg = LinearRegressionModel(lags_future_covariates=[0])
            resids = linreg.residuals(
                series, future_covariates=series.slice_intersect(future_covariates)
            )
            self._linreg = linreg
            target = resids
        else:
            target = series

        self.model.fit(
            target.values(copy=False).flatten(),
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
        forecast_df = self.model.predict(
            h=n,
            level=(68.27,),  # ask one std for the confidence interval
        )

        if future_covariates is not None:
            # TODO: match the future covariates to the index of the forecast
            linreg_forecast = self._linreg.predict(
                n, future_covariates=future_covariates
            )
            linreg_forecast_pd = linreg_forecast.pd_series()
            mu = forecast_df["mean"] + linreg_forecast_pd
        else:
            mu = forecast_df["mean"]

        if num_samples > 1:
            std = forecast_df["hi-68.27"] - mu
            samples = np.random.normal(loc=mu, scale=std, size=(num_samples, n)).T
            samples = np.expand_dims(samples, axis=1)
        else:
            samples = mu

        return self._build_forecast_series(samples)

    @property
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        return True

    def _is_probabilistic(self) -> bool:
        return True
