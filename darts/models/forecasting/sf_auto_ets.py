"""
StatsForecastAutoETS
-----------
"""

from typing import Optional

from statsforecast.models import AutoETS as SFAutoETS

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.components.statsforecast_utils import (
    create_normal_samples,
    one_sigma_rule,
    unpack_sf_dict,
)
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)


class StatsForecastAutoETS(FutureCovariatesLocalForecastingModel):
    def __init__(
        self, *autoets_args, add_encoders: Optional[dict] = None, **autoets_kwargs
    ):
        """ETS based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the :class:`ExponentialSmoothing` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        We refer to the `statsforecast AutoETS documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoets>`_
        for the exhaustive documentation of the arguments.

        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first
        regressing the series against the future covariates using the :class:'LinearRegressionModel' model and then
        running StatsForecast's AutoETS on the in-sample residuals from this original regression. This approach was
        inspired by 'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>'_.


        Parameters
        ----------
        autoets_args
            Positional arguments for ``statsforecasts.models.AutoETS``.
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
        autoets_kwargs
            Keyword arguments for ``statsforecasts.models.AutoETS``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastAutoETS
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define StatsForecastAutoETS parameters
        >>> model = StatsForecastAutoETS(season_length=12, model="AZZ")
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[441.40323676],
               [415.09871431],
               [448.90785391],
               [491.38584654],
               [493.11817462],
               [549.88974472]])
        """
        super().__init__(add_encoders=add_encoders)
        self.model = SFAutoETS(*autoets_args, **autoets_kwargs)
        self._linreg = None

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series

        if future_covariates is not None:
            # perform OLS and get in-sample residuals
            linreg = LinearRegressionModel(lags_future_covariates=[0])
            linreg.fit(series, future_covariates=future_covariates)
            fitted_values = linreg.model.predict(
                X=future_covariates.slice_intersect(series).values(copy=False)
            )
            fitted_values_ts = TimeSeries.from_times_and_values(
                times=series.time_index, values=fitted_values
            )
            resids = series - fitted_values_ts
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
        forecast_dict = self.model.predict(
            h=n,
            level=[one_sigma_rule],  # ask one std for the confidence interval
        )

        mu_ets, std = unpack_sf_dict(forecast_dict)

        if future_covariates is not None:
            mu_linreg = self._linreg.predict(n, future_covariates=future_covariates)
            mu_linreg_values = mu_linreg.values(copy=False).reshape(
                n,
            )
            mu = mu_ets + mu_linreg_values
        else:
            mu = mu_ets

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
