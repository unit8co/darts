"""
StatsForecast Models

- AutoETS
- AutoARIMA
- AutoTheta
- AutoCES
-----------
"""

from typing import Optional

import numpy as np
from statsforecast.models import ETS
from statsforecast.models import AutoARIMA as SFAutoARIMA
from statsforecast.models import AutoTheta as SFAutoTheta

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
    LocalForecastingModel,
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
            level=(68.27,),  # ask one std for the confidence interval
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
    def min_train_series_length(self) -> int:
        return 10

    def _supports_range_index(self) -> bool:
        return True

    def _is_probabilistic(self) -> bool:
        return True


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
        <https://nixtla.github.io/statsforecast/models.html#arima-methods>`_
        for the documentation of the arguments.

        Parameters
        ----------
        autoarima_args
            Positional arguments for ``statsforecasts.models.AutoARIMA``.
        autoarima_kwargs
            Keyword arguments for ``statsforecasts.models.AutoARIMA``.
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
        >>> from darts.models import StatsForecastAutoARIMA
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> model = StatsForecastAutoARIMA(season_length=12)
        >>> model.fit(series[:-36])
        >>> pred = model.predict(36, num_samples=100)
        """
        super().__init__(add_encoders=add_encoders)
        self.model = SFAutoARIMA(*autoarima_args, **autoarima_kwargs)

    def __str__(self):
        return "Auto-ARIMA-Statsforecasts"

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
            level=(68.27,),  # ask one std for the confidence interval.
        )

        mu, std = unpack_sf_dict(forecast_dict)
        if num_samples > 1:
            samples = create_normal_samples(mu, std, num_samples, n)
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


class StatsForecastAutoTheta(LocalForecastingModel):
    def __init__(
        self, *autotheta_args, add_encoders: Optional[dict] = None, **autotheta_kwargs
    ):
        """Auto-Theta based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best Theta (Standard Theta Model (‘STM’), Optimized Theta Model (‘OTM’),
        Dynamic Standard Theta Model (‘DSTM’), Dynamic Optimized Theta Model (‘DOTM’)) model using mse.
        <https://www.sciencedirect.com/science/article/pii/S0169207016300243>

        It is probabilistic, whereas :class:`FourTheta` is not.

        We refer to the `statsforecast AutoTheta documentation
        <https://nixtla.github.io/statsforecast/models.html#autotheta>`_
        for the documentation of the arguments.

        Parameters
        ----------
        autotheta_args
            Positional arguments for ``statsforecasts.models.AutoTheta``.
        autotheta_kwargs
            Keyword arguments for ``statsforecasts.models.AutoTheta``.

            ..

        Examples
        --------
        >>> from darts.models import StatsForecastAutoTheta
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> model = StatsForecastAutoTheta(season_length=12)
        >>> model.fit(series[:-36])
        >>> pred = model.predict(36, num_samples=100)
        """
        super().__init__()
        self.model = SFAutoTheta(*autotheta_args, **autotheta_kwargs)

    def __str__(self):
        return "Auto-Theta-Statsforecasts"

    def fit(self, series: TimeSeries):
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(
            series.values(copy=False).flatten(),
        )
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
    ):
        super().predict(n, num_samples)
        forecast_dict = self.model.predict(
            h=n,
            level=(68.27,),  # ask one std for the confidence interval.
        )

        mu, std = unpack_sf_dict(forecast_dict)
        if num_samples > 1:
            samples = create_normal_samples(mu, std, num_samples, n)
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


def create_normal_samples(
    mu: float,
    std: float,
    num_samples: int,
    n: int,
) -> np.array:
    """Generate samples assuming a Normal distribution."""
    samples = np.random.normal(loc=mu, scale=std, size=(num_samples, n)).T
    samples = np.expand_dims(samples, axis=1)
    return samples


def unpack_sf_dict(
    forecast_dict: dict,
):
    """Unpack the dictionary that is returned by the StatsForecast 'predict()' method."""
    mu = forecast_dict["mean"]
    std = forecast_dict["hi-68.27"] - mu
    return mu, std
