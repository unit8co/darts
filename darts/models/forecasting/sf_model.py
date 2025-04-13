"""
StatsForecastModel
------------------
"""

from typing import Optional

import numpy as np
from statsforecast.models import _TS

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.utils.likelihood_models.statsforecast import QuantilePrediction
from darts.utils.timeseries_generation import _build_forecast_series


class StatsForecastModel(TransferableFutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        model: _TS,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
    ):
        """StatsForecast Model.

        Can be used to fit any `StatsForecast` model. For more information on available models, see the
        `StatsForecast package <https://github.com/Nixtla/statsforecast>`_.

        All models come with future covariates support:

        - It either uses the model's native exogenous features, or
        - It adds future covariates support by first regressing the series against the future covariates using a
          :class:'LinearRegressionModel' model and then running the StatsForecast model on the in-sample residuals from
          this original regression. This approach was inspired by 'this post of Stephan Kolassa
          <https://stats.stackexchange.com/q/220885>'_.

        All models come with transferrable `series` support (applying the fitted model to a new input `series` at
        prediction time):

        - It either uses the model's native transferrable series support (StatsForecast models that have support the
          `forward()` method), or
        - It adds support by re-fitting a copy of the model on the new series and then generating the forecast for it
          using the StatsForecast model's `forecast()` method.

        .. note::
            Future covariates are not supported when the input series contain missing values.

        Parameters
        ----------
        model
            Any StatsForecast model.
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
        quantiles
            Optionally, produce quantile predictions at `quantiles` levels when performing probabilistic forecasting
            with `num_samples > 1` or `predict_likelihood_parameters=True`.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastModel
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> from statsforecast.models import AutoARIMA
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define AutoARIMA parameters
        >>> model = StatsForecastModel(model=AutoARIMA(season_length=12))
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

        self.model: _SFModel = model
        self._likelihood = QuantilePrediction(quantiles=quantiles)

        # future covariates support can be added through the use of a linear model
        self._linreg: Optional[LinearRegressionModel] = None
        super().__init__(add_encoders=add_encoders)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series

        if self._supports_native_future_covariates or future_covariates is None:
            target = series.values(copy=False).flatten()
        else:
            # perform OLS and get in-sample residuals
            target, self._linreg = self._get_target_residuals(
                series, future_covariates, fit=True
            )

        self.model.fit(
            target,
            X=(
                future_covariates.values(copy=False)
                if future_covariates is not None
                and self._supports_native_future_covariates
                else None
            ),
        )
        return self

    def _predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        historic_future_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        verbose: bool = False,
    ) -> TimeSeries:
        super()._predict(
            n, series, historic_future_covariates, future_covariates, num_samples
        )

        levels = (
            self.likelihood.levels
            if num_samples > 1 or predict_likelihood_parameters
            else None
        )
        model_output = self._estimator_predict(
            n=n,
            series=series,
            historic_future_covariates=historic_future_covariates,
            future_covariates=future_covariates,
            levels=levels,
        )

        series = series if series is not None else self.training_series
        comp_names_out = None

        pred_vals = self.likelihood.predict(
            model_output,
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
        )
        if predict_likelihood_parameters:
            comp_names_out = self.likelihood.component_names(series)

        return _build_forecast_series(
            points_preds=pred_vals,
            input_series=series,
            custom_columns=comp_names_out,
            with_static_covs=not predict_likelihood_parameters,
            with_hierarchy=not predict_likelihood_parameters,
        )

    def _estimator_predict(
        self,
        n: int,
        series: Optional[TimeSeries],
        historic_future_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        levels: Optional[list[float]],
    ) -> np.ndarray:
        """
        Computes the model output.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        historic_future_covariates
            Optionally,
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        levels
            The confidence levels (0. - 100.) for the prediction intervals.
        """
        native_cov_support = (
            future_covariates is not None and self._supports_native_future_covariates
        )
        custom_cov_support = (
            future_covariates is not None
            and not self._supports_native_future_covariates
        )

        x_future = future_covariates.values(copy=False) if native_cov_support else None

        linreg_model = self._linreg
        if series is None:
            forecast_dict = self.model.predict(h=n, X=x_future, level=levels)
        else:
            # if model has a `forward` method, it supports transferable prediction series
            # (uses fitted model to forecast new series). Otherwise, we use the `forecast` method which
            # performs `fit` (re-fit) and `predict()`
            has_forward = hasattr(self.model, "forward")
            if custom_cov_support:
                target, linreg_model = self._get_target_residuals(
                    series,
                    historic_future_covariates,
                    fit=not has_forward,  # refit if underlying model will be re-fit
                )
            else:
                target = series.values(copy=False).flatten()

            x_historic = (
                historic_future_covariates.values(copy=False)
                if native_cov_support
                else None
            )
            fc_method = self.model.forward if has_forward else self.model.forecast
            forecast_dict = fc_method(
                y=target, h=n, X=x_historic, X_future=x_future, level=levels
            )

        vals = _unpack_sf_dict(forecast_dict, levels=levels)
        if custom_cov_support:
            # sf model was trained on residuals, add back the remainder
            mu_linreg = linreg_model.predict(n, future_covariates=future_covariates)
            mu_linreg_values = mu_linreg.values(copy=False).reshape(n, 1)
            vals += mu_linreg_values
        return vals

    def _get_target_residuals(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries,
        fit: bool,
    ) -> tuple[np.ndarray, LinearRegressionModel]:
        """Computes the OLS residuals for predicting the target series from `future_covariates`."""
        if fit:
            model = LinearRegressionModel(lags_future_covariates=[0])
            model.fit(series, future_covariates=future_covariates)
        else:
            model = self._linreg
        fitted_values = model.model.predict(
            X=future_covariates.slice_intersect_values(series, copy=False)[:, :, 0]
        )
        residuals = series.values(copy=False).flatten() - fitted_values
        return residuals, model

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
        return self.likelihood is not None

    @property
    def _supports_native_future_covariates(self) -> bool:
        return self.model.uses_exog

    @property
    def likelihood(self) -> QuantilePrediction:
        return self._likelihood


class _SFModel(_TS):
    def fit(self, *args, **kwargs): ...

    def predict(self, *args, **kwargs) -> dict: ...

    def forecast(self, *args, **kwargs) -> dict: ...

    def forward(self, *args, **kwargs) -> dict: ...


def _unpack_sf_dict(
    forecast_dict: dict,
    levels: Optional[list[float]],
) -> np.ndarray:
    """Unpack the dictionary that is returned by the StatsForecast 'predict()' method.

    Into an array of quantile predictions with shape (n (horizon), n quantiles) ordered by increasing quantile.
    """
    mu = np.expand_dims(forecast_dict["mean"], -1)
    if levels is None:
        return mu

    lows = np.concatenate(
        [np.expand_dims(forecast_dict[f"lo-{level}"], -1) for level in levels], axis=1
    )
    highs = np.concatenate(
        [np.expand_dims(forecast_dict[f"hi-{level}"], -1) for level in levels[::-1]],
        axis=1,
    )
    return np.concatenate([lows, mu, highs], axis=1)
