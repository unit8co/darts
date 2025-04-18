"""
StatsForecastModel
------------------
"""

from typing import Optional

import numpy as np
from statsforecast.models import _TS

from darts import TimeSeries, concatenate
from darts.logging import get_logger
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.utils.likelihood_models.statsforecast import QuantilePrediction
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.utils import random_method

logger = get_logger(__name__)


class StatsForecastModel(TransferableFutureCovariatesLocalForecastingModel):
    @random_method
    def __init__(
        self,
        model: _TS,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
    ):
        """StatsForecast Model.

        Can be used to fit any `StatsForecast` base model. For more information on available models, see the
        `StatsForecast package <https://nixtlaverse.nixtla.io/statsforecast/index.html>`_.

        In addition to univariate deterministic forecasting, our `StatsForecastModel` comes with additional support:

        - **Future covariates:** Use exogenous features to potentially improve predictive accuracy.

          - It either uses the base model's native exogenous features, or

          - It adds future covariates support by first regressing the series against the future covariates using a
            :class:`~darts.models.forecasting.linear_regression_model.LinearRegressionModel` model and then running the
            StatsForecast model on the in-sample residuals from this original regression. This approach was inspired by
            `this post of Stephan Kolassa <https://stats.stackexchange.com/q/220885>`_.

        - **Probabilstic forecasting:** Some base models might require setting `prediction_intervals` at `model`
          creation to support probabilistic forecasting. To generate probabilistic forecasts, you can set the following
          parameters when calling :meth:`~darts.models.forecasting.sf_model.StatsForecastModel.predict`:

          - Forecast quantile values directly by setting `predict_likelihood_parameters=True`.

          - Generate sampled forecasts from these quantiles by setting `num_samples >> 1`.

        - **Conformal prediction:** In addition to the native probabilistic support, you can perform conformal
          prediction / forecasting by setting `prediction_intervals` at model creation. Then predict the in the same
          way as described above.

        - **Transferable series forecasting:** Apply the fitted model to a new input `series` at prediction time.

          - It either uses the base model's native transferrable series support (StatsForecast models that support the
            `forward()` method), or

          - It adds support by first fitting a copy of the model on the new series, and then using that model to
            generate the corresponding forecast.

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
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.

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
        array([[445.4276575 ],
               [420.04912881],
               [448.7142377 ],
               [491.23406559],
               [502.67834069],
               [566.04774778]])
        """
        if not isinstance(model, _TS):
            raise ValueError(
                "`model` must be a StatsForecast model imported from `statsforecast.models`."
            )
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
            self._linreg = LinearRegressionModel(lags_future_covariates=[0])
            self._linreg.fit(series, future_covariates=future_covariates)
            target = self._get_target_residuals(series, future_covariates)

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

    @random_method
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

        if series is not None and not self._supports_native_transferable_series:
            # if model doesn't support transferable series forecasts, we fit and predict using a copy of the model
            if future_covariates is not None:
                # merge covariates
                future_covariates = concatenate(
                    [historic_future_covariates, future_covariates], axis=0
                )

                encoders = self.encoders
                if encoders is not None and encoders.encoding_available:
                    # drop encoded covariates from the covariates
                    future_covariates = encoders._drop_encoded_components(
                        covariates=future_covariates,
                        components=encoders.future_components,
                    )

            return (
                self.untrained_model()
                .fit(
                    series=series,
                    future_covariates=future_covariates,
                )
                .predict(
                    n=n,
                    num_samples=num_samples,
                    predict_likelihood_parameters=predict_likelihood_parameters,
                    verbose=verbose,
                )
            )

        # confidence levels for prediction intervals
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

        # likelihood takes care of probabilistic forecasts
        pred_vals = self.likelihood.predict(model_output, num_samples=num_samples)

        comp_names_out = None
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

        When this method is called, it is guaranteed that either `series` is None, or that the
        model supports transferable series forecasting (model has a forward method).

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        series
            The series whose future values will be predicted. If the statsforecast model has a `forward` method then
            the previously fitted model will be used for prediction, otherwise a copy of the model will be fitted to
            predict the new series.
        historic_future_covariates
            Optionally, the historic part of the future-known covariates series that is overlapping with `series`.
            They must match the covariates used for training in terms of dimension.
        future_covariates
            Optionally, the future-known covariates series that extends to at least `n` steps after the end of `series`.
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

        if series is None:
            forecast_dict = self.model.predict(h=n, X=x_future, level=levels)
        else:
            # model has a `forward` method, it supports transferable prediction series (uses
            # fitted model to forecast new series)
            if custom_cov_support:
                target = self._get_target_residuals(
                    series,
                    historic_future_covariates,
                )
            else:
                target = series.values(copy=False).flatten()

            x_historic = (
                historic_future_covariates.values(copy=False)
                if native_cov_support
                else None
            )
            forecast_dict = self.model.forward(
                y=target, h=n, X=x_historic, X_future=x_future, level=levels
            )

        vals = _unpack_sf_dict(forecast_dict, levels=levels)
        if custom_cov_support:
            # sf model was trained on residuals, add back the remainder
            mu_linreg = self._linreg.predict(
                n, series=series, future_covariates=future_covariates
            )
            mu_linreg_values = mu_linreg.values(copy=False).reshape(n, 1)
            vals += mu_linreg_values
        return vals

    def _get_target_residuals(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries,
    ) -> np.ndarray:
        """Computes the OLS residuals for predicting the target series from `future_covariates`."""
        fitted_values = self._linreg.model.predict(
            X=future_covariates.slice_intersect_values(series, copy=False)[:, :, 0]
        )
        residuals = series.values(copy=False).flatten() - fitted_values
        return residuals

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
    def _supports_native_transferable_series(self) -> bool:
        return hasattr(self.model, "forward")

    @property
    def _supports_native_future_covariates(self) -> bool:
        return self.model.uses_exog

    @property
    def likelihood(self) -> QuantilePrediction:
        return self._likelihood

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return self._supports_native_transferable_series


class _SFModel(_TS):
    """This serves as a protocol for expected StatsForecast model API."""

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
