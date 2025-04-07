"""
AutoARIMA
---------
"""

from typing import Optional

import numpy as np
from statsforecast.models import _TS

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)
from darts.utils.likelihood_models.statsforecast import StatsForecastLikelihood
from darts.utils.timeseries_generation import _build_forecast_series


class StatsForecastModel(FutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        model: _TS,
        *args,
        likelihood: Optional[StatsForecastLikelihood] = None,
        add_encoders: Optional[dict] = None,
        **kwargs,
    ):
        """StatsForecast Model.

        Can be used to fit any `StatsForecast` model. For more information on available models, see the
        `StatsForecast package <https://github.com/Nixtla/statsforecast>`_.

        In addition to the StatsForecast models that do not support exogenous features natively, this model can handle
        future covariates. It does so by first regressing the series against the future covariates using the
        :class:'LinearRegressionModel' model and then running the StatsForecast model on the in-sample residuals from
        this original regression. This approach was inspired by 'this post of Stephan Kolassa
        <https://stats.stackexchange.com/q/220885>'_.

        .. note::
            Future covariates are not supported when the input series contain missing values.

        Parameters
        ----------
        model
            Any StatsForecast model.
        args
            Positional arguments to create an instance of `model`.
        likelihood
            Any of Darts' `StatsForecastLikelihood`.
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
        kwargs
            Keyword arguments to create an instance of `model`.
        """

        self.model = model
        self._likelihood = likelihood

        # future covariates support can be added through the use of a linear model
        self._linreg: Optional[LinearRegressionModel] = None
        super().__init__(add_encoders=add_encoders)

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._assert_univariate(series)
        series = self.training_series

        if self._supports_native_future_covariates or future_covariates is None:
            target = series
        else:
            # perform OLS and get in-sample residuals
            self._linreg = LinearRegressionModel(lags_future_covariates=[0])
            self._linreg.fit(series, future_covariates=future_covariates)
            fitted_values = self._linreg.model.predict(
                X=future_covariates.slice_intersect(series).values(copy=False)
            )
            fitted_values_ts = TimeSeries.from_times_and_values(
                times=series.time_index, values=fitted_values
            )
            residuals = series - fitted_values_ts
            target = residuals

        self.model.fit(
            target.values(copy=False).flatten(),
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
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super()._predict(n, future_covariates, num_samples)

        series = self.training_series
        comp_names_out = None
        if self.likelihood is not None:
            # probabilistic forecast
            pred_vals = self.likelihood.predict(
                model=self,
                n=n,
                future_covariates=future_covariates,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                **kwargs,
            )
            comp_names_out = self.likelihood.component_names(series)
        else:
            # mean forecast
            pred_vals = self.model.predict(
                h=n,
                X=future_covariates.values(copy=False) if future_covariates else None,
                level=None,
            )
            pred_vals = np.expand_dims(pred_vals["mean"], -1)

            if (
                future_covariates is not None
                and not self._supports_native_future_covariates
            ):
                # statsforecast model was trained on residuals, add the linear model predictions
                mu_linreg = self._linreg.predict(n, future_covariates=future_covariates)
                mu_linreg_values = mu_linreg.values(copy=False).reshape(n, 1)
                pred_vals += mu_linreg_values

        return _build_forecast_series(
            points_preds=pred_vals,
            input_series=self.training_series,
            custom_columns=comp_names_out,
            with_static_covs=not predict_likelihood_parameters,
            with_hierarchy=not predict_likelihood_parameters,
        )

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
    def likelihood(self) -> StatsForecastLikelihood:
        return self._likelihood
