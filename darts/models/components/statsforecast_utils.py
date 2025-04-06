from abc import ABC, abstractmethod
from typing import Optional

from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
)
from darts.utils.likelihood_models.statsforecast import StatsForecastLikelihood
from darts.utils.timeseries_generation import _build_forecast_series


class StatsForecastFutureCovariatesLocalModel(
    FutureCovariatesLocalForecastingModel, ABC
):
    def __init__(
        self,
        model,
        likelihood: Optional[StatsForecastLikelihood],
        add_encoders: Optional[dict] = None,
    ):
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
        if self.likelihood is not None:
            pred_vals = self.likelihood.predict(
                model=self,
                n=n,
                future_covariates=future_covariates,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                **kwargs,
            )
        else:
            pred_vals = self.model.predict(
                h=n,
                X=future_covariates.values(copy=False) if future_covariates else None,
                level=None,
            )["mean"]

        series = self.training_series
        comp_names_out = (
            self.likelihood.component_names(series)
            if predict_likelihood_parameters
            else None
        )
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
    @abstractmethod
    def _supports_native_future_covariates(self) -> bool:
        return False

    @property
    def likelihood(self) -> StatsForecastLikelihood:
        return self._likelihood
