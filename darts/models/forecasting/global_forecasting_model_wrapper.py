"""
Regression ensemble model
-------------------------

An ensemble model which uses a regression model to compute the ensemble forecast.
"""
from typing import List, Optional, Sequence, Tuple, Union

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
    GlobalForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries, concatenate
from darts.utils.utils import seq2series, series2seq

logger = get_logger(__name__)


class GlobalForecastingModelWrapper(GlobalForecastingModel):
    def __init__(self, model: FutureCovariatesLocalForecastingModel):
        """
        Use a regression model for ensembling individual models' predictions using the stacking technique [1]_.

        The provided regression model must implement ``fit()`` and ``predict()`` methods
        (e.g. scikit-learn regression models). Note that here the regression model is used to learn how to
        best ensemble the individual forecasting models' forecasts. It is not the same usage of regression
        as in :class:`RegressionModel`, where the regression model is used to produce forecasts based on the
        lagged series.

        If `future_covariates` or `past_covariates` are provided at training or inference time,
        they will be passed only to the forecasting models supporting them.

        The regression model does not leverage the covariates passed to ``fit()`` and ``predict()``.

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        regression_train_n_points
            The number of points to use to train the regression model
        regression_model
            Any regression model with ``predict()`` and ``fit()`` methods (e.g. from scikit-learn)
            Default: ``darts.model.LinearRegressionModel(fit_intercept=False)``

            .. note::
                if `regression_model` is probabilistic, the `RegressionEnsembleModel` will also be probabilistic.
            ..
        regression_train_num_samples
            Number of prediction samples from each forecasting model to train the regression model (samples are
            averaged). Should be set to 1 for deterministic models. Default: 1.

            .. note::
                if `forecasting_models` contains a mix of probabilistic and deterministic models,
                `regression_train_num_samples will be passed only to the probabilistic ones.
            ..
        regression_train_samples_reduction
            If `forecasting models` are probabilistic and `regression_train_num_samples` > 1, method used to
            reduce the samples before passing them to the regression model. Possible values: "mean", "median"
            or float value corresponding to the desired quantile. Default: "median"
        show_warnings
            Whether to show warnings related to forecasting_models covariates support.
        References
        ----------
        .. [1] D. H. Wolpert, “Stacked generalization”, Neural Networks, vol. 5, no. 2, pp. 241–259, Jan. 1992
        """
        super().__init__()

        self.base_model: FutureCovariatesLocalForecastingModel = model
        self._trained_models: List[List[FutureCovariatesLocalForecastingModel]] = []

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):
        self._trained_models = []
        super().fit(series)

        for multivariate_ts in series2seq(series):
            components = self._split_multivariate(multivariate_ts)
            if self.supports_future_covariates:
                series_models = [
                    self.base_model.untrained_model().fit(c, future_covariates)
                    for c in components
                ]
            else:
                series_models = [
                    self.base_model.untrained_model().fit(c) for c in components
                ]
            self._trained_models.append(series_models)

        return self

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        result = []
        for ts_models in self._trained_models:
            if self.supports_future_covariates:
                predictions = [
                    model.predict(n=n, future_covariates=future_covariates)
                    for model in ts_models
                ]
            else:
                predictions = [model.predict(n=n) for model in ts_models]
            multivariate_series = concatenate(predictions, axis=1)
            result.append(multivariate_series)
        return seq2series(result)

    def _split_multivariate(self, time_series: TimeSeries):
        """ " split multivariate TimeSeries into a list of univariate TimeSeries"""
        return [
            time_series.univariate_component(i) for i in range(time_series.n_components)
        ]

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        return self.base_model.extreme_lags

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return None, None, False, self.supports_future_covariates, None, None

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return self.base_model.supports_future_covariates

    @property
    def supports_static_covariates(self) -> bool:
        return False

    def _is_probabilistic(self) -> bool:
        """
        A RegressionEnsembleModel is probabilistic if its regression
        model is probabilistic (ensembling layer)
        """
        return self.base_model._is_probabilistic()

    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return isinstance(
            self.base_model, TransferableFutureCovariatesLocalForecastingModel
        )

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        return isinstance(
            self.base_model, TransferableFutureCovariatesLocalForecastingModel
        )
