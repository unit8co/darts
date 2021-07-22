"""
Regression Model
----------------

A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an exogenous series. They can wrap around any regression model having a `fit()`
and `predict()` functions accepting tabularized data (e.g. scikit-learn regression models), and are using
`sklearn.linear_model.LinearRegression` by default.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.
"""
from typing import Union, Optional, Sequence
import numpy as np

from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import GlobalForecastingModel
from ..logging import raise_if, raise_if_not, get_logger, raise_log
from darts.utils.data.lagged_dataset import (
    LaggedDataset,
    LaggedInferenceDataset,
    _process_lags,
)
from darts.utils.data.matrix_dataset import MatrixTrainingDataset


logger = get_logger(__name__)


def _consume_column(m: np.ndarray) -> Union[None, np.ndarray]:
    if m.shape[1] == 1:
        return None
    else:
        return m[:, 1:]


class RegressionModel(GlobalForecastingModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_covariates: Union[int, list] = None,
        model=None,
    ):
        """Regression Model

        Can be used to fit any scikit-learn-like regressor class to predict the target
        time series from lagged values.

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be strictly positive (>0).
        lags_covariates : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive). Otherwise a list of integers with lags is required.
            The integers must be positive (>=0).
        model
            Scikit-learn-like model with `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """

        super().__init__()
        if model is None:
            model = LinearRegression()

        if not callable(getattr(model, "fit", None)):
            raise_log(
                Exception("Provided model object must have a fit() method", logger)
            )
        if not callable(getattr(model, "predict", None)):
            raise_log(
                Exception("Provided model object must have a predict() method", logger)
            )

        self.model = model

        # turning lags into array of int or None
        self.lags, self.lags_covariates = _process_lags(lags, lags_covariates)

        if self.lags is not None:
            self.lags_indices = np.array(self.lags) * (-1)
        else:
            self.lags_indices = None

        if self.lags_covariates is not None:
            if 0 in self.lags_covariates:
                self.cov_lags_indices = (np.array(self.lags_covariates) + 1) * (-1)
            else:
                self.cov_lags_indices = (np.array(self.lags_covariates)) * (-1)
        else:
            self.cov_lags_indices = None

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts=None,
        **kwargs
    ) -> None:
        """Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series : Union[TimeSeries, Sequence[TimeSeries]]
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        covariates : Union[TimeSeries, Sequence[TimeSeries]], optional
            TimeSeries or Sequence[TimeSeries] object containing the exogenous values.
        max_samples_per_ts : int
            # TODO describe param
        """
        super().fit(series, covariates)
        raise_if(
            covariates is not None and self.lags_covariates is None,
            "`covariates` not None in `fit()` method call, but `lags_covariates` is None in constructor. ",
        )
        raise_if(
            covariates is None and self.lags_covariates is not None,
            "`covariates` is None in `fit()` method call, but `lags_covariates` is not None in constructor. ",
        )
        lagged_dataset = LaggedDataset(
            series,
            covariates,
            self.lags,
            self.lags_covariates,
            max_samples_per_ts,
        )
        self.fit_from_dataset(lagged_dataset, **kwargs)

        series = [series] if isinstance(series, TimeSeries) else series
        covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        self.input_dim = (0 if covariates is None else covariates[0].width) + series[0].width

    def fit_from_dataset(self, dataset: MatrixTrainingDataset, **kwargs):
        training_x, training_y = dataset.get_data()
        self.model.fit(training_x, training_y, **kwargs)

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n : int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        # TODO add doc
        num_samples
            Currently this parameter is ignored for regression models.
        **kwargs
            Additional keyword arguments passed to the `predict` method of the model.
        """
        super().predict(n, series, covariates, num_samples)

        if series is None:
            # then there must be a single TS, and that was saved in super().fit as self.training_series
            raise_if(
                self.training_series is None,
                "Input series has to be provided after fitting on multiple series.",
            )
            series = self.training_series

        if covariates is None and self.covariate_series is not None:
            covariates = self.covariate_series

        called_with_single_series = False
        if isinstance(series, TimeSeries):
            called_with_single_series = True
            series = [series]

        covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        # check that the input sizes match
        in_dim = (0 if covariates is None else covariates[0].width) + series[0].width

        raise_if_not(
            in_dim == self.input_dim,
            "The dimensionality of the series provided for prediction does not match the dimensionality "
            "of the series this model has been trained on. Provided input dim = {}, "
            "model input dim = {}".format(in_dim, self.input_dim),
        )

        dataset = LaggedInferenceDataset(
            series, covariates, self.lags, self.lags_covariates, n
        )

        predictions = self.predict_from_dataset(dataset, n, **kwargs)

        return predictions[0] if called_with_single_series else predictions

    def predict_from_dataset(self, dataset: LaggedInferenceDataset, n: int = 1, **kwargs):

        target_matrix, covariates_matrix, future_covariates_matrix = self._get_matrix_data_from_dataset(dataset)

        predictions = []

        for i in range(n):
            # getting training matrix
            X = []
            if self.lags_indices is not None:
                target_series = target_matrix[:, self.lags_indices]
                X.append(target_series)

            if self.cov_lags_indices is not None:
                covariates = covariates_matrix[:, self.cov_lags_indices]
                # reshaping since we could have multivariate covariates
                X.append(covariates.reshape(covariates.shape[0], -1))

            X = np.concatenate(X, axis=1)
            # predicting
            prediction = self.model.predict(X, **kwargs)
            prediction = prediction.reshape(-1, 1)
            # appending prediction to final predictions
            predictions.append(prediction)


            # discard oldest target
            if target_matrix is not None:
                target_matrix = _consume_column(target_matrix)
                # adding new prediction to the target series
                if target_matrix is None:
                    target_matrix = np.asarray(prediction)
                else:
                    target_matrix = np.concatenate([target_matrix, prediction], axis=1)

            # shifting matrices for the next step
            if i < n - 1:

                # discarding oldest covariate
                if covariates_matrix is not None:
                    covariates_matrix = _consume_column(covariates_matrix)
                    new_cov_matrix = []
                    if covariates_matrix is not None:
                        new_cov_matrix = [covariates_matrix]

                    if future_covariates_matrix is not None:
                        first_future = future_covariates_matrix[:, 0, :]
                        first_future = first_future.reshape(first_future.shape[0], 1, first_future.shape[1])
                        new_cov_matrix.append(first_future)

                        covariates_matrix = np.concatenate(new_cov_matrix,axis=1)
                        future_covariates_matrix = _consume_column(future_covariates_matrix)

                        # TODO move this to a check
                        raise_if( future_covariates_matrix is None and i != n - 2, "future covariates not "
                                                                                   "sufficiently long")

        predictions = np.concatenate(predictions, axis=1)
        return [self._build_forecast_series(row, input) for row, (input, _, _) in zip(predictions, dataset)]

    def _get_matrix_data_from_dataset(self, dataset: LaggedInferenceDataset):

        target_matrix = []
        covariates_matrix = []
        future_covariates_matrix = []

        for tgt_series, past_covariates, future_covariates in dataset:
            target_matrix.append(tgt_series.values().T)

            if past_covariates is not None:
                covariates_matrix.append(past_covariates.values())

            if future_covariates is not None:
                future_covariates_matrix.append(future_covariates.values())


        target_matrix = np.concatenate(target_matrix, axis=0)
        covariates_matrix = None if len(covariates_matrix) == 0 else np.asarray(covariates_matrix)
        future_covariates_matrix = None if len(future_covariates_matrix) == 0 else np.asarray(future_covariates_matrix)

        return target_matrix, covariates_matrix, future_covariates_matrix

    def __str__(self):
        return self.model.__str__()
