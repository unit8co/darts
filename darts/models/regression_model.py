"""
Regression Model
----------------

A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an covariate series. They can wrap around any regression model having a `fit()`
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
    LaggedTrainingDataset,
    LaggedInferenceDataset,
    _process_lags,
)


logger = get_logger(__name__)


def _consume_column(m: np.ndarray) -> Optional[np.ndarray]:
    """
    Deletes the first column of the given matrix. In case the column is only one, returns `None`.

    Params
    ------
    m
        np.array representing a matrix.

    Returns
    -------
    Optional[np.ndarray]
        The passed matrix with the first column removed. `None` in case the initial matrix was single-columned.
    """

    raise_if_not(
        len(m.shape) >= 2,
        f"The passed array must have at least 2 dimensions, found {len(m.shape)}"
    )

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
        lags
            Number of lagged target values used to predict the next time step. If an integer is given the last `lags`
            lags are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
        lags_covariates
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
            integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t`
            should be used. Note that the 0 index is not included when passing a single interger value > 0.
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

        # getting the indices from the lags
        if self.lags is not None:
            self.lags_indices = np.array(self.lags) * (-1)
        else:
            self.lags_indices = None

        if self.lags_covariates is not None:
            if 0 in self.lags_covariates:
                # +1 since -0 must be turned into -1 and the other must be shifted
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
            This is an upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
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
        lagged_dataset = LaggedTrainingDataset(
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

    def fit_from_dataset(
        self,
        dataset: LaggedTrainingDataset, **kwargs
    ):
        """
        Fit the model against the given `LaggedTrainingDataset`.
        """
        training_x, training_y = dataset.get_data()
        self.model.fit(training_x, training_y, **kwargs)

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n : int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series
            Optionally, one or several input `TimeSeries`, representing the history of the target series whose future
            is to be predicted. If specified, the method returns the forecasts of these series. Otherwise, the method
            returns the forecast of the (single) training series.
        covariates
            Optionally, the covariates series needed as inputs for the model. They must match the covariates used
            for training in terms of dimension and type.
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

    def predict_from_dataset(
        self,
        dataset: LaggedInferenceDataset,
        n: int = 1,
        **kwargs
    ):
        """
        Forecasts values for `n` time steps after the end of the series contained in the given dataset.
        """

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

                        covariates_matrix = np.concatenate(new_cov_matrix, axis=1)
                        future_covariates_matrix = _consume_column(future_covariates_matrix)

                        # TODO move this to a check
                        raise_if(future_covariates_matrix is None and i != n - 2, "future covariates not sufficiently"
                                 "long")

        predictions = np.concatenate(predictions, axis=1)
        return [self._build_forecast_series(row, input) for row, (input, _, _) in zip(predictions, dataset)]

    def _get_matrix_data_from_dataset(
        self,
        dataset: LaggedInferenceDataset
    ):
        """
        Helper function which turns a LaggedInferenceDataset into 3 matrices.
        """
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
