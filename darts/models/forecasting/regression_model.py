"""
Regression Model
----------------
A `RegressionModel` forecasts future values of a target series based on lagged values of

* The target series (past lags only)

* An optional past_covariates series (past lags only)

* An optional future_covariates series (possibly past and future lags)


The regression models are learned in a supervised way, and they can wrap around any "scikit-learn like" regression model
acting on tabular data having `fit()` and `predict()` functions.

Darts also provides `LinearRegressionModel` and `RandomForest`, which are regression models wrapping around
scikit-learn linear regression and random forest regression, respectively.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.

The lags can be specified either using an integer - in which case it represents the _number_ of (past or future) lags
to take into consideration, or as a list - in which case the lags have to be enumerated (strictly negative values
denoting past lags and positive values including 0 denoting future lags).
"""
from typing import Union, Sequence, Optional, Tuple, List, Dict
import numpy as np

from darts.timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.logging import raise_if, raise_if_not, get_logger, raise_log
from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset
from darts.utils.data.inference_dataset import MixedCovariatesInferenceDataset


logger = get_logger(__name__)


def _shift_matrices(past_matrix: Optional[np.ndarray],
                    future_matrix: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Given two matrices, consumes the first column of the past_matrix (if available), and moves the first column of
    the future_matrix to the end of the past_matrix. This can be seen as a time shift, sending the new past value
    to the past_matrix. `None` will be returned in case a matrix is empty after performing the operation.
    """
    raise_if(future_matrix is None,
             "Future matrix to be shifted cannot be None")

    new_past_matrix = []

    # if more than one timestep in past_matrix add all except for the oldest timestep to new_past_matrix
    if past_matrix.shape[1] > 1:
        new_past_matrix.append(past_matrix[:, 1:, :])

    first_future = future_matrix[:, 0, :]
    first_future = first_future.reshape(first_future.shape[0], 1, first_future.shape[1])
    new_past_matrix.append(first_future)
    future_matrix = future_matrix[:, 1:, :] if future_matrix.shape[1] > 1 else None
    new_past_matrix = np.concatenate(new_past_matrix, axis=1)

    return new_past_matrix, future_matrix


def _update_min_max(current_min: Union[int, None], current_max: Union[int, None],
                    new_values: Sequence[int]) -> Tuple[int, int]:
    """
    Helper function that, given min, max and some values, updates min and max. Min and max can be set to None, if not
    set yet.
    """
    new_min = min(min(new_values), current_min) if current_min is not None else min(new_values)
    new_max = max(max(new_values), current_max) if current_max is not None else max(new_values)
    return new_min, new_max


class RegressionModel(GlobalForecastingModel):
    def __init__(self,
                 lags: Union[int, list] = None,
                 lags_past_covariates: Union[int, List[int]] = None,
                 lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
                 model=None):
        """Regression Model
        Can be used to fit any scikit-learn-like regressor class to predict the target time series from lagged values.

        Parameters
        ----------
        lags
            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags
            are used (from -1 backward). Otherwise a list of integers with lags is required (each lag must be < 0).
        lags_past_covariates
            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last
            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers
            with lags < 0 is required.
        lags_future_covariates
            Number of lagged future_covariates values used to predict the next time step. If an tuple (past, future) is
            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first
            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list
            of integers with lags is required.
        model
            Scikit-learn-like model with `fit()` and `predict()` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
            If None, defaults to: `sklearn.linear_model.LinearRegression(n_jobs=-1)`
        """

        super().__init__()

        self.model = model
        self.lags = None
        self.lags_past_covariates = None
        self.lags_historical_covariates = None
        self.lags_future_covariates = None
        self.min_lag = None
        self.max_lag = None

        # model checks
        if self.model is None:
            self.model = LinearRegression(n_jobs=-1)

        if not callable(getattr(self.model, "fit", None)):
            raise_log(Exception("Provided model object must have a fit() method", logger))
        if not callable(getattr(self.model, "predict", None)):
            raise_log(Exception("Provided model object must have a predict() method", logger))

        # lags checks and processing to arrays
        raise_if((lags is None) and (lags_future_covariates is None) and (lags_past_covariates is None),
                 "At least one of `lags`, `lags_future_covariates` or `lags_past_covariates` must be not None.")

        lags_type_checks = [
            (lags, 'lags'),
            (lags_past_covariates, 'lags_past_covariates'),
        ]

        # checking types
        for _lags, lags_name in lags_type_checks:
            raise_if_not(isinstance(_lags, (int, list)) or _lags is None,
                         f"`{lags_name}` must be of type int or list. Given: {type(_lags)}.")

            raise_if(isinstance(lags, bool),
                     f"`{lags_name}` must be of type int or list, not bool.")

        raise_if_not(isinstance(lags_future_covariates, (tuple, list)) or lags_future_covariates is None,
                     f"`lags_future_covariates` must be of type tuple or list. Given: {type(lags_future_covariates)}.")

        raise_if(isinstance(lags_future_covariates, bool),
                 "`lags_future_covariates` must be of type int or list, not bool.")

        if isinstance(lags_future_covariates, tuple):
            raise_if_not(
                len(lags_future_covariates) == 2 and isinstance(lags_future_covariates[0], int)
                and isinstance(lags_future_covariates[1], int),
                "`lags_future_covariates` tuple must be of length 2, and must contain two integers"
            )
            raise_if(isinstance(lags_future_covariates[0], bool) or isinstance(lags_future_covariates[1], bool),
                     "`lags_future_covariates` tuple must contain intergers, not bool")

        if isinstance(lags, int):
            raise_if_not(
                lags > 0,
                f"`lags` must be strictly positive. Given: {lags}.")
            # selecting last `lags` lags, starting from position 1 (skipping current, pos 0, the one we want to predict)
            self.lags = list(range(-lags, 0))

        elif isinstance(lags, list):
            for lag in lags:
                raise_if(not isinstance(lag, int) or (lag >= 0),
                         f"Every element of `lags` must be a strictly negative integer. Given: {lags}.")
            self.lags = sorted(lags)

        if isinstance(lags_past_covariates, int):
            raise_if_not(
                lags_past_covariates > 0,
                f"`lags_past_covariates` must be an integer > 0. Given: {lags_past_covariates}.")
            self.lags_past_covariates = list(range(-lags_past_covariates, 0))

        elif isinstance(lags_past_covariates, list):
            for lag in lags_past_covariates:
                raise_if(
                    not isinstance(lag, int) or (lag >= 0),
                    f"Every element of `lags_covariates` must be an integer < 0. Given: {lags_past_covariates}.")
            self.lags_past_covariates = sorted(lags_past_covariates)

        if isinstance(lags_future_covariates, tuple):
            raise_if_not(
                lags_future_covariates[0] >= 0 and lags_future_covariates[1] >= 0,
                f"`lags_past_covariates` tuple must contain integers >= 0. Given: {lags_future_covariates}.")

            if lags_future_covariates[0] is not None and lags_future_covariates[0] > 0:
                self.lags_historical_covariates = list(range(-lags_future_covariates[0], 0))
            if lags_future_covariates[1] is not None and lags_future_covariates[1] > 0:
                self.lags_future_covariates = list(range(0, lags_future_covariates[1]))

        elif isinstance(lags_future_covariates, list):
            for lag in lags_future_covariates:
                raise_if(
                    not isinstance(lag, int) or isinstance(lag, bool),
                    f"Every element of `lags_future_covariates` must be an integer. Given: {lags_future_covariates}.")

            lags_future_covariates = np.array(sorted(lags_future_covariates))
            self.lags_historical_covariates = list(lags_future_covariates[lags_future_covariates < 0])
            self.lags_future_covariates = list(lags_future_covariates[lags_future_covariates >= 0])

            if len(self.lags_historical_covariates) == 0:
                self.lags_historical_covariates = None

            if len(self.lags_future_covariates) == 0:
                self.lags_future_covariates = None

        # retrieving min and max lags, since they will be necessary for choosing appropriate input_chunk_size and
        # output_chunk_size later

        if lags is not None:
            # min (index 0) and max (index -1) are enough since the array is already sorted
            self.min_lag, self.max_lag = _update_min_max(self.min_lag, self.max_lag, [self.lags[0], self.lags[-1]])

        lags = [
            self.lags_past_covariates,
            self.lags_historical_covariates,
            self.lags_future_covariates
        ]
        for lag in lags:
            if lag is not None:
                self.min_lag, self.max_lag = _update_min_max(
                    self.min_lag,
                    self.max_lag,
                    [lag[0], lag[-1]]
                )

    def _get_last_prediction_time(self, series, forecast_horizon, overlap_end):
        # overrides the ForecastingModel _get_last_prediction_time, taking care of future lags if any
        extra_shift = max(0, self.max_lag)

        if overlap_end:
            last_valid_pred_time = series.time_index[-1 - extra_shift]
        else:
            last_valid_pred_time = series.time_index[-forecast_horizon - extra_shift]

        return last_valid_pred_time

    def _get_training_data(self, training_dataset: MixedCovariatesSequentialDataset):
        """
        Helper function turning a MixedCovariatesSequentialDataset into training matrices X and y, like required by
        sklearn models.
        """

        training_samples = []
        training_labels = []

        for past_target, past_covariate, historic_future_covariate, future_covariate, future_target in training_dataset:
            row = []
            if self.lags is not None:
                row.append(past_target[self.lags].reshape(1, -1))
            covariates = [
                (past_covariate, self.lags_past_covariates),
                (historic_future_covariate, self.lags_historical_covariates),
                (future_covariate, self.lags_future_covariates)
            ]
            for covariate, lags in covariates:
                if lags is not None:
                    row.append(covariate[lags].reshape(1, -1))

            training_samples.append(np.concatenate(row, axis=1))
            training_labels.append(future_target[0].reshape(1, -1))

        training_samples = np.concatenate(training_samples, axis=0)
        training_labels = np.concatenate(training_labels, axis=0)
        return training_samples, training_labels

    def _create_lagged_data(self, target_series, past_covariates, future_covariates, max_samples_per_ts):
        """
        Helper function that creates training/validation matrices (X and y as required in sklearn), given series and
        max_samples_per_ts.
        """

        # setting proper input and output chunk length. We need to find all necessary lags in the dataset. Thus, we
        # need an input chunk length that contains the oldest past lag (min_lag), and we need to find the most future
        # lag in the future_covariates as well. In case no future covariate are required, we still need an output
        # chunk length of 1 for having the prediction target value.
        training_dataset = MixedCovariatesSequentialDataset(
            target_series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=max(0, -self.min_lag),
            output_chunk_length=max(1, self.max_lag + 1),  # max lag + 1 since we need to access that index
            max_samples_per_ts=max_samples_per_ts
        )

        return self._get_training_data(training_dataset=training_dataset)

    def _fit_model(self, target_series, past_covariates, future_covariates, max_samples_per_ts, **kwargs):
        """
        Function that fit the model. Deriving classes can override this method for adding additional parameters (e.g.,
        adding validation data), keeping the sanity checks on series performed by fit().
        """
        training_samples, training_labels = self._create_lagged_data(
            target_series, past_covariates, future_covariates, max_samples_per_ts)

        # if training_labels is of shape (n_samples, 1) we flatten it to have shape (n_samples,)
        if len(training_labels.shape) == 2 and training_labels.shape[1] == 1:
            training_labels = training_labels.ravel()
        self.model.fit(training_samples, training_labels, **kwargs)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series : TimeSeries or list of TimeSeries
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        past_covariates : TimeSeries or list of TimeSeries, optional
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates : TimeSeries or list of TimeSeries, optional
            Optionally, a series or sequence of series specifying future-known covariates
        max_samples_per_ts : int, optional
            This is an integer upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `fit` method of the model.
        """
        super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates)

        raise_if(
            past_covariates is not None and self.lags_past_covariates is None,
            "`past_covariates` not None in `fit()` method call, but `lags_past_covariates` is None in constructor")

        raise_if(
            past_covariates is None and self.lags_past_covariates is not None,
            "`past_covariates` is None in `fit()` method call, but `lags_past_covariates` is not None in"
            "constructor.")

        raise_if(
            future_covariates is not None and (self.lags_historical_covariates is None
                                               and self.lags_future_covariates is None),
            "`future_covariates` not None in `fit()` method call, but `lags_future_covariates` is None in "
            "constructor.")

        raise_if(
            future_covariates is None and (self.lags_future_covariates is not None
                                           or self.lags_historical_covariates is not None),
            "`future_covariates` is None in `fit()` method call, but `lags_future_covariates` is not None in"
            "constructor.")

        # saving the input dim, so that we can perform the dim check at prediction time
        series_dim = series.width if isinstance(series, TimeSeries) else series[0].width
        covariates_dim = 0
        for covariates in [past_covariates, future_covariates]:
            if covariates is not None:
                if isinstance(covariates, TimeSeries):
                    covariates_dim += covariates.width
                else:
                    covariates_dim += covariates[0].width

        self.input_dim = series_dim + covariates_dim

        # if series is multivariate wrap model with MultiOutputRegressor
        if not series[0].is_univariate:
            self.model = MultiOutputRegressor(self.model, n_jobs=1)

        self._fit_model(series, past_covariates, future_covariates, max_samples_per_ts, **kwargs)

    def _get_prediction_data(
        self,
        prediction_dataset: MixedCovariatesInferenceDataset
    ):
        """
        Helper function which turns a MixedCovariatesInferenceDataset into 5 matrices. The matrices simply contain all
        the samples stacked in the first dimension.
        """
        target_matrix = []
        past_covariates_matrix = []
        historic_future_covariates_matrix = []
        future_covariates_matrix = []
        future_past_covariates_matrix = []

        for (past_target, past_covariates, historic_future_covariates, future_covariates,
             future_past_covariates, _) in prediction_dataset:

            target_matrix.append(past_target)

            if past_covariates is not None:
                past_covariates_matrix.append(past_covariates)
            if future_covariates is not None:
                future_covariates_matrix.append(future_covariates)
            if historic_future_covariates is not None:
                historic_future_covariates_matrix.append(historic_future_covariates)
            if future_past_covariates is not None:
                future_past_covariates_matrix.append(future_past_covariates)

        covariates_matrices = [
            past_covariates_matrix,
            historic_future_covariates_matrix,
            future_covariates_matrix,
            future_past_covariates_matrix
        ]

        for i in range(len(covariates_matrices)):
            covariates_matrices[i] = None if len(covariates_matrices[i]) == 0 else np.asarray(covariates_matrices[i])

        return (
            np.asarray(target_matrix),
            *covariates_matrices
        )

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts values for `n` time steps after the end of the series.
        Parameters
        ----------
        n : int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series : TimeSeries or list of TimeSeries, optional
            Optionally, one or several input `TimeSeries`, representing the history of the target series whose future
            is to be predicted. If specified, the method returns the forecasts of these series. Otherwise, the method
            returns the forecast of the (single) training series.
        past_covariates : TimeSeries or list of TimeSeries, optional
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension and type.
        future_covariates : TimeSeries or list of TimeSeries, optional
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension and type.
        num_samples : int, default: 1
            Currently this parameter is ignored for regression models.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `predict` method of the model. Only works with
            univariate target series.
        """
        super().predict(n, series, past_covariates, future_covariates, num_samples)
        if series is None:
            # then there must be a single TS, and that was saved in super().fit as self.training_series
            raise_if(
                self.training_series is None,
                "Input series has to be provided after fitting on multiple series.",
            )
            series = self.training_series

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = self.past_covariate_series
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = self.future_covariate_series

        called_with_single_series = False

        if isinstance(series, TimeSeries):
            called_with_single_series = True
            series = [series]
            past_covariates = [past_covariates] if past_covariates is not None else None
            future_covariates = [future_covariates] if future_covariates is not None else None

        # check that the input sizes match
        series_dim = series[0].width
        covariates_dim = 0
        for covariates in [past_covariates, future_covariates]:
            if covariates is not None:
                covariates_dim += covariates[0].width
        in_dim = series_dim + covariates_dim

        raise_if_not(
            in_dim == self.input_dim,
            "The dimensionality of the series provided for prediction does not match the dimensionality "
            "of the series this model has been trained on. Provided input dim = {}, "
            "model input dim = {}".format(in_dim, self.input_dim),
        )

        # checking if there are enough future values in the future_covariates. MixedCovariatesInferenceDataset would
        # detect the problem in any case, but the error message would be meaningless.

        if future_covariates is not None:
            for sample in range(len(series)):
                last_req_ts = series[sample].end_time() + (max(n + self.max_lag, n)) * series[sample].freq
                raise_if_not(
                    future_covariates[sample].end_time() >= last_req_ts,
                    "When forecasting future values for a horizon n and lags_future_covariates >= 0, future_covariates"
                    "are requires to be at least `n + max_lags`, with `max_lag` being the furthest lag in the future"
                    f"For the {sample}-th sample, last future covariate timestamp is"
                    f"{future_covariates[sample].end_time()}, whereas it should be at least {last_req_ts}."
                )

        inference_dataset = MixedCovariatesInferenceDataset(
            target_series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=max(n + self.max_lag, n),  # required for retrieving enough future covariates
            input_chunk_length=max(0, -self.min_lag),
            output_chunk_length=1
        )

        # all matrices have dimensions (n_series, time, n_components)
        (
            target_matrix,
            past_covariates_matrix,
            historic_future_covariates_matrix,
            future_covariates_matrix,
            future_past_covariates_matrix
        ) = self._get_prediction_data(inference_dataset)

        predictions = []

        """
        The columns of the prediction matrix need to have the same column order as during the training step, which is
        as follows: lags | lag_cov_0 | lag_cov_1 | .. where each lag_cov_X is a shortcut for
        lag_cov_X_dim_0 | lag_cov_X_dim_1 | .., that means, the lag X value of all the dimension of the covariate
        series (when multivariate).
        """
        for i in range(n):
            # building prediction matrix
            X = []
            if self.lags is not None:
                target_series = target_matrix[:, self.lags]

                X.append(target_series.reshape(len(series), -1))

            covariates_matrices = [
                (past_covariates_matrix, self.lags_past_covariates),
                (historic_future_covariates_matrix, self.lags_historical_covariates),
                (future_covariates_matrix, self.lags_future_covariates)
            ]

            for covariate_matrix, lags in covariates_matrices:
                if lags is not None:
                    covariates = covariate_matrix[:, lags]
                    X.append(covariates.reshape(len(series), -1))

            X = np.concatenate(X, axis=1)
            prediction = self.model.predict(X, **kwargs)
            # reshape to (n_series, time (always 1), n_components)
            prediction = prediction.reshape(len(series), 1, -1)
            # appending prediction to final predictions
            predictions.append(prediction)

            if i < n - 1:
                # discard oldest target
                if target_matrix is not None:
                    target_matrix = target_matrix[:, 1:] if target_matrix.shape[1] > 1 else None
                    # adding new prediction to the target series
                    if target_matrix is None:
                        target_matrix = np.asarray(prediction)
                    else:
                        target_matrix = np.concatenate([target_matrix, prediction], axis=1)

                # discarding oldest covariate
                if past_covariates_matrix is not None:
                    past_covariates_matrix, future_past_covariates_matrix = _shift_matrices(
                        past_covariates_matrix, future_past_covariates_matrix)
                if historic_future_covariates_matrix is not None:
                    historic_future_covariates_matrix, future_covariates_matrix = _shift_matrices(
                        historic_future_covariates_matrix, future_covariates_matrix)

        predictions = np.concatenate(predictions, axis=1)
        predictions = [self._build_forecast_series(row, input_tgt) for row, input_tgt in zip(predictions, series)]

        return predictions[0] if called_with_single_series else predictions

    def __str__(self):
        return self.model.__str__()
