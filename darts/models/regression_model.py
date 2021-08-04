"""
Regression Model
----------------
A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an covariate series. They can wrap around any regression model having a `fit()`
and `predict()` functions accepting tabularized data (e.g. scikit-learn regression models), and are using
`sklearn.linear_model.LinearRegression` by default.
Behind the scenes this model is tabularizing the time series data to make it work with regression models.
"""
from typing import Union, Sequence, Optional, Tuple
import numpy as np

from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import GlobalForecastingModel
from ..logging import raise_if, raise_if_not, get_logger, raise_log
from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset
from darts.utils.data.inference_dataset import MixedCovariatesInferenceDataset


logger = get_logger(__name__)


def shift_matrices(past_matrix, future_matrix):
    """
    TODO add doc
    """
    raise_if(future_matrix is None,
             "Future matrix to be shifted cannot be None")

    new_past_matrix = []

    if past_matrix is not None:
        past_matrix = past_matrix[:, 1:, :] if past_matrix.shape[1] > 1 else None

    if past_matrix is not None:
        new_past_matrix = [past_matrix]

    first_future = future_matrix[:, 0, :]
    first_future = first_future.reshape(first_future.shape[0], 1, first_future.shape[1])
    new_past_matrix.append(first_future)
    future_matrix = future_matrix[:, 1:, :] if future_matrix.shape[1] > 1 else None
    new_past_matrix = np.concatenate(new_past_matrix, axis=1)

    return new_past_matrix, future_matrix


def min_with_none(arr: list):
    return min(filter(lambda x: x is not None, arr))


def max_with_none(arr: list):
    return max(filter(lambda x: x is not None, arr))


class RegressionModel(GlobalForecastingModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, list] = None,
        lags_future_covariates: Union[Tuple[int, int], list] = None,
        model=None,
    ):
        """Regression Model
        Can be used to fit any scikit-learn-like regressor class to predict the target
        time series from lagged values.
        Parameters
        ----------
        lags TODO fix
            Number of lagged target values used to predict the next time step. If an integer is given the last `lags`
            lags are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
        lags_covariates TODO fix
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
            integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t`
            should be used. Note that the 0 index is not included when passing a single interger value > 0.
        model
            Scikit-learn-like model with `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
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
            self.model = LinearRegression()

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

        # setting min_lag and max_lag != None just if we have some lags in those directions. In other words,
        # leaving min_lag to None if only future lags are required (>= prediction time), and leaving max_lag to None,
        # in case only past lags are required (< prediction time)
        if lags is not None:
            self.min_lag = self.lags[0]  # 0 index will be the min

        if self.lags_past_covariates is not None:
            self.min_lag = min_with_none([self.min_lag, self.lags_past_covariates[0]])

        if self.lags_historical_covariates is not None:
            self.min_lag = min_with_none([self.min_lag, self.lags_historical_covariates[0]])

        if self.lags_future_covariates is not None:
            self.max_lag = self.lags_future_covariates[-1] + 1  # + 1 since lag 0 is already in the future of 1 step

        self.input_chunk_length = -min_with_none([self.min_lag, 0])
        self.training_output_chunk_length = max_with_none([self.max_lag, 1])

    def _get_training_data(self, training_dataset: MixedCovariatesSequentialDataset):
        """
        TODO write doc
        """

        training_samples = []
        training_labels = []

        # TODO parallelise matrix building. We could check if stacking everything and selecting the lags in the final
        # matrix is more efficient then masking single rows.

        # TODO fix view thingy, appending is ok, but np stuff is probably creating a copy
        for past_target, past_covariate, historic_future_covariate, future_covariate, future_target in training_dataset:
            row = []
            if self.lags is not None:
                row.append(past_target[self.lags].T)

            covariates = [
                (past_covariate, self.lags_past_covariates),
                (historic_future_covariate, self.lags_historical_covariates),
                (future_covariate, self.lags_future_covariates)
            ]

            for covariate, lags in covariates:
                if covariate is not None:
                    row.append(covariate[lags].reshape(1, -1))

            training_samples.append(np.concatenate(row, axis=1))
            # discard other future values which were retrived just because we need the covariates
            training_labels.append(future_target[0])
        training_samples = np.concatenate(training_samples, axis=0)
        training_labels = np.concatenate(training_labels, axis=0).ravel()

        return training_samples, training_labels

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts=None,
        **kwargs
    ) -> None:
        """Fits/trains the model using the provided list of features time series and the target time series.
        Parameters
        ----------
        series : Union[TimeSeries, Sequence[TimeSeries]]
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        covariates : Union[TimeSeries, Sequence[TimeSeries]], optional TODO fix
            TimeSeries or Sequence[TimeSeries] object containing the exogenous values.
        max_samples_per_ts : int
            This is an upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """
        super().fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates)

        checks = [
            (past_covariates, 'past_covariates', self.lags_past_covariates, 'self.lags_past_covariates'),
            (future_covariates, 'future_covariates', self.lags_future_covariates, 'self.lags_future_covariates')
        ]

        for s, s_name, l, l_name in checks:
            raise_if(
                s is not None and l is None,
                f"`{s_name}` not None in `fit()` method call, but `{l_name}` is None in constructor. ",
            )
            raise_if(
                s is None and l is not None,
                f"`{s_name}` is None in `fit()` method call, but `{l_name}` is not None in constructor. ",
            )

        # setting proper input and output chunk length. We need to find all necessary lags in the dataset. Thus, we
        # need an input chunk length that contains the oldest past lag (min_lag), and we need to find the most future
        # lag in the future_covariates as well. In case no future covariate are required, we still need an output
        # chunk length of 1 for having the prediction target value. Both min_lag and max_lag could be set to None.

        training_dataset = MixedCovariatesSequentialDataset(
            target_series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.training_output_chunk_length,
            max_samples_per_ts=max_samples_per_ts
        )

        training_samples, training_labels = self._get_training_data(training_dataset=training_dataset)
        self.model.fit(training_samples, training_labels, **kwargs)

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

    def _get_prediction_data(
        self,
        prediction_dataset: MixedCovariatesInferenceDataset
    ):
        """
        Helper function which turns a LaggedInferenceDataset into 5 matrices and a List[TimeSeries].
        """
        target_matrix = []
        past_covariates_matrix = []
        historic_future_covariates_matrix = []
        future_covariates_matrix = []
        future_past_covariates_matrix = []

        for (past_target, past_covariates, historic_future_covariates, future_covariates,
             future_past_covariates, _) in prediction_dataset:

            # past target will have ndim = 2 (time, dim), we remove dim since we have the univariate assumption
            target_matrix.append(past_target.ravel())

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
        series
            Optionally, one or several input `TimeSeries`, representing the history of the target series whose future
            is to be predicted. If specified, the method returns the forecasts of these series. Otherwise, the method
            returns the forecast of the (single) training series.
        covariates # TODO update doc
            Optionally, the covariates series needed as inputs for the model. They must match the covariates used
            for training in terms of dimension and type.
        num_samples
            Currently this parameter is ignored for regression models.
        **kwargs
            Additional keyword arguments passed to the `predict` method of the model.
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

        prediction_output_chunk_length = n + max_with_none([self.max_lag, 0])

        # checking if there are enough future values in the future_covariates. MixedCovariatesInferenceDataset would
        # detect the problem in any case, but the error message would be meaningless.

        if future_covariates is not None:
            for sample in range(len(series)):
                last_req_ts = series[sample].end_time() + prediction_output_chunk_length * series[sample].freq
                raise_if_not(
                    future_covariates[sample].end_time() >= last_req_ts,
                    "When forecasting future values for a horizon n using lags_future_covariates >= 0, future_covariates "
                    "are requires to be at least `n + max_lags`, with `max_lag` being the futhest lag in the future"
                    f"For the {sample}-th sample, last future covariate timestamp is"
                    f"{future_covariates[sample].end_time()}, whereas it should be at least {last_req_ts}."
                )

        inference_dataset = MixedCovariatesInferenceDataset(
            target_series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=1
        )

        # ------------------
        (
            target_matrix,
            past_covariates_matrix,
            historic_future_covariates_matrix,
            future_covariates_matrix,
            future_past_covariates_matrix
        ) = self._get_prediction_data(inference_dataset)

        predictions = []

        """
        The columns of the prediction matrix has to have the same column order as during the training step, which is
        as follows: lags | lag_cov_0 | lag_cov_1 | .. where each lag_cov_X is a shortcut for
        lag_cov_X_dim_0 | lag_cov_X_dim_1 | .., that means, the lag X value of all the dimension of the covariate
        series (when multivariate).
        """
        for i in range(n):
            # building training matrix
            X = []
            if self.lags is not None:
                target_series = target_matrix[:, self.lags]
                X.append(target_series)

            covariates_matrices = [
                (past_covariates_matrix, self.lags_past_covariates),
                (historic_future_covariates_matrix, self.lags_historical_covariates),
                (future_covariates_matrix, self.lags_future_covariates)
            ]

            for covariate_matrix, lags in covariates_matrices:
                if lags is not None:
                    covariates = covariate_matrix[:, lags]
                    X.append(covariates.reshape(covariates.shape[0], -1))

            X = np.concatenate(X, axis=1)

            prediction = self.model.predict(X, **kwargs)
            prediction = prediction.reshape(-1, 1)
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
                    past_covariates_matrix, future_past_covariates_matrix = shift_matrices(
                        past_covariates_matrix, future_past_covariates_matrix)
                if historic_future_covariates_matrix is not None:
                    historic_future_covariates_matrix, future_covariates_matrix = shift_matrices(
                        historic_future_covariates_matrix, future_covariates_matrix)

        predictions = np.concatenate(predictions, axis=1)
        predictions = [self._build_forecast_series(row, input_tgt) for row, input_tgt in zip(predictions, series)]

        return predictions[0] if called_with_single_series else predictions

    def __str__(self):
        return self.model.__str__()
