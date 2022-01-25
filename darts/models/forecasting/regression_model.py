"""
Regression Model
----------------
A `RegressionModel` forecasts future values of a target series based on lagged values of

* The target series (past lags only)

* An optional past_covariates series (past lags only)

* An optional future_covariates series (possibly past and future lags)


The regression models are learned in a supervised way, and they can wrap around any "scikit-learn like" regression model
acting on tabular data having ``fit()`` and ``predict()`` methods.

Darts also provides :class:`LinearRegressionModel` and :class:`RandomForest`, which are regression models
wrapping around scikit-learn linear regression and random forest regression, respectively.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.

The lags can be specified either using an integer - in which case it represents the _number_ of (past or future) lags
to take into consideration, or as a list - in which case the lags have to be enumerated (strictly negative values
denoting past lags and positive values including 0 denoting future lags).
"""

from typing import Union, Sequence, Optional, Tuple, List
import numpy as np
import pandas as pd
import math

from darts.timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.logging import raise_if, raise_if_not, get_logger, raise_log
from darts.utils.data.inference_dataset import MixedCovariatesInferenceDataset


logger = get_logger(__name__)


def _update_min_max(
    current_min: Union[int, None],
    current_max: Union[int, None],
    new_values: Sequence[int],
) -> Tuple[int, int]:
    """
    Helper function that, given min, max and some values, updates min and max. Min and max can be set to None, if not
    set yet.
    """
    new_min = (
        min(min(new_values), current_min)
        if current_min is not None
        else min(new_values)
    )
    new_max = (
        max(max(new_values), current_max)
        if current_max is not None
        else max(new_values)
    )
    return new_min, new_max


class RegressionModel(GlobalForecastingModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        model=None,
    ):
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
            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
        """

        super().__init__()

        self.model = model
        self.lags = None
        self.lags_past_covariates = None
        self.lags_historical_covariates = None
        self.lags_future_covariates = None
        self.output_chunk_length = None
        self.min_lag = None
        self.max_lag = None

        # model checks
        if self.model is None:
            self.model = LinearRegression(n_jobs=-1)

        if not callable(getattr(self.model, "fit", None)):
            raise_log(
                Exception("Provided model object must have a fit() method", logger)
            )
        if not callable(getattr(self.model, "predict", None)):
            raise_log(
                Exception("Provided model object must have a predict() method", logger)
            )

        # parameter checks and processing
        raise_if(
            (lags is None)
            and (lags_future_covariates is None)
            and (lags_past_covariates is None),
            "At least one of `lags`, `lags_future_covariates` or `lags_past_covariates` must be not None.",
        )

        lags_type_checks = [
            (lags, "lags"),
            (lags_past_covariates, "lags_past_covariates"),
        ]

        # checking types
        for _lags, lags_name in lags_type_checks:
            raise_if_not(
                isinstance(_lags, (int, list)) or _lags is None,
                f"`{lags_name}` must be of type int or list. Given: {type(_lags)}.",
            )

            raise_if(
                isinstance(lags, bool),
                f"`{lags_name}` must be of type int or list, not bool.",
            )

        raise_if_not(
            isinstance(lags_future_covariates, (tuple, list))
            or lags_future_covariates is None,
            f"`lags_future_covariates` must be of type tuple or list. Given: {type(lags_future_covariates)}.",
        )

        if isinstance(lags_future_covariates, tuple):
            raise_if_not(
                len(lags_future_covariates) == 2
                and isinstance(lags_future_covariates[0], int)
                and isinstance(lags_future_covariates[1], int),
                "`lags_future_covariates` tuple must be of length 2, and must contain two integers",
            )
            raise_if(
                isinstance(lags_future_covariates[0], bool)
                or isinstance(lags_future_covariates[1], bool),
                "`lags_future_covariates` tuple must contain intergers, not bool",
            )

        if isinstance(lags, int):
            raise_if_not(lags > 0, f"`lags` must be strictly positive. Given: {lags}.")
            # selecting last `lags` lags, starting from position 1 (skipping current, pos 0, the one we want to predict)
            self.lags = list(range(-lags, 0))

        elif isinstance(lags, list):
            for lag in lags:
                raise_if(
                    not isinstance(lag, int) or (lag >= 0),
                    f"Every element of `lags` must be a strictly negative integer. Given: {lags}.",
                )
            self.lags = sorted(lags)

        if isinstance(lags_past_covariates, int):
            raise_if_not(
                lags_past_covariates > 0,
                f"`lags_past_covariates` must be an integer > 0. Given: {lags_past_covariates}.",
            )
            self.lags_past_covariates = list(range(-lags_past_covariates, 0))

        elif isinstance(lags_past_covariates, list):
            for lag in lags_past_covariates:
                raise_if(
                    not isinstance(lag, int) or (lag >= 0),
                    f"Every element of `lags_covariates` must be an integer < 0. Given: {lags_past_covariates}.",
                )
            self.lags_past_covariates = sorted(lags_past_covariates)

        if isinstance(lags_future_covariates, tuple):
            raise_if_not(
                lags_future_covariates[0] >= 0 and lags_future_covariates[1] >= 0,
                f"`lags_past_covariates` tuple must contain integers >= 0. Given: {lags_future_covariates}.",
            )

            if lags_future_covariates[0] is not None and lags_future_covariates[0] > 0:
                self.lags_historical_covariates = list(
                    range(-lags_future_covariates[0], 0)
                )
            if lags_future_covariates[1] is not None and lags_future_covariates[1] > 0:
                self.lags_future_covariates = list(range(0, lags_future_covariates[1]))

        elif isinstance(lags_future_covariates, list):
            for lag in lags_future_covariates:
                raise_if(
                    not isinstance(lag, int) or isinstance(lag, bool),
                    f"Every element of `lags_future_covariates` must be an integer. Given: {lags_future_covariates}.",
                )

            lags_future_covariates = np.array(sorted(lags_future_covariates))
            self.lags_historical_covariates = list(
                lags_future_covariates[lags_future_covariates < 0]
            )
            self.lags_future_covariates = list(
                lags_future_covariates[lags_future_covariates >= 0]
            )

            if len(self.lags_historical_covariates) == 0:
                self.lags_historical_covariates = None

            if len(self.lags_future_covariates) == 0:
                self.lags_future_covariates = None

        raise_if_not(isinstance(output_chunk_length, int) and output_chunk_length > 0,
                     f"output_chunk_length must be an integer greater than 0. Given: {output_chunk_length}")

        self.output_chunk_length = output_chunk_length

        # retrieving min and max lags, since they will be necessary for choosing appropriate input_chunk_size and
        # output_chunk_size later

        if lags is not None:
            # min (index 0) and max (index -1) are enough since the array is already sorted
            self.min_lag, self.max_lag = _update_min_max(
                self.min_lag, self.max_lag, [self.lags[0], self.lags[-1]]
            )

        lags = [
            self.lags_past_covariates,
            self.lags_historical_covariates,
            self.lags_future_covariates,
        ]
        for lag in lags:
            if lag is not None:
                self.min_lag, self.max_lag = _update_min_max(
                    self.min_lag, self.max_lag, [lag[0], lag[-1]]
                )

    def _get_last_prediction_time(self, series, forecast_horizon, overlap_end):
        # overrides the ForecastingModel _get_last_prediction_time, taking care of future lags if any
        extra_shift = max(0, self.max_lag)

        if overlap_end:
            last_valid_pred_time = series.time_index[-1 - extra_shift]
        else:
            last_valid_pred_time = series.time_index[-forecast_horizon - extra_shift]

        return last_valid_pred_time

    def _create_lagged_data(
            self, target_series, past_covariates, future_covariates, max_samples_per_ts
    ):
        """
        Helper function that creates training/validation matrices (X and y as required in sklearn), given series and
        max_samples_per_ts.

        X has the following structure:
        lags_target | lags_past_covariates | lags_future_covariates

        Where each lags_X has the following structure (lags_X=[-2,-1] and X has 2 components):
        lag_-2_comp_1_X | lag_-2_comp_2_X | lag_-1_comp_1_X | lag_-1_comp_2_X
        """

        target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        past_covariates = (
            [past_covariates]
            if isinstance(past_covariates, TimeSeries)
            else past_covariates
        )
        future_covariates = (
            [future_covariates]
            if isinstance(future_covariates, TimeSeries)
            else future_covariates
        )

        Xs, ys = [], []
        # iterate over series
        for idx, target_ts in enumerate(target_series):
            covariates = [
                (past_covariates[idx].pd_dataframe(copy=False), self.lags_past_covariates)
                if past_covariates
                else (None, None),
                (
                    future_covariates[idx].pd_dataframe(copy=False),
                    (
                        self.lags_historical_covariates
                        if self.lags_historical_covariates
                        else []
                    )
                    + (
                        self.lags_future_covariates
                        if self.lags_future_covariates
                        else []
                    ),
                )
                if future_covariates
                else (None, None),
            ]

            df_X = []
            df_y = []
            df_target = target_ts.pd_dataframe(copy=False)

            # y: output chunk length lags
            for future_target_lag in range(self.output_chunk_length):
                df_y.append(df_target.shift(-future_target_lag))

            # X: target lags
            if self.lags:
                for lag in self.lags:
                    df_X.append(df_target.shift(-lag))

            # X: covariate lags
            for df_cov, lags in covariates:
                if lags:
                    for lag in lags:
                        df_X.append(df_cov.shift(-lag))

            # combine lags
            df_X = pd.concat(df_X, axis=1)
            df_y = pd.concat(df_y, axis=1)
            df_X_y = pd.concat([df_X, df_y], axis=1)
            X_y = df_X_y.dropna().values

            # keep most recent max_samples_per_ts samples
            if max_samples_per_ts:
                X_y = X_y[-max_samples_per_ts:]

            raise_if(
                X_y.shape[0] == 0,
                f"Unable to build any training samples; the {idx}th target and the corresponding "
                f"covariate series overlap too little.",
            )

            X, y = np.split(X_y, [df_X.shape[1]], axis=1)
            Xs.append(X)
            ys.append(y)

        # combine samples from all series
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return X, y

    def _fit_model(
        self,
        target_series,
        past_covariates,
        future_covariates,
        max_samples_per_ts,
        **kwargs,
    ):
        """
        Function that fit the model. Deriving classes can override this method for adding additional parameters (e.g.,
        adding validation data), keeping the sanity checks on series performed by fit().
        """
        training_samples, training_labels = self._create_lagged_data(
            target_series, past_covariates, future_covariates, max_samples_per_ts
        )

        # if training_labels is of shape (n_samples, 1) flatten it to shape (n_samples,)
        if len(training_labels.shape) == 2 and training_labels.shape[1] == 1:
            training_labels = training_labels.ravel()
        self.model.fit(training_samples, training_labels, **kwargs)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Fit/train the model on one or multiple series.

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
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        raise_if(
            past_covariates is not None and self.lags_past_covariates is None,
            "`past_covariates` not None in `fit()` method call, but `lags_past_covariates` is None in constructor",
        )

        raise_if(
            past_covariates is None and self.lags_past_covariates is not None,
            "`past_covariates` is None in `fit()` method call, but `lags_past_covariates` is not None in"
            "constructor.",
        )

        raise_if(
            future_covariates is not None
            and (
                self.lags_historical_covariates is None
                and self.lags_future_covariates is None
            ),
            "`future_covariates` not None in `fit()` method call, but `lags_future_covariates` is None in "
            "constructor.",
        )

        raise_if(
            future_covariates is None
            and (
                self.lags_future_covariates is not None
                or self.lags_historical_covariates is not None
            ),
            "`future_covariates` is None in `fit()` method call, but `lags_future_covariates` is not None in"
            "constructor.",
        )

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

        # if multi-target regression
        if not series[0].is_univariate or self.output_chunk_length > 1:
            # check whether model supports it
            if not (callable(getattr(self.model, '_get_tags', None)) and isinstance(self.model._get_tags(),
                                                                                    dict) and self.model._get_tags().get(
                'multioutput', False)):
                # if not, wrap model with multioutputregressor
                self.model = MultiOutputRegressor(self.model, n_jobs=1)

        self._fit_model(
            series, past_covariates, future_covariates, max_samples_per_ts, **kwargs
        )

    def _get_prediction_data(self, prediction_dataset: MixedCovariatesInferenceDataset):
        """
        Helper function which turns a MixedCovariatesInferenceDataset into 5 matrices. The matrices simply contain all
        the samples stacked in the first dimension.
        """
        target_matrix = []
        past_covariates_matrix = []
        historic_future_covariates_matrix = []
        future_covariates_matrix = []
        future_past_covariates_matrix = []

        for (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            future_past_covariates,
            _,
        ) in prediction_dataset:

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
            future_past_covariates_matrix,
        ]

        for i in range(len(covariates_matrices)):
            covariates_matrices[i] = (
                None
                if len(covariates_matrices[i]) == 0
                else np.asarray(covariates_matrices[i])
            )

        return (np.asarray(target_matrix), *covariates_matrices)

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        **kwargs,
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
            future_covariates = (
                [future_covariates] if future_covariates is not None else None
            )

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

        # check if we have sufficient covariate data for given horizon, lags and output_chunk_length
        n_pred_steps = math.ceil(n / self.output_chunk_length)

        covariates = {
            "past": (past_covariates, self.lags_past_covariates),
            "future": (future_covariates,
                       (self.lags_historical_covariates if self.lags_historical_covariates else []) + (
                           self.lags_future_covariates if self.lags_future_covariates else []))
        }

        covariate_matrices = {}
        relative_lags = {}
        for cov_type, (covs, lags) in covariates.items():
            if covs is not None:
                relative_lags[cov_type] = np.array(lags) - min(lags)
                covariate_matrices[cov_type] = []
                for idx, (ts, cov) in enumerate(zip(series, covs)):
                    first_pred_ts = ts.end_time() + 1 * ts.freq
                    last_pred_ts = first_pred_ts + ((n_pred_steps - 1) * self.output_chunk_length) * ts.freq
                    first_req_ts = first_pred_ts + min(lags) * ts.freq
                    last_req_ts = last_pred_ts + max(lags) * ts.freq

                    raise_if_not(
                        cov.start_time() <= first_req_ts and cov.end_time() >= last_req_ts,
                        f"The corresponding {cov_type}_covariate of the {idx}-th series isn't sufficiently long. "
                        f"Given horizon `n={n}`, `min(lags_{cov_type}_covariates)={min(lags)}`, "
                        f"`max(lags_{cov_type}_covariates)={max(lags)}` and `output_chunk_length={self.output_chunk_length}` "
                        f"the {cov_type}_covariate has to range from {first_req_ts} until {last_req_ts} (inclusive), "
                        f"but it ranges only from {cov.start_time()} until {cov.end_time()}."
                    )

                    # slicing integer indices doesn't include the end while it does with datetime indices
                    if cov.has_datetime_index:
                        covariate_matrices[cov_type].append(cov[first_req_ts:last_req_ts].values())
                    else:
                        covariate_matrices[cov_type].append(cov[first_req_ts:last_req_ts + 1].values())

                covariate_matrices[cov_type] = np.stack(covariate_matrices[cov_type])

        """
        The columns of the prediction matrix need to have the same column order as during the training step, which is
        as follows: lags | lag_cov_0 | lag_cov_1 | .. where each lag_cov_X is a shortcut for
        lag_cov_X_dim_0 | lag_cov_X_dim_1 | .., that means, the lag X value of all the dimension of the covariate
        series (when multivariate).
        """

        series_matrix = np.stack([ts[min(self.lags):].values() for ts in series]) if self.lags else None
        predictions = []
        for t_after_end in range(0, n, self.output_chunk_length):

            np_X = []
            if self.lags:
                target_matrix = np.concatenate([series_matrix, np.concatenate(predictions, axis=1)],
                                               axis=1) if predictions else series_matrix
                np_X.append(target_matrix[:, self.lags].reshape(len(series), -1))
            for cov_type in covariate_matrices:
                np_X.append(
                    covariate_matrices[cov_type][:, relative_lags[cov_type] + t_after_end].reshape(len(series), -1))
            X = np.concatenate(np_X, axis=1)
            # X has shape (n_series, n_regression_features)
            prediction = self.model.predict(X, **kwargs)
            # reshape to (n_series, time (output_chunk_length), n_components)
            prediction = prediction.reshape(len(series), self.output_chunk_length, -1)
            # appending prediction to final predictions
            predictions.append(prediction)

        # concatenate and use first n points as prediction
        predictions = np.concatenate(predictions, axis=1)[:, :n]
        predictions = [
            self._build_forecast_series(row, input_tgt)
            for row, input_tgt in zip(predictions, series)
        ]

        return predictions[0] if called_with_single_series else predictions

    def __str__(self):
        return self.model.__str__()
