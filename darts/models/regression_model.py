"""
Regression Model
----------------

A `RegressionModel` forecasts future values of a target series based on lagged values of the target values
and possibly lags of an covariate series. They can wrap around any regression model having a `fit()`
and `predict()` functions accepting tabularized data (e.g. scikit-learn regression models), and are using
`sklearn.linear_model.LinearRegression` by default.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.
"""
from typing import Union, Sequence, Optional, Tuple, List
import numpy as np

from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import GlobalForecastingModel
from ..logging import raise_if, raise_if_not, get_logger, raise_log
from darts.utils.data.sequential_dataset import SequentialDataset
from darts.utils.data.simple_inference_dataset import SimpleInferenceDataset

logger = get_logger(__name__)


def _process_lags(lags: Optional[Union[int, List[int]]] = None,
                  lags_covariates: Optional[Union[int, List[int]]] = None
                  ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Process lags and lags_covariate.

    Params
    ------
    lags
        Number of lagged target values used to predict the next time step. If an integer is given the last `lags` lags
        are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
    lags_covariates
        Number of lagged covariates values used to predict the next time step. If an integer is given
        the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
        integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t` should
        be used. Note that the 0 index is not included when passing a single interger value > 0.

    Returns
    -------
    Optional[List[int]]
        Processed `lags`, as a list of integers. If no lags are used, then `None` is returned.
    Optional[List[int]]
        Processed `lags_covariates` as a list of integers. If no lags covariates are used, then `None` is returned.

    Raises
    ------
    ValueError
        In case at least one of the required conditions is not met.
    """

    raise_if(
        (lags is None) and (lags_covariates is None),
        "At least one of `lags` or `lags_covariates` must be not None."
    )

    raise_if_not(
        isinstance(lags, (int, list)) or lags is None,
        f"`lags` must be of type int or list. Given: {type(lags)}."
    )

    raise_if_not(
        isinstance(lags_covariates, (int, list)) or lags_covariates is None,
        f"`lags_covariates` must be of type int or list. Given: {type(lags_covariates)}."
    )

    raise_if(
        isinstance(lags, bool) or isinstance(lags_covariates, bool),
        "`lags` and `lags_covariates` must be of type int or list, not bool."
    )

    if isinstance(lags, int):
        raise_if_not(
            lags > 0,
            f"`lags` must be strictly positive. Given: {lags}."
        )
        # selecting last `lags` lags, starting from position 1 (skipping current, pos 0, the one we want to predict)
        lags = list(range(1, lags + 1))

    elif isinstance(lags, list):
        for lag in lags:
            raise_if(
                not isinstance(lag, int) or (lag <= 0),
                f"Every element of `lags` must be a strictly positive integer. Given: {lags}."
            )
    # using only the current current covariates, at position 0, which is the same timestamp as the prediction
    if isinstance(lags_covariates, int) and lags_covariates == 0:
        lags_covariates = [0]

    elif isinstance(lags_covariates, int):
        raise_if_not(
            lags_covariates > 0,
            f"`lags_covariates` must be an integer >= 0. Given: {lags_covariates}."
        )
        lags_covariates = list(range(1, lags_covariates + 1))

    elif isinstance(lags_covariates, list):
        for lag in lags_covariates:
            raise_if(
                not isinstance(lag, int) or (lag < 0),
                f"Every element of `lags_covariates` must be an integer >= 0. Given: {lags_covariates}."
            )

    return lags, lags_covariates


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


class LaggedTrainingDataset:
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        lags: Optional[Union[int, List[int]]] = None,
        lags_covariates: Optional[Union[int, List[int]]] = None,
        max_samples_per_ts: Optional[int] = None
    ):

        """Lagged Dataset
        A time series dataset wrapping around `SequentialDataset` containing tuples of (input_target, output_target,
        input_covariates) arrays, where "input_target" is #lags long, "input_covariates" is #lags_covariates long,
        and "output" has length 1.

        Params
        ------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`.
        lags
            Number of lagged target values used to predict the next time step. If an integer is given the last `lags`
            lags are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
        lags_covariates
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
            integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t`
            should be used. Note that the 0 index is not included when passing a single interger value > 0.
        """

        # the Sequential dataset will take care of handling series properly, and it is supporting
        # multiple TS

        super().__init__()

        self.lags, self.lags_covariates = _process_lags(lags, lags_covariates)

        if self.lags is not None and self.lags_covariates is not None:
            max_lags = max(max(self.lags), max(self.lags_covariates))
        elif self.lags_covariates is not None:
            max_lags = max(self.lags_covariates)
        else:
            max_lags = max(self.lags)

        if self.lags_covariates is not None and 0 in self.lags_covariates:
            # adding one for 0 covariate trick
            max_lags += 1

        self.sequential_dataset = SequentialDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=max_lags,
            output_chunk_length=1,
            max_samples_per_ts=max_samples_per_ts
        )

    def __len__(self):
        return len(self.sequential_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        arrays = self.sequential_dataset[idx]

        input_target, output_target, input_covariates = [
            ar.copy() if ar is not None else None for ar in arrays
        ]

        if self.lags_covariates is not None and 0 in self.lags_covariates:
            """
            In case we need the 'time 0' covariate, we have to adjust the data with the following trick

            T5 T4 T3 T2 T1 T0 -> P | T5 T4 T3 T2 T1     -> T0~P'
            C5 C4 C3 C2 C1 C0      | C5 C4 C3 C2 C1 C0

            """
            # overwrite the prediction
            output_target = np.array(input_target[-1]).reshape(1, 1)
            # shortening the input_target by one
            input_target = input_target[:-1]

        # evaluating indexes from the end
        if self.lags is not None:
            lags_indices = np.array(self.lags) * (-1)
            input_target = input_target[lags_indices]
        else:
            input_target = None

        if self.lags_covariates is not None:
            if 0 in self.lags_covariates:
                cov_lags_indices = (np.array(self.lags_covariates) + 1) * (-1)
            else:
                cov_lags_indices = (np.array(self.lags_covariates)) * (-1)
            input_covariates = input_covariates[cov_lags_indices]
        else:
            input_covariates = None
        return input_target, output_target, input_covariates

    def get_data(self):
        """
        The function returns a training matrix X with shape (n_samples, lags + lags_covariates*covariates.width)
        and y with shape (n_sample,).

        The columns of the resulting matrix have the following order: lags | lag_cov_0 | lag_cov_1 | .. where each
        lag_cov_X is a shortcut for lag_cov_X_dim_0 | lag_cov_X_dim_1 | .., that means, the lag X value of all the
        dimension of the covariate series (when multivariate).
        """
        x = []
        y = []

        for input_target, output_target, input_covariates in self:
            row = []
            if input_target is not None:
                row.append(input_target.T)
            if input_covariates is not None:
                row.append(input_covariates.reshape(1, -1))

            x.append(np.concatenate(row, axis=1))
            y.append(output_target)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        return x, y.ravel()


class LaggedInferenceDataset:
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        lags: Union[int, list] = None,
        lags_covariates: Union[int, list] = None,
        n: int = 1
    ):
        """
        A time series dataset wrapping around `SimpleInferenceDataset`. The `input_chunk_length` is inferred through
        lags and lags_covariates.

        Params
        ------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`.
        lags
            Number of lagged target values used to predict the next time step. If an integer is given the last `lags`
            lags are used (inclusive). Otherwise a list of integers with lags is required (each lag must be > 0).
        lags_covariates
            Number of lagged covariates values used to predict the next time step. If an integer is given
            the last `lags_covariates` lags are used (inclusive, starting from lag 1). Otherwise a list of
            integers with lags >= 0 is required. The special index 0 is supported, in case the covariate at time `t`
            should be used. Note that the 0 index is not included when passing a single interger value > 0.
        n
            The number of time steps after the end of the training time series for which to produce predictions.
        """
        super().__init__()

        self.lags, self.lags_covariates = _process_lags(lags, lags_covariates)

        if self.lags is not None and self.lags_covariates is None:
            max_lag = max(self.lags)
        elif self.lags is None and self.lags_covariates is not None:
            max_lag = max(self.lags_covariates)
        else:
            max_lag = max([max(self.lags), max(self.lags_covariates)])

        input_chunk_length = max(max_lag, 1)

        self.inference_dataset = SimpleInferenceDataset(
            series=target_series,
            covariates=covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            model_is_recurrent=True if self.lags_covariates is not None and 0 in self.lags_covariates else False,
            add_prediction_covariate=True if self.lags_covariates is not None and 0 in self.lags_covariates else False
        )

    def __len__(self):
        return len(self.inference_dataset)

    def __getitem__(self, idx):
        return self.inference_dataset[idx]


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

        raise_if_not(
            (dataset.lags == self.lags) and (dataset.lags_covariates == self.lags_covariates),
            "Either lags or lags_covariates not matching with the one used during training."
        )

        target_matrix, covariates_matrix, future_covariates_matrix = self._get_matrix_data_from_dataset(dataset)
        predictions = []

        """
        The columns of the prediction matrix has to have the same column order as during the training step, which is
        as follows: lags | lag_cov_0 | lag_cov_1 | .. where each lag_cov_X is a shortcut for
        lag_cov_X_dim_0 | lag_cov_X_dim_1 | .., that means, the lag X value of all the dimension of the covariate
        series (when multivariate).
        """

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
