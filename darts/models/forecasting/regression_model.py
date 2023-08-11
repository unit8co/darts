"""
Regression Model
----------------
A `RegressionModel` forecasts future values of a target series based on

* The target series (past lags only)

* An optional past_covariates series (past lags only)

* An optional future_covariates series (possibly past and future lags)

* Available static covariates


The regression models are learned in a supervised way, and they can wrap around any "scikit-learn like" regression model
acting on tabular data having ``fit()`` and ``predict()`` methods.

Darts also provides :class:`LinearRegressionModel` and :class:`RandomForest`, which are regression models
wrapping around scikit-learn linear regression and random forest regression, respectively.

Behind the scenes this model is tabularizing the time series data to make it work with regression models.

The lags can be specified either using an integer - in which case it represents the _number_ of (past or future) lags
to take into consideration, or as a list - in which case the lags have to be enumerated (strictly negative values
denoting past lags and positive values including 0 denoting future lags).
When static covariates are present, they are appended to the lagged features. When multiple time series are passed,
if their static covariates do not have the same size, the shorter ones are padded with 0 valued features.
"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.data.tabularization import (
    add_static_covariates_to_lagged_data,
    create_lagged_component_names,
    create_lagged_training_data,
)
from darts.utils.historical_forecasts import (
    _optimized_historical_forecasts_regression_all_points,
    _optimized_historical_forecasts_regression_last_points_only,
)
from darts.utils.multioutput import MultiOutputRegressor
from darts.utils.utils import (
    _check_quantiles,
    get_single_series,
    seq2series,
    series2seq,
)

logger = get_logger(__name__)


class RegressionModel(GlobalForecastingModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        add_encoders: Optional[dict] = None,
        model=None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
    ):
        """Regression Model
        Can be used to fit any scikit-learn-like regressor class to predict the target time series from lagged values.

        Parameters
        ----------
        lags
            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags
            are used (from -1 backward). Otherwise, a list of integers with lags is required (each lag must be < 0).
        lags_past_covariates
            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last
            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers
            with lags < 0 is required.
        lags_future_covariates
            Number of lagged future_covariates values used to predict the next time step. If a tuple (past, future) is
            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first
            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list
            of integers with lags is required.
        output_chunk_length
            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast
            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may
            be useful if the covariates don't extend far enough into the future.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        model
            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model is
            trained to predict at step 'output_chunk_length' in the future. Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        """

        super().__init__(add_encoders=add_encoders)

        self.model = model
        self.lags: Dict[str, List[int]] = {}
        self.input_dim = None
        self.multi_models = multi_models
        self._considers_static_covariates = use_static_covariates
        self._static_covariates_shape: Optional[Tuple[int, int]] = None
        self._lagged_feature_names: Optional[List[str]] = None

        # check and set output_chunk_length
        raise_if_not(
            isinstance(output_chunk_length, int) and output_chunk_length > 0,
            f"output_chunk_length must be an integer greater than 0. Given: {output_chunk_length}",
            logger=logger,
        )
        self._output_chunk_length = output_chunk_length

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

        # check lags
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

        for _lags, lags_name in lags_type_checks:
            raise_if_not(
                isinstance(_lags, (int, list)) or _lags is None,
                f"`{lags_name}` must be of type int or list. Given: {type(_lags)}.",
            )
            raise_if(
                isinstance(_lags, bool),
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
                "`lags_future_covariates` tuple must contain integers, not bool",
            )

        # set lags
        if isinstance(lags, int):
            raise_if_not(lags > 0, f"`lags` must be strictly positive. Given: {lags}.")
            # selecting last `lags` lags, starting from position 1 (skipping current, pos 0, the one we want to predict)
            self.lags["target"] = list(range(-lags, 0))
        elif isinstance(lags, list):
            for lag in lags:
                raise_if(
                    not isinstance(lag, int) or (lag >= 0),
                    f"Every element of `lags` must be a strictly negative integer. Given: {lags}.",
                )
            if lags:
                self.lags["target"] = sorted(lags)

        if isinstance(lags_past_covariates, int):
            raise_if_not(
                lags_past_covariates > 0,
                f"`lags_past_covariates` must be an integer > 0. Given: {lags_past_covariates}.",
            )
            self.lags["past"] = list(range(-lags_past_covariates, 0))
        elif isinstance(lags_past_covariates, list):
            for lag in lags_past_covariates:
                raise_if(
                    not isinstance(lag, int) or (lag >= 0),
                    f"Every element of `lags_covariates` must be an integer < 0. Given: {lags_past_covariates}.",
                )
            if lags_past_covariates:
                self.lags["past"] = sorted(lags_past_covariates)

        if isinstance(lags_future_covariates, tuple):
            raise_if_not(
                lags_future_covariates[0] >= 0 and lags_future_covariates[1] >= 0,
                f"`lags_future_covariates` tuple must contain integers >= 0. Given: {lags_future_covariates}.",
            )
            if (
                lags_future_covariates[0] is not None
                and lags_future_covariates[1] is not None
            ):
                if not (
                    lags_future_covariates[0] == 0 and lags_future_covariates[1] == 0
                ):
                    self.lags["future"] = list(
                        range(-lags_future_covariates[0], lags_future_covariates[1])
                    )
        elif isinstance(lags_future_covariates, list):
            for lag in lags_future_covariates:
                raise_if(
                    not isinstance(lag, int) or isinstance(lag, bool),
                    f"Every element of `lags_future_covariates` must be an integer. Given: {lags_future_covariates}.",
                )
            if lags_future_covariates:
                self.lags["future"] = sorted(lags_future_covariates)

        self.pred_dim = self.output_chunk_length if self.multi_models else 1

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[int, int, bool, bool, Optional[List[int]], Optional[List[int]]]:
        target_lags = self.lags.get("target", [0])
        lags_past_covariates = self.lags.get("past", None)
        if lags_past_covariates is not None:
            lags_past_covariates = [
                min(lags_past_covariates)
                - int(not self.multi_models) * (self.output_chunk_length - 1),
                max(lags_past_covariates),
            ]
        lags_future_covariates = self.lags.get("future", None)
        if lags_future_covariates is not None:
            lags_future_covariates = [
                min(lags_future_covariates)
                - int(not self.multi_models) * (self.output_chunk_length - 1),
                max(lags_future_covariates),
            ]
        return (
            abs(min(target_lags)),
            self.output_chunk_length,
            lags_past_covariates is not None,
            lags_future_covariates is not None,
            lags_past_covariates,
            lags_future_covariates,
        )

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
        min_target_lag = self.lags.get("target")[0] if "target" in self.lags else None
        max_target_lag = self.output_chunk_length - 1
        min_past_cov_lag = self.lags.get("past")[0] if "past" in self.lags else None
        max_past_cov_lag = self.lags.get("past")[-1] if "past" in self.lags else None
        min_future_cov_lag = (
            self.lags.get("future")[0] if "future" in self.lags else None
        )
        max_future_cov_lag = (
            self.lags.get("future")[-1] if "future" in self.lags else None
        )
        return (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            max_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
        )

    @property
    def supports_multivariate(self) -> bool:
        """
        If available, uses `model`'s native multivariate support. If not available, obtains multivariate support by
        wrapping the univariate model in a `sklearn.multioutput.MultiOutputRegressor`.
        """
        return True

    @property
    def min_train_series_length(self) -> int:
        return max(
            3,
            -self.lags["target"][0] + self.output_chunk_length
            if "target" in self.lags
            else self.output_chunk_length,
        )

    @property
    def min_train_samples(self) -> int:
        return 2

    @property
    def output_chunk_length(self) -> int:
        return self._output_chunk_length

    def get_multioutput_estimator(self, horizon, target_dim):
        raise_if_not(
            isinstance(self.model, MultiOutputRegressor),
            "The sklearn model is not a MultiOutputRegressor object.",
        )

        return self.model.estimators_[horizon + target_dim]

    def _get_last_prediction_time(self, series, forecast_horizon, overlap_end):
        # overrides the ForecastingModel _get_last_prediction_time, taking care of future lags if any
        extra_shift = max(0, max(lags[-1] for lags in self.lags.values()))

        if overlap_end:
            last_valid_pred_time = series.time_index[-1 - extra_shift]
        else:
            last_valid_pred_time = series.time_index[-forecast_horizon - extra_shift]

        return last_valid_pred_time

    def _create_lagged_data(
        self, target_series, past_covariates, future_covariates, max_samples_per_ts
    ):
        lags = self.lags.get("target")
        lags_past_covariates = self.lags.get("past")
        lags_future_covariates = self.lags.get("future")

        (
            features,
            labels,
            _,
            self._static_covariates_shape,
        ) = create_lagged_training_data(
            target_series=target_series,
            output_chunk_length=self.output_chunk_length,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            uses_static_covariates=self.uses_static_covariates,
            last_static_covariates_shape=None,
            max_samples_per_ts=max_samples_per_ts,
            multi_models=self.multi_models,
            check_inputs=False,
            concatenate=False,
        )

        for i, (X_i, y_i) in enumerate(zip(features, labels)):
            features[i] = X_i[:, :, 0]
            labels[i] = y_i[:, :, 0]

        training_samples = np.concatenate(features, axis=0)
        training_labels = np.concatenate(labels, axis=0)

        return training_samples, training_labels

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
            target_series,
            past_covariates,
            future_covariates,
            max_samples_per_ts,
        )

        # if training_labels is of shape (n_samples, 1) flatten it to shape (n_samples,)
        if len(training_labels.shape) == 2 and training_labels.shape[1] == 1:
            training_labels = training_labels.ravel()
        self.model.fit(training_samples, training_labels, **kwargs)

        # generate and store the lagged components names (for feature importance analysis)
        self._lagged_feature_names, _ = create_lagged_component_names(
            target_series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags=self.lags.get("target"),
            lags_past_covariates=self.lags.get("past"),
            lags_future_covariates=self.lags.get("future"),
            output_chunk_length=self.output_chunk_length,
            concatenate=False,
            use_static_covariates=self.uses_static_covariates,
        )

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        **kwargs,
    ):
        """
        Fit/train the model on one or multiple series.

        Parameters
        ----------
        series
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        max_samples_per_ts
            This is an integer upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        n_jobs_multioutput_wrapper
            Number of jobs of the MultiOutputRegressor wrapper to run in parallel. Only used if the model doesn't
            support multi-output regression natively.
        **kwargs
            Additional keyword arguments passed to the `fit` method of the model.
        """
        # guarantee that all inputs are either list of TimeSeries or None
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        self._verify_static_covariates(series[0].static_covariates)

        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_fit_encodings(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True
        if (
            get_single_series(series).static_covariates is not None
            and self.supports_static_covariates
            and self.considers_static_covariates
        ):
            self._uses_static_covariates = True

        for covs, name in zip([past_covariates, future_covariates], ["past", "future"]):
            raise_if(
                covs is not None and name not in self.lags,
                f"`{name}_covariates` not None in `fit()` method call, but `lags_{name}_covariates` is None in "
                f"constructor.",
            )

            raise_if(
                covs is None and name in self.lags,
                f"`{name}_covariates` is None in `fit()` method call, but `lags_{name}_covariates` is not None in "
                "constructor.",
            )

        # saving the dims of all input series to check at prediction time
        self.input_dim = {
            "target": series[0].width,
            "past": past_covariates[0].width if past_covariates else None,
            "future": future_covariates[0].width if future_covariates else None,
        }

        # if multi-output regression
        if not series[0].is_univariate or (
            self.output_chunk_length > 1 and self.multi_models
        ):
            # and model isn't wrapped already
            if not isinstance(self.model, MultiOutputRegressor):
                # check whether model supports multi-output regression natively
                if not (
                    callable(getattr(self.model, "_get_tags", None))
                    and isinstance(self.model._get_tags(), dict)
                    and self.model._get_tags().get("multioutput")
                ):
                    # if not, wrap model with MultiOutputRegressor
                    self.model = MultiOutputRegressor(
                        self.model, n_jobs=n_jobs_multioutput_wrapper
                    )
                elif self.model.__class__.__name__ == "CatBoostRegressor":
                    if (
                        self.model.get_params()["loss_function"]
                        == "RMSEWithUncertainty"
                    ):
                        self.model = MultiOutputRegressor(
                            self.model, n_jobs=n_jobs_multioutput_wrapper
                        )

        # warn if n_jobs_multioutput_wrapper was provided but not used
        if (
            not isinstance(self.model, MultiOutputRegressor)
            and n_jobs_multioutput_wrapper is not None
        ):
            logger.warning("Provided `n_jobs_multioutput_wrapper` wasn't used.")

        super().fit(
            series=seq2series(series),
            past_covariates=seq2series(past_covariates),
            future_covariates=seq2series(future_covariates),
        )

        self._fit_model(
            series, past_covariates, future_covariates, max_samples_per_ts, **kwargs
        )

        return self

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
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
            Number of times a prediction is sampled from a probabilistic model. Should be set to 1
            for deterministic models.
        verbose
            Optionally, whether to print progress.
        predict_likelihood_parameters
            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``
        **kwargs : dict, optional
            Additional keyword arguments passed to the `predict` method of the model. Only works with
            univariate target series.
        """
        if series is None:
            # then there must be a single TS, and that was saved in super().fit as self.training_series
            if self.training_series is None:
                raise_log(
                    ValueError(
                        "Input `series` must be provided. This is the result either from fitting on multiple series, "
                        "or from not having fit the model yet."
                    ),
                    logger,
                )
            series = self.training_series

        called_with_single_series = True if isinstance(series, TimeSeries) else False

        # guarantee that all inputs are either list of TimeSeries or None
        series = series2seq(series)

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = [self.past_covariate_series] * len(series)
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = [self.future_covariate_series] * len(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        self._verify_static_covariates(series[0].static_covariates)

        # encoders are set when calling fit(), but not when calling fit_from_dataset()
        # when covariates are loaded from model, they already contain the encodings: this is not a problem as the
        # encoders regenerate the encodings
        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_predict_encodings(
                n=n,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
        super().predict(
            n,
            series,
            past_covariates,
            future_covariates,
            num_samples,
            verbose,
            predict_likelihood_parameters,
        )

        # check that the input sizes of the target series and covariates match
        pred_input_dim = {
            "target": series[0].width,
            "past": past_covariates[0].width if past_covariates else None,
            "future": future_covariates[0].width if future_covariates else None,
        }
        raise_if_not(
            pred_input_dim == self.input_dim,
            f"The number of components of the target series and the covariates provided for prediction doesn't "
            f"match the number of components of the target series and the covariates this model has been "
            f"trained on.\n"
            f"Provided number of components for prediction: {pred_input_dim}\n"
            f"Provided number of components for training: {self.input_dim}",
        )

        # prediction preprocessing
        covariates = {
            "past": (past_covariates, self.lags.get("past")),
            "future": (future_covariates, self.lags.get("future")),
        }

        # prepare one_shot shift and step
        if self.multi_models:
            shift = 0
            step = self.output_chunk_length
        else:
            shift = self.output_chunk_length - 1
            step = 1

        # dictionary containing covariate data over time span required for prediction
        covariate_matrices = {}
        # dictionary containing covariate lags relative to minimum covariate lag
        relative_cov_lags = {}
        for cov_type, (covs, lags) in covariates.items():
            if covs is None:
                continue

            relative_cov_lags[cov_type] = np.array(lags) - lags[0]
            covariate_matrices[cov_type] = []
            for idx, (ts, cov) in enumerate(zip(series, covs)):
                # how many steps to go back from end of target series for start of covariates
                steps_back = -(min(lags) + 1) + shift
                lags_diff = max(lags) - min(lags) + 1
                # over how many steps the covariates range
                n_steps = lags_diff + max(0, n - self.output_chunk_length) + shift

                # calculate first and last required covariate time steps
                start_ts = ts.end_time() - ts.freq * steps_back
                end_ts = start_ts + ts.freq * (n_steps - 1)

                # check for sufficient covariate data
                if not (cov.start_time() <= start_ts and cov.end_time() >= end_ts):
                    raise_log(
                        ValueError(
                            f"The corresponding {cov_type}_covariate of the series at index {idx} isn't sufficiently "
                            f"long. Given horizon `n={n}`, `min(lags_{cov_type}_covariates)={lags[0]}`, "
                            f"`max(lags_{cov_type}_covariates)={lags[-1]}` and "
                            f"`output_chunk_length={self.output_chunk_length}`, the {cov_type}_covariate has to range "
                            f"from {start_ts} until {end_ts} (inclusive), but it ranges only from {cov.start_time()} "
                            f"until {cov.end_time()}."
                        ),
                        logger=logger,
                    )

                # use slice() instead of [] as for integer-indexed series [] does not act on time index
                # for range indexes, we make the end timestamp inclusive here
                end_ts = end_ts + ts.freq if ts.has_range_index else end_ts
                covariate_matrices[cov_type].append(
                    cov.slice(start_ts, end_ts).values(copy=False)
                )

            covariate_matrices[cov_type] = np.stack(covariate_matrices[cov_type])

        series_matrix = None
        if "target" in self.lags:
            series_matrix = np.stack(
                [
                    ts.values(copy=False)[self.lags["target"][0] - shift :, :]
                    for ts in series
                ]
            )

        # repeat series_matrix to shape (num_samples * num_series, n_lags, n_components)
        # [series 0 sample 0, series 0 sample 1, ..., series n sample k]
        series_matrix = np.repeat(series_matrix, num_samples, axis=0)

        # same for covariate matrices
        for cov_type, data in covariate_matrices.items():
            covariate_matrices[cov_type] = np.repeat(data, num_samples, axis=0)
        # prediction
        predictions = []
        last_step_shift = 0

        # t_pred indicates the number of time steps after the first prediction
        for t_pred in range(0, n, step):
            # in case of autoregressive forecast `(t_pred > 0)` and if `n` is not a round multiple of `step`,
            # we have to step back `step` from `n` in the last iteration
            if 0 < n - t_pred < step and t_pred > 0:
                last_step_shift = t_pred - (n - step)
                t_pred = n - step

            np_X = []
            # retrieve target lags
            if "target" in self.lags:
                if predictions:
                    series_matrix = np.concatenate(
                        [series_matrix, predictions[-1]], axis=1
                    )
                np_X.append(
                    series_matrix[
                        :,
                        [
                            lag - (shift + last_step_shift)
                            for lag in self.lags["target"]
                        ],
                    ].reshape(len(series) * num_samples, -1)
                )
            # retrieve covariate lags, enforce order (dict only preserves insertion order for python 3.6+)
            for cov_type in ["past", "future"]:
                if cov_type in covariate_matrices:
                    np_X.append(
                        covariate_matrices[cov_type][
                            :, relative_cov_lags[cov_type] + t_pred
                        ].reshape(len(series) * num_samples, -1)
                    )

            # concatenate retrieved lags
            X = np.concatenate(np_X, axis=1)
            # Need to split up `X` into three equally-sized sub-blocks
            # corresponding to each timeseries in `series`, so that
            # static covariates can be added to each block; valid since
            # each block contains same number of observations:
            X_blocks = np.split(X, len(series), axis=0)
            X_blocks, _ = add_static_covariates_to_lagged_data(
                X_blocks,
                series,
                uses_static_covariates=self.uses_static_covariates,
                last_shape=self._static_covariates_shape,
            )
            X = np.concatenate(X_blocks, axis=0)

            # X has shape (n_series * n_samples, n_regression_features)
            prediction = self._predict_and_sample(
                X, num_samples, predict_likelihood_parameters, **kwargs
            )
            # prediction shape (n_series * n_samples, output_chunk_length, n_components)
            # append prediction to final predictions
            predictions.append(prediction[:, last_step_shift:])

        # concatenate and use first n points as prediction
        predictions = np.concatenate(predictions, axis=1)[:, :n]

        # bring into correct shape: (n_series, output_chunk_length, n_components, n_samples)
        predictions = np.moveaxis(
            predictions.reshape(len(series), num_samples, n, -1), 1, -1
        )

        # build time series from the predicted values starting after end of series
        predictions = [
            self._build_forecast_series(
                points_preds=row,
                input_series=input_tgt,
                custom_components=self._likelihood_components_names(input_tgt)
                if predict_likelihood_parameters
                else None,
                with_static_covs=False if predict_likelihood_parameters else True,
                with_hierarchy=False if predict_likelihood_parameters else True,
            )
            for idx_ts, (row, input_tgt) in enumerate(zip(predictions, series))
        ]

        return predictions[0] if called_with_single_series else predictions

    def _predict_and_sample(
        self,
        x: np.ndarray,
        num_samples: int,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        """By default, the regression model returns a single sample."""
        prediction = self.model.predict(x, **kwargs)
        k = x.shape[0]
        return prediction.reshape(k, self.pred_dim, -1)

    @property
    def lagged_feature_names(self) -> Optional[List[str]]:
        """The lagged feature names the model has been trained on.

        The naming convention for target, past and future covariates is: ``"{name}_{type}_lag{i}"``, where:

            - ``{name}`` the component name of the (first) series
            - ``{type}`` is the feature type, one of "target", "pastcov", and "futcov"
            - ``{i}`` is the lag value

        The naming convention for static covariates is: ``"{name}_statcov_target_{comp}"``, where:

            - ``{name}`` the static covariate name of the (first) series
            - ``{comp}`` the target component name of the (first) that the static covariate act on. If the static
                covariate acts globally on a multivariate target series, will show "global".
        """
        return self._lagged_feature_names

    def __str__(self):
        return self.model.__str__()

    @property
    def supports_past_covariates(self) -> bool:
        return len(self.lags.get("past", [])) > 0

    @property
    def supports_future_covariates(self) -> bool:
        return len(self.lags.get("future", [])) > 0

    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        return True

    def _check_optimizable_historical_forecasts(
        self,
        forecast_horizon: int,
        retrain: Union[bool, int, Callable[..., bool]],
        show_warnings=bool,
    ) -> bool:
        """
        Historical forecast can be optimized only if `retrain=False` and `forecast_horizon <= self.output_chunk_length`
        (no auto-regression required).
        """

        supported_retrain = (retrain is False) or (retrain == 0)
        supported_forecast_horizon = forecast_horizon <= self.output_chunk_length
        if supported_retrain and supported_forecast_horizon:
            return True

        if show_warnings:
            if not supported_retrain:
                logger.warning(
                    "`enable_optimization=True` is ignored because `retrain` is not `False`"
                    "To hide this warning, set `show_warnings=False` or `enable_optimization=False`."
                )
            if not supported_forecast_horizon:
                logger.warning(
                    "`enable_optimization=True` is ignored because "
                    "`forecast_horizon > self.output_chunk_length`."
                    "To hide this warning, set `show_warnings=False` or `enable_optimization=False`."
                )

        return False

    def _optimized_historical_forecasts(
        self,
        series: Optional[Sequence[TimeSeries]],
        past_covariates: Optional[Sequence[TimeSeries]] = None,
        future_covariates: Optional[Sequence[TimeSeries]] = None,
        num_samples: int = 1,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        forecast_horizon: int = 1,
        stride: int = 1,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
    ) -> Union[
        TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
    ]:
        """
        TODO: support forecast_horizon > output_chunk_length (auto-regression)
        """
        if not self._fit_called:
            raise_log(
                ValueError("Model has not been fit yet."),
                logger,
            )
        if forecast_horizon > self.output_chunk_length:
            raise_log(
                ValueError(
                    "`forecast_horizon > model.output_chunk_length` requires auto-regression which is not "
                    "supported in this optimized routine."
                ),
                logger,
            )

        # manage covariates, usually handled by RegressionModel.predict()
        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = [self.past_covariate_series] * len(series)
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = [self.future_covariate_series] * len(series)

        self._verify_static_covariates(series[0].static_covariates)

        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_fit_predict_encodings(
                n=forecast_horizon,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        # TODO: move the loop here instead of duplicated code in each sub-routine?
        if last_points_only:
            return _optimized_historical_forecasts_regression_last_points_only(
                model=self,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                overlap_end=overlap_end,
                show_warnings=show_warnings,
                predict_likelihood_parameters=predict_likelihood_parameters,
            )
        else:
            return _optimized_historical_forecasts_regression_all_points(
                model=self,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                overlap_end=overlap_end,
                show_warnings=show_warnings,
                predict_likelihood_parameters=predict_likelihood_parameters,
            )


class _LikelihoodMixin:
    """
    A class containing functions supporting quantile, poisson and gaussian regression, to be used as a mixin for some
    `RegressionModel` subclasses.
    """

    @staticmethod
    def _check_likelihood(likelihood, available_likelihoods):
        raise_if_not(
            likelihood in available_likelihoods,
            f"If likelihood is specified it must be one of {available_likelihoods}",
        )

    @staticmethod
    def _get_model_container():
        return _QuantileModelContainer()

    @staticmethod
    def _prepare_quantiles(quantiles):
        if quantiles is None:
            quantiles = [
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        else:
            quantiles = sorted(quantiles)
            _check_quantiles(quantiles)
        median_idx = quantiles.index(0.5)

        return quantiles, median_idx

    def _likelihood_components_names(
        self, input_series: TimeSeries
    ) -> Optional[List[str]]:
        if self.likelihood == "quantile":
            return self._quantiles_generate_components_names(input_series)
        elif self.likelihood == "poisson":
            return self._likelihood_generate_components_names(input_series, ["lambda"])
        else:
            return None

    def _predict_quantile(
        self,
        x: np.ndarray,
        num_samples: int,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        """
        X is of shape (n_series * n_samples, n_regression_features)
        """
        k = x.shape[0]

        # if predict_likelihood_parameters is True, all the quantiles must be predicted
        if num_samples == 1 and not predict_likelihood_parameters:
            # return median
            fitted = self._model_container[0.5]
            return fitted.predict(x, **kwargs).reshape(k, self.pred_dim, -1)

        model_outputs = []
        for quantile, fitted in self._model_container.items():
            self.model = fitted
            # model output has shape (n_series * n_samples, output_chunk_length, n_components)
            model_output = fitted.predict(x, **kwargs).reshape(k, self.pred_dim, -1)
            model_outputs.append(model_output)
        model_outputs = np.stack(model_outputs, axis=-1)
        # shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)
        return model_outputs

    def _predict_poisson(
        self,
        x: np.ndarray,
        num_samples: int,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        """
        X is of shape (n_series * n_samples, n_regression_features)
        """
        k = x.shape[0]
        # shape (n_series * n_samples, output_chunk_length, n_components)
        return self.model.predict(x, **kwargs).reshape(k, self.pred_dim, -1)

    def _predict_normal(
        self,
        x: np.ndarray,
        num_samples: int,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        """Method intended for CatBoost's RMSEWithUncertainty loss. Returns samples
        computed from double-valued inputs [mean, variance].
        X is of shape (n_series * n_samples, n_regression_features)
        """
        k = x.shape[0]

        # model_output shape:
        # if univariate & output_chunk_length = 1: (num_samples, 2)
        # else: (2, num_samples, n_components * output_chunk_length)
        # where the axis with 2 dims is mu, sigma
        model_output = self.model.predict(x, **kwargs)
        output_dim = len(model_output.shape)

        # deterministic case: we return the mean only
        if num_samples == 1 and not predict_likelihood_parameters:
            # univariate & single-chunk output
            if output_dim <= 2:
                output_slice = model_output[:, 0]
            else:
                output_slice = model_output[0, :, :]
            return output_slice.reshape(k, self.pred_dim, -1)

        # probabilistic case
        # univariate & single-chunk output
        if output_dim <= 2:
            # embedding well shaped 2D output into 3D
            model_output = np.expand_dims(model_output, axis=0)

        else:
            # we transpose to get mu, sigma couples on last axis
            # shape becomes: (n_components * output_chunk_length, num_samples, 2)
            model_output = model_output.transpose()

        # shape (n_components * output_chunk_length, num_samples, 2)
        return model_output

    def _sampling_quantile(self, model_output: np.ndarray) -> np.ndarray:
        """
        Sample uniformly between [0, 1] (for each batch example) and return the linear interpolation between the fitted
        quantiles closest to the sampled value.

        model_output is of shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)
        """
        k, n_timesteps, n_components, n_quantiles = model_output.shape

        # obtain samples
        probs = self._rng.uniform(
            size=(
                k,
                n_timesteps,
                n_components,
                1,
            )
        )

        # add dummy dim
        probas = np.expand_dims(probs, axis=-2)

        # tile and transpose
        p = np.tile(probas, (1, 1, 1, n_quantiles, 1)).transpose((0, 1, 2, 4, 3))

        # prepare quantiles
        tquantiles = np.array(self.quantiles).reshape((1, 1, 1, -1))

        # calculate index of the largest quantile smaller than the sampled value
        left_idx = np.sum(p > tquantiles, axis=-1)

        # obtain index of the smallest quantile larger than the sampled value
        right_idx = left_idx + 1

        # repeat the model output on the edges
        repeat_count = [1] * n_quantiles
        repeat_count[0] = 2
        repeat_count[-1] = 2
        repeat_count = np.array(repeat_count)
        shifted_output = np.repeat(model_output, repeat_count, axis=-1)

        # obtain model output values corresponding to the quantiles left and right of the sampled value
        left_value = np.take_along_axis(shifted_output, left_idx, axis=-1)
        right_value = np.take_along_axis(shifted_output, right_idx, axis=-1)

        # add 0 and 1 to quantiles
        ext_quantiles = [0.0] + self.quantiles + [1.0]
        expanded_q = np.tile(np.array(ext_quantiles), left_idx.shape)

        # calculate closest quantiles to the sampled value
        left_q = np.take_along_axis(expanded_q, left_idx, axis=-1)
        right_q = np.take_along_axis(expanded_q, right_idx, axis=-1)

        # linear interpolation
        weights = (probs - left_q) / (right_q - left_q)
        inter = left_value + weights * (right_value - left_value)

        # shape (n_series * n_samples, output_chunk_length, n_components * n_quantiles)
        return inter.squeeze(-1)

    def _sampling_poisson(self, model_output: np.ndarray) -> np.ndarray:
        """
        model_output is of shape (n_series * n_samples, output_chunk_length, n_components)
        """
        return self._rng.poisson(lam=model_output).astype(float)

    def _sampling_normal(self, model_output: np.ndarray) -> np.ndarray:
        """Sampling method for CatBoost's [mean, variance] output.
        model_output is of shape (n_components * output_chunk_length, n_samples, 2) where the last dimension
        contain mu and sigma.
        """
        n_entries, n_samples, n_params = model_output.shape

        # treating each component separately
        mu_sigma_list = [model_output[i, :, :] for i in range(n_entries)]

        list_of_samples = [
            self._rng.normal(
                mu_sigma[:, 0],  # mean vector
                mu_sigma[:, 1],  # diagonal covariance matrix
            )
            for mu_sigma in mu_sigma_list
        ]

        samples_transposed = np.array(list_of_samples).transpose()
        samples_reshaped = samples_transposed.reshape(n_samples, self.pred_dim, -1)

        return samples_reshaped

    def _params_quantile(self, model_output: np.ndarray) -> np.ndarray:
        """Quantiles on the last dimension, grouped by component"""
        k, n_timesteps, n_components, n_quantiles = model_output.shape
        # last dim : [comp_1_q_1, ..., comp_1_q_n, ..., comp_n_q_1, ..., comp_n_q_n]
        return model_output.reshape(k, n_timesteps, n_components * n_quantiles)

    def _params_poisson(self, model_output: np.ndarray) -> np.ndarray:
        """Lambdas on the last dimension, grouped by component"""
        return model_output

    def _params_normal(self, model_output: np.ndarray) -> np.ndarray:
        """[mu, sigma] on the last dimension, grouped by component"""
        shape = model_output.shape
        n_samples = shape[1]

        # extract mu and sigma for each component
        mu_sigma_list = [model_output[i, :, :] for i in range(shape[0])]

        # reshape to (n_samples, output_chunk_length, 2)
        params_transposed = np.array(mu_sigma_list).transpose()
        params_reshaped = params_transposed.reshape(n_samples, self.pred_dim, -1)
        return params_reshaped

    def _predict_and_sample_likelihood(
        self,
        x: np.ndarray,
        num_samples: int,
        likelihood: str,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        model_output = getattr(self, f"_predict_{likelihood}")(
            x, num_samples, predict_likelihood_parameters, **kwargs
        )
        if predict_likelihood_parameters:
            return getattr(self, f"_params_{likelihood}")(model_output)
        else:
            if num_samples == 1:
                return model_output
            else:
                return getattr(self, f"_sampling_{likelihood}")(model_output)

    def _num_parameters_quantile(self) -> int:
        return len(self.quantiles)

    def _num_parameters_poisson(self) -> int:
        return 1

    def _num_parameters_normal(self) -> int:
        return 2

    @property
    def num_parameters(self) -> int:
        """Mimic function of Likelihood class"""
        likelihood = self.likelihood
        if likelihood is None:
            return 0
        elif likelihood in ["gaussian", "RMSEWithUncertainty"]:
            return self._num_parameters_normal()
        else:
            return getattr(self, f"_num_parameters_{likelihood}")()

    def _quantiles_generate_components_names(
        self, input_series: TimeSeries
    ) -> List[str]:
        return self._likelihood_generate_components_names(
            input_series,
            [f"q{quantile:.2f}" for quantile in self._model_container.keys()],
        )

    def _likelihood_generate_components_names(
        self, input_series: TimeSeries, parameter_names: List[str]
    ) -> List[str]:
        return [
            f"{tgt_name}_{param_n}"
            for tgt_name in input_series.components
            for param_n in parameter_names
        ]


class _QuantileModelContainer(OrderedDict):
    def __init__(self):
        super().__init__()


class RegressionModelWithCategoricalCovariates(RegressionModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        add_encoders: Optional[dict] = None,
        model=None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        categorical_past_covariates: Optional[Union[str, List[str]]] = None,
        categorical_future_covariates: Optional[Union[str, List[str]]] = None,
        categorical_static_covariates: Optional[Union[str, List[str]]] = None,
    ):
        """
        Extension of `RegressionModel` for regression models that support categorical covariates.

        Parameters
        ----------
        lags
            Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags
            are used (from -1 backward). Otherwise, a list of integers with lags is required (each lag must be < 0).
        lags_past_covariates
            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last
            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers
            with lags < 0 is required.
        lags_future_covariates
            Number of lagged future_covariates values used to predict the next time step. If a tuple (past, future) is
            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first
            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list
            of integers with lags is required.
        output_chunk_length
            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast
            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may
            be useful if the covariates don't extend far enough into the future.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        model
            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model is
            trained to predict at step 'output_chunk_length' in the future. Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        categorical_past_covariates
            Optionally, component name or list of component names specifying the past covariates that should be treated
            as categorical.
        categorical_future_covariates
            Optionally, component name or list of component names specifying the future covariates that should be
            treated as categorical.
        categorical_static_covariates
            Optionally, string or list of strings specifying the static covariates that should be treated as
            categorical.
        """
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            add_encoders=add_encoders,
            model=model,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
        )
        self.categorical_past_covariates = (
            [categorical_past_covariates]
            if isinstance(categorical_past_covariates, str)
            else categorical_past_covariates
        )
        self.categorical_future_covariates = (
            [categorical_future_covariates]
            if isinstance(categorical_future_covariates, str)
            else categorical_future_covariates
        )
        self.categorical_static_covariates = (
            [categorical_static_covariates]
            if isinstance(categorical_static_covariates, str)
            else categorical_static_covariates
        )

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        **kwargs,
    ):
        self._validate_categorical_covariates(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            n_jobs_multioutput_wrapper=n_jobs_multioutput_wrapper,
            **kwargs,
        )

    @property
    def _categorical_fit_param(self) -> Tuple[str, Any]:
        """
        Returns the name, and default value of the categorical features parameter from model's `fit` method .
        Can be overridden in subclasses.
        """
        return "categorical_feature", "auto"

    def _validate_categorical_covariates(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> None:
        """
        Checks that the categorical covariates are valid. Specifically, checks that the categorical covariates
        of the model are a subset of all covariates.

        Parameters
        ----------
        series
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        """
        for categorical_covariates, covariates, cov_type in zip(
            [self.categorical_past_covariates, self.categorical_future_covariates],
            [past_covariates, future_covariates],
            ["past_covariates", "future_covariates"],
        ):
            if categorical_covariates:
                if not covariates:
                    raise_log(
                        ValueError(
                            f"`categorical_{cov_type}` were declared at model creation but no "
                            f"`{cov_type}` are passed to the `fit()` call."
                        ),
                    )
                s = get_single_series(covariates)
                if not set(categorical_covariates).issubset(set(s.components)):
                    raise_log(
                        ValueError(
                            f"Some `categorical_{cov_type}` components "
                            f"({set(categorical_covariates) - set(s.components)}) "
                            f"declared at model creation are not present in the `{cov_type}` "
                            f"passed to the `fit()` call."
                        )
                    )
        if self.categorical_static_covariates:
            s = get_single_series(series)
            covariates = s.static_covariates
            if not s.has_static_covariates:
                raise_log(
                    ValueError(
                        "`categorical_static_covariates` were declared at model creation but `series`"
                        "passed to the `fit()` call does not contain `static_covariates`."
                    ),
                )
            if not set(self.categorical_static_covariates).issubset(
                set(covariates.columns)
            ):
                raise_log(
                    ValueError(
                        f"Some `categorical_static_covariates` components "
                        f"({set(self.categorical_static_covariates) - set(covariates.columns)}) "
                        f"declared at model creation are not present in the series' `static_covariates` "
                        f"passed to the `fit()` call."
                    )
                )

    def _get_categorical_features(
        self,
        series: Union[List[TimeSeries], TimeSeries],
        past_covariates: Optional[Union[List[TimeSeries], TimeSeries]] = None,
        future_covariates: Optional[Union[List[TimeSeries], TimeSeries]] = None,
    ) -> Tuple[List[int], List[str]]:
        """
        Returns the indices and column names of the categorical features in the regression model.

        Steps:
        1. Get the list of features used in the model. We keep the creation order of the different lags/features
            in create_lagged_data.
        2. Get the indices of the categorical features in the list of features.
        """

        categorical_covariates = (
            (
                self.categorical_past_covariates
                if self.categorical_past_covariates
                else []
            )
            + (
                self.categorical_future_covariates
                if self.categorical_future_covariates
                else []
            )
            + (
                self.categorical_static_covariates
                if self.categorical_static_covariates
                else []
            )
        )

        if not categorical_covariates:
            return [], []
        else:
            target_ts = get_single_series(series)
            past_covs_ts = get_single_series(past_covariates)
            fut_covs_ts = get_single_series(future_covariates)

            # We keep the creation order of the different lags/features in create_lagged_data
            feature_list = (
                [
                    f"target_{component}_lag{lag}"
                    for lag in self.lags.get("target", [])
                    for component in target_ts.components
                ]
                + [
                    f"past_cov_{component}_lag{lag}"
                    for lag in self.lags.get("past", [])
                    for component in past_covs_ts.components
                ]
                + [
                    f"fut_cov_{component}_lag{lag}"
                    for lag in self.lags.get("future", [])
                    for component in fut_covs_ts.components
                ]
                + (
                    list(target_ts.static_covariates.columns)
                    if target_ts.has_static_covariates
                    # if isinstance(target_ts.static_covariates, pd.DataFrame)
                    else []
                )
            )

            indices = [
                i
                for i, col in enumerate(feature_list)
                for cat in categorical_covariates
                if cat and cat in col
            ]
            col_names = [feature_list[i] for i in indices]

            return indices, col_names

    def _fit_model(
        self,
        target_series,
        past_covariates,
        future_covariates,
        max_samples_per_ts,
        **kwargs,
    ):
        """
        Custom fit function for `RegressionModelWithCategoricalCovariates` models, adding logic to let the model
        handle categorical features directly.
        """
        cat_col_indices, _ = self._get_categorical_features(
            target_series,
            past_covariates,
            future_covariates,
        )

        cat_param_name, cat_param_default = self._categorical_fit_param
        kwargs[cat_param_name] = (
            cat_col_indices if cat_col_indices else cat_param_default
        )
        super()._fit_model(
            target_series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            **kwargs,
        )
