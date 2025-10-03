"""
SKLearn-Like Models
-------------------

Darts provides a comprehensive set of forecasting models that wrap around scikit-learn-like machine learning algorithms.
These models can forecast future values of a target series based on lagged features of the following inputs:

* **Target series** (past lags only) - the main time series to be forecasted
* **Past covariates** (past lags only) - external variables known only up to the prediction time
* **Future covariates** (past and future lags) - external variables known or forecasted into the future
* **Static covariates** - time-invariant features associated with each target series

The models learn in a supervised way by tabularizing time series data into (lagged) feature-target pairs, making them
compatible with any scikit-learn-like model that has ``fit()`` and ``predict()`` methods.

Lags can be specified in multiple ways:

* As an integer for the target and past covariates series: represents the number of past lags to consider
* As a tuple (integer, integer) for the future covariates series: represents the number of (past, future) lags to
  consider
* As a list: explicitly enumerate specific lags (negative values for past, non-negative for future)
* As a dictionary: specify different lags for different series components (columns)

We offer both regression and classification models, each tailored for specific forecasting tasks.

**Regression Models:** These models predict continuous numerical values, making them ideal for forecasting future
trends and patterns in time series data. Utilize these models to gain insights into potential future outcomes based on
historical data.

* :class:`~darts.models.forecasting.sklearn_model.SKLearnModel` - Generic wrapper around any scikit-learn regression
  model (default: `LinearRegression`)
* :class:`~darts.models.forecasting.linear_regression_model.LinearRegressionModel` - Wrapper around scikit-learn's
  `LinearRegression`
* :class:`~darts.models.forecasting.random_forest.RandomForestModel` - Wrapper around scikit-learn's
  `RandomForestRegressor`
* :class:`~darts.models.forecasting.catboost_model.CatBoostModel` - Wrapper around CatBoost's `CatBoostRegressor`
* :class:`~darts.models.forecasting.lgbm.LightGBMModel` - Wrapper around LightGBM's `LGBMRegressor`
* :class:`~darts.models.forecasting.xgboost.XGBModel` - Wrapper around XGBoost's `XGBRegressor`

**Classification Models:** These models predict categorical class labels, enabling effective time series labeling and
future class prediction. These models are perfect for scenarios where identifying distinct categories or states over
time is crucial.

* :class:`~darts.models.forecasting.sklearn_model.SKLearnClassifierModel` - Generic wrapper around any scikit-learn
  classification model (default: `LogisticRegression`)
* :class:`~darts.models.forecasting.catboost_model.CatBoostClassifierModel` - Wrapper around CatBoost's
  `CatBoostClassifier`
* :class:`~darts.models.forecasting.lgbm.LightGBMClassifierModel` - Wrapper around LightGBM's `LGBMClassifier`
* :class:`~darts.models.forecasting.xgboost.XGBClassifierModel` - Wrapper around XGBoost's `XGBClassifier`

For detailed examples and tutorials, see:

* `SKLearn-Like Regression Model Examples <https://unit8co.github.io/darts/examples/20-SKLearnModel-examples.html>`_
* `SKLearn-Like Classification Model Examples
  <https://unit8co.github.io/darts/examples/24-SKLearnClassifierModel-examples.html>`_
"""

import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import has_fit_parameter

from darts import TimeSeries
from darts.logging import (
    get_logger,
    raise_deprecation_warning,
    raise_if,
    raise_if_not,
    raise_log,
)
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.data.tabularization import (
    _create_lagged_data_autoregression,
    create_lagged_component_names,
    create_lagged_training_data,
)
from darts.utils.historical_forecasts import (
    _check_optimizable_historical_forecasts_global_models,
    _optimized_historical_forecasts_all_points,
    _optimized_historical_forecasts_last_points_only,
    _process_historical_forecast_input,
)
from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.sklearn import (
    QuantileRegression,
    SKLearnLikelihood,
    _get_likelihood,
)
from darts.utils.multioutput import MultiOutputMixin, get_multioutput_estimator_cls
from darts.utils.ts_utils import get_single_series, seq2series, series2seq
from darts.utils.utils import ModelType, random_method

logger = get_logger(__name__)

LAGS_TYPE = Union[int, list[int], dict[str, Union[int, list[int]]]]
FUTURE_LAGS_TYPE = Union[
    tuple[int, int], list[int], dict[str, Union[tuple[int, int], list[int]]]
]


class SKLearnModel(GlobalForecastingModel):
    @random_method
    def __init__(
        self,
        lags: Optional[LAGS_TYPE] = None,
        lags_past_covariates: Optional[LAGS_TYPE] = None,
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        model=None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        random_state: Optional[int] = None,
    ):
        """Regression Model
        Can be used to fit any scikit-learn-like regressor class to predict the target time series from lagged values.

        Parameters
        ----------
        lags
            Lagged target `series` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`
            corresponds the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `series` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_past_covariates
            Lagged `past_covariates` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,
            where `0` corresponds to the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_future_covariates
            Lagged `future_covariates` values used to predict the next time step/s. The lags are always relative to the
            first step in the output chunk, even when `output_chunk_shift > 0`.
            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`
            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0` corresponds the first
            predicted time step of each sample. If `output_chunk_shift > 0`, the position of negative lags differ from
            those of `lags` and `lags_past_covariates`. In this case a future lag `-5` would point at the same
            step as a target lag of `-5 + output_chunk_shift`.
            If a list of integers, uses only the specified values as lags.
            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key
            'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using a
            one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression. This is
            useful when the covariates don't extend far enough into the future, or to prohibit the model from using
            future values of past and / or future covariates for prediction (depending on the model's covariate
            support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input (history of target and past covariates) and
            output. If the model supports `future_covariates`, the `lags_future_covariates` are relative to the first
            step in the shifted output chunk. Predictions will start `output_chunk_shift` steps after the end of the
            target `series`. If `output_chunk_shift` is set, the model cannot generate autoregressive predictions
            (`n > output_chunk_length`).
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

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        model
            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model
            is trained to predict all the steps in 'output_chunk_length' (features lags are shifted back by
            `output_chunk_length - n` for each step `n`). Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        random_state
            Controls the randomness for reproducible forecasting.

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import SKLearnModel
        >>> from sklearn.linear_model import Ridge
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # wrap around the sklearn Ridge model
        >>> model = SKLearnModel(
        >>>     model=Ridge(),
        >>>     lags=12,
        >>>     lags_past_covariates=4,
        >>>     lags_future_covariates=(0,6),
        >>>     output_chunk_length=6
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[1005.73340676],
               [1005.71159051],
               [1005.7322616 ],
               [1005.76314504],
               [1005.82204348],
               [1005.89100967]])
        """

        super().__init__(add_encoders=add_encoders)

        self.model = model
        self.lags: dict[str, list[int]] = {}
        self.component_lags: dict[str, dict[str, list[int]]] = {}
        self.input_dim = None
        self.multi_models = True if multi_models or output_chunk_length == 1 else False
        self._considers_static_covariates = use_static_covariates
        self._static_covariates_shape: Optional[tuple[int, int]] = None
        self._lagged_feature_names: Optional[list[str]] = None
        self._lagged_label_names: Optional[list[str]] = None

        # optionally, the model can be wrapped in a likelihood model
        self._likelihood: Optional[SKLearnLikelihood] = getattr(
            self, "_likelihood", None
        )
        # for quantile likelihood models, the model container is a dict of quantile -> model
        self._model_container: Optional[_QuantileModelContainer] = getattr(
            self, "_model_container", None
        )

        # check and set output_chunk_length
        raise_if_not(
            isinstance(output_chunk_length, int) and output_chunk_length > 0,
            f"output_chunk_length must be an integer greater than 0. Given: {output_chunk_length}",
            logger=logger,
        )
        self._output_chunk_length = output_chunk_length
        self._output_chunk_shift = output_chunk_shift

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
        self._validate_lags(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
        )

        # convert lags arguments to list of int
        # lags attribute should always be accessed with self._get_lags(), not self.lags.get()
        self.lags, self.component_lags = self._generate_lags(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_shift=output_chunk_shift,
        )

        self.pred_dim = self.output_chunk_length if self.multi_models else 1

    def _validate_lags(
        self,
        lags: Optional[LAGS_TYPE],
        lags_past_covariates: Optional[LAGS_TYPE],
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE],
    ):
        raise_if(
            (lags is None)
            and (lags_future_covariates is None)
            and (lags_past_covariates is None),
            "At least one of `lags`, `lags_future_covariates` or `lags_past_covariates` must be not None.",
        )

    @staticmethod
    def _generate_lags(
        lags: Optional[LAGS_TYPE],
        lags_past_covariates: Optional[LAGS_TYPE],
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE],
        output_chunk_shift: int,
    ) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
        """
        Based on the type of the argument and the nature of the covariates, perform some sanity checks before
        converting the lags to a list of integer.

        If lags are provided as a dictionary, the lags values are contained in self.component_lags and the self.lags
        attributes contain only the extreme values
        If the lags are provided as integer, list, tuple or dictionary containing only the 'default_lags' keys, the lags
        values are contained in the self.lags attribute and the self.component_lags is an empty dictionary.

        If `output_chunk_shift > 0`, the `lags_future_covariates` are shifted into the future.
        """
        processed_lags: dict[str, list[int]] = dict()
        processed_component_lags: dict[str, dict[str, list[int]]] = dict()
        for lags_values, lags_name, lags_abbrev in zip(
            [lags, lags_past_covariates, lags_future_covariates],
            ["lags", "lags_past_covariates", "lags_future_covariates"],
            ["target", "past", "future"],
        ):
            if lags_values is None:
                continue

            # converting to dictionary to run sanity checks
            if not isinstance(lags_values, dict):
                lags_values = {"default_lags": lags_values}
            elif len(lags_values) == 0:
                raise_log(
                    ValueError(
                        f"When passed as a dictionary, `{lags_name}` must contain at least one key."
                    ),
                    logger,
                )

            invalid_type = False
            supported_types = ""
            min_lags = None
            max_lags = None
            tmp_components_lags: dict[str, list[int]] = dict()
            for comp_name, comp_lags in lags_values.items():
                if lags_name == "lags_future_covariates":
                    if isinstance(comp_lags, tuple):
                        raise_if_not(
                            len(comp_lags) == 2
                            and isinstance(comp_lags[0], int)
                            and isinstance(comp_lags[1], int),
                            f"`{lags_name}` - `{comp_name}`: tuple must be of length 2, and must contain two integers",
                            logger,
                        )

                        raise_if(
                            isinstance(comp_lags[0], bool)
                            or isinstance(comp_lags[1], bool),
                            f"`{lags_name}` - `{comp_name}`: tuple must contain integers, not bool",
                            logger,
                        )

                        raise_if_not(
                            comp_lags[0] >= 0 and comp_lags[1] >= 0,
                            f"`{lags_name}` - `{comp_name}`: tuple must contain positive integers. Given: {comp_lags}.",
                            logger,
                        )
                        raise_if(
                            comp_lags[0] == 0 and comp_lags[1] == 0,
                            f"`{lags_name}` - `{comp_name}`: tuple cannot be (0, 0) as it corresponds to an empty "
                            f"list of lags.",
                            logger,
                        )
                        tmp_components_lags[comp_name] = list(
                            range(-comp_lags[0], comp_lags[1])
                        )
                    elif isinstance(comp_lags, list):
                        for lag in comp_lags:
                            raise_if(
                                not isinstance(lag, int) or isinstance(lag, bool),
                                f"`{lags_name}` - `{comp_name}`: list must contain only integers. Given: {comp_lags}.",
                                logger,
                            )
                        tmp_components_lags[comp_name] = sorted(comp_lags)
                    else:
                        invalid_type = True
                        supported_types = "tuple or a list"
                else:
                    if isinstance(comp_lags, int):
                        raise_if_not(
                            comp_lags > 0,
                            f"`{lags_name}` - `{comp_name}`: integer must be strictly positive . Given: {comp_lags}.",
                            logger,
                        )
                        tmp_components_lags[comp_name] = list(range(-comp_lags, 0))
                    elif isinstance(comp_lags, list):
                        for lag in comp_lags:
                            raise_if(
                                not isinstance(lag, int) or (lag >= 0),
                                f"`{lags_name}` - `{comp_name}`: list must contain only strictly negative integers. "
                                f"Given: {comp_lags}.",
                                logger,
                            )
                        tmp_components_lags[comp_name] = sorted(comp_lags)
                    else:
                        invalid_type = True
                        supported_types = "strictly positive integer or a list"

                if invalid_type:
                    raise_log(
                        ValueError(
                            f"`{lags_name}` - `{comp_name}`: must be either a {supported_types}. "
                            f"Given : {type(comp_lags)}."
                        ),
                        logger,
                    )

                # extracting min and max lags va
                if min_lags is None:
                    min_lags = tmp_components_lags[comp_name][0]
                else:
                    min_lags = min(min_lags, tmp_components_lags[comp_name][0])

                if max_lags is None:
                    max_lags = tmp_components_lags[comp_name][-1]
                else:
                    max_lags = max(max_lags, tmp_components_lags[comp_name][-1])

            # Check if only default lags are provided
            has_default_lags = list(tmp_components_lags.keys()) == ["default_lags"]

            # revert to shared lags logic when applicable
            if has_default_lags:
                processed_lags[lags_abbrev] = tmp_components_lags["default_lags"]
            else:
                processed_lags[lags_abbrev] = [min_lags, max_lags]
                processed_component_lags[lags_abbrev] = tmp_components_lags

            # if output chunk is shifted, shift future covariates lags with it
            if output_chunk_shift and lags_abbrev == "future":
                processed_lags[lags_abbrev] = [
                    lag_ + output_chunk_shift for lag_ in processed_lags[lags_abbrev]
                ]
                if processed_component_lags and not has_default_lags:
                    processed_component_lags[lags_abbrev] = {
                        comp_: [lag_ + output_chunk_shift for lag_ in lags_]
                        for comp_, lags_ in processed_component_lags[
                            lags_abbrev
                        ].items()
                    }
        return processed_lags, processed_component_lags

    def _get_lags(self, lags_type: str):
        """
        If lags were specified in a component-wise manner, they are contained in self.component_lags and
        the values in self.lags should be ignored as they correspond just the extreme values.
        """
        if lags_type in self.component_lags:
            return self.component_lags[lags_type]
        else:
            return self.lags.get(lags_type, None)

    def _get_lagged_features(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
    ) -> list[list[tuple[str, str, int]]]:
        """Returns a list of lagged features for the target, past, future, and static covariates.

        The returned lagged features are in the same order as the features passed to the underlying model.
        """
        feature_list = []
        for series_, lags_name, feature_type, series_name in zip(
            [series, past_covariates, future_covariates],
            ["target", "past", "future"],
            ["target", "past_cov", "fut_cov"],
            ["series", "past_covariates", "future_covariates"],
        ):
            series_ = get_single_series(series_)
            lags = self._get_lags(lags_name)
            if lags is None:
                feature_list.append([])
                continue

            if series_ is None:
                raise_log(
                    ValueError(
                        f"`{series_name}` cannot be `None` when lags are specified for it."
                    ),
                    logger=logger,
                )

            is_comp_specific = self.component_lags.get(lags_name) is not None
            series_comps = series_.components.tolist()
            component_lags = {}

            # create a mapping of lags to components (components are in the same order as in the series)
            if is_comp_specific:
                # component-specific lags are specific to each component
                for comp_name in series_comps:
                    comp_lags = lags[comp_name]
                    for lag in comp_lags:
                        if lag not in component_lags:
                            component_lags[lag] = [comp_name]
                        else:
                            component_lags[lag].append(comp_name)
            else:
                # global lags are identical for each component
                for lag in lags:
                    component_lags[lag] = series_comps

            # create the feature list (in order of increasing lag)
            lags_sorted = sorted(component_lags.keys())
            feature_list.append([
                (feature_type, component, lag)
                for lag in lags_sorted
                for component in component_lags[lag]
            ])

        # add static covariates at the end
        target_ts = get_single_series(series)
        if target_ts.has_static_covariates:
            feature_list.append([
                ("static_cov", component, 0)
                for component in list(target_ts.static_covariates.columns)
            ])
        else:
            feature_list.append([])
        return feature_list

    @property
    def _model_encoder_settings(
        self,
    ) -> tuple[int, int, bool, bool, Optional[list[int]], Optional[list[int]]]:
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
            self.output_chunk_length + self.output_chunk_shift,
            lags_past_covariates is not None,
            lags_future_covariates is not None,
            lags_past_covariates,
            lags_future_covariates,
        )

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
    ]:
        min_target_lag = self.lags["target"][0] if "target" in self.lags else None
        max_target_lag = self.output_chunk_length - 1 + self.output_chunk_shift
        min_past_cov_lag = self.lags["past"][0] if "past" in self.lags else None
        max_past_cov_lag = self.lags["past"][-1] if "past" in self.lags else None
        min_future_cov_lag = self.lags["future"][0] if "future" in self.lags else None
        max_future_cov_lag = self.lags["future"][-1] if "future" in self.lags else None
        return (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            max_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
            self.output_chunk_shift,
        )

    @property
    def supports_multivariate(self) -> bool:
        """
        If available, uses `model`'s native multivariate support. If not available, obtains multivariate support by
        wrapping the univariate model in a `sklearn.multioutput.MultiOutputRegressor`.
        """
        return True

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        input_chunk_length = -self.lags["target"][0] if "target" in self.lags else 0
        return input_chunk_length, self.output_chunk_length + self.output_chunk_shift

    @property
    def min_train_samples(self) -> int:
        # some models require more than 1 sample to train (e.g. CatBoost); for consistency, we set the minimum to 2
        return 2

    @property
    def output_chunk_length(self) -> int:
        return self._output_chunk_length

    @property
    def output_chunk_shift(self) -> int:
        return self._output_chunk_shift

    def get_estimator(
        self, horizon: int, target_dim: int, quantile: Optional[float] = None
    ):
        """Returns the estimator that forecasts the `horizon`th step of the `target_dim`th target component.

        For probabilistic models fitting quantiles, a desired `quantile` can also be passed. If not passed, it will
        return the model predicting the median (quantile=0.5).

        If the (quantile) model supports multi-output natively, it will return the model that can predict the entire
        horizon and all target components jointly.

        Note: Internally, estimators are grouped by `output_chunk_length` position, then by component. For probabilistic
        models fitting quantiles, there is an additional abstraction layer, grouping the estimators by `quantile`.

        Parameters
        ----------
        horizon
            The index of the forecasting point within `output_chunk_length`.
        target_dim
            The index of the target component.
        quantile
            Optionally, for probabilistic model with `likelihood="quantile"`, the desired quantile value. If `None` and
            `likelihood="quantile"`, returns the model predicting the median (quantile=0.5).
        """
        likelihood = self.likelihood
        if isinstance(likelihood, QuantileRegression):
            # for quantile-models, the estimators are grouped by quantiles
            if quantile is None:
                quantile = likelihood.quantiles[likelihood._median_idx]
            elif quantile not in self._model_container:
                raise_log(
                    ValueError(
                        f"Invalid `quantile={quantile}`. Must be one of the fitted quantiles "
                        f"`{list(self._model_container.keys())}`."
                    ),
                    logger,
                )
            model = self._model_container[quantile]
        elif quantile is not None:
            raise_log(
                ValueError(
                    "`quantile` is only supported for probabilistic models that use `likelihood='quantile'`."
                ),
                logger=logger,
            )
        else:
            model = self.model

        if not isinstance(model, MultiOutputMixin):
            logger.warning(
                "Model supports multi-output; a single estimator forecasts all the horizons and components."
            )
            return model

        if not 0 <= horizon < self.output_chunk_length:
            raise_log(
                ValueError(
                    f"`horizon` must be `>= 0` and `< output_chunk_length={self.output_chunk_length}`."
                ),
                logger,
            )
        if not 0 <= target_dim < self.input_dim["target"]:
            raise_log(
                ValueError(
                    f"`target_dim` must be `>= 0`, and `< n_target_components={self.input_dim['target']}`."
                ),
                logger,
            )

        # when multi_models=True, one model per horizon and target component
        idx_estimator = (
            self.multi_models * self.input_dim["target"] * horizon + target_dim
        )
        return model.estimators_[idx_estimator]

    def _add_val_set_to_kwargs(
        self,
        kwargs: dict,
        val_series: Sequence[TimeSeries],
        val_past_covariates: Optional[Sequence[TimeSeries]],
        val_future_covariates: Optional[Sequence[TimeSeries]],
        val_sample_weight: Optional[Union[Sequence[TimeSeries], str]],
        max_samples_per_ts: int,
        stride: int,
    ) -> dict:
        """Creates a validation set and returns a new set of kwargs passed to `self.model.fit()` including the
        validation set. This method can be overridden if the model requires a different logic to add the eval set."""
        val_samples, val_labels, val_weight = self._create_lagged_data(
            series=val_series,
            past_covariates=val_past_covariates,
            future_covariates=val_future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            sample_weight=val_sample_weight,
            last_static_covariates_shape=self._static_covariates_shape,
            stride=stride,
        )
        # create validation sets for MultiOutputMixin
        if val_labels.ndim == 2 and isinstance(self.model, MultiOutputMixin):
            val_sets, val_weights = [], []
            for i in range(val_labels.shape[1]):
                val_sets.append((val_samples, val_labels[:, i]))
                if val_weight is not None:
                    val_weights.append(val_weight[:, i])
            val_weights = val_weights or None
        else:
            val_sets = [(val_samples, val_labels)]
            val_weights = [val_weight]

        val_set_name, val_weight_name = self.val_set_params
        return dict(kwargs, **{val_set_name: val_sets, val_weight_name: val_weights})

    def _create_lagged_data(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries],
        future_covariates: Sequence[TimeSeries],
        max_samples_per_ts: int,
        sample_weight: Optional[Union[TimeSeries, str]] = None,
        stride: int = 1,
        last_static_covariates_shape: Optional[tuple[int, int]] = None,
    ):
        (
            features,
            labels,
            _,
            self._static_covariates_shape,
            sample_weights,
        ) = create_lagged_training_data(
            target_series=series,
            output_chunk_length=self.output_chunk_length,
            output_chunk_shift=self.output_chunk_shift,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            lags=self._get_lags("target"),
            lags_past_covariates=self._get_lags("past"),
            lags_future_covariates=self._get_lags("future"),
            uses_static_covariates=self.uses_static_covariates,
            last_static_covariates_shape=last_static_covariates_shape,
            max_samples_per_ts=max_samples_per_ts,
            multi_models=self.multi_models,
            check_inputs=False,
            concatenate=False,
            sample_weight=sample_weight,
            stride=stride,
        )

        expected_nb_feat = (
            features[0].shape[1]
            if isinstance(features, Sequence)
            else features.shape[1]
        )
        for i, (X_i, y_i) in enumerate(zip(features, labels)):
            # TODO: account for scenario where two wrong shapes can silently hide the problem
            if expected_nb_feat != X_i.shape[1]:
                shape_error_msg = []
                for ts, cov_name, arg_name in zip(
                    [series, past_covariates, future_covariates],
                    ["target", "past", "future"],
                    ["series", "past_covariates", "future_covariates"],
                ):
                    if ts is not None and ts[i].width != self.input_dim[cov_name]:
                        shape_error_msg.append(
                            f"Expected {self.input_dim[cov_name]} components but received "
                            f"{ts[i].width} components at index {i} of `{arg_name}`."
                        )
                raise_log(ValueError("\n".join(shape_error_msg)), logger)
            features[i] = X_i[:, :, 0]
            labels[i] = y_i[:, :, 0]
            if sample_weights is not None:
                sample_weights[i] = sample_weights[i][:, :, 0]

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        if sample_weights is not None:
            sample_weights = np.concatenate(sample_weights, axis=0)

        # if labels are of shape (n_samples, 1) flatten it to shape (n_samples,)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.ravel()
        if (
            sample_weights is not None
            and sample_weights.ndim == 2
            and sample_weights.shape[1] == 1
        ):
            sample_weights = sample_weights.ravel()

        features, labels = self._format_samples(features, labels)

        return features, labels, sample_weights

    def _format_samples(
        self, samples: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> tuple[Any, Any]:
        """
        Subclasses can override this method to format the samples and labels before fit and predict.
        """
        return samples, labels

    def _fit_model(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Sequence[TimeSeries],
        future_covariates: Sequence[TimeSeries],
        max_samples_per_ts: int,
        sample_weight: Optional[Union[Sequence[TimeSeries], str]],
        stride: int,
        val_series: Optional[Sequence[TimeSeries]] = None,
        val_past_covariates: Optional[Sequence[TimeSeries]] = None,
        val_future_covariates: Optional[Sequence[TimeSeries]] = None,
        val_sample_weight: Optional[Union[Sequence[TimeSeries], str]] = None,
        verbose: Optional[bool] = None,
        **kwargs,
    ):
        """
        Function that fit the model. Deriving classes can override this method for adding additional
        parameters (e.g., adding validation data), keeping the sanity checks on series performed by fit().
        """
        training_samples, training_labels, sample_weights = self._create_lagged_data(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            sample_weight=sample_weight,
            last_static_covariates_shape=None,
            stride=stride,
        )

        if self._supports_val_series and val_series is not None:
            kwargs = self._add_val_set_to_kwargs(
                kwargs=kwargs,
                val_series=val_series,
                val_past_covariates=val_past_covariates,
                val_future_covariates=val_future_covariates,
                val_sample_weight=val_sample_weight,
                max_samples_per_ts=max_samples_per_ts,
                stride=stride,
            )

        # only use `sample_weight` if model supports it
        sample_weight_kwargs = dict()
        if sample_weights is not None:
            if self.supports_sample_weight:
                sample_weight_kwargs = {"sample_weight": sample_weights}
            else:
                logger.warning(
                    "`sample_weight` was ignored since underlying regression model's "
                    "`fit()` method does not support it."
                )

        # only pass `verbose` if provided and the underlying model supports it;
        # we always pass it to MultiOutputMixin as it will handle it there
        if verbose is not None and (
            isinstance(self.model, MultiOutputMixin)
            or "verbose" in inspect.signature(self.model.fit).parameters
        ):
            kwargs["verbose"] = verbose

        self.model.fit(
            training_samples,
            training_labels,
            **sample_weight_kwargs,
            **kwargs,
        )

        # generate and store the lagged components names (for feature importance analysis)
        self._lagged_feature_names, self._lagged_label_names = (
            create_lagged_component_names(
                target_series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                lags=self._get_lags("target"),
                lags_past_covariates=self._get_lags("past"),
                lags_future_covariates=self._get_lags("future"),
                output_chunk_length=self.output_chunk_length,
                concatenate=False,
                use_static_covariates=self.uses_static_covariates,
            )
        )

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
        stride: int = 1,
        verbose: Optional[bool] = None,
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
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
        stride
            The number of time steps between consecutive samples, applied starting from the end of the series. The same
            stride will be applied to both the training and evaluation set (if supplied and supported). This should be
            used with caution as it might introduce bias in the forecasts.
        verbose
            Optionally, set the fit verbosity. Not effective for all models.
        **kwargs
            Additional keyword arguments passed to the `fit` method of the model.
        """
        # guarantee that all inputs are either list of TimeSeries or None
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        val_series = series2seq(kwargs.pop("val_series", None))
        val_past_covariates = series2seq(kwargs.pop("val_past_covariates", None))
        val_future_covariates = series2seq(kwargs.pop("val_future_covariates", None))

        if not isinstance(sample_weight, str):
            sample_weight = series2seq(sample_weight)
        val_sample_weight = kwargs.pop("val_sample_weight", None)
        if not isinstance(val_sample_weight, str):
            val_sample_weight = series2seq(val_sample_weight)

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
            self._verify_static_covariates(get_single_series(series).static_covariates)
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

        if self._supports_val_series:
            val_series, val_past_covariates, val_future_covariates = (
                self._process_validation_set(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    val_series=val_series,
                    val_past_covariates=val_past_covariates,
                    val_future_covariates=val_future_covariates,
                )
            )

        # saving the dims of all input series to check at prediction time
        self.input_dim = {
            "target": series[0].width,
            "past": past_covariates[0].width if past_covariates else None,
            "future": future_covariates[0].width if future_covariates else None,
        }

        # Check if multi-output regression is required
        requires_multioutput = not series[0].is_univariate or (
            self.output_chunk_length > 1 and self.multi_models
        )

        # If multi-output required and model doesn't support it natively, wrap it in a MultiOutputMixin
        if (
            requires_multioutput
            and not isinstance(self.model, MultiOutputMixin)
            and (
                not self._supports_native_multioutput
                or sample_weight
                is not None  # we have 2D sample (and time) weights, only supported in Darts
            )
        ):
            val_set_name, val_weight_name = self.val_set_params
            mor_kwargs = {
                "eval_set_name": val_set_name,
                "eval_weight_name": val_weight_name,
                "n_jobs": n_jobs_multioutput_wrapper,
            }
            mor_kwargs["output_chunk_length"] = self.output_chunk_length

            self.model = get_multioutput_estimator_cls(self._model_type)(
                estimator=self.model, **mor_kwargs
            )

        if (
            not isinstance(self.model, MultiOutputMixin)
            and n_jobs_multioutput_wrapper is not None
        ):
            logger.warning("Provided `n_jobs_multioutput_wrapper` wasn't used.")

        super().fit(
            series=seq2series(series),
            past_covariates=seq2series(past_covariates),
            future_covariates=seq2series(future_covariates),
            verbose=verbose,
        )
        variate2arg = {
            "target": "lags",
            "past": "lags_past_covariates",
            "future": "lags_future_covariates",
        }

        # if provided, component-wise lags must be defined for all the components of the first series
        component_lags_error_msg = []
        for variate_type, variate in zip(
            ["target", "past", "future"], [series, past_covariates, future_covariates]
        ):
            if variate_type not in self.component_lags:
                continue

            # ignore the fallback lags entry
            provided_components = set(self.component_lags[variate_type].keys())
            required_components = set(variate[0].components)

            wrong_components = list(
                provided_components - {"default_lags"} - required_components
            )
            missing_keys = list(required_components - provided_components)
            # lags were specified for unrecognized components
            if len(wrong_components) > 0:
                component_lags_error_msg.append(
                    f"The `{variate2arg[variate_type]}` dictionary specifies lags for components that are not "
                    f"present in the series : {wrong_components}. They must be removed to avoid any ambiguity."
                )
            elif len(missing_keys) > 0 and "default_lags" not in provided_components:
                component_lags_error_msg.append(
                    f"The {variate2arg[variate_type]} dictionary is missing the lags for the following components "
                    f"present in the series: {missing_keys}. The key 'default_lags' can be used to provide lags for "
                    f"all the non-explicitely defined components."
                )
            else:
                # reorder the components based on the input series, insert the default when necessary
                self.component_lags[variate_type] = {
                    comp_name: (
                        self.component_lags[variate_type][comp_name]
                        if comp_name in self.component_lags[variate_type]
                        else self.component_lags[variate_type]["default_lags"]
                    )
                    for comp_name in variate[0].components
                }

        # single error message for all the lags arguments
        if len(component_lags_error_msg) > 0:
            raise_log(ValueError("\n".join(component_lags_error_msg)), logger)

        self._fit_model(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            val_series=val_series,
            val_past_covariates=val_past_covariates,
            val_future_covariates=val_future_covariates,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            max_samples_per_ts=max_samples_per_ts,
            stride=stride,
            verbose=verbose,
            **kwargs,
        )

        likelihood = self.likelihood
        if likelihood is not None:
            likelihood.fit(self)
        return self

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: Optional[bool] = None,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
        random_state: Optional[int] = None,
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
            Optionally, set the prediction verbosity. Not effective for all models.
        predict_likelihood_parameters
            If set to `True`, the model predicts the parameters of its `likelihood` instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``
        show_warnings
            Optionally, control whether warnings are shown. Not effective for all models.
        random_state
            Controls the randomness of probabilistic predictions.
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
                        "from not having fit the model yet, or from loading a model saved with `clean=True`."
                    ),
                    logger,
                )
            series = self.training_series

        called_with_single_series = isinstance(series, TimeSeries)

        # guarantee that all inputs are either list of TimeSeries or None
        series = series2seq(series)

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = [self.past_covariate_series] * len(series)
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = [self.future_covariate_series] * len(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        if self.uses_static_covariates:
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
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
            predict_likelihood_parameters=predict_likelihood_parameters,
            show_warnings=show_warnings,
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
                    index_text = (
                        " "
                        if called_with_single_series
                        else f" at list/sequence index {idx} "
                    )
                    raise_log(
                        ValueError(
                            f"The `{cov_type}_covariates`{index_text}are not long enough. "
                            f"Given horizon `n={n}`, `min(lags_{cov_type}_covariates)={lags[0]}`, "
                            f"`max(lags_{cov_type}_covariates)={lags[-1]}` and "
                            f"`output_chunk_length={self.output_chunk_length}`, the `{cov_type}_covariates` have to "
                            f"range from {start_ts} until {end_ts} (inclusive), but they only range from "
                            f"{cov.start_time()} until {cov.end_time()}."
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
            series_matrix = np.stack([
                ts.values(copy=False)[self.lags["target"][0] - shift :, :]
                for ts in series
            ])

        # repeat series_matrix to shape (num_samples * num_series, n_lags, n_components)
        # [series 0 sample 0, series 0 sample 1, ..., series n sample k]
        series_matrix = np.repeat(series_matrix, num_samples, axis=0)

        # same for covariate matrices
        for cov_type, data in covariate_matrices.items():
            covariate_matrices[cov_type] = np.repeat(data, num_samples, axis=0)

        # for concatenating target with predictions (or quantile parameters)
        likelihood = self.likelihood
        if predict_likelihood_parameters and likelihood is not None:
            # with `multi_models=False`, the predictions are concatenated with the past target, even if `n<=ocl`
            # to make things work, we just append the first predicted parameter (it will never be accessed)
            sample_slice = slice(0, None, likelihood.num_parameters)
        else:
            sample_slice = slice(None)

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

            # concatenate previous iteration forecasts
            if "target" in self.lags and predictions:
                series_matrix = np.concatenate(
                    [series_matrix, predictions[-1][:, :, sample_slice]], axis=1
                )

            # extract and concatenate lags from target and covariates series
            X = _create_lagged_data_autoregression(
                target_series=series,
                t_pred=t_pred,
                shift=shift,
                last_step_shift=last_step_shift,
                series_matrix=series_matrix,
                covariate_matrices=covariate_matrices,
                lags=self.lags,
                component_lags=self.component_lags,
                relative_cov_lags=relative_cov_lags,
                num_samples=num_samples,
                uses_static_covariates=self.uses_static_covariates,
                last_static_covariates_shape=self._static_covariates_shape,
            )

            # X has shape (n_series * n_samples, n_regression_features)
            prediction = self._predict(
                x=X,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                **kwargs,
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
                custom_components=(
                    self.likelihood.component_names(series=input_tgt)
                    if predict_likelihood_parameters
                    else None
                ),
                with_static_covs=False if predict_likelihood_parameters else True,
                with_hierarchy=False if predict_likelihood_parameters else True,
                pred_start=input_tgt.end_time()
                + (1 + self.output_chunk_shift) * input_tgt.freq,
            )
            for idx_ts, (row, input_tgt) in enumerate(zip(predictions, series))
        ]

        return predictions[0] if called_with_single_series else predictions

    @random_method
    def _predict(
        self,
        x: np.ndarray,
        num_samples: int,
        predict_likelihood_parameters: bool,
        random_state: Optional[int] = None,
        verbose: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate predictions.

        Generates deterministic predictions if no `Likelihood` was used.
        Otherwise, generates probabilistic predictions. Either sampled from the predicted distribution,
        or the predicted distribution parameters directly.
        """
        x, _ = self._format_samples(x)
        if self.likelihood is not None:
            return self.likelihood.predict(
                model=self,
                x=x,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                **kwargs,
            )

        prediction = self.model.predict(x, **kwargs)
        k = x.shape[0]
        return prediction.reshape(k, self.pred_dim, -1)

    @property
    def lagged_feature_names(self) -> Optional[list[str]]:
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

    @property
    def lagged_label_names(self) -> Optional[list[str]]:
        """The lagged label name for the model's estimators.

        The naming convention is: ``"{name}_target_hrz{i}"``, where:

            - ``{name}`` the component name of the (first) series
            - ``{i}`` is the position in output_chunk_length (label lag)
        """
        return self._lagged_label_names

    def __str__(self):
        return self.model.__str__()

    @property
    def likelihood(self) -> Optional[SKLearnLikelihood]:
        return getattr(self, "_likelihood", None)

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
    def supports_probabilistic_prediction(self) -> bool:
        return self.likelihood is not None

    @property
    def supports_sample_weight(self) -> bool:
        """Whether the model supports a validation set during training."""
        return (
            self.model.supports_sample_weight
            if isinstance(self.model, MultiOutputMixin)
            else has_fit_parameter(self.model, "sample_weight")
        )

    @property
    def val_set_params(self) -> tuple[Optional[str], Optional[str]]:
        """Returns the parameter names for the validation set, and validation sample weights if it supports
        a validation set."""
        return None, None

    def _check_optimizable_historical_forecasts(
        self,
        forecast_horizon: int,
        retrain: Union[bool, int, Callable[..., bool]],
        show_warnings: bool,
    ) -> bool:
        """
        Historical forecast can be optimized only if `retrain=False` and `forecast_horizon <= model.output_chunk_length`
        (no auto-regression required).
        """
        return _check_optimizable_historical_forecasts_global_models(
            model=self,
            forecast_horizon=forecast_horizon,
            retrain=retrain,
            show_warnings=show_warnings,
            allow_autoregression=False,
        )

    def _optimized_historical_forecasts(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]] = None,
        future_covariates: Optional[Sequence[TimeSeries]] = None,
        num_samples: int = 1,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
        stride: int = 1,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        random_state: Optional[int] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
        """
        For SKLearnModels we create the lagged prediction data once per series using a moving window.
        With this, we can avoid having to recreate the tabular input data and call `model.predict()` for each
        forecastable index and series.
        Additionally, there is a dedicated subroutines for `last_points_only=True` and `last_points_only=False`.

        TODO: support forecast_horizon > output_chunk_length (auto-regression)
        """
        series, past_covariates, future_covariates = _process_historical_forecast_input(
            model=self,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            allow_autoregression=False,
        )

        # TODO: move the loop here instead of duplicated code in each sub-routine?
        if last_points_only:
            hfc = _optimized_historical_forecasts_last_points_only(
                model=self,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
                start=start,
                start_format=start_format,
                forecast_horizon=forecast_horizon,
                stride=stride,
                overlap_end=overlap_end,
                show_warnings=show_warnings,
                verbose=verbose,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                predict_kwargs=predict_kwargs,
            )
        else:
            hfc = _optimized_historical_forecasts_all_points(
                model=self,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
                start=start,
                start_format=start_format,
                forecast_horizon=forecast_horizon,
                stride=stride,
                overlap_end=overlap_end,
                show_warnings=show_warnings,
                verbose=verbose,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                predict_kwargs=predict_kwargs,
            )
        return hfc

    @property
    def _supports_native_multioutput(self) -> bool:
        """
        Returns True if the model supports multi-output regression natively.
        """
        model = (
            self.model.estimator
            if isinstance(self.model, MultiOutputMixin)
            else self.model
        )
        return model.__sklearn_tags__().target_tags.multi_output

    @property
    def _model_type(self) -> ModelType:
        return ModelType.FORECASTING_REGRESSOR


class _QuantileModelContainer(OrderedDict):
    def __init__(self):
        super().__init__()


class SKLearnModelWithCategoricalFeatures(SKLearnModel, ABC):
    def __init__(
        self,
        model,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, list[int]] = None,
        lags_future_covariates: Union[tuple[int, int], list[int]] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        categorical_past_covariates: Optional[Union[str, list[str]]] = None,
        categorical_future_covariates: Optional[Union[str, list[str]]] = None,
        categorical_static_covariates: Optional[Union[str, list[str]]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Extension of `SKLearnModel` for regression models that support categorical features.

        Parameters
        ----------
        model
            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
        lags
            Lagged target `series` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`
            corresponds the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `series` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_past_covariates
            Lagged `past_covariates` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,
            where `0` corresponds to the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_future_covariates
            Lagged `future_covariates` values used to predict the next time step/s. The lags are always relative to the
            first step in the output chunk, even when `output_chunk_shift > 0`.
            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`
            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0` corresponds the first
            predicted time step of each sample. If `output_chunk_shift > 0`, the position of negative lags differ from
            those of `lags` and `lags_past_covariates`. In this case a future lag `-5` would point at the same
            step as a target lag of `-5 + output_chunk_shift`.
            If a list of integers, uses only the specified values as lags.
            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key
            'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using a
            one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression. This is
            useful when the covariates don't extend far enough into the future, or to prohibit the model from using
            future values of past and / or future covariates for prediction (depending on the model's covariate
            support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input (history of target and past covariates) and
            output. If the model supports `future_covariates`, the `lags_future_covariates` are relative to the first
            step in the shifted output chunk. Predictions will start `output_chunk_shift` steps after the end of the
            target `series`. If `output_chunk_shift` is set, the model cannot generate autoregressive predictions
            (`n > output_chunk_length`).
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

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
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
        random_state
            Controls the randomness for reproducible forecasting.
        """
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            model=model,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
        )

        if categorical_static_covariates is not None and not use_static_covariates:
            raise_log(
                ValueError(
                    "`categorical_static_covariates` is declared but `use_static_covariates` is set to False."
                ),
                logger,
            )
        if categorical_past_covariates is not None and lags_past_covariates is None:
            raise_log(
                ValueError(
                    "`categorical_past_covariates` is declared but `lags_past_covariates` is not set."
                ),
                logger,
            )
        if categorical_future_covariates is not None and lags_future_covariates is None:
            raise_log(
                ValueError(
                    "`categorical_future_covariates` is declared but `lags_future_covariates` is not set."
                ),
                logger,
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
        self._categorical_indices = []  # Indices are set on fit

    def _get_categorical_features(
        self,
        series: Union[Sequence[TimeSeries], TimeSeries],
        past_covariates: Optional[Union[Sequence[TimeSeries], TimeSeries]] = None,
        future_covariates: Optional[Union[Sequence[TimeSeries], TimeSeries]] = None,
    ) -> tuple[list[int], list[str]]:
        """
        Returns the indices and column names of the categorical features in the regression model.

        Steps:
        1. Get the list of features used in the model. We keep the creation order of the different lags/features
            in create_lagged_data.
        2. Get the indices of the categorical features in the list of features.
        """
        target_ts = get_single_series(series)
        categorical_covariates = [
            list(target_ts.components)
            if self._get_lags("target") is not None
            and self._model_type == ModelType.FORECASTING_CLASSIFIER
            else [],
            self.categorical_past_covariates
            if self.categorical_past_covariates
            else [],
            self.categorical_future_covariates
            if self.categorical_future_covariates
            else [],
            self.categorical_static_covariates
            if self.categorical_static_covariates
            else [],
        ]

        # if no categorical covariates are declared, return empty lists
        if sum(len(cat_cov) for cat_cov in categorical_covariates) == 0:
            return [], []

        # keep track of feature list index to refer to the columns indices
        feature_list = self._get_lagged_features(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        index = 0
        indices = []
        col_names = []
        series_type = [
            "series",
            "past_covariates",
            "future_covariates",
            "static_covariates",
        ]
        for cat_covs, features, s_type in zip(
            categorical_covariates, feature_list, series_type
        ):
            # extract all categorical feature indices
            extracted_categorical_features = []
            for prefix, component, lag in features:
                if component in cat_covs:
                    indices.append(index)
                    col_names.append(f"{prefix}_{component}_lag{lag}")
                    extracted_categorical_features.append(component)
                index += 1

            # check that all cat components were extracted
            missing_comps = [
                comp for comp in cat_covs if comp not in extracted_categorical_features
            ]
            if missing_comps:
                raise_log(
                    ValueError(
                        f"Some `categorical_{s_type}` components "
                        f"({missing_comps}) declared at model creation are "
                        f"not present in the `{s_type}` passed to the `fit()` call. "
                        f"Available feature(s) are: {set([comp for _, comp, _ in features])}"
                    ),
                    logger=logger,
                )
        return indices, col_names

    def _fit_model(
        self,
        series,
        past_covariates,
        future_covariates,
        max_samples_per_ts,
        sample_weight,
        verbose: Optional[bool] = None,
        **kwargs,
    ):
        """
        Custom fit function for `SKLearnModelWithCategoricalFeatures` models, adding logic to let the model
        handle categorical features directly.

        Sub-classes can override `_format_samples` to format the columns listed in self._categorical_features
        according to the model's requirements.
        """
        cat_col_indices, _ = self._get_categorical_features(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        self._categorical_indices = cat_col_indices

        # cat_param_name is None if model has no attribute to specify categorical features in `fit()`
        cat_param_name = self._categorical_fit_param

        # Add attributes in kwargs if applicable for model's `fit()` method
        if cat_param_name is not None and len(cat_col_indices) > 0:
            kwargs[cat_param_name] = cat_col_indices

        super()._fit_model(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            sample_weight=sample_weight,
            verbose=verbose,
            **kwargs,
        )

    def _validate_categorical_components(self, samples):
        """Check if categorical features are integer-encoded"""
        if np.any(samples[:, self._categorical_indices] % 1 != 0):
            raise_log(
                ValueError(
                    "Categorical features must be integer-encoded, decimal values found instead."
                ),
                logger=logger,
            )

    def _format_samples(
        self, samples: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> tuple[Any, Any]:
        """
        Validate and format the categorical columns listed in self._categorical_indices accordingly to the model's
        requirements.
        """
        if len(self._categorical_indices) != 0:
            self._validate_categorical_components(samples)
        return samples, labels

    @property
    @abstractmethod
    def _categorical_fit_param(self) -> Optional[str]:
        """
        Returns the name of the categorical features parameter from model's `fit` method .
        """


class RegressionModel(SKLearnModel):
    def __init__(
        self,
        lags: Optional[LAGS_TYPE] = None,
        lags_past_covariates: Optional[LAGS_TYPE] = None,
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        model=None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        random_state: Optional[int] = None,
    ):
        """Regression Model
        Can be used to fit any scikit-learn-like regressor class to predict the target time series from lagged values.

        Note: `RegressionModel` is deprecated and will be removed in a future version. Use
        :class:`~darts.models.forecasting.sklearn_model.SKLearnModel` instead.

        Parameters
        ----------
        lags
            Lagged target `series` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`
            corresponds the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `series` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_past_covariates
            Lagged `past_covariates` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,
            where `0` corresponds to the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_future_covariates
            Lagged `future_covariates` values used to predict the next time step/s. The lags are always relative to the
            first step in the output chunk, even when `output_chunk_shift > 0`.
            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`
            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0` corresponds the first
            predicted time step of each sample. If `output_chunk_shift > 0`, the position of negative lags differ from
            those of `lags` and `lags_past_covariates`. In this case a future lag `-5` would point at the same
            step as a target lag of `-5 + output_chunk_shift`.
            If a list of integers, uses only the specified values as lags.
            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key
            'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using a
            one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression. This is
            useful when the covariates don't extend far enough into the future, or to prohibit the model from using
            future values of past and / or future covariates for prediction (depending on the model's covariate
            support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input (history of target and past covariates) and
            output. If the model supports `future_covariates`, the `lags_future_covariates` are relative to the first
            step in the shifted output chunk. Predictions will start `output_chunk_shift` steps after the end of the
            target `series`. If `output_chunk_shift` is set, the model cannot generate autoregressive predictions
            (`n > output_chunk_length`).
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

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        model
            Scikit-learn-like model with ``fit()`` and ``predict()`` methods. Also possible to use model that doesn't
            support multi-output regression for multivariate timeseries, in which case one regressor
            will be used per component in the multivariate series.
            If None, defaults to: ``sklearn.linear_model.LinearRegression(n_jobs=-1)``.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model
            is trained to predict all the steps in 'output_chunk_length' (features lags are shifted back by
            `output_chunk_length - n` for each step `n`). Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        random_state
            Controls the randomness for reproducible forecasting.

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import RegressionModel
        >>> from sklearn.linear_model import Ridge
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # wrap around the sklearn Ridge model
        >>> model = RegressionModel(
        >>>     model=Ridge(),
        >>>     lags=12,
        >>>     lags_past_covariates=4,
        >>>     lags_future_covariates=(0,6),
        >>>     output_chunk_length=6
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[1005.73340676],
                [1005.71159051],
                [1005.7322616 ],
                [1005.76314504],
                [1005.82204348],
                [1005.89100967]])
        """
        raise_deprecation_warning(
            "`RegressionModel` is deprecated and will be removed in a future version. "
            "Use `SKLearnModel` instead.",
            logger,
        )

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            model=model,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
        )


class _ClassifierMixin:
    """
    Mixin for sklearn-like classification forecasting models
    """

    model: Any
    multi_models: bool
    class_labels: Union[list[np.ndarray], np.ndarray]
    _output_chunk_length: int

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
        stride: int = 1,
        verbose: Optional[bool] = None,
        **kwargs,
    ):
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            n_jobs_multioutput_wrapper=n_jobs_multioutput_wrapper,
            sample_weight=sample_weight,
            stride=stride,
            verbose=verbose,
            **kwargs,
        )

        classes = self.class_labels
        if self.multi_models and self._output_chunk_length > 1:
            # make sure that all individual models have seen the same classes during training
            num_components = len(classes) // self._output_chunk_length

            # classes are ordered by chunk then by component: [classes_comp0_chunk0, classes_comp1_chunk0, ..]
            for i in range(self._output_chunk_length):
                if any(
                    not np.array_equal(classes[i], estimator_classes)
                    for estimator_classes in classes[i::num_components]
                ):
                    raise_log(
                        ValueError(
                            "Estimators for the same target component received different class labels "
                            "during training. In most cases this occurs for shorter target series. "
                            f"For component `{i}`, the observed classes for each step in the output chunk are: "
                            f"`{[estimator_classes for estimator_classes in classes[i::num_components]]}`"
                        )
                    )
        return self

    @property
    def class_labels(self):
        """Returns the classes of the classifier model if the model was previously trained."""
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        # unify `class_labels` to a list of arrays of classes for simplicity
        # - single-labels and single-model tasks return a single array of classes
        # - multi-labels or multi-models tasks return a list of arrays of classes
        if not isinstance(classes, list):
            classes = [classes]
        return classes

    def _validate_lags(
        self,
        lags: Optional[LAGS_TYPE],
        lags_past_covariates: Optional[LAGS_TYPE],
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE],
    ):
        super()._validate_lags(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
        )

        if lags is not None and not isinstance(
            self, SKLearnModelWithCategoricalFeatures
        ):
            logger.warning(
                "This model will treat target `series` lagged values as "
                "numeric input features (and not categorical)."
            )

    @property
    def _model_type(self) -> ModelType:
        return ModelType.FORECASTING_CLASSIFIER


class SKLearnClassifierModel(_ClassifierMixin, SKLearnModel):
    def __init__(
        self,
        model=None,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, list[int]] = None,
        lags_future_covariates: Union[tuple[int, int], list[int]] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = LikelihoodType.ClassProbability.value,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        random_state: Optional[int] = None,
    ):
        """SKLearn Classifier Model

        Can be used to fit any scikit-learn-like classifier class to predict the target time series with categorical
        values from lagged values.

        .. note::
            If target series `lags` are provided, they will be treated as numeric input features (and not categorical).
            For categorical input feature support, consider models such as
            :class:`~darts.models.forecasting.lgbm.LightGBMClassifierModel` or
            :class:`~darts.models.forecasting.catboost_model.CatBoostClassifierModel`.

        Parameters
        ----------
        model
            Scikit-learn-like classifier model with ``fit()`` and ``predict()`` methods. Also possible to use model
            that doesn't support multi-label classification for multivariate timeseries, in which case one classifier
            will be used per component in the multivariate series.
            If `None`, defaults to: ``sklearn.linear_model.LogisticRegression(n_jobs=-1)``.
        lags
            Lagged target `series` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`
            corresponds the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `series` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
            This model treats the target `series` as numerical features when lags are provided.
        lags_past_covariates
            Lagged `past_covariates` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,
            where `0` corresponds to the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_future_covariates
            Lagged `future_covariates` values used to predict the next time step/s. The lags are always relative to the
            first step in the output chunk, even when `output_chunk_shift > 0`.
            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`
            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0` corresponds the first
            predicted time step of each sample. If `output_chunk_shift > 0`, the position of negative lags differ from
            those of `lags` and `lags_past_covariates`. In this case a future lag `-5` would point at the same
            step as a target lag of `-5 + output_chunk_shift`.
            If a list of integers, uses only the specified values as lags.
            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key
            'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using a
            one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression. This is
            useful when the covariates don't extend far enough into the future, or to prohibit the model from using
            future values of past and / or future covariates for prediction (depending on the model's covariate
            support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input (history of target and past covariates) and
            output. If the model supports `future_covariates`, the `lags_future_covariates` are relative to the first
            step in the shifted output chunk. Predictions will start `output_chunk_shift` steps after the end of the
            target `series`. If `output_chunk_shift` is set, the model cannot generate autoregressive predictions
            (`n > output_chunk_length`).
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

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        likelihood
            'classprobability' or ``None``. If set to 'classprobability', setting `predict_likelihood_parameters`
            in `predict()` will forecast class probabilities.
            Default: 'classprobability'
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model is
            trained to predict at step 'output_chunk_length' in the future. Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        random_state
            Controls the randomness for reproducible forecasting.

        Examples
        --------
        >>> import numpy as np
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import SKLearnClassifierModel
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> series = WeatherDataset().load().resample("1D", method="mean")
        >>> # predicting if it will rain or not
        >>> target =  series['rain (mm)'][:105].map(lambda x: np.where(x > 0, 1, 0))
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 105)
        >>> past_cov = series['T (degC)'][:105]
        >>> # optionally, use future pressure (pretending this component is a forecast)
        >>> future_cov = series['p (mbar)'][:111]
        >>> # predict 6 "will rain" values using the 12 past values of pressure and temperature,
        >>> # as well as the 6 pressure values corresponding to the forecasted period
        >>> model = SKLearnClassifierModel(
        >>>     model=KNeighborsClassifier(),
        >>>     lags_past_covariates=12,
        >>>     lags_future_covariates=[0,1,2,3,4,5],
        >>>     output_chunk_length=6
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[0.],
               [0.],
               [1.],
               [1.],
               [1.],
               [0.]])
        """

        model = model if model is not None else LogisticRegression(n_jobs=-1)
        if not is_classifier(model):
            raise_log(
                ValueError(
                    "`SKLearnClassifierModel` must be initialized with a classifier `model`."
                ),
                logger,
            )
        if not hasattr(model, "predict_proba"):
            logger.warning(
                "`model` has no method with name `predict_proba()`. Probabilistic forecasting support deactivated."
            )
            likelihood = None

        self._likelihood = _get_likelihood(
            likelihood=likelihood,
            n_outputs=output_chunk_length if multi_models else 1,
            available_likelihoods=[LikelihoodType.ClassProbability],
        )

        super().__init__(
            model=model,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            random_state=random_state,
        )
