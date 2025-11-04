"""
XGBoost Models
--------------

This module offers wrappers around XGBoost's Gradient Boosted Trees algorithms.

* :class:`~darts.models.forecasting.xgboost.XGBModel` - Wrapper around XGBoost's `XGBRegressor`
* :class:`~darts.models.forecasting.xgboost.XGBClassifierModel` - Wrapper around XGBoost's
  `XGBClassifier`

The wrappers come with all capabilities of Darts' `SKLearn*Model`.

For detailed examples and tutorials, see:

* `SKLearn-Like Regression Model Examples
  <https://unit8co.github.io/darts/examples/20-SKLearnModel-examples.html>`_
* `SKLearn-Like Classification Model Examples
  <https://unit8co.github.io/darts/examples/24-SKLearnClassifierModel-examples.html>`_
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import xgboost as xgb

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.sklearn_model import (
    FUTURE_LAGS_TYPE,
    LAGS_TYPE,
    SKLearnModel,
    _ClassifierMixin,
    _QuantileModelContainer,
)
from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.sklearn import QuantileRegression, _get_likelihood

logger = get_logger(__name__)


def xgb_quantile_loss(labels: np.ndarray, preds: np.ndarray, quantile: float):
    """Custom loss function for XGBoost to compute quantile loss gradient.

    Inspired from: https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7

    This computes the gradient of the pinball loss between predictions and target labels.
    """
    raise_if_not(0 <= quantile <= 1, "Quantile must be between 0 and 1.", logger)

    errors = preds - labels
    left_mask = errors < 0
    right_mask = errors > 0

    grad = -quantile * left_mask + (1 - quantile) * right_mask
    hess = np.ones_like(preds)

    return grad, hess


class XGBModel(SKLearnModel):
    def __init__(
        self,
        lags: Optional[LAGS_TYPE] = None,
        lags_past_covariates: Optional[LAGS_TYPE] = None,
        lags_future_covariates: Optional[FUTURE_LAGS_TYPE] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        dir_rec: Optional[bool] = False,
        **kwargs,
    ):
        """XGBoost Model

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
        likelihood
            Can be set to `poisson` or `quantile`. If set, the model will be probabilistic, allowing sampling at
            prediction time. This will overwrite any `objective` parameter.
        quantiles
            Fit the model to these quantiles if the `likelihood` is set to `quantile`.
        random_state
            Controls the randomness for reproducible forecasting.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model
            is trained to predict all the steps in 'output_chunk_length' (features lags are shifted back by
            `output_chunk_length - n` for each step `n`). Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        dir_rec
            Whether to use direct-recursive strategy for multi-step forecasting. When True, each forecast
            horizon uses predictions from previous horizons as additional input features. This creates a
            chained prediction where step t+2 uses the prediction for step t+1 as a feature. Default: False.
        **kwargs
            Additional keyword arguments passed to `xgb.XGBRegressor`.

        Examples
        --------
        Deterministic forecasting, using past/future covariates (optional)

        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import XGBModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # predict 6 pressure values using the 12 past values of pressure and rainfall, as well as the 6 temperature
        >>> # values corresponding to the forecasted period
        >>> model = XGBModel(
        >>>     lags=12,
        >>>     lags_past_covariates=12,
        >>>     lags_future_covariates=[0,1,2,3,4,5],
        >>>     output_chunk_length=6,
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[1005.9185 ],
               [1005.8315 ],
               [1005.7878 ],
               [1005.72626],
               [1005.7475 ],
               [1005.76074]])
        """
        kwargs["random_state"] = random_state  # seed for tree learner
        self.kwargs = kwargs

        self._set_likelihood(
            likelihood=likelihood,
            output_chunk_length=output_chunk_length,
            multi_models=multi_models,
            quantiles=quantiles,
        )

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            multi_models=multi_models,
            model=self._create_model(**self.kwargs),
            use_static_covariates=use_static_covariates,
            random_state=random_state,
            dir_rec=dir_rec,
        )

    @staticmethod
    def _create_model(**kwargs):
        return xgb.XGBRegressor(**kwargs)

    def _set_likelihood(
        self,
        likelihood: Optional[str],
        output_chunk_length: int,
        multi_models: bool,
        quantiles: Optional[list[float]] = None,
    ):
        self._likelihood = _get_likelihood(
            likelihood=likelihood,
            n_outputs=output_chunk_length if multi_models else 1,
            quantiles=quantiles,
            available_likelihoods=[LikelihoodType.Quantile, LikelihoodType.Poisson],
        )

        if likelihood is None:
            return

        if likelihood == LikelihoodType.Poisson.value:
            self.kwargs["objective"] = f"count:{likelihood}"
        else:  # quantile
            self.kwargs["objective"] = "reg:quantileerror"
            self._model_container = _QuantileModelContainer()

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        n_jobs_multioutput_wrapper: Optional[int] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
        val_sample_weight: Optional[
            Union[TimeSeries, Sequence[TimeSeries], str]
        ] = None,
        **kwargs,
    ):
        """
        Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        val_series
            TimeSeries or Sequence[TimeSeries] object containing the target values for evaluation dataset
        val_past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates for evaluation dataset
        val_future_covariates :
            Optionally, a series or sequence of series specifying future-known covariates for evaluation dataset
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
        val_sample_weight
            Same as for `sample_weight` but for the evaluation dataset.
        **kwargs
            Additional kwargs passed to `xgb.XGBRegressor.fit()`
        """
        # TODO: XGBRegressor supports multi quantile reqression which we could leverage in the future
        #  see https://xgboost.readthedocs.io/en/latest/python/examples/quantile_regression.html
        likelihood = self.likelihood
        if isinstance(likelihood, QuantileRegression):
            # empty model container in case of multiple calls to fit, e.g. when backtesting
            self._model_container.clear()
            for quantile in likelihood.quantiles:
                self.kwargs["quantile_alpha"] = quantile
                self.model = self._create_model(**self.kwargs)
                super().fit(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    val_series=val_series,
                    val_past_covariates=val_past_covariates,
                    val_future_covariates=val_future_covariates,
                    max_samples_per_ts=max_samples_per_ts,
                    n_jobs_multioutput_wrapper=n_jobs_multioutput_wrapper,
                    sample_weight=sample_weight,
                    val_sample_weight=val_sample_weight,
                    **kwargs,
                )
                # store the trained model in the container as it might have been wrapped by MultiOutputRegressor
                self._model_container[quantile] = self.model
            return self

        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            val_series=val_series,
            val_past_covariates=val_past_covariates,
            val_future_covariates=val_future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            n_jobs_multioutput_wrapper=n_jobs_multioutput_wrapper,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            **kwargs,
        )
        return self

    @property
    def _supports_val_series(self) -> bool:
        return True

    @property
    def val_set_params(self) -> tuple[Optional[str], Optional[str]]:
        return "eval_set", "sample_weight_eval_set"

    @property
    def _supports_native_multioutput(self) -> bool:
        # since xgboost==2.1.0, likelihoods do not support native multi output regression
        return super()._supports_native_multioutput and self.likelihood is None


class XGBClassifierModel(_ClassifierMixin, XGBModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, list[int]] = None,
        lags_future_covariates: Union[tuple[int, int], list[int]] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = LikelihoodType.ClassProbability.value,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        dir_rec: Optional[bool] = False,
        **kwargs,
    ):
        """XGBoost Model for classification forecasting

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
            This model treats the target `series` as numerical features when lags are provided,
            as support for categorical features is not available for XGBModel.
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
        random_state
            Controls the randomness for reproducible forecasting.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model
            is trained to predict all the steps in 'output_chunk_length' (features lags are shifted back by
            `output_chunk_length - n` for each step `n`). Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        dir_rec
            Whether to use direct-recursive strategy for multi-step forecasting. When True, each forecast
            horizon uses predictions from previous horizons as additional input features. This creates a
            chained prediction where step t+2 uses the prediction for step t+1 as a feature. Default: False.
        **kwargs
            Additional keyword arguments passed to `xgb.XGBClassifier`.

        Examples
        --------
        >>> import numpy as np
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import XGBClassifierModel
        >>> series = WeatherDataset().load().resample("1D", method="mean")
        >>> # predicting if it will rain or not
        >>> target =  series['rain (mm)'][:105].map(lambda x: np.where(x > 0, 1, 0))
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 105)
        >>> past_cov = series['T (degC)'][:105]
        >>> # optionally, use future pressure (pretending this component is a forecast)
        >>> future_cov = series['p (mbar)'][:111]
        >>> # predict 6 "will rain" values using the 12 past values of pressure and temperature,
        >>> # as well as the 6 pressure values corresponding to the forecasted period
        >>> model = XGBClassifierModel(
        >>>     lags_past_covariates=12,
        >>>     lags_future_covariates=[0,1,2,3,4,5],
        >>>     output_chunk_length=6,
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[0.],
               [0.],
               [0.],
               [1.],
               [1.],
               [1.]])
        """
        # likelihood always set to ClassProbability as it's the only supported classifiaction likelihood
        # this allow users to predict class probabilities,
        # by setting `predict_likelihood_parameters`to `True` in `predict()`
        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            likelihood=likelihood,
            random_state=random_state,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            dir_rec=dir_rec,
            **kwargs,
        )

    @staticmethod
    def _create_model(**kwargs):
        return xgb.XGBClassifier(**kwargs)

    def _set_likelihood(
        self,
        likelihood: Optional[str],
        output_chunk_length: int,
        multi_models: bool,
        quantiles: Optional[list[float]] = None,
    ):
        self._likelihood = _get_likelihood(
            likelihood=likelihood,
            n_outputs=output_chunk_length if multi_models else 1,
            available_likelihoods=[LikelihoodType.ClassProbability],
        )

    @property
    def _supports_native_multioutput(self) -> bool:
        return False
