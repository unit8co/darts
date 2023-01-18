"""
CatBoost model
--------------

CatBoost based regression model.

This implementation comes with the ability to produce probabilistic forecasts.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from catboost import CatBoostRegressor

from darts.logging import get_logger
from darts.models.forecasting.regression_model import RegressionModel, _LikelihoodMixin
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class CatBoostModel(RegressionModel, _LikelihoodMixin):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        add_encoders: Optional[dict] = None,
        likelihood: str = None,
        quantiles: List = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        **kwargs,
    ):
        """CatBoost Model

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
        likelihood
            Can be set to 'quantile', 'poisson' or 'gaussian'. If set, the model will be probabilistic,
            allowing sampling at prediction time. When set to 'gaussian', the model will use CatBoost's
            'RMSEWithUncertainty' loss function. When using this loss function, CatBoost returns a mean
            and variance couple, which capture data (aleatoric) uncertainty.
            This will overwrite any `objective` parameter.
        quantiles
            Fit the model to these quantiles if the `likelihood` is set to `quantile`.
        random_state
            Control the randomness in the fitting procedure and for sampling.
            Default: ``None``.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model is
            trained to predict at step 'output_chunk_length' in the future. Default: True.
        **kwargs
            Additional keyword arguments passed to `catboost.CatBoostRegressor`.
        """
        kwargs["random_state"] = random_state  # seed for tree learner
        self.kwargs = kwargs
        self._median_idx = None
        self._model_container = None
        self._rng = None
        self.likelihood = likelihood
        self.quantiles = None

        self.output_chunk_length = output_chunk_length

        likelihood_map = {
            "quantile": None,
            "poisson": "Poisson",
            "gaussian": "RMSEWithUncertainty",
            "RMSEWithUncertainty": "RMSEWithUncertainty",
        }

        available_likelihoods = list(likelihood_map.keys())

        if likelihood is not None:
            self._check_likelihood(likelihood, available_likelihoods)
            self._rng = np.random.default_rng(seed=random_state)  # seed for sampling

            if likelihood == "quantile":
                self.quantiles, self._median_idx = self._prepare_quantiles(quantiles)
                self._model_container = self._get_model_container()

            else:
                self.kwargs["loss_function"] = likelihood_map[likelihood]

        # suppress writing catboost info files when user does not specifically ask to
        if "allow_writing_files" not in kwargs:
            kwargs["allow_writing_files"] = False

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            add_encoders=add_encoders,
            multi_models=multi_models,
            model=CatBoostRegressor(**kwargs),
        )

    def __str__(self):
        return f"CatBoostModel(lags={self.lags})"

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
        verbose: Optional[Union[int, bool]] = 0,
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
        val_future_covariates : Union[TimeSeries, Sequence[TimeSeries]]
            Optionally, a series or sequence of series specifying future-known covariates for evaluation dataset
        max_samples_per_ts
            This is an integer upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        verbose
            An integer or a boolean that can be set to 1 to display catboost's default verbose output
        **kwargs
            Additional kwargs passed to `catboost.CatboostRegressor.fit()`
        """

        if val_series is not None:
            kwargs["eval_set"] = self._create_lagged_data(
                target_series=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
                max_samples_per_ts=max_samples_per_ts,
            )

        if self.likelihood == "quantile":
            # empty model container in case of multiple calls to fit, e.g. when backtesting
            self._model_container.clear()
            for quantile in self.quantiles:
                this_quantile = str(quantile)
                # translating to catboost argument
                self.kwargs["loss_function"] = f"Quantile:alpha={this_quantile}"
                self.model = CatBoostRegressor(**self.kwargs)

                super().fit(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    max_samples_per_ts=max_samples_per_ts,
                    verbose=verbose,
                    **kwargs,
                )

                self._model_container[quantile] = self.model

            return self

        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            verbose=verbose,
            **kwargs,
        )

        return self

    def _predict_and_sample(
        self, x: np.ndarray, num_samples: int, **kwargs
    ) -> np.ndarray:
        """Override of RegressionModel's predict method,
        to allow for the probabilistic case
        """
        if self.likelihood == "quantile":
            return self._predict_quantiles(x, num_samples, **kwargs)
        elif self.likelihood == "poisson":
            return self._predict_poisson(x, num_samples, **kwargs)
        elif self.likelihood in ["gaussian", "RMSEWithUncertainty"]:
            return self._predict_normal(x, num_samples, **kwargs)
        else:
            return super()._predict_and_sample(x, num_samples, **kwargs)

    def _is_probabilistic(self) -> bool:
        return self.likelihood is not None

    @property
    def min_train_series_length(self) -> int:
        # Catboost requires a minimum of 2 train samples, therefore the min_train_series_length should be one more than
        # for other regression models
        return max(
            3,
            -self.lags["target"][0] + self.output_chunk_length + 1
            if "target" in self.lags
            else self.output_chunk_length,
        )
