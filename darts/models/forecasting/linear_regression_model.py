"""
Linear Regression model
-----------------------

A forecasting model using a linear regression of some of the target series' lags, as well as optionally some
covariate series' lags in order to obtain a forecast.
"""
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression, PoissonRegressor, QuantileRegressor

from darts.logging import get_logger
from darts.models.forecasting.regression_model import RegressionModel, _LikelihoodMixin
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class LinearRegressionModel(RegressionModel, _LikelihoodMixin):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        likelihood: str = None,
        quantiles: List[float] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Linear regression model.

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
        likelihood
            Can be set to `quantile` or 'poisson'. If set, the model will be probabilistic, allowing sampling at
            prediction time. If set to `quantile`, the `sklearn.linear_model.QuantileRegressor` is used. Similarly, if
            set to `poisson`, the `sklearn.linear_model.PoissonRegressor` is used.
        quantiles
            Fit the model to these quantiles if the `likelihood` is set to `quantile`.
        random_state
            Control the randomness of the sampling. Used as seed for
            `link <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`_ . Ignored when
             no`likelihood` is set.
            Default: ``None``.
        **kwargs
            Additional keyword arguments passed to `sklearn.linear_model.LinearRegression` (by default), to
            `sklearn.linear_model.PoissonRegressor` (if `likelihood="poisson"`), or to
             `sklearn.linear_model.QuantileRegressor` (if `likelihood="quantile"`).
        """
        self.kwargs = kwargs
        self._median_idx = None
        self._model_container = None
        self.quantiles = None
        self.likelihood = likelihood
        self._rng = None

        # parse likelihood
        available_likelihoods = ["quantile", "poisson"]  # to be extended
        if likelihood is not None:
            self._check_likelihood(likelihood, available_likelihoods)
            self._rng = np.random.default_rng(seed=random_state)

            if likelihood == "poisson":
                model = PoissonRegressor(**kwargs)
            if likelihood == "quantile":
                model = QuantileRegressor(**kwargs)
                self.quantiles, self._median_idx = self._prepare_quantiles(quantiles)
                self._model_container = self._get_model_container()
        else:
            model = LinearRegression(**kwargs)

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            model=model,
        )

    def __str__(self):
        return f"LinearRegression(lags={self.lags})"

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

        if self.likelihood == "quantile":
            # empty model container in case of multiple calls to fit, e.g. when backtesting
            self._model_container.clear()

            # set solver for linear program
            if "solver" not in self.kwargs:
                # set default fast solver
                self.kwargs["solver"] = "highs"

            # test solver availability with dummy problem
            c = [1]
            try:
                linprog(c=c, method=self.kwargs["solver"])
            except ValueError as ve:
                logger.warning(
                    f"{ve}. Upgrading scipy enables significantly faster solvers"
                )
                # set solver to slow legacy
                self.kwargs["solver"] = "interior-point"

            for quantile in self.quantiles:
                self.kwargs["quantile"] = quantile
                self.model = QuantileRegressor(**self.kwargs)
                super().fit(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    max_samples_per_ts=max_samples_per_ts,
                    **kwargs,
                )

                self._model_container[quantile] = self.model

            return self

        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            **kwargs,
        )

        return self

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
            Specifies the numer of samples to obtain from the model. Should be set to 1 if no `likelihood` is specified.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `predict` method of the model. Only works with
            univariate target series.
        """

        if self.likelihood == "quantile":
            model_outputs = []
            for quantile, fitted in self._model_container.items():
                self.model = fitted
                prediction = super().predict(
                    n, series, past_covariates, future_covariates, **kwargs
                )
                model_outputs.append(prediction.all_values(copy=False))
            model_outputs = np.concatenate(model_outputs, axis=-1)
            samples = self._sample_quantiles(model_outputs, num_samples)

            # build timeseries from samples
            return self._ts_like(prediction, samples)

        elif self.likelihood == "poisson":
            prediction = super().predict(
                n, series, past_covariates, future_covariates, **kwargs
            )
            samples = self._sample_poisson(
                np.array(prediction.all_values(copy=False)), num_samples
            )

            # build timeseries from samples
            return self._ts_like(prediction, samples)

        else:
            return super().predict(
                n, series, past_covariates, future_covariates, num_samples, **kwargs
            )
