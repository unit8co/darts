"""
LightGBM Model
--------------

This is a LightGBM implementation of Gradient Boosted Trees algorithm.

To enable LightGBM support in Darts, follow the detailed install instructions for LightGBM in the README:
https://github.com/unit8co/darts/blob/master/README.md
"""

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy
import numpy as np
import xarray as xr

from darts.logging import get_logger
from darts.models.forecasting.regression_model import RegressionModel
from darts.timeseries import TimeSeries
from darts.utils.utils import raise_if_not

logger = get_logger(__name__)


class LightGBMModel(RegressionModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        likelihood: str = None,
        quantiles: List[float] = None,
        **kwargs,
    ):
        """Light Gradient Boosted Model

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
            The objective used by the model. Currently, only `quantile` and 'poisson' are available. Allows sampling
            from the model.
        quantiles
            If the `likelihood` is set to `quantile`, use these quantiles to samples from.
        **kwargs
            Additional keyword arguments passed to `lightgbm.LGBRegressor`.
        """
        self.kwargs = kwargs
        self._median_idx = None
        self._model_container = _LightGBMModelContainer()
        self.quantiles = quantiles
        self.likelihood = likelihood
        self._rng = None

        # parse likelihood
        available_likelihoods = ["quantile", "poisson"]  # to be extended
        if likelihood is not None:
            raise_if_not(
                likelihood in available_likelihoods,
                f"If likelihood is specified it must be one of {available_likelihoods}",
            )
            self.kwargs["objective"] = likelihood
            self._rng = np.random.default_rng(seed=420)
            if likelihood == "quantile":
                if quantiles is None:
                    self.quantiles = [
                        0.01,
                        0.05,
                        0.1,
                        0.15,
                        0.2,
                        0.25,
                        0.3,
                        0.4,
                        0.45,
                        0.5,
                        0.55,
                        0.6,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                        0.95,
                        0.99,
                    ]
                else:
                    self.quantiles = sorted(self.quantiles)
                    self._check_quantiles(self.quantiles)
                self._median_idx = self.quantiles.index(0.5)

        super().__init__(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            model=lgb.LGBMRegressor(**kwargs),
        )

    def __str__(self):
        return f"LGBModel(lags={self.lags})"

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        max_samples_per_ts: Optional[int] = None,
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
                self.kwargs["alpha"] = quantile
                self.model = lgb.LGBMRegressor(**self.kwargs)

                super().fit(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    max_samples_per_ts=max_samples_per_ts,
                    **kwargs,
                )

                self._model_container[quantile] = self.model

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
                model_outputs.append(prediction._xa.to_numpy())
            model_outputs = np.concatenate(model_outputs, axis=-1)
            samples = self._sample_quantiles(model_outputs, num_samples)
            # build timeseries from samples
            new_xa = xr.DataArray(
                samples, dims=prediction._xa.dims, coords=prediction._xa.coords
            )
            return TimeSeries(new_xa)

        if self.likelihood == "poisson":
            prediction = super().predict(
                n, series, past_covariates, future_covariates, **kwargs
            )
            samples = self._sample_poisson(
                np.array(prediction._xa.to_numpy()), num_samples
            )
            # build timeseries from samples
            new_xa = xr.DataArray(
                samples, dims=prediction._xa.dims, coords=prediction._xa.coords
            )
            return TimeSeries(new_xa)

        return super().predict(
            n, series, past_covariates, future_covariates, num_samples, **kwargs
        )

    def _sample_quantiles(
        self, model_output: numpy.ndarray, num_samples: int
    ) -> numpy.ndarray:
        """
        This method is ported to numpy from the probabilistic torch models module
        model_output is of shape (n_timesteps, n_components, n_quantiles)
        """
        raise_if_not(all([isinstance(num_samples, int), num_samples > 0]))
        quantiles = np.tile(np.array(self.quantiles), (num_samples, 1))
        probas = np.tile(
            self._rng.uniform(size=(num_samples,)), (len(self.quantiles), 1)
        )

        quantile_idxs = np.sum(probas.T > quantiles, axis=1)

        # To make the sampling symmetric around the median, we assign the two "probability buckets" before and after
        # the median to the median value. If we don't do that, the highest quantile would be wrongly sampled
        # too often as it would capture the "probability buckets" preceding and following it.
        #
        # Example; the arrows shows how the buckets map to values: [--> 0.1 --> 0.25 --> 0.5 <-- 0.75 <-- 0.9 <--]
        quantile_idxs = np.where(
            quantile_idxs <= self._median_idx, quantile_idxs, quantile_idxs - 1
        )

        return model_output[:, :, quantile_idxs]

    def _sample_poisson(
        self, model_output: numpy.ndarray, num_samples: int
    ) -> numpy.ndarray:
        raise_if_not(all([isinstance(num_samples, int), num_samples > 0]))
        return self._rng.poisson(
            lam=model_output, size=(*model_output.shape[:2], num_samples)
        ).astype(float)

    @staticmethod
    def _check_quantiles(quantiles):
        raise_if_not(
            all([0 < q < 1 for q in quantiles]),
            "All provided quantiles must be between 0 and 1.",
        )

        # we require the median to be present and the quantiles to be symmetric around it,
        # for correctness of sampling.
        median_q = 0.5
        raise_if_not(
            median_q in quantiles, "median quantile `q=0.5` must be in `quantiles`"
        )
        is_centered = [
            -1e-6 < (median_q - left_q) + (median_q - right_q) < 1e-6
            for left_q, right_q in zip(quantiles, quantiles[::-1])
        ]
        raise_if_not(
            all(is_centered),
            "quantiles lower than `q=0.5` need to share same difference to `0.5` as quantiles "
            "higher than `q=0.5`",
        )


class _LightGBMModelContainer(OrderedDict):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"_LightGBMModelContainer(quantiles={list(self.keys())})"
