"""
VARIMA
-----

Models for VARIMA (Vector Autoregressive moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Vector_autoregression
"""
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VARMAX as staVARMA

from darts.logging import get_logger, raise_if
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class VARIMA(TransferableFutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 0,
        trend: Optional[str] = None,
        add_encoders: Optional[dict] = None,
    ):
        """VARIMA

        Parameters
        ----------
        p : int
            Order (number of time lags) of the autoregressive model (AR)
        d : int
            The order of differentiation; i.e., the number of times the data
            have had past values subtracted. (I) Note that Darts only supports d <= 1 because for
            d > 1 the optimizer often does not result in stable predictions. If results are not stable
            for d = 1 try to set d = 0 and enable the trend parameter
            to account for possible non-stationarity.
        q : int
            The size of the moving average window (MA).
        trend: str
            Parameter controlling the deterministic trend. 'n' indicates no trend,
            'c' a constant term, 't' linear trend in time, and 'ct' includes both.
            Default is 'c' for models without integration, and no trend for models with integration.
        add_encoders
            A large number of future covariates can be automatically generated with `add_encoders`.
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
                    'position': {'future': ['relative']},
                    'custom': {'future': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        """
        super().__init__(add_encoders=add_encoders)
        self.p = p
        self.d = d
        self.q = q
        self.trend = trend
        self.model = None

        assert d <= 1, "d > 1 not supported."

    def __str__(self):
        if self.d == 0:
            return f"VARMA({self.p},{self.q})"
        return f"VARIMA({self.p},{self.d},{self.q})"

    def _differentiate_series(self, series: TimeSeries) -> TimeSeries:
        """Differentiate the series self.d times"""
        for _ in range(self.d):
            series = TimeSeries.from_dataframe(
                df=series.pd_dataframe(copy=False).diff().dropna(),
                static_covariates=series.static_covariates,
                hierarchy=series.hierarchy,
            )
        return series

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        # for VARIMA we need to process target `series` before calling
        # TransferableFutureCovariatesLocalForecastingModel's fit() method
        self._last_values = (
            series.last_values()
        )  # needed for back-transformation when d=1

        series = self._differentiate_series(series)

        super().fit(series, future_covariates)

        return self

    def _fit(
        self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None
    ) -> None:
        super()._fit(series, future_covariates)

        # storing to restore the statsmodels model results object
        self.training_historic_future_covariates = future_covariates

        m = staVARMA(
            endog=series.values(copy=False),
            exog=future_covariates.values(copy=False) if future_covariates else None,
            order=(self.p, self.q),
            trend=self.trend,
        )

        self.model = m.fit(disp=0)

    def _predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        historic_future_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> TimeSeries:

        if num_samples > 1 and self.trend:
            logger.warning(
                "Trends are not well supported yet for getting probabilistic forecasts with ARIMA."
                "If you run into issues, try calling fit() with num_samples=1 or removing the trend from"
                "your model."
            )

        self._last_num_samples = num_samples

        super()._predict(
            n, series, historic_future_covariates, future_covariates, num_samples
        )

        if series is not None:
            self._training_last_values = self._last_values
            # store new _last_values of the new target series
            self._last_values = (
                series.last_values()
            )  # needed for back-transformation when d=1

            series = self._differentiate_series(series)

            # if the series is differentiated, the new len will be = len - 1, we have to adjust the future covariates
            if historic_future_covariates and self.d > 0:
                historic_future_covariates = historic_future_covariates.slice_intersect(
                    series
                )

            # updating statsmodels results object state

            self.model = self.model.apply(
                series.values(copy=False),
                exog=historic_future_covariates.values(copy=False)
                if historic_future_covariates
                else None,
            )

        # forecast before restoring the training state
        if num_samples == 1:
            forecast = self.model.forecast(
                steps=n,
                exog=future_covariates.values(copy=False)
                if future_covariates
                else None,
            )
        else:
            forecast = self.model.simulate(
                nsimulations=n,
                repetitions=num_samples,
                initial_state=self.model.states.predicted[-1, :],
                exog=future_covariates.values(copy=False)
                if future_covariates
                else None,
            )

        forecast = self._invert_transformation(forecast)

        # restoring statsmodels results object state and last values
        if series is not None:
            self.model = self.model.apply(
                self._orig_training_series.values(copy=False),
                exog=self.training_historic_future_covariates.values(copy=False)
                if self.training_historic_future_covariates
                else None,
            )

            self._last_values = self._training_last_values

        return self._build_forecast_series(np.array(forecast))

    def _invert_transformation(self, series_df: pd.DataFrame):
        if self.d == 0:
            return series_df
        if self._last_num_samples > 1:
            series_df = np.tile(
                self._last_values, (self._last_num_samples, 1)
            ).T + series_df.cumsum(axis=0)
        else:
            series_df = self._last_values + series_df.cumsum(axis=0)
        return series_df

    @property
    def min_train_series_length(self) -> int:
        return 30

    def _is_probabilistic(self) -> bool:
        return True

    def _supports_range_index(self) -> bool:
        raise_if(
            self.trend and self.trend != "c",
            "'trend' is not None. Range indexing is not supported in that case.",
            logger,
        )
        return True
