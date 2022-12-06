"""
Neural Prophet
------------
"""

from typing import Optional, Sequence, Tuple, Union

import neuralprophet
import pandas as pd

from darts.logging import raise_if_not
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries


class NeuralProphet(GlobalForecastingModel):
    def __init__(self, n_lags: int = 0, n_forecasts: int = 1, *kwargs):
        super().__init__()

        raise_if_not(n_lags >= 0, "Argument n_lags should be a non-negative integer")

        self.model = neuralprophet.NeuralProphet(
            n_lags=n_lags, n_forecasts=n_forecasts, *kwargs
        )

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> "GlobalForecastingModel":
        super().fit(series, past_covariates, future_covariates)

        # TODO change to accept multivariate
        fit_df = pd.DataFrame(
            data={"ds": series.time_index, "y": series.univariate_values()}
        )
        self.model.fit(fit_df, freq=series.freq_str)
        return self

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().predict(
            n, series, past_covariates, future_covariates, num_samples, verbose
        )
        print(series.columns)
        predict_df = pd.DataFrame(
            data={"ds": series.time_index, "y": series.univariate_values()}
        )
        predict_df = self.model.make_future_dataframe(df=predict_df, periods=n)
        future_df = self.model.predict(predict_df)

        future_df = future_df[["ds", "yhat1"]].rename(
            columns={"yhat1": series.columns[0]}
        )
        future_ts = TimeSeries.from_dataframe(df=future_df[-n:], time_col="ds")

        return future_ts

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        raise NotImplementedError()

    def __str__(self):
        return "Neural Prophet"
