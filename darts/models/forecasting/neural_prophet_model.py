"""
Neural Prophet
------------
"""

from typing import Optional, Sequence, Tuple, Union

import neuralprophet
import pandas as pd
from neuralprophet.utils import fcst_df_to_latest_forecast

from darts.logging import raise_if_not
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries, concatenate


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
    ) -> "NeuralProphet":
        super().fit(series, past_covariates, future_covariates)
        self.training_series = series

        fit_df = self._convert_ts_to_df(series)
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

        if series is None:
            series = self.training_series
        df = self._convert_ts_to_df(series)

        future_df = self.model.make_future_dataframe(df=df, periods=n)
        forecast_df = self.model.predict(future_df)

        return self._convert_df_to_ts(forecast_df, series.end_time(), series.components)

    def _convert_ts_to_df(self, series: TimeSeries):
        dfs = []

        for component in series.components:
            new_df = (
                series[component].pd_dataframe(copy=False).reset_index(names=["ds"])
            )
            component_df = (
                new_df[["ds", component]]
                .copy(deep=True)
                .rename(columns={component: "y"})
            )
            component_df["ID"] = component
            dfs.append(component_df)

        return pd.concat(dfs)

    def _convert_df_to_ts(self, forecast, last_train_date, components):
        groups = []
        for component in components:
            simple_df = fcst_df_to_latest_forecast(
                forecast[forecast["ID"] == component].copy(deep=True),
                quantiles=[0.5],
                n_last=1,
            )
            simple_df = simple_df[["ds", "origin-0"]].rename(
                columns={"origin-0": component}
            )
            groups.append(simple_df[simple_df["ds"] > last_train_date])

        return concatenate(
            [TimeSeries.from_dataframe(group, time_col="ds") for group in groups],
            axis=1,
        )

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        raise NotImplementedError()

    def __str__(self):
        return "Neural Prophet"
