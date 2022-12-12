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

        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
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

        self.training_series = self._as_sequence(series)
        fit_df = self._convert_ts_to_df(self.training_series)

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
            series_list = self.training_series
        else:
            series_list = self._as_sequence(series)

        raise_if_not(
            self.n_lags == 0 or n <= self.n_forecasts,
            "Auto-regression has been configured. `n` must be smaller than or equal to"
            "`n_forecasts` parameter in the constructor.",
        )
        # TODO consider time series indexed by ints
        # TODO check if series was used during training

        predictions = []
        for series in series_list:
            df = self._convert_ts_to_df([series])
            future_df = self.model.make_future_dataframe(df=df, periods=n)
            forecast_df = self.model.predict(future_df)

            predictions.append(
                self._convert_df_to_ts(
                    forecast_df, series.end_time(), series.components
                )
            )
        return self._from_sequence(predictions)

    def _convert_ts_to_df(self, series_list: Sequence[TimeSeries]):
        dfs = []

        for series in series_list:
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

    def _convert_df_to_ts(self, forecast: pd.DataFrame, last_train_date, components):
        groups = []
        for component in components:
            if self.n_lags == 0:
                # output format is different when AR is not used
                groups.append(
                    forecast[
                        (forecast["ID"] == component)
                        & (forecast["ds"] > last_train_date)
                    ]
                    .filter(items=["ds", "yhat1"])
                    .rename(columns={"yhat1": component})
                )
            else:
                df = fcst_df_to_latest_forecast(
                    forecast[(forecast["ID"] == component)],
                    quantiles=[0.5],
                    n_last=1,
                )
                groups.append(
                    df[df["ds"] > last_train_date]
                    .filter(items=["ds", "origin-0"])
                    .rename(columns={"origin-0": component})
                )

        return concatenate(
            [TimeSeries.from_dataframe(group, time_col="ds") for group in groups],
            axis=1,
        )

    def _as_sequence(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Sequence[TimeSeries]:
        if isinstance(series, TimeSeries):
            return [series]

        if isinstance(series, Sequence[TimeSeries]):
            return series

        raise ValueError("Invalid type. Expected TimeSeries or Sequence[TimeSeries]")

    def _from_sequence(
        self, series_list: Sequence[TimeSeries]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if len(series_list) == 1:
            return series_list[0]
        return series_list

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        raise NotImplementedError()

    def __str__(self):
        return "Neural Prophet"
