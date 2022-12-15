"""
Neural Prophet
------------
"""

from typing import Optional, Sequence, Union

import neuralprophet
import pandas as pd
from neuralprophet.utils import fcst_df_to_latest_forecast

from darts.logging import raise_if_not
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries, concatenate


class NeuralProphet(ForecastingModel):
    def __init__(self, n_lags: int = 0, n_forecasts: int = 1, **kwargs):
        super().__init__()

        raise_if_not(n_lags >= 0, "Argument n_lags should be a non-negative integer")

        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.model = neuralprophet.NeuralProphet(
            n_lags=n_lags, n_forecasts=n_forecasts, **kwargs
        )

    def fit(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
    ) -> "NeuralProphet":
        super().fit(series)

        raise_if_not(
            series.has_datetime_index,
            "NeuralProphet model is limited to TimeSeries index with DatetimeIndex",
        )

        raise_if_not(
            past_covariates is None or self.n_lags > 0,
            "Past covariates are only supported when auto-regression is enabled (n_lags > 1)",
        )

        self.training_series = series
        fit_df = self._convert_ts_to_df(series)

        if past_covariates is not None:
            fit_df = self._add_past_covariate(self.model, fit_df, past_covariates)

        # TODO add future covariates to df

        self.model.fit(fit_df, freq=series.freq_str, minimal=True)

        self.fit_df = fit_df
        return self

    def predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().predict(n, num_samples)

        raise_if_not(
            self.n_lags == 0 or n <= self.n_forecasts,
            "Auto-regression has been configured. `n` must be smaller than or equal to"
            "`n_forecasts` parameter in the constructor.",
        )

        future_df = self.model.make_future_dataframe(df=self.fit_df, periods=n)
        forecast_df = self.model.predict(future_df)

        return self._convert_df_to_ts(
            forecast_df,
            self.training_series.end_time(),
            self.training_series.components,
        )

    def _convert_ts_to_df(self, series: TimeSeries) -> pd.DataFrame:
        """Convert TimeSeries to pandas DataFrame format required by Neural Prophet"""
        dfs = []  # ID y

        for component in series.components:
            component_df = (
                series[component]
                .pd_dataframe(copy=False)
                .reset_index(names=["ds"])
                .filter(items=["ds", component])
                .rename(columns={component: "y"})
            )
            component_df["ID"] = component
            dfs.append(component_df)

        return pd.concat(dfs).copy(deep=True)

    def _add_past_covariate(
        self,
        model: neuralprophet.NeuralProphet,
        df: pd.DataFrame,
        past_covariates: TimeSeries,
    ) -> pd.DataFrame:
        """Convert past covariates from TimeSeries and add them to DataFrame"""

        # TODO add checks if past covariate Time series has enough coverage and the same frequency

        for component in past_covariates.components:
            covariate_df = (
                past_covariates[component]
                .pd_dataframe(copy=False)
                .reset_index(names=["ds"])
                .filter(items=["ds", component])
            )

            df = df.merge(covariate_df, how="left", on="ds")

            model.add_lagged_regressor(names=component)

        return df

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

    def __str__(self):
        return "Neural Prophet"
