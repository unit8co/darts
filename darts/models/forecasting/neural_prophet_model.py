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
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> "NeuralProphet":
        super().fit(series, past_covariates, future_covariates)

        # TODO accept list of univariate series or one multivariate ???
        # TODO series have to have the same frequency

        self.training_series = self._as_sequence(series)
        self.train_past_cov = (
            self._as_sequence(past_covariates) if past_covariates is not None else None
        )
        self.train_future_cov = (
            self._as_sequence(future_covariates)
            if future_covariates is not None
            else None
        )

        fit_df = self._convert_ts_to_df(
            self.model, self.training_series, self.train_past_cov, self.train_future_cov
        )

        # TODO check if all time series has common frequency string
        self.model.fit(fit_df, freq=series[0].freq_str)

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

    def _convert_ts_to_df(
        self,
        model: neuralprophet.NeuralProphet,
        series_list: Sequence[TimeSeries],
        past_cov: Optional[Sequence[TimeSeries]],
        future_cov: Optional[Sequence[TimeSeries]],
    ):
        raise_if_not(
            len(past_cov) == 0 or len(past_cov) == len(series_list),
            "Number of past covariates has to be zero or equal to number of fit time series.",
        )

        dfs = []

        for i, series in enumerate(series_list):
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

            if past_cov is not None:
                for component in past_cov[i].components:
                    covaraite_df = (
                        past_cov[i].pd_dataframe(copy=False).reset_index(names=["ds"])
                    )
                    covaraite_df = covaraite_df[["ds", component]].copy(deep=True)

                    # TODO add checks if past covariate has full coverage
                    component_df = component_df.merge(covaraite_df, how="left", on="ds")
                    model.add_lagged_regressor(names=component)

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
        self, series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]
    ) -> Sequence[TimeSeries]:
        if series is None:
            return []

        if isinstance(series, TimeSeries):
            return [series]

        return series

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
