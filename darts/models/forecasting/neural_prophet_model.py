"""
Neural Prophet
------------
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import neuralprophet
import pandas as pd
from neuralprophet.utils import fcst_df_to_latest_forecast

from darts.logging import raise_if_not
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries, concatenate


class NeuralProphet(ForecastingModel):
    def __init__(
        self,
        n_lags: int = 0,
        n_forecasts: int = 1,
        add_encoders: Optional[Dict] = None,
        **kwargs,
    ):
        """Neural Prophet

        This class provides a basic wrapper around `NeuralProphet <https://github.com/ourownstory/neural_prophet>`_.
        It extends approach similar to Facebook Prophet model with auto-regressive feed-forward neural network
        It supports also supports past and future covariates. For more parameters refer to the original documentation.

        Parameters
        ----------
        n_lags
            Number of lagged values provided to AR-Net. If equal to 0 then only trend
            and seasonality will be used for forecasting.

        n_forecast
            Output size chunk of the AR-Net. Limits how far into the future is is possible to forecast.

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
        super().__init__(add_encoders=add_encoders, **kwargs)
        # TODO improve passing arguments to the model

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
            "NeuralProphet model is limited to TimeSeries indexed with DatetimeIndex",
        )

        raise_if_not(
            past_covariates is None or self.n_lags > 0,
            "Past covariates are only supported when auto-regression is enabled (n_lags > 0)",
        )

        self.training_series = series
        fit_df = self._convert_ts_to_df(series)

        if past_covariates is not None:
            fit_df = self._add_past_covariates(self.model, fit_df, past_covariates)

        if future_covariates is not None:
            fit_df = self._add_future_covariates(self.model, fit_df, future_covariates)
            self.future_components = future_covariates.components
        else:
            self.future_components = None

        with warnings.catch_warnings():
            self.model.fit(fit_df, freq=series.freq_str)

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
            "Auto-regression has been enabled. `n` must be smaller than or equal to"
            "`n_forecasts` parameter in the constructor.",
        )

        self._future_covariates_checks(future_covariates)

        regressors_df = (
            self._future_covariates_df(future_covariates)
            if self.future_components is not None
            else None
        )

        future_df = self.model.make_future_dataframe(
            df=self.fit_df, regressors_df=regressors_df, periods=n
        )

        with warnings.catch_warnings():
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

    def _add_past_covariates(
        self,
        model: neuralprophet.NeuralProphet,
        df: pd.DataFrame,
        covariates: TimeSeries,
    ):
        df = self._add_covariate(df, covariates)
        model.add_lagged_regressor(names=list(covariates.components))
        return df

    def _add_future_covariates(
        self,
        model: neuralprophet.NeuralProphet,
        df: pd.DataFrame,
        covariates: TimeSeries,
    ):
        df = self._add_covariate(df, covariates)
        for component in covariates.components:
            model.add_future_regressor(name=component)

        return df

    def _add_covariate(
        self,
        df: pd.DataFrame,
        covariates: TimeSeries,
    ) -> pd.DataFrame:
        """Convert past covariates from TimeSeries and add them to DataFrame"""

        raise_if_not(
            self.training_series.freq == covariates.freq,
            "Covariate TimeSeries has to have the same frequency as the TimeSeries that model is fitted on.",
        )

        raise_if_not(
            covariates.start_time() <= self.training_series.start_time()
            and self.training_series.end_time() <= covariates.end_time(),
            "Covaraite TimeSeries has to span across all TimeSeries that model is fitted on",
        )

        for component in covariates.components:
            covariate_df = (
                covariates[component]
                .pd_dataframe(copy=False)
                .reset_index(names=["ds"])
                .filter(items=["ds", component])
            )

            df = df.merge(covariate_df, how="left", on="ds")

        return df

    def _convert_df_to_ts(self, forecast: pd.DataFrame, last_train_date, components):
        groups = []
        for component in components:
            if self.n_lags == 0:
                # output format is different when AR is not enabled
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

    def _future_covariates_df(self, series: TimeSeries) -> pd.DataFrame:
        component_dfs = []
        for component in series.components:
            component_dfs.append(series[component].pd_dataframe())

        return pd.concat(component_dfs, axis=1).reset_index(names=["ds"])

    def _future_covariates_checks(self, future_covariates: Optional[TimeSeries]):
        raise_if_not(
            self.future_components is None
            or (
                future_covariates is not None
                and set(self.future_components) == set(future_covariates.components)
            ),
            f"Missing future covariate TimeSeries. Model was trained with {self.future_components} "
            "future components",
        )

        raise_if_not(
            self.future_components is None
            or future_covariates.freq == self.training_series.freq,
            "Invalid frequency in future covariate TimeSeries",
        )

    def uses_future_covariates(self):
        return True

    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return (None, None, True, True, None, None)

    def __str__(self):
        return "Neural Prophet"
