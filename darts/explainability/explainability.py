"""
Forecasting Model Explainer Base Class
------------------------------
A forecasting model explainer captures an already fitted forecasting model, and apply an Explainability model
to this forecasting model. Its purpose is to be able to explain each past input contribution to a given model forecast.
This 'explanation' depends on the characteristics of the XAI model chosen (shap, lime etc...).

"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

from numpy import integer

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils import retain_period_common_to_all
from darts.utils.statistics import stationarity_tests

logger = get_logger(__name__)


class ForecastingModelExplainer(ABC):
    @abstractmethod
    def __init__(
        self,
        model: ForecastingModel,
        background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        background_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        background_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
    ):
        """The base class for forecasting model explainers. It defines the *minimal* behavior that all
        forecasting model explainers support.

        Nomenclature:
        - A background time series is a time series with which we train the Explainer model.
        - A foreground time series is the time series we will explain according to the fitted Explainer model.

        Parameters
        ----------
        model
            A ForecastingModel we want to explain. It has to be fitted first.
        background_series
            A TimeSeries or a list of time series we want to use to compare with any foreground we want to explain.
            This is optional, for 2 reasons:
                - In general we want to keep the training_series of the model and this is the default one,
                but in case of multiple time series training (global or meta learning) the ForecastingModel doesn't
                save them. In this case we need to feed a background time series.
                - We might want to consider a reduced well chosen background in order to reduce computation
                time.
        background_past_covariates
            A past covariates TimeSeries or list of TimeSeries that the model needs once fitted.
        background_future_covariates
            A future covariates TimeSeries or list of TimeSeries that the model needs once fitted.
        """
        if not model._fit_called:
            raise_log(
                ValueError(
                    "The model must be fitted before instantiating a ForecastingModelExplainer."
                ),
                logger,
            )

        if model._is_probabilistic():
            logger.warning(
                "The model is probabilistic, but n_sample=1 will be used for explainability."
            )

        self.model = model

        # In case we don't want to fit the Explainer with a specific background time series, we use the one
        # already existing in the fitted model input.
        if background_series is None:

            raise_if(
                (background_past_covariates is not None)
                or (background_future_covariates is not None),
                "There is background past or future covariates but no background series. Please provide one.",
            )

            raise_if(
                self.model.training_series is None,
                "A background time series has to be provided for a model fitted on multiple time series, as"
                "no training series has been saved by the model.",
            )

            self.background_series = self.model.training_series
            self.background_past_covariates = self.model.past_covariate_series
            self.background_future_covariates = self.model.future_covariate_series

        else:

            # ensure list of TimeSeries format
            if isinstance(background_series, TimeSeries):
                self.background_series = [background_series]
                self.background_past_covariates = (
                    [background_past_covariates] if background_past_covariates else None
                )
                self.background_future_covariates = (
                    [background_future_covariates]
                    if background_future_covariates
                    else None
                )

            for idx in range(len(self.background_series)):
                if not all(
                    self.background_series[idx].has_same_time_as(
                        self.background_past_covariates[idx]
                    ),
                    self.background_series[idx].has_same_time_as(
                        self.background_future_covariates[idx]
                    ),
                ):

                    logger.warniplainabilityng(
                        "Some series and their covariates don't share the same time index. We will take "
                        "the time index common to all."
                    )

                (
                    self.background_series[idx],
                    self.background_past_covariates[idx],
                    self.background_future_covariates[idx],
                ) = retain_period_common_to_all(
                    [
                        self.background_series[idx],
                        self.background_past_covariates[idx],
                        self.background_future_covariates[idx],
                    ]
                )

        self.target_names = self.background_series.columns
        if self.background_past_covariates is not None:
            self.past_covariates_names = self.background_past_covariates.columns
        if self.background_future_covariates is not None:
            self.future_covariates_names = self.background_future_covariates.columns

        raise_if(
            self.model._expect_past_covariates
            and self.background_past_covariates is None,
            "A background past covariates is not provided, but the model needs past covariates.",
        )

        raise_if(
            self.model._expect_future_covariates
            and self.background_future_covariates is None,
            "A background future covariates is not provided, but the model needs future covariates.",
        )

        if not self.test_stationarity():
            logger.warning(
                "One time series component of the background time series is not stationary."
                " Beware of wrong interpretation with chosen explainability."
            )

    @abstractmethod
    def explain(
        self,
        foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        foreground_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        foreground_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
    ) -> Union[
        Dict[integer, Dict[str, TimeSeries]],
        Sequence[Dict[integer, Dict[str, TimeSeries]]],
    ]:
        """
        Main method of the ForecastingExplainer object.
        Return a dictionary of dictionaries (or a list of dictionaries of dictionaries, il multiple TimeSeries list):
        - the first dimension corresponds to the n forecasts ahead we want to explain (Horizon).
        - the second dimension corresponds to each component of the target time series.


        The value of the second dimension dictionary is a (multivariate) TimeSeries object giving the 'explanation'
        for a given forecast (horizon, target) at any timestamp forecastable given the foreground TimeSeries
        time dimension.

        The name convention for each component of this multivariate TimeSeries is:
        `name`_`type_of_cov`_lag_`int` where:
        - `name` is the existing name of the component in the original foreground TimeSeries (target or past or future).
        - `type_of_cov` is the type of covariates. It can take 3 different values: `target`, `past`, `future`.
        - `int` is the lag index.

        Example:
        Let's say we have a model with 2 targets (multivariates) named "T_1" and "T_2", three past covariates we didn't
        name and one future covariate we didn't name. Also, n = 2.
        The model is a regression model, with lags = 3, lags_past_covariates=[-1, -3], lags_future_covariates = [0]

        We provide a foreground_series (not a list), past covariates, future covariates, of length 5.

        Then the output will be the following:

        output[0]['T_1'] a multivariate TimeSeries containing the 'explanations' of the chosen Explainer, with
        component names:
            - T_0_target_lag-1
            - T_0_target_lag-2
            - T_0_target_lag-3
            - T_1_target_lag-1
            - T_1_target_lag-2
            - T_1_target_lag-3
            - 0_past_cov_lag-1 (we didn't name the past covariate so it took the default name)
            - 0_past_cov_lag-3 (we didn't name the past covariate so it took the default name)
            - 1_past_cov_lag-1 (we didn't name the past covariate so it took the default name)
            - 1_past_cov_lag-3 (we didn't name the past covariate so it took the default name)
            - 2_past_cov_lag-1 (we didn't name the past covariate so it took the default name)
            - 2_past_cov_lag-3 (we didn't name the past covariate so it took the default name)
            - 0_fut_cov_lag_0  (we didn't name the future covariate so it took the default name)

        of length 3, as we can explain 5-3+1 forecasts (basically timestamp indexes 4, 5, and 6)


        Parameters
        ----------
        foreground_series
            Optionally, target timeseries we want to explain. Can be multivariate.
            If none is provided, explain will automatically provide the whole background TimeSeries explanation.
        foreground_past_covariates
            Optionally, past covariate timeseries if needed by model.
        foreground_future_covariates
            Optionally, future covariate timeseries if needed by model.

        Returns
        -------
        a dictionary of dictionary of Timeseries (or a list of such) of explaining values :
            - each element of the first dimension dictionary is corresponding to a forecast horizon
            - each element of the second dimension dictionary is corresponding to a target
        """

        pass

    def test_stationarity(self):
        return all(
            [
                stationarity_tests(self.background_series[c])
                for c in self.background_series.components
            ]
        )
