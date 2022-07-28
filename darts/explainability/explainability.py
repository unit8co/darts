"""
Explainability Base Class
------------------------------

TODO
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
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

                    logger.warning(
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
        foreground_series: TimeSeries,
        foreground_past_covariates: Optional[TimeSeries] = None,
        foreground_future_covariates: Optional[TimeSeries] = None,
        horizons: Optional[Sequence[int]] = None,
        target_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, TimeSeries]]:
        """
        Return a dictionary of dictionaries:
        - the first dimension corresponds to each component of the target time series.
        - the second dimension corresponds to the i element of the n forecast ahead we want to explain.

        The value of the second dimension dictionary is a (multivariate) TimeSeries object giving the 'explanation'
        for a given forecast (target, i future forecast) at any timestamp forecastable given the foreground TimeSeries time dimension.

        The name convention for each component of this multivariate TimeSeries is:
        `name`_`type_of_cov`_lag_`int` where:
        - `name` is the existing name of the component in the original foreground TimeSeries (target or past or future).
        - `type_of_cov` is the type of covariates. It can take 3 different values: `target`, `past`, `future`.
        - `int` is the lag index.

        Example:
        Let's say we have a model with 2 targets (multivariates) named T_1 and T_2, three past covariates we didn't name and one
        future covariate we didn't name. Also, n = 2 and past_step_explained = 3.

        We provide a foreground_series, past covariates, future covariates, of length 5.

        Then the output will be the following:

        output['T_1'][0] a multivariate TimeSeries containing the 'explanations' of the chosen Explainer, with component names:
            - T_1_target_lag-1
            - T_1_target_lag-2
            - T_1_target_lag-3
            - 0_past_cov_lag-1 (we didn't name the past covariate so it took the default name)
            - 0_past_cov_lag-2
            - 0_past_cov_lag-3
            - 1_past_cov_lag-1
            - 1_past_cov_lag-2
            - 1_past_cov_lag-3
            - 2_past_cov_lag-1
            - 2_past_cov_lag-2
            - 2_past_cov_lag-3
            - 0_fut_cov_lag_0

        of length 3, as we can explain 5-3+1 forecasts (basically timestamp indexes 4, 5, and 6)



        Parameters
        ----------
        foreground_series
            TimeSeries target we want to explain. Can be multivariate.
        foreground_past_covariates
            Optionally, past covariate timeseries if needed by model.
        foreground_future_covariates
            Optionally, future covariate timeseries if needed by model.
        horizons
            Optionally, a list of integer values representing which elements in the future
            we want to explain, starting from the first timestamp prediction at 0.
            For now we consider only models with output_chunk_length and it can't be bigger than
            output_chunk_length.
            If no input, then all elements of output_chunk_length will be explained.
        target_names
            Optionally, a list of strings naming the components of `foreground_series` we want to explain.
            If no input, then all targets will be explained.

        Returns
        -------
        a TimeSeries or dictionary of Timeseries of explaining values :
            - each element of the first dictionary is corresponding to an horizon
            - each element of the second layer dictionary is corresponding to a target
        """
        pass

    def test_stationarity(self):
        return all(
            [
                stationarity_tests(self.background_series[c])
                for c in self.background_series.components
            ]
        )
