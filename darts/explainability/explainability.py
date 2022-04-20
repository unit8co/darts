"""
Explainability Base Class
------------------------------

TODO
"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

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

        if not model._fit_called:
            raise_log(
                ValueError(
                    "The model must be fitted before instantiating a ForecastingModelExplainer."
                ),
                logger,
            )

        if model._is_probabilistic():
            # TODO: We can probably add explainability to probabilistic models, by taking the mean output.
            raise_log(
                ValueError(
                    "Explainability is only available for non-probabilistic models."
                ),
                logger,
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
                background_series = [background_series]
                background_past_covariates = (
                    [background_past_covariates] if background_past_covariates else None
                )
                background_future_covariates = (
                    [background_future_covariates]
                    if background_future_covariates
                    else None
                )
            for idx in range(len(background_series)):
                if not all(
                    len(
                        background_series[idx].time_index.difference(
                            background_past_covariates[idx].time_index
                        )
                    )
                    == 0,
                    len(
                        background_past_covariates[idx].time_index.difference(
                            background_future_covariates[idx].time_index
                        )
                    )
                    == 0,
                    len(
                        background_future_covariates[idx].time_index.difference(
                            background_series[idx].time_index
                        )
                    )
                    == 0,
                ):
                    logger.warning(
                        "Some series and their covariates don't share the same time index. We will take "
                        "the time index common to all."
                    )

                (
                    background_series[idx],
                    background_past_covariates[idx],
                    background_future_covariates[idx],
                ) = retain_period_common_to_all(
                    [
                        background_series[idx],
                        background_past_covariates[idx],
                        background_future_covariates[idx],
                    ]
                )

            self.background_series = background_series
            self.background_past_covariates = background_past_covariates
            self.background_future_covariates = background_past_covariates

        self.target_names = self.background_series.columns
        self.past_covariates_names = self.background_past_covariates.columns
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

        self.model = model

        # For now we won't consider further time step that output_chunk_length, even though we could in
        # theory explain anything in the future by auto regressive process.
        if hasattr(self.model, "output_chunk_length"):
            self.n = self.model.output_chunk_length
        else:
            self.n = 1

        if hasattr(self.model, "input_chunk_length"):
            self.past_steps_explained = self.model.input_chunk_length
        else:
            self.past_steps_explained = 1

        if not self.test_stationarity():
            logger.warning(
                "One time series component of the background time series is not stationary."
                " Beware of wrong interpretation with chosen explainability."
            )

    @abstractmethod
    def explain_from_input(
        foreground_series: TimeSeries,
        foreground_past_covariates: Optional[TimeSeries] = None,
        foreground_future_covariates: Optional[TimeSeries] = None,
        horizons: Optional[Sequence[int]] = None,
        target_names: Optional[Sequence[str]] = None,
    ) -> Sequence[Sequence[TimeSeries]]:
        """
        Return explanations values for each target and covariates lag, in a multivariate TimeSeries format.
        Each timestamp of the foreground TimeSeries is explained in the output TimeSeries, with the following
        notation:
        `name`_`type_of_cov`_lag_`int`

        Example:
        Let's say we have a model with 2 targets (multivariates) names T_1 and T_2, one past covariate and one
        future covariate. Also, n = 2 and past_step_explained = 2.

        Then the function is supposed to return a dictionary time series, with for example

        output[0]['T_1'] (but also output[1]['T_1'], output[0]['T_2'] and output[1]['T_2']a TimeSeries
        with the following components:
            - T_1_target_lag-1
            - T_1_target_lag-2
            - 0_past_cov_lag-1 (we didn't name the past covariate so it took the default name)
            - 0_past_cov_lag-2
            - 0_fut_cov_lag_0 (could be also lag_1 if output[1])


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
