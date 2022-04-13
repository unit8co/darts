"""
Explainability Base Class
------------------------------

TODO
"""

from pyexpat import model
from xmlrpc.client import Boolean
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.regression_model import RegressionModel
from darts.utils.statistics import stationarity_tests

from darts import models
from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if


logger = get_logger(__name__)


class ForecastingModelExplainer(ABC):

    @abstractmethod
    def __init__(
        self,
        model: ForecastingModel, 
        background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        background_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        background_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: Optional[int]=1,
        past_steps_explained: Optional[int] = 2
    ):

        if not issubclass(type(model), RegressionModel):
            raise_log(
                ValueError('Invalid model type. For now, only RegressionModel type can be explained.'),
                logger
        )

        if not model._fit_called:
            raise_log(
                ValueError('The model must be fitted before instantiating a ForecastingModelExplainer.'),
                logger
                )

        if model._is_probabilistic():
            # TODO: We can probably add explainability to probabilistic models, by taking the mean output.
            raise_log(
                ValueError('Explainability is only available for non-probabilistic models.'),
                logger
                )

        self.model = model

        if background_series is None:

            raise_if(
                (not background_past_covariates is None) or (not background_future_covariates is None), 
                "There is background past or future covariates but no background series. Please provide one.")
            
            raise_if(
                self.model.training_series is None, 
                "A background time series has to be provided for a model fitted on multiple time series, as"
                "no training series has been saved by the model.")
        
        self.background_series = self.model.training_series
        self.background_past_covariates = self.model.past_covariate_series
        self.background_future_covariates = self.model.future_covariate_series

        self.target_names = self.background_series.columns
        self.past_covariates_names = self.background_past_covariates.columns
        self.future_covariates_names = self.background_future_covariates.columns

        raise_if(
            self.model._expect_past_covariates and self.background_past_covariates is None, 
            "A background past covariates is not provided, but the model needs past covariates.")

        raise_if(
            self.model._expect_future_covariates and self.background_future_covariates is None, 
                "A background future covariates is not provided, but the model needs future covariates.")
        
        self.model = model
        self.past_steps_explained = past_steps_explained
        self.n = n

        # For now we won't consider further time step that output_chunk_length, even though we could in 
        # theory explain anything in the future by auto regressive process.
        if hasattr(self.model, 'output_chunk_length'):
            self.n = self.model.output_chunk_length
        
        if hasattr(self.model, 'input_chunk_length'):
            self.past_steps_explained = self.model.input_chunk_length

        if not self.test_stationarity():
            logger.warning("The background time series is not stationary. Beware of wrong interpretation "
            "with chosen explainability")

    @abstractmethod
    def explain_from_input(        
        foreground_series: TimeSeries,
        foreground_past_covariates: Optional[TimeSeries],
        foreground_future_covariates: Optional[TimeSeries],
        horizons: Optional[Sequence[int]]=None, 
        target_names: Optional[Sequence[str]]=None
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
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
            Optionally, a list of string values naming the targets we want to explain.
            If no input, then all targets will be explained.
        
        Returns
        -------
        a shap Explanation dictionary of dictionaries of shap Explanation objects:
            - each element of the first dictionary is corresponding to an horizon
            - each element of the second layer dictionary is corresponding to a target
        """
        pass
    
    def test_stationarity(self):
        return all(
            [stationarity_tests(self.background_series[c]) for c in self.background_series.components]
        )








