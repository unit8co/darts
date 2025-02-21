"""
Ensemble Model Base Class
"""

import os
import sys
from abc import abstractmethod
from collections.abc import Sequence
from typing import BinaryIO, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.models.utils import TORCH_AVAILABLE
from darts.timeseries import TimeSeries, concatenate
from darts.utils.ts_utils import series2seq

if TORCH_AVAILABLE:
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
else:
    TorchForecastingModel = None

logger = get_logger(__name__)


class EnsembleModel(GlobalForecastingModel):
    """
    Abstract base class for ensemble models.
    Ensemble models take in a list of forecasting models and ensemble their predictions
    to make a single one according to the rule defined by their `ensemble()` method.

    If `future_covariates` or `past_covariates` are provided at training or inference time,
    they will be passed only to the models supporting them.

    Parameters
    ----------
    forecasting_models
        List of forecasting models whose predictions to ensemble

        .. note::
                if all the models are probabilistic, the `EnsembleModel` will also be probabilistic.
        ..
    train_num_samples
        Number of prediction samples from each forecasting model for multi-level ensembles. The n_samples
        dimension will be reduced using the `train_samples_reduction` method.
    train_samples_reduction
        If `forecasting_models` are probabilistic and `train_num_samples` > 1, method used to reduce the
        samples dimension to 1. Possible values: "mean", "median" or float value corresponding to the
        desired quantile.
    train_forecasting_models
        If set to `False`, the `forecasting_models` are not retrained when calling `fit()` (only supported
        if all the `forecasting_models` are pretrained `GlobalForecastingModels`). Default: ``True``.
    show_warnings
        Whether to show warnings related to models covariates support.
    """

    def __init__(
        self,
        forecasting_models: list[ForecastingModel],
        train_num_samples: int,
        train_samples_reduction: Optional[Union[str, float]],
        train_forecasting_models: bool = True,
        show_warnings: bool = True,
    ):
        raise_if_not(
            isinstance(forecasting_models, list) and forecasting_models,
            "Cannot instantiate EnsembleModel with an empty list of `forecasting_models`",
            logger,
        )

        is_local_model = [
            isinstance(model, LocalForecastingModel) for model in forecasting_models
        ]
        is_global_model = [
            isinstance(model, GlobalForecastingModel) for model in forecasting_models
        ]

        self.is_local_ensemble = all(is_local_model)
        self.is_global_ensemble = all(is_global_model)

        raise_if_not(
            all([
                local_model or global_model
                for local_model, global_model in zip(is_local_model, is_global_model)
            ]),
            "All models must be of type `GlobalForecastingModel`, or `LocalForecastingModel`. "
            "Also, make sure that all `forecasting_models` are instantiated.",
            logger,
        )

        model_fit_status = [m._fit_called for m in forecasting_models]
        self.all_trained = all(model_fit_status)
        some_trained = any(model_fit_status)

        raise_if(
            (not self.is_global_ensemble and some_trained)
            or (self.is_global_ensemble and not (self.all_trained or not some_trained)),
            "Cannot instantiate EnsembleModel with a mixture of unfitted and fitted `forecasting_models`. "
            "Consider resetting all models with `my_model.untrained_model()` or using only trained "
            "GlobalForecastingModels together with `train_forecasting_models=False`.",
            logger,
        )

        if train_forecasting_models:
            # prevent issues with pytorch-lightning trainer during retraining
            raise_if(
                some_trained,
                "`train_forecasting_models=True` but some `forecasting_models` were already fitted. "
                "Consider resetting all the `forecasting_models` with `my_model.untrained_model()` "
                "before passing them to the `EnsembleModel`.",
                logger,
            )
        else:
            raise_if_not(
                self.is_global_ensemble and self.all_trained,
                "`train_forecasting_models=False` is supported only if all the `forecasting_models` are "
                "already trained `GlobalForecastingModels`.",
                logger,
            )

        raise_if(
            train_num_samples is not None
            and train_num_samples > 1
            and all([
                not m.supports_probabilistic_prediction for m in forecasting_models
            ]),
            "`train_num_samples` is greater than 1 but the `RegressionEnsembleModel` "
            "contains only deterministic `forecasting_models`.",
            logger,
        )

        supported_reduction = ["mean", "median"]
        if train_samples_reduction is None:
            pass
        elif isinstance(train_samples_reduction, float):
            raise_if_not(
                0.0 < train_samples_reduction < 1.0,
                f"if a float, `train_samples_reduction` must be between "
                f"0 and 1, received ({train_samples_reduction})",
                logger,
            )
        elif isinstance(train_samples_reduction, str):
            raise_if(
                train_samples_reduction not in supported_reduction,
                f"if a string, `train_samples_reduction` must be one of {supported_reduction}, "
                f"received ({train_samples_reduction})",
                logger,
            )
        else:
            raise_log(
                ValueError(
                    f"`train_samples_reduction` type not supported "
                    f"({train_samples_reduction}). Must be `float` "
                    f" or one of {supported_reduction}."
                ),
                logger,
            )

        super().__init__()
        self.forecasting_models = forecasting_models
        self.train_num_samples = train_num_samples
        self.train_samples_reduction = train_samples_reduction
        self.train_forecasting_models = train_forecasting_models
        self.show_warnings = show_warnings

        if show_warnings:
            if (
                self.supports_past_covariates
                and not self._full_past_covariates_support()
            ):
                logger.warning(
                    "Some `forecasting_models` in the ensemble do not support past covariates, the past covariates "
                    "will be provided only to the models supporting them when calling fit()` or `predict()`. "
                    "To hide these warnings, set `show_warnings=False`."
                )

            if (
                self.supports_future_covariates
                and not self._full_future_covariates_support()
            ):
                logger.warning(
                    "Some `forecasting_models` in the ensemble do not support future covariates, the future covariates"
                    " will be provided only to the models supporting them when calling `fit()` or `predict()`. "
                    "To hide these warnings, set `show_warnings=False`."
                )

    @abstractmethod
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """

        is_single_series = isinstance(series, TimeSeries)

        # local models OR mix of local and global models
        raise_if(
            not self.is_global_ensemble and not is_single_series,
            "The `forecasting_models` contain at least one LocalForecastingModel, which does not support training "
            "on multiple series.",
            logger,
        )

        # check that if timeseries is single series, that covariates are as well and vice versa
        error_past_cov = False
        error_future_cov = False

        if past_covariates is not None:
            error_past_cov = is_single_series != isinstance(past_covariates, TimeSeries)

        if future_covariates is not None:
            error_future_cov = is_single_series != isinstance(
                future_covariates, TimeSeries
            )

        raise_if(
            error_past_cov or error_future_cov,
            "Both series and covariates have to be either single TimeSeries or sequences of TimeSeries.",
            logger,
        )

        self._verify_past_future_covariates(past_covariates, future_covariates)
        super().fit(series, past_covariates, future_covariates)
        return self

    def _stack_ts_seq(self, predictions):
        # stacks list of predictions into one multivariate timeseries
        return concatenate(predictions, axis=1)

    def _stack_ts_multiseq(self, predictions_list):
        # stacks multiple sequences of timeseries elementwise
        return [self._stack_ts_seq(ts_list) for ts_list in zip(*predictions_list)]

    @property
    def _model_encoder_settings(self):
        raise NotImplementedError(
            "Encoders are not supported by EnsembleModels. Instead add encoders to the underlying `forecasting_models`."
        )

    def _make_multiple_predictions(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        is_single_series = isinstance(series, TimeSeries) or series is None
        # maximize covariate usage
        predictions = [
            model._predict_wrapper(
                n=n,
                series=series,
                past_covariates=(
                    past_covariates if model.supports_past_covariates else None
                ),
                future_covariates=(
                    future_covariates if model.supports_future_covariates else None
                ),
                num_samples=(
                    num_samples if model.supports_probabilistic_prediction else 1
                ),
                predict_likelihood_parameters=predict_likelihood_parameters,
            )
            for model in self.forecasting_models
        ]

        # reduce the probabilistics series
        if self.train_samples_reduction is not None and self.train_num_samples > 1:
            predictions = [
                self._predictions_reduction(prediction) for prediction in predictions
            ]

        return (
            self._stack_ts_seq(predictions)
            if is_single_series
            else self._stack_ts_multiseq(predictions)
        )

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # ensure forecasting models all rely on the same series during inference
        if series is None:
            series = self.training_series
        if past_covariates is None:
            past_covariates = self.past_covariate_series
        if future_covariates is None:
            future_covariates = self.future_covariate_series

        super().predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
            predict_likelihood_parameters=predict_likelihood_parameters,
            show_warnings=show_warnings,
        )

        # for single-level ensemble, probabilistic forecast is obtained directly from forecasting models
        if self.train_samples_reduction is None:
            pred_num_samples = num_samples
            forecast_models_pred_likelihood_params = predict_likelihood_parameters
        # for multi-levels ensemble, forecasting models can generate arbitrary number of samples
        else:
            pred_num_samples = self.train_num_samples
            # second layer model (regression) cannot be trained on likelihood parameters
            forecast_models_pred_likelihood_params = False

        self._verify_past_future_covariates(past_covariates, future_covariates)

        predictions = self._make_multiple_predictions(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=pred_num_samples,
            predict_likelihood_parameters=forecast_models_pred_likelihood_params,
        )

        return self.ensemble(
            predictions,
            series=series,
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
        )

    @abstractmethod
    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Defines how to ensemble the individual models' predictions to produce a single prediction.

        Parameters
        ----------
        predictions
            Individual predictions to ensemble
        series
            Sequence of timeseries to predict on. Optional, since it only makes sense for sequences of timeseries -
            local models retain timeseries for prediction.

        Returns
        -------
        TimeSeries or Sequence[TimeSeries]
            The predicted ``TimeSeries`` or sequence of ``TimeSeries`` obtained by ensembling the individual predictions
        """
        pass

    def _predictions_reduction(
        self, predictions: Union[Sequence[TimeSeries], TimeSeries]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Reduce the sample dimension of the forecasting models predictions"""
        is_single_series = isinstance(predictions, TimeSeries)
        predictions = series2seq(predictions)
        if self.train_samples_reduction == "median":
            predictions = [pred.median(axis=2) for pred in predictions]
        elif self.train_samples_reduction == "mean":
            predictions = [pred.mean(axis=2) for pred in predictions]
        else:
            predictions = [
                pred.quantile(self.train_samples_reduction) for pred in predictions
            ]
        return predictions[0] if is_single_series else predictions

    def _clean(self) -> Self:
        """Cleans the model and sub-models."""
        cleaned_model = super()._clean()
        cleaned_model.forecasting_models = [
            model._clean() for model in self.forecasting_models
        ]
        return cleaned_model

    def save(
        self,
        path: Optional[Union[str, os.PathLike, BinaryIO]] = None,
        clean: bool = False,
        **pkl_kwargs,
    ) -> None:
        """
        Saves the ensemble model under a given path or file handle.

        Additionally, two files are stored for each `TorchForecastingModel` under the forecasting models.

        Example for saving and loading a :class:`RegressionEnsembleModel`:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RegressionEnsembleModel, LinearRegressionModel, TiDEModel

                model = RegressionEnsembleModel(
                    forecasting_models = [
                        LinearRegressionModel(lags=4),
                        TiDEModel(input_chunk_length=4, output_chunk_length=4),
                        ],
                        regression_train_n_points=10,
                )

                model.save("my_ensemble_model.pkl")
                model_loaded = RegressionEnsembleModel.load("my_ensemble_model.pkl")
            ..

        Parameters
        ----------
        path
            Path or file handle under which to save the ensemble model at its current state. If no path is specified,
            the ensemble model is automatically saved under ``"{RegressionEnsembleModel}_{YYYY-mm-dd_HH_MM_SS}.pkl"``.
            If the i-th model of `forecasting_models` is a TorchForecastingModel, two files (model object and
            checkpoint) are saved under ``"{path}.{ithModelClass}_{i}.pt"`` and ``"{path}.{ithModelClass}_{i}.ckpt"``.
        clean
            Whether to store a cleaned version of the model. If `True`, the training series and covariates are removed.
            If the underlying `forecasting_models` contain any `TorchForecastingModel`, will additionally remove all of
            their Lightning Trainer-related parameters.

            Note: After loading a model stored with `clean=True`, a `series` must be passed 'predict()',
            `historical_forecasts()` and other forecasting methods.
        pkl_kwargs
            Keyword arguments passed to `pickle.dump()`
        """

        if path is None:
            # default path
            path = self._default_save_path() + ".pkl"

        super().save(path, clean=clean, **pkl_kwargs)

        for i, m in enumerate(self.forecasting_models):
            if TORCH_AVAILABLE and issubclass(type(m), TorchForecastingModel):
                path_tfm = f"{path}.{type(m).__name__}_{i}.pt"
                m.save(path=path_tfm, clean=clean)

    @staticmethod
    def load(
        path: Union[str, os.PathLike, BinaryIO],
        pl_trainer_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> "EnsembleModel":
        """
        Loads a model from a given path or file handle.

        Parameters
        ----------
        path
            Path or file handle from which to load the model.
        pl_trainer_kwargs
            Only effective if the underlying forecasting models contain a `TorchForecastingModel`.
            Optionally, a set of kwargs to create a new Lightning Trainer used to configure the model for downstream
            tasks (e.g. prediction).
            Some examples include specifying the batch size or moving the model to CPU/GPU(s). Check the
            `Lightning Trainer documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_
            for more information about the supported kwargs.
        **kwargs
            Only effective if the underlying forecasting models contain a `TorchForecastingModel`.
            Additional kwargs for PyTorch Lightning's :func:`LightningModule.load_from_checkpoint()` method,
            For more information, read the `official documentation <https://pytorch-lightning.readthedocs.io/en/stable/
            common/lightning_module.html#load-from-checkpoint>`_.
        """
        model: EnsembleModel = GlobalForecastingModel.load(path)

        for i, m in enumerate(model.forecasting_models):
            if TORCH_AVAILABLE and issubclass(type(m), TorchForecastingModel):
                path_tfm = f"{path}.{type(m).__name__}_{i}.pt"
                model.forecasting_models[i] = TorchForecastingModel.load(
                    path_tfm, pl_trainer_kwargs=pl_trainer_kwargs, **kwargs
                )
        return model

    @property
    def min_train_series_length(self) -> int:
        return max(model.min_train_series_length for model in self.forecasting_models)

    @property
    def min_train_samples(self) -> int:
        return max(model.min_train_samples for model in self.forecasting_models)

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        def find_max_lag_or_none(lag_id, aggregator) -> Optional[int]:
            max_lag = None
            for model in self.forecasting_models:
                curr_lag = model.extreme_lags[lag_id]
                if max_lag is None:
                    max_lag = curr_lag
                elif curr_lag is not None:
                    max_lag = aggregator(max_lag, curr_lag)
            return max_lag

        lag_aggregators = (min, max, min, max, min, max, max, max)
        return tuple(
            find_max_lag_or_none(i, agg) for i, agg in enumerate(lag_aggregators)
        )

    @property
    def output_chunk_length(self) -> Optional[int]:
        """Return `None` if none of the forecasting models have a `output_chunk_length`,
        otherwise return the smallest output_chunk_length.
        """
        tmp = [
            m.output_chunk_length
            for m in self.forecasting_models
            if m.output_chunk_length is not None
        ]

        if len(tmp) == 0:
            return None
        else:
            return min(tmp)

    @property
    def _models_are_probabilistic(self) -> bool:
        return all([
            model.supports_probabilistic_prediction for model in self.forecasting_models
        ])

    @property
    def _models_same_likelihood(self) -> bool:
        """Return `True` if all the `forecasting_models` are probabilistic and fit the same distribution."""
        if not self._models_are_probabilistic:
            return False

        models_likelihood = set()
        lkl_same_params = True
        tmp_quantiles = None
        for m in self.forecasting_models:
            # regression model likelihood is a string, torch-based model likelihoods is an object
            likelihood = getattr(m, "likelihood")
            is_obj_lkl = not isinstance(likelihood, str)
            lkl_simplified_name = (
                likelihood.simplified_name() if is_obj_lkl else likelihood
            )
            models_likelihood.add(lkl_simplified_name)

            # check the quantiles
            if lkl_simplified_name == "quantile":
                quantiles: list[str] = (
                    likelihood.quantiles if is_obj_lkl else m.quantiles
                )
                if tmp_quantiles is None:
                    tmp_quantiles = quantiles
                elif tmp_quantiles != quantiles:
                    lkl_same_params = False

        return len(models_likelihood) == 1 and lkl_same_params

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        """EnsembleModel can predict likelihood parameters if all its forecasting models were fitted with the
        same likelihood.
        """
        return (
            all([
                m.supports_likelihood_parameter_prediction
                for m in self.forecasting_models
            ])
            and self._models_same_likelihood
        )

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return self._models_are_probabilistic

    @property
    def supports_multivariate(self) -> bool:
        return all([model.supports_multivariate for model in self.forecasting_models])

    @property
    def supports_past_covariates(self) -> bool:
        return any([
            model.supports_past_covariates for model in self.forecasting_models
        ])

    @property
    def supports_future_covariates(self) -> bool:
        return any([
            model.supports_future_covariates for model in self.forecasting_models
        ])

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        """
        Whether the model supports optimized historical forecasts
        """
        return False

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return self.is_global_ensemble

    def _full_past_covariates_support(self) -> bool:
        return all([
            model.supports_past_covariates for model in self.forecasting_models
        ])

    def _full_future_covariates_support(self) -> bool:
        return all([
            model.supports_future_covariates for model in self.forecasting_models
        ])

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        """
        Verify that any non-None covariates comply with the model type.
        """
        raise_if(
            past_covariates is not None and not self.supports_past_covariates,
            "`past_covariates` were provided to an `EnsembleModel` but none of its "
            "`forecasting_models` support such covariates.",
            logger,
        )
        raise_if(
            future_covariates is not None and not self.supports_future_covariates,
            "`future_covariates` were provided to an `EnsembleModel` but none of its "
            "`forecasting_models` support such covariates.",
            logger,
        )
