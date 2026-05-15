"""
Base Ensemble Model
-------------------
"""

import copy
import os
import sys
from abc import abstractmethod
from collections import defaultdict
from typing import BinaryIO

from darts.models.forecasting.sklearn_model import SKLearnModel
from darts.utils.likelihood_models.base import LikelihoodType

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from darts import TimeSeries, concatenate
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.typing import TimeSeriesLike
from darts.utils.ts_utils import series2seq
from darts.utils.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from darts.models.forecasting.torch_forecasting_model import (
        TFM_ATTRS_NO_PICKLE,
        TorchForecastingModel,
    )
else:
    TorchForecastingModel, TFM_ATTRS_NO_PICKLE = None, None

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
    train_n_points
        The number of points per series to use to train the ensemble model. Can be set to `-1` to use the
        entire series to train the regressor if `forecasting_models` are already fitted and
        `train_forecasting_models=False`.
    show_warnings
        Whether to show warnings related to models covariates support.
    """

    def __init__(
        self,
        forecasting_models: list[ForecastingModel],
        ensemble_model: SKLearnModel | None,
        train_num_samples: int,
        train_samples_reduction: str | float | None,
        train_forecasting_models: bool = True,
        train_n_points: int = 0,
        show_warnings: bool = True,
    ):
        super().__init__()

        if not isinstance(forecasting_models, list) or len(forecasting_models) == 0:
            raise_log(
                ValueError(
                    "`forecasting_models` must be a non-empty list of forecasting models."
                ),
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

        if not all([
            local_model or global_model
            for local_model, global_model in zip(is_local_model, is_global_model)
        ]):
            raise_log(
                ValueError(
                    "All models must be of type `GlobalForecastingModel`, or `LocalForecastingModel`. "
                    "Also, make sure that all `forecasting_models` are instantiated."
                ),
                logger,
            )

        model_fit_status = [m._fit_called for m in forecasting_models]
        self.all_trained = all(model_fit_status)
        some_trained = any(model_fit_status)

        if not self.is_global_ensemble and some_trained:
            raise_log(
                ValueError(
                    "Some models in `forecasting_models` are already fitted. Using pre-trained models is "
                    "only supported if all models are of type `GlobalForecastingModel`. "
                    "Consider resetting all models with `my_model.untrained_model()`."
                ),
                logger,
            )
        elif self.is_global_ensemble and not (self.all_trained or not some_trained):
            raise_log(
                ValueError(
                    "All `forecasting_models` are global but there is a mixture of fitted and unfitted models. "
                    "Consider resetting all models with `my_model.untrained_model()` or using only trained "
                    "`GlobalForecastingModel` together with `train_forecasting_models=False`."
                ),
                logger,
            )

        if train_forecasting_models:
            # prevent issues with pytorch-lightning trainer during retraining
            if some_trained:
                raise_log(
                    ValueError(
                        "`train_forecasting_models=True` but some `forecasting_models` were already fitted. "
                        "Consider resetting all the `forecasting_models` with `my_model.untrained_model()` "
                        "before passing them to the `EnsembleModel`."
                    ),
                    logger,
                )
        else:
            if not (self.is_global_ensemble and self.all_trained):
                raise_log(
                    ValueError(
                        "`train_forecasting_models=False` is supported only if all the `forecasting_models` are "
                        "already trained `GlobalForecastingModels`."
                    ),
                    logger,
                )

        if (
            train_num_samples is not None
            and train_num_samples > 1
            and all([
                not m.supports_probabilistic_prediction for m in forecasting_models
            ])
        ):
            raise_log(
                ValueError(
                    "`train_num_samples` is greater than 1 but the `RegressionEnsembleModel` "
                    "contains only deterministic `forecasting_models`."
                ),
                logger,
            )

        supported_reduction = ["mean", "median"]
        if train_samples_reduction is None:
            pass
        elif isinstance(train_samples_reduction, float):
            if not (0.0 < train_samples_reduction < 1.0):
                raise_log(
                    ValueError(
                        f"if a float, `train_samples_reduction` must be between "
                        f"0 and 1, received ({train_samples_reduction})"
                    ),
                    logger,
                )
        elif isinstance(train_samples_reduction, str):
            if train_samples_reduction not in supported_reduction:
                raise_log(
                    ValueError(
                        f"if a string, `train_samples_reduction` must be one of {supported_reduction}, "
                        f"received ({train_samples_reduction})"
                    ),
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

        if train_n_points == -1 and not (
            self.all_trained and (not train_forecasting_models)
        ):
            raise_log(
                ValueError(
                    "`regression_train_n_points` can only be `-1` if `retrain_forecasting_model=False` and "
                    "all `forecasting_models` are already fitted."
                ),
                logger,
            )

        # ensemble model checks
        self.forecasting_models = forecasting_models
        self.ensemble_model = ensemble_model
        self.train_num_samples = train_num_samples
        self.train_samples_reduction = train_samples_reduction
        self.train_forecasting_models = train_forecasting_models
        self.show_warnings = show_warnings
        # regression_train_n_points=-1 is converted to actual n points at fitting time
        self.train_n_points: int = train_n_points

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

    def untrained_model(self):
        model = self.__class__(**copy.deepcopy(self.model_params))
        if not self.train_forecasting_models:
            # torch models drop the underlying network when calling `untrained_model()`;
            # add them back in case the models are not retrained
            for sub_model, sub_model_orig in zip(
                model.forecasting_models, self.forecasting_models
            ):
                if TORCH_AVAILABLE and isinstance(sub_model, TorchForecastingModel):
                    for attr in TFM_ATTRS_NO_PICKLE:
                        setattr(sub_model, attr, getattr(sub_model_orig, attr))
        return model

    @abstractmethod
    def fit(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None = None,
        future_covariates: TimeSeriesLike | None = None,
        verbose: bool | None = None,
    ):
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """

        is_single_series = isinstance(series, TimeSeries)

        # local models OR mix of local and global models
        if not self.is_global_ensemble and not is_single_series:
            raise_log(
                ValueError(
                    "The `forecasting_models` contain at least one LocalForecastingModel, "
                    "which does not support training on multiple series."
                ),
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

        if error_past_cov or error_future_cov:
            raise_log(
                ValueError(
                    "Both series and covariates have to be either single TimeSeries or sequences of TimeSeries."
                ),
                logger,
            )

        self._verify_past_future_covariates(past_covariates, future_covariates)

        # the minimum train series length includes the training requirements from `forecasting_models` as
        # well as the ones from the ensemble model
        min_train_series_length = self.min_train_series_length
        if is_single_series:
            series_too_short = len(series) < min_train_series_length
        else:
            series_too_short = any([len(s) < min_train_series_length for s in series])

        if series_too_short:
            raise_log(
                ValueError(
                    f"{'All time series in ' if not is_single_series else ''}`series` must have "
                    f"a minimum length of `{min_train_series_length}` to fit the model."
                ),
                logger,
            )

        super().fit(series, past_covariates, future_covariates, verbose=verbose)
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

    def _base_model_predict_n(self, n: int) -> int:
        """Minimum prediction horizon for base models to satisfy the ensemble
        (regression) model's future covariate requirements during predict.

        The base model predictions start at ``series.end_time() + (shift + 1) * freq``
        (shifted output). The regression model needs covariates from
        ``series.end_time() + (min(lags) + 1) * freq``. Since all models share the
        same shift, ``min(lags) >= shift``, and we subtract the shift to avoid
        requesting autoregression the base models cannot perform.
        """
        if self.ensemble_model is None:
            return n
        ens_lags = self.ensemble_model.lags["future"]
        base_shift = self.output_chunk_shift
        return (
            max(ens_lags)
            + 1
            - base_shift
            + max(0, n - self.ensemble_model.output_chunk_length)
        )

    def _make_multiple_predictions(
        self,
        n: int,
        series: TimeSeriesLike | None = None,
        past_covariates: TimeSeriesLike | None = None,
        future_covariates: TimeSeriesLike | None = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        random_state: int | None = None,
        verbose: bool | None = None,
    ) -> TimeSeriesLike:
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
                random_state=random_state,
                verbose=verbose,
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
        series: TimeSeriesLike | None = None,
        past_covariates: TimeSeriesLike | None = None,
        future_covariates: TimeSeriesLike | None = None,
        num_samples: int = 1,
        verbose: bool | None = None,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
        random_state: int | None = None,
    ) -> TimeSeriesLike:
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
            random_state=random_state,
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

        base_n = self._base_model_predict_n(n)
        predictions = self._make_multiple_predictions(
            n=base_n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=pred_num_samples,
            predict_likelihood_parameters=forecast_models_pred_likelihood_params,
            random_state=random_state,
            verbose=verbose,
        )

        return self.ensemble(
            predictions,
            series=series,
            n=n,
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
            random_state=random_state,
            verbose=verbose,
        )

    @abstractmethod
    def ensemble(
        self,
        predictions: TimeSeriesLike,
        series: TimeSeriesLike,
        n: int,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        random_state: int | None = None,
        verbose: bool | None = None,
    ) -> TimeSeriesLike:
        """
        Defines how to ensemble the individual models' predictions to produce a single prediction.

        Parameters
        ----------
        predictions
            Individual predictions to ensemble
        series
            Sequence of timeseries to predict on. Optional, since it only makes sense for sequences of timeseries -
            local models retain timeseries for prediction.
        n
            The number of output time steps the ensemble should produce.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        predict_likelihood_parameters
            If set to `True`, the model predicts the parameters of its `likelihood` instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``
        random_state
            Controls the randomness of probabilistic predictions.
        verbose
            Optionally, set the prediction verbosity. Not effective for all models.

        Returns
        -------
        TimeSeries or Sequence[TimeSeries]
            The predicted ``TimeSeries`` or sequence of ``TimeSeries`` obtained by ensembling the individual predictions
        """
        pass

    def _predictions_reduction(self, predictions: TimeSeriesLike) -> TimeSeriesLike:
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
        path: str | os.PathLike | BinaryIO | None = None,
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
        path: str | os.PathLike | BinaryIO,
        pl_trainer_kwargs: dict | None = None,
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
            `Lightning Trainer documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__
            for more information about the supported kwargs.
        **kwargs
            Only effective if the underlying forecasting models contain a `TorchForecastingModel`.
            Additional kwargs for PyTorch Lightning's :func:`LightningModule.load_from_checkpoint()` method,
            For more information, read the `official documentation <https://pytorch-lightning.readthedocs.io/en/stable/
            common/lightning_module.html#load-from-checkpoint>`__.
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
    def min_train_samples(self) -> int:
        train_n_points = abs(self.train_n_points)
        if self.train_forecasting_models:
            # if base models are re-trained, it is the max of the sub-models' min samples + train_n_points
            min_train_samples = (
                max(model.min_train_samples for model in self.forecasting_models)
                + train_n_points
            )
        else:
            # if base models not re-trained, we might already have some training points within the base model's
            # first output chunk; if we need more, we add them as additional required samples
            base_ocl = max(self.extreme_lags[1] + 1, 0)
            min_train_samples = max(train_n_points - base_ocl, 0) + 1
        return min_train_samples

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        extreme_lags = self.extreme_lags
        input_length = abs(extreme_lags[0]) if extreme_lags[0] is not None else 0
        output_length = max(extreme_lags[1] + 1, 0)
        return input_length, output_length

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int,
    ]:
        # the extreme lags are:
        # - min target lag
        # - max target lag
        # - min past covariate lag
        # - max past covariate lag
        # - min future covariate lag
        # - max future covariate lag
        # - output shift

        if self.ensemble_model is not None:
            # use the max of the ensemble model's max target lag
            ft_lag = self.ensemble_model.extreme_lags[1]
        else:
            # or simulate a local forecasting model max target lag
            ft_lag = -1

        # adjust the right-bound lags if the ensemble models has a larger max target lag than the submodels
        extreme_lags_adjusted = defaultdict(list)
        for model in self.forecasting_models:
            model_extreme_lags = model.extreme_lags
            model_ft_lag = model_extreme_lags[1]

            # only adjust global models (model_ft_lag >= 0); local models
            # (model_ft_lag < 0) have no fixed output window and their training
            # data requirements should not be inflated
            ft_lag_diff = max(ft_lag - model_ft_lag, 0) if model_ft_lag >= 0 else 0

            extreme_lags_adjusted[0].append(model_extreme_lags[0])
            extreme_lags_adjusted[1].append(model_extreme_lags[1] + ft_lag_diff)
            extreme_lags_adjusted[2].append(model_extreme_lags[2])
            extreme_lags_adjusted[3].append(
                model_extreme_lags[3] + ft_lag_diff
                if model_extreme_lags[3] is not None
                else None
            )
            extreme_lags_adjusted[4].append(model_extreme_lags[4])
            extreme_lags_adjusted[5].append(
                model_extreme_lags[5] + ft_lag_diff
                if model_extreme_lags[5] is not None
                else None
            )
            extreme_lags_adjusted[6].append(model_extreme_lags[6])

        def find_max_lag_or_none(lag_id, aggregator) -> int | None:
            max_lag = None
            for curr_lag in extreme_lags_adjusted[lag_id]:
                if max_lag is None:
                    max_lag = curr_lag
                elif curr_lag is not None:
                    max_lag = aggregator(max_lag, curr_lag)
            return max_lag

        # extreme lags is given by the min or max of the extreme lags of the sub-models
        return (
            find_max_lag_or_none(0, min),
            find_max_lag_or_none(1, max),
            find_max_lag_or_none(2, min),
            find_max_lag_or_none(3, max),
            find_max_lag_or_none(4, min),
            find_max_lag_or_none(5, max),
            find_max_lag_or_none(6, max),
        )

    @property
    def output_chunk_length(self) -> int | None:
        # either it's the ensemble model's output_chunk_length
        if self.ensemble_model is not None:
            return self.ensemble_model.output_chunk_length

        # or the smallest base model output chunk length
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
    def output_chunk_shift(self) -> int:
        # either use the ensemble model's output shift
        if self.ensemble_model is not None:
            return self.ensemble_model.output_chunk_shift

        # or the output shift of the sub models (enforced to be identical at model creation)
        return self.extreme_lags[6]

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
            likelihood = m.likelihood
            lkl_type = likelihood.type
            models_likelihood.add(lkl_type)

            # check the quantiles
            if lkl_type is LikelihoodType.Quantile:
                quantiles: list[str] = likelihood.quantiles
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
        if past_covariates is not None and not self.supports_past_covariates:
            raise_log(
                ValueError(
                    "`past_covariates` were provided to an `EnsembleModel` but none of its "
                    "`forecasting_models` support such covariates."
                ),
                logger,
            )
        if future_covariates is not None and not self.supports_future_covariates:
            raise_log(
                ValueError(
                    "`future_covariates` were provided to an `EnsembleModel` but none of its "
                    "`forecasting_models` support such covariates."
                ),
                logger,
            )
