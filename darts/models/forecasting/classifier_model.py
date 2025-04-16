from typing import Optional, Union

from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from darts.logging import (
    get_logger,
    raise_log,
)
from darts.models.forecasting.regression_model import (
    RegressionModel,
    RegressionModelWithCategoricalFeatures,
)
from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.sklearn import (
    _get_classification_likelihood,
)
from darts.utils.utils import ModelType

logger = get_logger(__name__)


class _ForecastingClassifierMixin:
    """
    Mixin for sklearn-like classification forecasting models
    """

    @property
    def classes_(self):
        """Returns the classes of the classifier model if the model was previously trained."""
        if not hasattr(self.model, "classes_") or self.model.classes_ is None:
            raise AttributeError("Model is not trained")
        return self.model.classes_

    def _validate_lags(self, lags, lags_future_covariates, lags_past_covariates):
        super()._validate_lags(
            lags=lags,
            lags_future_covariates=lags_future_covariates,
            lags_past_covariates=lags_past_covariates,
        )

        if lags is not None and not isinstance(
            self, RegressionModelWithCategoricalFeatures
        ):
            logger.warning(
                "This model will treat the target `series` data/label "
                "as a numerical feature when taking it as an input."
            )

    @property
    def _model_type(self) -> ModelType:
        return ModelType.FORECASTING_CLASSIFIER


class SKLearnClassifierModel(_ForecastingClassifierMixin, RegressionModel):
    def __init__(
        self,
        model=None,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, list[int]] = None,
        lags_future_covariates: Union[tuple[int, int], list[int]] = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        add_encoders: Optional[dict] = None,
        random_state: Optional[int] = None,
        likelihood: Optional[str] = LikelihoodType.ClassProbability.value,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
    ):
        """SKLearn Classifier Model

        Can be used to fit any scikit-learn-like classifier class to predict the target time series with categorical
        values from lagged values.

        Parameters
        ----------
        model
            Scikit-learn-like classifier model with ``fit()`` and ``predict()`` methods. Also possible to use model
            that doesn't support multi-label classification for multivariate timeseries, in which case one classifier
            will be used per component in the multivariate series.
            If `None`, defaults to: ``sklearn.linear_model.LogisticRegression(n_jobs=-1)``.
        lags
            Lagged target `series` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`
            corresponds the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `series` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
            This model treats the target `series` as numerical features when lags are provided.
        lags_past_covariates
            Lagged `past_covariates` values used to predict the next time step/s.
            If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,
            where `0` corresponds to the first predicted time step of each sample. If `output_chunk_shift > 0`, then
            lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
            If a list of integers, each value must be < 0. Uses only the specified values as lags.
            If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (integer or list of integers). The
            key 'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        lags_future_covariates
            Lagged `future_covariates` values used to predict the next time step/s. The lags are always relative to the
            first step in the output chunk, even when `output_chunk_shift > 0`.
            If a tuple of `(past, future)`, both values must be > 0. Uses the last `n=past` past lags and `n=future`
            future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0` corresponds the first
            predicted time step of each sample. If `output_chunk_shift > 0`, the position of negative lags differ from
            those of `lags` and `lags_past_covariates`. In this case a future lag `-5` would point at the same
            step as a target lag of `-5 + output_chunk_shift`.
            If a list of integers, uses only the specified values as lags.
            If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when
            using multiple series) and the values correspond to the component lags (tuple or list of integers). The key
            'default_lags' can be used to provide default lags for un-specified components. Raises and error if some
            components are missing and the 'default_lags' key is not provided.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same as forecast
            horizon `n` used in `predict()`, which is the desired number of prediction points generated using a
            one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents auto-regression. This is
            useful when the covariates don't extend far enough into the future, or to prohibit the model from using
            future values of past and / or future covariates for prediction (depending on the model's covariate
            support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input (history of target and past covariates) and
            output. If the model supports `future_covariates`, the `lags_future_covariates` are relative to the first
            step in the shifted output chunk. Predictions will start `output_chunk_shift` steps after the end of the
            target `series`. If `output_chunk_shift` is set, the model cannot generate autoregressive predictions
            (`n > output_chunk_length`).
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        likelihood
            'classprobability' or ``None``. If set to 'classprobability', setting `predict_likelihood_parameters`
            in `predict()` will forecast class probabilities.
            Default: 'classprobability'
        random_state
            Control the randomness in the fitting procedure and for sampling.
            Default: ``None``.
        multi_models
            If True, a separate model will be trained for each future lag to predict. If False, a single model is
            trained to predict at step 'output_chunk_length' in the future. Default: True.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.

        Examples
        --------
        >>> import numpy as np
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import SKLearnClassifierModel
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> series = WeatherDataset().load().resample("1D", method="mean")
        >>> # predicting if it will rain or not
        >>> target =  series['rain (mm)'][:105].map(lambda x: np.where(x > 0, 1, 0))
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 105)
        >>> past_cov = series['T (degC)'][:105]
        >>> # optionally, use future pressure (pretending this component is a forecast)
        >>> future_cov = series['p (mbar)'][:111]
        >>> # predict 6 "will rain" values using the 12 past values of pressure and temperature,
        >>> # as well as the 6 pressure values corresponding to the forecasted period
        >>> model = SKLearnClassifierModel(
        >>>     model=KNeighborsClassifier(),
        >>>     lags=12,
        >>>     lags_past_covariates=12,
        >>>     lags_future_covariates=[0,1,2,3,4,5],
        >>>     output_chunk_length=6
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[0.],
                [0.],
                [0.],
                [1.],
                [1.],
                [0.]])
        """

        model = model if model is not None else LogisticRegression(n_jobs=-1)
        if not is_classifier(model):
            raise_log(
                ValueError(
                    "`SKLearnClassifierModel` must be initialized with a classifier `model`."
                ),
                logger,
            )
        if not hasattr(model, "predict_proba"):
            logger.warning(
                "Set 'probability' to 'True' in SVC to be able to predict class probabilities "
                if isinstance(model) == SVC
                else "Received model has no 'predict_proba' function"
                " this model won't be able to predict class probabilities"
            )

        self._likelihood = _get_classification_likelihood(
            likelihood=likelihood,
            n_outputs=output_chunk_length if multi_models else 1,
            random_state=random_state,
        )

        super().__init__(
            model=model,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            add_encoders=add_encoders,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
        )
