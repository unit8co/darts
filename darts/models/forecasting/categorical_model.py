from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression

from darts.logging import (
    get_logger,
    raise_log,
)
from darts.models.forecasting.regression_model import (
    RegressionModel,
)

logger = get_logger(__name__)


class CategoricalForecastingMixin:
    @property
    def class_(self):
        """Returns the classes of the classifier model if the model was previously trained."""
        if not hasattr(self.model, "classes_") or self.model.classes_ is None:
            raise AttributeError("Model is not trained")
        return self.model.classes_

    def _validate_lags(self, lags):
        if lags is not None and not self._is_target_categorical:
            logger.warning(
                "This model will treat the target `series` data/label "
                "as a numerical feature when taking it as an input."
            )


class CategoricalModel(RegressionModel, CategoricalForecastingMixin):
    def __init__(self, model=None, lags=None, **kwargs):
        """Categorical Model
        Can be used to fit any scikit-learn-like classifier class to predict
        categorical target time series from lagged values."
        """
        model = model if model is not None else LogisticRegression(n_jobs=-1)
        if not is_classifier(model):
            raise_log(
                ValueError(
                    "CategoricalModel must be initialized with a classifier model"
                ),
                logger,
            )

        self._validate_lags(lags)

        super().__init__(model=model, lags=lags, **kwargs)

    @property
    def _is_target_categorical(self) -> bool:
        """ "
        Returns if the target serie will be treated as categorical features when `lags` are provided.
        """
        return False
