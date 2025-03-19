from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.regression_model import (
    RegressionModel,
)


class CategoricalForecastingMixin:
    model: ForecastingModel

    @property
    def class_labels(self):
        """Returns the classes of the classifier model if the model was previously trained."""
        if not hasattr(self.model, "classes_") or self.model.classes_ is None:
            raise AttributeError("Model is not trained")
        return self.model.classes_


class CategoricalModel(RegressionModel, CategoricalForecastingMixin):
    def __init__(self, model=None, **kwargs):
        """Categorical Model
        Can be used to fit any scikit-learn-like classifier class to predict
        categorical target time series from lagged values."
        """
        if model is None:
            model = LogisticRegression(n_jobs=-1)
        if not is_classifier(model):
            raise ValueError(
                "CategoricalModel must be initialized with a classifier model"
            )

        super().__init__(model=model, **kwargs)

    # TODO settings lags should either probably raise a warning about how lags will be assumed to have ordinal meaning
