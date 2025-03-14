from sklearn.base import is_classifier

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.regression_model import (
    RegressionModel,
)


class CategoricalForecastingMixin:
    model: ForecastingModel

    @property
    def class_labels(self):
        """Returns the classes of the classifier model if the model was previously trained."""
        if not hasattr(self.model, "classes_"):
            raise AttributeError("Model is not trained")
        return self.model.classes_


class CategoricalModel(RegressionModel, CategoricalForecastingMixin):
    def __init__(self, model=None, **kwargs):
        """
        TODO explain categorical input not supported yet
        supposed to use covariates as input and target as output
        """
        if not is_classifier(model):
            raise ValueError(
                "CategoricalModel must be initialized with a classifier model"
            )

        super().__init__(model=model, **kwargs)
