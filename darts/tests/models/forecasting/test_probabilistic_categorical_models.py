import numpy as np
import pytest

from darts.logging import get_logger
from darts.models.forecasting.catboost_model import CatBoostCategoricalModel
from darts.models.forecasting.lgbm import LightGBMCategoricalModel
from darts.models.forecasting.xgboost import XGBCategoricalModel
from darts.utils import timeseries_generation as tg
from darts.utils.utils import NotImportedModule

lgbm_available = not isinstance(LightGBMCategoricalModel, NotImportedModule)
cb_available = not isinstance(CatBoostCategoricalModel, NotImportedModule)

logger = get_logger(__name__)


class TestProbabilisticCategoricalModels:
    np.random.seed(42)
    # shift sines to positive values so that they can be used as target for classification with classes [0, 1, 2]
    sine_univariate1 = tg.sine_timeseries(length=100) + 1
    sine_univariate2 = tg.sine_timeseries(length=100, value_phase=1.5705) + 1
    sine_univariate3 = tg.sine_timeseries(length=100, value_phase=0.78525) + 1
    sine_univariate4 = tg.sine_timeseries(length=100, value_phase=0.392625) + 1
    sine_univariate5 = tg.sine_timeseries(length=100, value_phase=0.1963125) + 1
    sine_univariate6 = tg.sine_timeseries(length=100, value_phase=0.09815625) + 1

    sine_univariate1_cat = sine_univariate1.map(lambda x: np.round(x))
    sine_univariate2_cat = sine_univariate2.map(lambda x: np.round(x))
    sine_univariate3_cat = sine_univariate3.map(lambda x: np.round(x))

    sine_multivariate1_cat = sine_univariate1_cat.stack(sine_univariate2_cat)
    sine_multivariate2 = sine_univariate2.stack(sine_univariate3)

    sine_multiseries1_cat = [
        sine_univariate1_cat,
        sine_univariate2_cat,
        sine_univariate3_cat,
    ]
    sine_multiseries2 = [sine_univariate4, sine_univariate5, sine_univariate6]

    probabilistic_classifiers = [
        (
            XGBCategoricalModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "max_leaves": 1,
                "random_state": 42,
            },
        ),
    ]
    probabilistic_accuracies = [1]
    probabilistic_multioutput = [False]

    if lgbm_available:
        probabilistic_classifiers.append((
            LightGBMCategoricalModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "num_leaves": 2,
                "verbosity": -1,
                "random_state": 42,
            },
        ))
        probabilistic_accuracies.append(1)
        probabilistic_multioutput.append(False)

    if cb_available:
        probabilistic_classifiers.append((
            CatBoostCategoricalModel,
            {
                "iterations": 1,
                "depth": 1,
                "verbose": -1,
                "random_state": 42,
            },
        ))
        probabilistic_accuracies.append(1)
        probabilistic_multioutput.append(False)

    @pytest.mark.parametrize(
        "clf_params",
        probabilistic_classifiers,
    )
    def test_class_proba_likelihood_median_pred_is_same_than_no_likelihood(
        self, clf_params
    ):
        clf, kwargs = clf_params
        model = clf(lags=2, **kwargs)
        model_likelihood = clf(
            lags=2,
            likelihood="classprobability",
            **kwargs,
        )
        model.fit(self.sine_univariate1_cat)
        model_likelihood.fit(self.sine_univariate1_cat)

        assert model_likelihood.predict(5) == model.predict(5)
