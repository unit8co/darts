"""
Explainability
--------------

Tools for explaining and interpreting forecasting model predictions, including SHAP-based explainers and
model-specific explainability methods.

`SHAP <https://github.com/slundberg/shap>`__-Based Explainers
-------------------------------------------------------------
- :class:`~darts.explainability.shap_explainer.ShapExplainer`: SHAP-based explainer for Darts' SKLearn
  and Torch Models.

Model-Specific Explainers
-------------------------
- :class:`~darts.explainability.tft_explainer.TFTExplainer`: Explainer for
  :class:`TFTModel <darts.models.forecasting.tft_model.TFTModel>`.

"""

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.explainability.explainability_result import (
        ShapExplainabilityResult as ShapExplainabilityResult,
    )
    from darts.explainability.explainability_result import (
        ShapSingleExplainabilityResult as ShapSingleExplainabilityResult,
    )
    from darts.explainability.explainability_result import (
        TFTExplainabilityResult as TFTExplainabilityResult,
    )
    from darts.explainability.shap_explainer import ShapExplainer as ShapExplainer
    from darts.explainability.tft_explainer import TFTExplainer as TFTExplainer

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "ShapExplainabilityResult": ("darts.explainability.explainability_result", None),
    "ShapSingleExplainabilityResult": (
        "darts.explainability.explainability_result",
        None,
    ),
    "TFTExplainabilityResult": ("darts.explainability.explainability_result", None),
    "ShapExplainer": ("darts.explainability.shap_explainer", None),
    "TFTExplainer": ("darts.explainability.tft_explainer", "(Py)Torch"),
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
