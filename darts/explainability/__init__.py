"""
Explainability
--------------

Tools for explaining and interpreting forecasting model predictions, including SHAP-based explainers and
model-specific explainability methods.

`SHAP <https://github.com/slundberg/shap>`__-Based Explainers
-------------------------------------------------------------
- :class:`~darts.explainability.sklearn_explainer.SKLearnExplainer`: SHAP-based explainer for SKLearn models.
- :class:`~darts.explainability.torch_explainer.TorchExplainer`: SHAP-based explainer for PyTorch models.

Model-Specific Explainers
-------------------------
- :class:`~darts.explainability.tft_explainer.TFTExplainer`: Explainer for
  :class:`TFTModel <darts.models.forecasting.tft_model.TFTModel>`.

"""

from darts.explainability.explainability_result import (
    SHAPExplainabilityResult,
    SHAPSingleExplainabilityResult,
    TFTExplainabilityResult,
)
from darts.explainability.sklearn_explainer import SKLearnExplainer
from darts.logging import get_logger
from darts.utils.utils import NotImportedModule

logger = get_logger(__name__)
try:
    from darts.explainability.tft_explainer import TFTExplainer
    from darts.explainability.torch_explainer import TorchExplainer
except ModuleNotFoundError:
    logger.warning(
        "Support for Torch based explainers not available. "
        'To enable them, install "darts[torch]" or "darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )
    TFTExplainer = NotImportedModule(module_name="(Py)Torch", warn=False)
    TorchExplainer = NotImportedModule(module_name="(Py)Torch", warn=False)

__all__ = [
    "SHAPExplainabilityResult",
    "SHAPSingleExplainabilityResult",
    "TFTExplainabilityResult",
    "SKLearnExplainer",
    "TFTExplainer",
    "TorchExplainer",
]
