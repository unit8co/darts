"""
Explainability
--------------
"""

from darts.logging import get_logger

logger = get_logger(__name__)

from darts.explainability.explainability_result import (
    ShapExplainabilityResult,
    TFTExplainabilityResult,
    _ExplainabilityResult,
)
from darts.explainability.shap_explainer import ShapExplainer

try:
    from darts.explainability.tft_explainer import TFTExplainer
except ModuleNotFoundError:
    logger.warning(
        "Support for Torch based explainers not available. "
        'To enable them, install "darts", "u8darts[torch]" or "u8darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )
