"""
Explainability
--------------

Tools for explaining and interpreting forecasting model predictions, including SHAP-based explainers and
model-specific explainability methods.
"""

import importlib

from darts.utils.utils import NotImportedModule

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "ShapExplainabilityResult": ("darts.explainability.explainability_result", None),
    "TFTExplainabilityResult": ("darts.explainability.explainability_result", None),
    "_ExplainabilityResult": ("darts.explainability.explainability_result", None),
    "ShapExplainer": ("darts.explainability.shap_explainer", None),
    "TFTExplainer": ("darts.explainability.tft_explainer", "(Py)Torch"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, optional_dep = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path)
        value = getattr(module, name)
    except (ModuleNotFoundError, ImportError):
        if optional_dep is not None:
            value = NotImportedModule(module_name=optional_dep, warn=False)
        else:
            raise

    globals()[name] = value
    return value


def __dir__():
    return __all__
