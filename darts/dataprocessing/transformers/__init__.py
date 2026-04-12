"""
Data Transformers
-----------------

Data transformers for preprocessing time series data, including scalers, missing value fillers,
differencing, BoxCox transformations, and hierarchical reconciliation methods.
"""

from darts.utils._lazy import setup_lazy_imports

_LAZY_IMPORTS: dict[str, str] = {
    "BaseDataTransformer": "darts.dataprocessing.transformers.base_data_transformer",
    "BoxCox": "darts.dataprocessing.transformers.boxcox",
    "Diff": "darts.dataprocessing.transformers.diff",
    "FittableDataTransformer": "darts.dataprocessing.transformers.fittable_data_transformer",
    "InvertibleDataTransformer": "darts.dataprocessing.transformers.invertible_data_transformer",
    "InvertibleMapper": "darts.dataprocessing.transformers.mappers",
    "Mapper": "darts.dataprocessing.transformers.mappers",
    "MIDAS": "darts.dataprocessing.transformers.midas",
    "MissingValuesFiller": "darts.dataprocessing.transformers.missing_values_filler",
    "BottomUpReconciliator": "darts.dataprocessing.transformers.reconciliation",
    "MinTReconciliator": "darts.dataprocessing.transformers.reconciliation",
    "TopDownReconciliator": "darts.dataprocessing.transformers.reconciliation",
    "Scaler": "darts.dataprocessing.transformers.scaler",
    "StaticCovariatesTransformer": "darts.dataprocessing.transformers.static_covariates_transformer",
    "WindowTransformer": "darts.dataprocessing.transformers.window_transformer",
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
