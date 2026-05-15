"""
Data Transformers
-----------------

Data transformers for preprocessing time series data, including scalers, missing value fillers,
differencing, BoxCox transformations, and hierarchical reconciliation methods.
"""

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.dataprocessing.transformers.base_data_transformer import (
        BaseDataTransformer as BaseDataTransformer,
    )
    from darts.dataprocessing.transformers.boxcox import BoxCox as BoxCox
    from darts.dataprocessing.transformers.diff import Diff as Diff
    from darts.dataprocessing.transformers.fittable_data_transformer import (
        FittableDataTransformer as FittableDataTransformer,
    )
    from darts.dataprocessing.transformers.invertible_data_transformer import (
        InvertibleDataTransformer as InvertibleDataTransformer,
    )
    from darts.dataprocessing.transformers.mappers import (
        InvertibleMapper as InvertibleMapper,
    )
    from darts.dataprocessing.transformers.mappers import Mapper as Mapper
    from darts.dataprocessing.transformers.midas import MIDAS as MIDAS
    from darts.dataprocessing.transformers.missing_values_filler import (
        MissingValuesFiller as MissingValuesFiller,
    )
    from darts.dataprocessing.transformers.reconciliation import (
        BottomUpReconciliator as BottomUpReconciliator,
    )
    from darts.dataprocessing.transformers.reconciliation import (
        MinTReconciliator as MinTReconciliator,
    )
    from darts.dataprocessing.transformers.reconciliation import (
        TopDownReconciliator as TopDownReconciliator,
    )
    from darts.dataprocessing.transformers.scaler import Scaler as Scaler
    from darts.dataprocessing.transformers.static_covariates_transformer import (
        StaticCovariatesTransformer as StaticCovariatesTransformer,
    )
    from darts.dataprocessing.transformers.window_transformer import (
        WindowTransformer as WindowTransformer,
    )

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
