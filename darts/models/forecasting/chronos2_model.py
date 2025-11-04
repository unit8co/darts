"""
Time Series Foundation Model (TSFM)
---------------------------------
"""

import os
from typing import Optional, Union

from darts.logging import get_logger
from darts.models.forecasting.foundation_model import (
    FoundationModel,
    HuggingFaceModelMixin,
)
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
)
from darts.utils.data.torch_datasets.utils import TorchTrainingSample

logger = get_logger(__name__)


class _Chronos2Module(PLForecastingModule):
    """
    Chronos2 module
    """


class Chronos2Model(FoundationModel, HuggingFaceModelMixin):
    _repo_id = "amazon/chronos-2"
    _repo_commit = "18128c7b4f3fd286f06d6d4efe1d252f1d2a9a7c"

    def __init__(
        self,
        local_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        self.local_dir = local_dir

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        module = self._load_model(
            _Chronos2Module,
        )
        return module
