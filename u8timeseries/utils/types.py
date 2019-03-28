from typing import NewType, List, Tuple
from ..timeseries import TimeSeries

# The features used by regressive model
RegrFeatures = NewType('RegrFeatures', List[TimeSeries])

# The "dataset" (features + target) used by regressive model
RegrDataset = NewType('RegrDataset', Tuple[RegrFeatures, TimeSeries])