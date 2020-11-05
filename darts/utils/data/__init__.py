from ...logging import get_logger
logger = get_logger(__name__)

try:
    from .torch_timeseries_datasets import HorizonBasedTrainDataset
except ModuleNotFoundError:
    logger.warning("Support for Torch not available. To enable it, install u8darts[torch] or u8darts[all].")
