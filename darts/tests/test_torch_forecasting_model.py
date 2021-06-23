from .base_test_class import DartsBaseTestClass
from ..logging import get_logger

logger = get_logger(__name__)

try:
    from ..models.rnn_model import _RNNModule, RNNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. RNN tests will be skipped.')
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TestTorchForecastingModel(DartsBaseTestClass):

        def test_create_instance_new_model_no_name_set(self):
            False

        def test_create_instance_new_model(self):
            False

        def test_create_instance_existing_model(self):
            False

        def test_load(self):
            # immutable params
            # mutable params
            # new model
            False

        def test_load_checkpoint_folder_does_not_exist(self):
            False
