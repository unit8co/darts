from darts.logging import get_logger, raise_log

logger = get_logger(__name__)

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NotImportedModule:
    """Helper class for handling import errors of optional dependencies."""

    usable = False

    def __init__(self, module_name: str, warn: bool = True):
        self.error_message = (
            f"The `{module_name}` module could not be imported. "
            f"To enable {module_name} support in Darts, follow the detailed "
            f"instructions in the installation guide: "
            f"https://github.com/unit8co/darts/blob/master/INSTALL.md"
        )
        if warn:
            logger.warning(self.error_message)

    def __call__(self, *args, **kwargs):
        raise_log(ImportError(self.error_message), logger=logger)
