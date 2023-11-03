from typing import Any, Dict, List

from darts.logging import get_logger, raise_log

logger = get_logger(__name__)


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


def _check_kwargs_keys(
    param_name: str, kwargs_dict: Dict[str, Any], invalid_keys: List[str]
):
    """Check if the dictionary contain any of the invalid key"""
    invalid_args_passed = set(invalid_keys).intersection(set(kwargs_dict.keys()))
    if len(invalid_args_passed) > 0:
        raise_log(
            f"`{param_name}` can't contain the following parameters : {list(invalid_args_passed)}.",
            logger,
        )
