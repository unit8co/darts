from darts.logging import get_logger

logger = get_logger(__name__)


class NotImportedModule:
    """Helper class for handling import errors of optional dependencies."""

    usable = False

    def __init__(self, module_name: str, warn: bool = True):
        if warn:
            txt = (
                f"The {module_name} module could not be imported. "
                "To enable LightGBM support in Darts, follow the detailed "
                "install instructions for LightGBM in the README: "
                "https://github.com/unit8co/darts/blob/master/INSTALL.md"
            )
            logger.warning(txt)
