import sys
import logging
from datetime import datetime
import pathlib


main_logger = logging.getLogger('main_logger')
stderr_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(name)s | %(levelname)s | %(message)s')
stderr_handler.setFormatter(formatter)
main_logger.addHandler(stderr_handler)

def assert_log(boolean_value, message="", logger=logging.getLogger('main_logger')):
    """
    Can be used to replace assert statements to make sure the error message is logged.
    This method was chosen in favor of overriding sys.excepthook because sys.excepthook
    cannot be overridden in Jupyter Notebook.
    """
    try:
        assert boolean_value, message
    except AssertionError:
        logger.error("AssertionError: " + message)
        raise
