import sys
import logging
from datetime import datetime
import pathlib

def get_logger(name):
    logger = logging.getLogger(name)
    stderr_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s | %(message)s')
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    return logger

def assert_log(boolean_value, message="", logger=get_logger('main_logger')):
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

def raise_log(exception, logger=get_logger('main_logger')):
    """
    Can be used to replace "raise" when throwing an exception to ensure the logging
    of the exception. After logging it, the exception is raised.
    """
    exception_type = str(type(exception)).split("'")[1]
    message = str(exception)
    logger.error(exception_type + ": " + message)

    raise exception