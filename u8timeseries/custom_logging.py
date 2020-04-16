import sys
import logging
from datetime import datetime
import pathlib
import time

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.INFO)
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

def time_log(logger = get_logger('main_logger')):
    def time_log_helper(method):
        def timed(*args):
            start_time = time.time()
            result = method(*args)
            end_time = time.time()
            duration = int((end_time - start_time) * 1000)

            logger.info(method.__name__ + " function ran for {} milliseconds".format(duration))

            
        return timed

    return time_log_helper