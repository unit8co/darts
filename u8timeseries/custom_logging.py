import sys
import logging
from datetime import datetime
import pathlib
import time

def get_logger(name):
    """
    Internally calls the logging.getLogger function with the 'name' argument to create or 
    retrieve a logger object. It is recommended to pass __name__ as argument when calling 
    get_logger. The returned logger object logs to the standard error stream and formats
    the messages appropriately.

    :param name: The name that gets passed to the logger.getLogger function.
    :return: A logger instance with the given name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stderr_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s | %(message)s')
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    return logger

def assert_log(boolean_value: bool, message: str = "", logger: logging.Logger = get_logger('main_logger')):
    """
    Can be used to replace assert statements to make sure the error message is logged.
    This method was chosen in favor of overriding sys.excepthook because sys.excepthook
    cannot be overridden in Jupyter Notebook.

    :param boolean_value: The boolean expression (or value) of the assert statement.
    :param message: The message of the assert statement.
    :param logger: The logger instance to log the error message if the boolean_value is False.
    """
    try:
        assert boolean_value, message
    except AssertionError:
        logger.error("AssertionError: " + message)
        raise

def raise_log(exception: Exception, logger: logging.Logger = get_logger('main_logger')):
    """
    Can be used to replace "raise" when throwing an exception to ensure the logging
    of the exception. After logging it, the exception is raised.

    :param exception: The exception instance to be raised.
    :param logger: The logger instance to log the exception type and message.
    """
    exception_type = str(type(exception)).split("'")[1]
    message = str(exception)
    logger.error(exception_type + ": " + message)

    raise exception

def time_log(logger: logging.Logger = get_logger('main_logger')):
    """
    A decorator function that logs the runtime of the function it is decorating
    to the logger object that is taken as an argument.

    :param logger: The logger instance to log the runtime of the function.
    """
    def time_log_helper(method):
        def timed(*args):
            start_time = time.time()
            result = method(*args)
            end_time = time.time()
            duration = int((end_time - start_time) * 1000)

            logger.info(method.__name__ + " function ran for {} milliseconds".format(duration))

            
        return timed

    return time_log_helper