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

def raise_if_not(condition: bool, message: str = "", logger: logging.Logger = get_logger('main_logger')):
    """
    Checks provided boolean condition and raises a ValueError if it evaluates to False.
    It logs the error to the provided logger before raising it.

    :param condition: The boolean condition to be checked.
    :param message: The message of the ValueError.
    :param logger: The logger instance to log the error message if 'condition' is False.
    """
    if (not condition):
        logger.error("ValueError: " + message)
        raise ValueError(message)

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
    Inspired by: https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    
    :param logger: The logger instance to log the runtime of the function.
    """
    def time_log_helper(method):
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args)
            end_time = time.time()
            duration = int((end_time - start_time) * 1000)

            logger.info(method.__name__ + " function ran for {} milliseconds".format(duration))

            
        return timed

    return time_log_helper