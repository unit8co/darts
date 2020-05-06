import logging
import time
import os


def get_logger(name: str, handler=logging.StreamHandler()):
    """
    Internally calls the logging.getLogger function with the 'name' argument to create or
    retrieve a logger object. It is recommended to pass __name__ as argument when calling
    get_logger. The returned logger object logs to the standard error stream and formats
    the messages appropriately.

    :param name: The name that gets passed to the logger.getLogger function.
    :param handler: The handler instance to be added to the logger.
    :return: A logger instance with the given name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def raise_if_not(condition: bool, message: str = "", logger: logging.Logger = get_logger('main_logger')):
    """
    Checks provided boolean condition and raises a ValueError if it evaluates to False.
    It logs the error to the provided logger before raising it.

    :param condition: The boolean condition to be checked.
    :param message: The message of the ValueError.
    :param logger: The logger instance to log the error message if 'condition' is False.
    """
    if not condition:
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
            _ = method(*args, **kwargs)
            end_time = time.time()
            duration = int((end_time - start_time) * 1000)

            logger.info(method.__name__ + " function ran for {} milliseconds".format(duration))

        return timed

    return time_log_helper


class SuppressStdoutStderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    source: 
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def execute_and_suppress_output(function, logger, suppression_threshold_level, *args):
    if (logger.level >= suppression_threshold_level):
        with SuppressStdoutStderr():
            return_value = function(*args)
    else:
        return_value = function(*args)
    return return_value