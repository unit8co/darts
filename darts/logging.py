import logging
import os
import sys
import threading
import time
import warnings
from typing import NoReturn


def get_logger(name):
    """
    Internally calls the logging.getLogger function with the `name` argument to create or
    retrieve a logger object. It is recommended to pass __name__ as argument when calling
    get_logger. The returned logger object logs to the standard error stream and formats
    the messages appropriately.

    Parameters
    ----------
    name
        The name that gets passed to the logger.getLogger function.

    Returns
    -------
    A logger instance with the given name.
    """

    logger = logging.getLogger(name)

    return logger


def raise_deprecation_warning(
    message: str = "",
    logger: logging.Logger = get_logger("main_logger"),
):
    """
    Raises a DeprecationWarning.

    Parameters
    ----------
    message
        The message of the ValueError.
    logger
        The logger instance to log the error message if 'condition' is False.

    Raises
    ------
    Warning
        DeprecationWarning
    """

    logger.warning("DeprecationWarning: " + message)


def raise_log(exception: Exception) -> NoReturn:
    """
    Can be used to replace "raise" when throwing an exception to ensure the logging
    of the exception. After logging it, the exception is raised.

    The logger is automatically resolved from the calling module's ``__name__``.

    Parameters
    ----------
    exception
        The exception instance to be raised.

    Raises
    ------
    Exception
        The provided exception
    """

    frame = sys._getframe(1)
    module_name = frame.f_globals.get("__name__", "darts")
    logger = logging.getLogger(module_name)

    exception_type = str(type(exception)).split("'")[1]
    message = str(exception)
    logger.error(exception_type + ": " + message)

    raise exception


def time_log(logger: logging.Logger = get_logger("main_logger")):
    """
    A decorator function that logs the runtime of the function it is decorating
    to the logger object that is taken as an argument.
    Inspired by:
    `https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d`_.

    Parameters
    ----------
    logger
        The logger instance to log the runtime of the function.
    """

    def time_log_helper(method):
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            duration = int((end_time - start_time) * 1000)
            logger.info(method.__name__ + f" function ran for {duration} milliseconds")
            return result

        return timed

    return time_log_helper


class SuppressStdoutStderr:
    """
    A context manager for "deep suppression" of stdout and stderr in
    Python, i.e. it suppresses all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This does not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    The redirection applies to the whole process, so nested and concurrent
    (multi-threaded) uses share it: the first context to enter saves the real
    stdout and stderr, and the last one to exit restores them. If stdout or
    stderr has no valid file descriptor, nothing is redirected.

    source:
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    """

    _lock = threading.Lock()
    _depth = 0
    _save_fds: list[int] = []

    def __enter__(self):
        with SuppressStdoutStderr._lock:
            if SuppressStdoutStderr._depth == 0:
                SuppressStdoutStderr._save_fds = self._redirect_to_devnull()
            SuppressStdoutStderr._depth += 1

    def __exit__(self, *_):
        with SuppressStdoutStderr._lock:
            SuppressStdoutStderr._depth -= 1
            if SuppressStdoutStderr._depth == 0:
                # Re-assign the real stdout/stderr back to (1) and (2)
                for target_fd, save_fd in zip((1, 2), SuppressStdoutStderr._save_fds):
                    os.dup2(save_fd, target_fd)
                    os.close(save_fd)
                SuppressStdoutStderr._save_fds = []

    @staticmethod
    def _redirect_to_devnull() -> list[int]:
        # Save the actual stdout (1) and stderr (2) file descriptors.
        save_fds = []
        try:
            for fd in (1, 2):
                save_fds.append(os.dup(fd))
        except OSError:
            for fd in save_fds:
                os.close(fd)
            return []
        # Assign a null file to stdout and stderr.
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        return save_fds


def execute_and_suppress_output(function, logger, suppression_threshold_level, *args):
    """
    This function conditionally executes the given function with the given arguments
    based on whether the current level of 'logger' is below, above or equal to
    'suppression_threshold_level'. In the latter two cases, all printouts to stdout
    and stderr will be suppressed.

    :param function: Function whose printouts will conditionally be suppressed.
    :param logger: Logger whose level will be checked.
    :param suppression_threshold_level: Minimum logger level triggering suppression.
    :param *args: Arguments passed to 'function'.
    :return: Outputs of 'function'.
    """
    if logger.level >= suppression_threshold_level:
        with SuppressStdoutStderr():
            return_value = function(*args)
    else:
        return_value = function(*args)
    return return_value


def suppress_lightning_warnings(suppress_all: bool = False):
    warnings.filterwarnings(
        "ignore", ".*You defined a `validation_step` but have no `val_dataloader`.*"
    )
    if suppress_all:
        warnings.filterwarnings(
            "ignore", ".*does not have many workers which may be a bottleneck.*"
        )
        warnings.filterwarnings(
            "ignore",
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
        )
