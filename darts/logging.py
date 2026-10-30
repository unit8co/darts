import logging
import os
import sys
import threading
import time
import warnings
from contextlib import suppress
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

    The redirection targets the process-level file descriptors 1 and 2, so
    instances used from different threads share a single redirection: the
    first context to enter saves the real descriptors and the last one to
    leave restores them, and a context entered while a redirection is
    already active never mistakes the redirected descriptors for the real
    ones. When the standard streams are not backed by usable file
    descriptors (e.g. under pythonw on Windows), the context degrades to
    doing nothing instead of raising.

    source:
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    """

    _lock = threading.Lock()
    _depth = 0
    _null_fds: tuple[int, int] | None = None
    _saved_fds: tuple[int, int] | None = None

    def __init__(self):
        # The descriptors are only touched on __enter__ / __exit__, so that
        # creating the context manager neither fails nor leaks descriptors
        # when the standard streams are not backed by real files.
        self._active = False

    def __enter__(self):
        with SuppressStdoutStderr._lock:
            if SuppressStdoutStderr._depth > 0:
                # Someone else already redirected the descriptors; joining
                # the active redirection must not overwrite the saved ones.
                SuppressStdoutStderr._depth += 1
                self._active = True
                return self

            null_a = null_b = saved_a = saved_b = -1
            try:
                null_a = os.open(os.devnull, os.O_RDWR)
                null_b = os.open(os.devnull, os.O_RDWR)
                saved_a = os.dup(1)
                saved_b = os.dup(2)
            except OSError:
                # No usable standard streams (e.g. windowed GUI
                # interpreters): nothing to save, nothing to restore.
                # Close anything opened before the failure so it does not
                # leak, and leave the redirection untouched.
                for fd in (null_a, null_b, saved_a):
                    if fd >= 0:
                        with suppress(OSError):
                            os.close(fd)
                return self

            os.dup2(null_a, 1)
            os.dup2(null_b, 2)
            SuppressStdoutStderr._null_fds = (null_a, null_b)
            SuppressStdoutStderr._saved_fds = (saved_a, saved_b)
            SuppressStdoutStderr._depth = 1
            self._active = True
        return self

    def __exit__(self, *_):
        with SuppressStdoutStderr._lock:
            if not self._active:
                return
            self._active = False
            SuppressStdoutStderr._depth -= 1
            if SuppressStdoutStderr._depth == 0:
                saved_fds = SuppressStdoutStderr._saved_fds
                null_fds = SuppressStdoutStderr._null_fds
                if saved_fds is not None and null_fds is not None:
                    # Re-assign the real stdout/stderr back to (1) and (2)
                    os.dup2(saved_fds[0], 1)
                    os.dup2(saved_fds[1], 2)
                    # Close all file descriptors
                    for fd in saved_fds + null_fds:
                        with suppress(OSError):
                            os.close(fd)
                    SuppressStdoutStderr._saved_fds = None
                    SuppressStdoutStderr._null_fds = None


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
