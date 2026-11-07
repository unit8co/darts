import errno
import logging
import os
import re
import threading

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from darts import TimeSeries
from darts.logging import (
    SuppressStdoutStderr,
    get_logger,
    raise_log,
    time_log,
)


@pytest.fixture(scope="module", autouse=True)
def setup_logging():
    logging.disable(logging.NOTSET)


def test_raise_log():
    exception_was_raised = False
    with LogCapture() as lc:
        logger = get_logger(__name__)
        logger.handlers = []
        try:
            raise_log(Exception("test"))
        except Exception:
            exception_was_raised = True

    # testing correct log message
    lc.check((__name__, "ERROR", "Exception: test"))

    # checking whether exception was properly raised
    assert exception_was_raised


def test_timeseries_constructor_error_log(caplog):
    # test assert error log when trying to construct a TimeSeries that is too short
    array_4d = np.zeros((10, 3, 1, 1))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as exc:
            _ = TimeSeries.from_values(values=array_4d)

    message_expected = (
        "TimeSeries require a `values` array that has or can be expanded to 3 "
        "dimensions (('time', 'component', 'sample'))."
    )
    assert str(exc.value) == message_expected
    assert f"ValueError: {message_expected}" in caplog.text


def test_timeseries_split_error_log():
    # test raised error log that occurs when trying to split TimeSeries at a point outside of the time index range
    times = pd.date_range(start="2000-01-01", periods=3, freq="D")
    values = np.array(range(3))
    ts = TimeSeries.from_times_and_values(times, values)
    with LogCapture() as lc:
        get_logger("darts.timeseries").handlers = []
        try:
            ts.split_after(pd.Timestamp("2020-02-01"))
        except Exception:
            pass

    lc.check((
        "darts.timeseries",
        "ERROR",
        "ValueError: Timestamp must be between 2000-01-01 00:00:00 and 2000-01-03 00:00:00",
    ))


def test_time_log():
    logger = get_logger(__name__)
    logger.handlers = []

    @time_log(logger)
    def _my_timed_fn():
        # do something for some time
        for _ in range(2):
            pass

    with LogCapture() as lc:
        _my_timed_fn()

    logged_message = lc.records[-1].getMessage()
    assert (
        re.fullmatch(
            "_my_timed_fn function ran for [0-9]+ milliseconds", logged_message
        )
        is not None
    )


class _PipeOnFd1:
    """
    Point file descriptor 1 at the write end of a fresh pipe, so that
    whatever a test writes to fd 1 can be read back afterwards. The saved
    stream is restored on exit, and the pipe is only readable once fd 1
    points elsewhere again.
    """

    def __enter__(self):
        self.read_fd, write_fd = os.pipe()
        self.saved_fd = os.dup(1)
        os.dup2(write_fd, 1)
        os.close(write_fd)
        return self

    def __exit__(self, *_):
        os.dup2(self.saved_fd, 1)
        os.close(self.saved_fd)

    def drain(self):
        data = b""
        while True:
            chunk = os.read(self.read_fd, 4096)
            if not chunk:
                break
            data += chunk
        os.close(self.read_fd)
        return data


def test_suppress_stdout_stderr_round_trip():
    with _PipeOnFd1() as pipe:
        with SuppressStdoutStderr():
            os.write(1, b"suppressed")
        os.write(1, b"restored")
    assert pipe.drain() == b"restored"


def test_suppress_stdout_stderr_threads_no_permanent_silencing():
    """
    Two threads use the context manager with overlapping lifetimes: the
    first one to enter is still inside the redirection when the second one
    enters, and the second one is the last to leave. After both are done,
    fd 1 must still point at the real stream; with the pre-fix behaviour
    the second context saved the devnull descriptors as if they were the
    real ones and silenced the process for good (#3276).
    """
    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()
    b_exited = threading.Event()

    def enter_and_exit(entered, may_exit, exited):
        with SuppressStdoutStderr():
            entered.set()
            assert may_exit.wait(timeout=30)
        exited.set()

    with _PipeOnFd1() as pipe:
        thread_a = threading.Thread(
            target=enter_and_exit, args=(a_entered, b_entered, a_exited)
        )
        thread_b = threading.Thread(
            target=enter_and_exit, args=(b_entered, a_exited, b_exited)
        )

        thread_a.start()
        assert a_entered.wait(timeout=30)
        thread_b.start()
        assert b_entered.wait(timeout=30)
        thread_a.join(timeout=30)
        thread_b.join(timeout=30)

        assert not thread_a.is_alive() and not thread_b.is_alive()
        os.write(1, b"survivor")

    assert pipe.drain() == b"survivor"


def test_suppress_stdout_stderr_nested_only_outer_restores():
    with _PipeOnFd1() as pipe:
        with SuppressStdoutStderr():
            os.write(1, b"outer")
            with SuppressStdoutStderr():
                os.write(1, b"inner")
            # leaving the inner context must not restore the descriptors
            # while the outer one is still active
            os.write(1, b"between")
        os.write(1, b"after")
    assert pipe.drain() == b"after"


def test_suppress_stdout_stderr_falls_back_without_real_streams(monkeypatch):
    """
    In interpreters without usable standard streams (pythonw on Windows,
    GUI-embedded hosts) os.dup on fd 1/2 raises. The context must degrade
    to a no-op instead of crashing, must not leak the descriptors opened
    before the failure, and must not leave the shared state corrupted for
    the next user.
    """
    opened_fds = []
    real_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened_fds.append(fd)
        return fd

    def bad_dup(fd):
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "dup", bad_dup)

    with SuppressStdoutStderr():
        pass

    monkeypatch.undo()

    for fd in opened_fds:
        with pytest.raises(OSError):
            os.fstat(fd)

    with _PipeOnFd1() as pipe:
        with SuppressStdoutStderr():
            os.write(1, b"suppressed")
        os.write(1, b"restored")
    assert pipe.drain() == b"restored"
