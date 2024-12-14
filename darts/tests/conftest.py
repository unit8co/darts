import logging
import os
import shutil
import tempfile

import pytest

from darts.logging import get_logger

logger = get_logger(__name__)

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - Some tests will be skipped.")
    TORCH_AVAILABLE = False

tfm_kwargs = {
    "pl_trainer_kwargs": {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
}


@pytest.fixture(scope="session", autouse=True)
def set_up_tests(request):
    logging.disable(logging.CRITICAL)

    def tear_down_tests():
        try:
            shutil.rmtree(".darts")
        except FileNotFoundError:
            pass

    request.addfinalizer(tear_down_tests)


@pytest.fixture(scope="module")
def tmpdir_module():
    """Sets up and moves into a temporary directory that will be deleted after the test module (script) finished."""
    temp_work_dir = tempfile.mkdtemp(prefix="darts")
    # remember origin
    cwd = os.getcwd()
    # move to temp dir
    os.chdir(temp_work_dir)
    # go into test with temp dir as input
    yield temp_work_dir
    # move back to origin
    shutil.rmtree(temp_work_dir)
    # remove temp dir
    os.chdir(cwd)


@pytest.fixture(scope="function")
def tmpdir_fn():
    """Sets up and moves into a temporary directory that will be deleted after the test function finished."""
    temp_work_dir = tempfile.mkdtemp(prefix="darts")
    # remember origin
    cwd = os.getcwd()
    # move to temp dir
    os.chdir(temp_work_dir)
    # go into test with temp dir as input
    yield temp_work_dir
    # move back to origin
    os.chdir(cwd)
    # remove temp dir
    shutil.rmtree(temp_work_dir)
