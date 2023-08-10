import logging
import shutil
import tempfile

import pytest


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
    """Sets up a temporary directory that will be dunped after the test module (script) finished."""
    temp_work_dir = tempfile.mkdtemp(prefix="darts")
    yield temp_work_dir
    shutil.rmtree(temp_work_dir)


@pytest.fixture(scope="session")
def tfm_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
    }
