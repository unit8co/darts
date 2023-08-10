# import logging
import shutil
import tempfile

import pytest

# import time


# Print something for all tests taking longer than this
DURATION_THRESHOLD = 2.0
# prevent PyTorch Lightning from using GPU (M1 system compatibility)
tfm_kwargs = {"pl_trainer_kwargs": {"accelerator": "cpu"}}


@pytest.fixture(scope="module")
def tmpdir_module():
    temp_work_dir = tempfile.mkdtemp(prefix="darts")
    yield temp_work_dir
    shutil.rmtree(temp_work_dir)
