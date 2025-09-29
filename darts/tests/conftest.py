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

try:
    import catboost
    import lightgbm
    import xgboost  # noqa: F401

    GBM_AVAILABLE = True
except ImportError:
    logger.warning(
        "Gradient Boosting Models not installed - Some tests will be skipped."
    )
    GBM_AVAILABLE = False

try:
    import xgboost  # noqa: F401

    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed - Some tests will be skipped.")
    XGB_AVAILABLE = False

try:
    import lightgbm  # noqa: F401

    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not installed - Some tests will be skipped.")
    LGBM_AVAILABLE = False

try:
    import catboost  # noqa: F401

    CB_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not installed - Some tests will be skipped.")
    CB_AVAILABLE = False

try:
    import prophet  # noqa: F401

    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not installed - Some tests will be skipped.")
    PROPHET_AVAILABLE = False

try:
    import statsforecast  # noqa: F401

    SF_AVAILABLE = True
except ImportError:
    logger.warning("StatsForecast not installed - Some tests will be skipped.")
    SF_AVAILABLE = False

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("Onnx not installed - Some tests will be skipped.")
    ONNX_AVAILABLE = False

try:
    import optuna  # noqa: F401

    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not installed - Some tests will be skipped.")
    OPTUNA_AVAILABLE = False

try:
    import ray  # noqa: F401

    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not installed - Some tests will be skipped.")
    RAY_AVAILABLE = False

try:
    import polars  # noqa: F401

    POLARS_AVAILABLE = True
except ImportError:
    logger.warning("Polars not installed - Some tests will be skipped.")
    POLARS_AVAILABLE = False

tfm_kwargs = {
    "pl_trainer_kwargs": {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
}

tfm_kwargs_dev = {
    "pl_trainer_kwargs": {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "fast_dev_run": True,
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
