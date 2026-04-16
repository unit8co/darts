import importlib.util
import logging
import os
import shutil
import tempfile
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from packaging import version

from darts.logging import get_logger

logger = get_logger(__name__)

PANDAS_30_OR_GREATER = version.parse(pd.__version__) >= version.parse("3.0.0")


def _package_available(*names: str) -> bool:
    return all(importlib.util.find_spec(n) is not None for n in names)


TORCH_AVAILABLE = _package_available("torch")
GBM_AVAILABLE = _package_available("catboost", "lightgbm", "xgboost")
XGB_AVAILABLE = _package_available("xgboost")
LGBM_AVAILABLE = _package_available("lightgbm")
CB_AVAILABLE = _package_available("catboost")
PROPHET_AVAILABLE = _package_available("prophet")
SF_AVAILABLE = _package_available("statsforecast")
NF_AVAILABLE = _package_available("neuralforecast")
ONNX_AVAILABLE = _package_available("onnx", "onnxruntime")
OPTUNA_AVAILABLE = _package_available("optuna")
RAY_AVAILABLE = _package_available("ray")
POLARS_AVAILABLE = _package_available("polars")
PLOTLY_AVAILABLE = _package_available("plotly")
IPYTHON_AVAILABLE = _package_available("IPython")

tfm_kwargs: dict[str, Any] = {
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


@pytest.fixture(scope="function")
def mpl_safe_plotting():
    """Patches plt.show() and closes all plots / figures from memory at the end of the test."""
    import matplotlib.pyplot as plt

    with patch("matplotlib.pyplot.show") as patched_show:
        yield patched_show
    plt.close("all")
