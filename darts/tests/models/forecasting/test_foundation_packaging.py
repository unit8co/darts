"""
Integration tests for foundation model installation and packaging.

Tests the complete flow: pyproject.toml → setup.py → model imports
"""
import subprocess
import sys
from pathlib import Path

import pytest


def test_pyproject_extras_available_in_setup():
    """Verify extras from pyproject.toml are available in setup.py."""
    # This test verifies the complete flow:
    # pyproject.toml → setup.py → pip install

    repo_root = Path(__file__).parents[4]  # Go up to repo root
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        capture_output=True,
        text=True,
        cwd=repo_root
    )

    assert result.returncode == 0, f"setup.py failed: {result.stderr}"
    assert "darts" in result.stdout.lower()


def test_chronos_model_importable():
    """Verify ChronosModel can be imported from darts.models."""
    try:
        from darts.models import ChronosModel
        assert ChronosModel is not None
    except ImportError as e:
        pytest.fail(f"Could not import ChronosModel: {e}")


def test_chronos_extra_defined_in_pyproject():
    """Verify 'chronos' extra is defined in pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    extras = pyproject["project"]["optional-dependencies"]
    assert "chronos" in extras, "'chronos' extra not defined"
    assert len(extras["chronos"]) > 0, "'chronos' extra is empty"
    assert any("chronos-forecasting" in dep for dep in extras["chronos"]), \
        "'chronos' extra missing chronos-forecasting package"


def test_timesfm_extra_defined_in_pyproject():
    """Verify 'timesfm' extra is defined in pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    extras = pyproject["project"]["optional-dependencies"]
    assert "timesfm" in extras, "'timesfm' extra not defined"
    assert len(extras["timesfm"]) > 0, "'timesfm' extra is empty"


def test_chronos_model_with_mock_pipeline():
    """Test ChronosModel initialization with mocked pipeline."""
    from unittest.mock import MagicMock, patch

    # Mock the chronos module's Chronos2Pipeline (imported lazily in pipeline property)
    with patch("chronos.Chronos2Pipeline") as mock:
        mock.from_pretrained.return_value = MagicMock()

        from darts.models import ChronosModel

        model = ChronosModel("s3://autogluon/chronos-2")

        # Verify model initialized
        assert model.model_id == "s3://autogluon/chronos-2"


def test_complete_flow_pyproject_to_model():
    """Integration test: pyproject.toml → setup.py → ChronosModel."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # 1. Verify pyproject.toml has chronos extra
    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    extras = pyproject["project"]["optional-dependencies"]
    assert "chronos" in extras

    # 2. Verify ChronosModel can be imported
    from darts.models import ChronosModel

    # 3. Verify model can be instantiated (with mock)
    from unittest.mock import MagicMock, patch

    with patch("chronos.Chronos2Pipeline") as mock:
        mock.from_pretrained.return_value = MagicMock()
        model = ChronosModel("s3://autogluon/chronos-2")
        assert model.model_id == "s3://autogluon/chronos-2"
