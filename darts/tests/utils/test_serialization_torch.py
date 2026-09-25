from unittest.mock import MagicMock, patch

import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.torch import GaussianLikelihood, TorchLikelihood
from darts.utils.serialization.base import dedupe_by_identity
from darts.utils.serialization.torch import (
    DartsCheckpointIO,
    likelihood_safe_globals,
    load_torch_safely,
    safe_globals_for_checkpoint,
)


class _UntrustedCheckpointClass:
    """Module-level stand-in for an attacker-chosen class outside trusted packages."""


class TestDedupeByIdentity:
    def test_removes_duplicates_preserving_order(self):
        a, b = object(), object()
        assert dedupe_by_identity([a, b, a, b]) == [a, b]

    def test_empty_input(self):
        assert dedupe_by_identity([]) == []


class TestDartsSafeGlobals:
    def test_includes_likelihood_types(self):
        globals_ = likelihood_safe_globals()
        assert LikelihoodType in globals_
        assert TorchLikelihood in globals_
        assert GaussianLikelihood in globals_

    def test_all_entries_are_classes(self):
        import inspect

        for obj in likelihood_safe_globals():
            assert inspect.isclass(obj)


class TestSafeGlobalsForCheckpoint:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            assert safe_globals_for_checkpoint("/nonexistent/path.ckpt") == []

    def test_allow_lists_referenced_darts_likelihood(self, tmp_path):
        ckpt_path = tmp_path / "likelihood.ckpt"
        torch.save({"likelihood": GaussianLikelihood()}, ckpt_path)

        resolved = safe_globals_for_checkpoint(ckpt_path)
        assert GaussianLikelihood in resolved

    def test_rejects_untrusted_global(self, tmp_path):
        ckpt_path = tmp_path / "evil.ckpt"
        torch.save({"payload": _UntrustedCheckpointClass()}, ckpt_path)

        resolved = safe_globals_for_checkpoint(ckpt_path)
        assert _UntrustedCheckpointClass not in resolved


class TestLoadCkptSafely:
    def test_invokes_load_fn_and_returns_result(self, tmp_path):
        ckpt_path = tmp_path / "dummy.ckpt"
        torch.save({"x": 1}, ckpt_path)
        called = []

        def _load(*args, **kwargs):
            called.append(True)
            return {"loaded": True}

        result = load_torch_safely(_load, ckpt_path)
        assert called == [True]
        assert result == {"loaded": True}

    def test_extra_globals_are_included(self, tmp_path):
        ckpt_path = tmp_path / "dummy.ckpt"
        torch.save({"x": 1}, ckpt_path)
        extra = [object()]

        with patch(
            "darts.utils.serialization.torch.torch.serialization.safe_globals"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            load_torch_safely(
                lambda *args, **kwargs: None, ckpt_path, extra_globals=extra
            )
            allow = mock_ctx.call_args[0][0]
            assert extra[0] in allow


class TestDartsCheckpointIO:
    def test_defaults_weights_only_to_true(self, tmp_path):
        ckpt_path = tmp_path / "test.ckpt"
        torch.save({"state": 0}, ckpt_path)

        io = DartsCheckpointIO()
        with patch.object(
            TorchCheckpointIO, "load_checkpoint", return_value={"ok": True}
        ) as mock_super:
            result = io.load_checkpoint(str(ckpt_path), map_location="cpu")

        assert result == {"ok": True}
        mock_super.assert_called_once()
        assert mock_super.call_args.kwargs["weights_only"] is True

    def test_weights_only_false_skips_safe_wrapper(self, tmp_path):
        ckpt_path = tmp_path / "test.ckpt"
        torch.save({"state": 0}, ckpt_path)

        io = DartsCheckpointIO()
        with patch.object(
            TorchCheckpointIO, "load_checkpoint", return_value={"ok": True}
        ) as mock_super:
            with patch("torch.serialization.safe_globals") as mock_safe:
                result = io.load_checkpoint(
                    str(ckpt_path), map_location="cpu", weights_only=False
                )

        assert result == {"ok": True}
        mock_safe.assert_not_called()
        assert mock_super.call_args.kwargs["weights_only"] is False
