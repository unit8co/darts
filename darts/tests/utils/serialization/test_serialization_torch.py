from unittest.mock import MagicMock, patch

import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import os

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from torch.utils.cpp_extension import load as cpp_load

from darts.logging import execute_and_suppress_output
from darts.utils.likelihood_models.torch import GaussianLikelihood
from darts.utils.serialization.base import UnpicklingError
from darts.utils.serialization.registry import safe_globals
from darts.utils.serialization.torch import (
    _DartsCheckpointIO,
    _safe_globals_for_torch_file,
    load_ckpt_safely,
    load_wrapper_safely,
)


class _UntrustedCheckpointClass:
    """Module-level stand-in for an attacker-chosen class outside trusted packages."""


class TestSafeGlobalsForCheckpoint:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _safe_globals_for_torch_file("/nonexistent/path.ckpt")

    def test_allow_lists_referenced_darts_likelihood(self, tmp_path):
        ckpt_path = tmp_path / "likelihood.ckpt"
        torch.save({"likelihood": GaussianLikelihood()}, ckpt_path)

        resolved = _safe_globals_for_torch_file(ckpt_path)
        assert GaussianLikelihood in resolved.values()

    def test_rejects_untrusted_global(self, tmp_path):
        ckpt_path = tmp_path / "evil.ckpt"
        torch.save({"payload": _UntrustedCheckpointClass()}, ckpt_path)

        with pytest.raises(UnpicklingError):
            _safe_globals_for_torch_file(ckpt_path)


class TestLoadCkptSafely:
    def test_invokes_load_fn_and_returns_result(self, tmp_path):
        ckpt_path = tmp_path / "dummy.ckpt"
        torch.save({"x": 1}, ckpt_path)
        called = []

        def _load(*args, **kwargs):
            called.append(True)
            return {"loaded": True}

        result = load_ckpt_safely(_load, ckpt_path)
        assert called == [True]
        assert result == {"loaded": True}

    def test_extra_globals_are_included(self, tmp_path):
        ckpt_path = tmp_path / "dummy.ckpt"
        torch.save({"x": 1}, ckpt_path)

        class TrustedClass:
            pass

        extra = [TrustedClass]

        with patch(
            "darts.utils.serialization.torch.torch.serialization.safe_globals"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            with safe_globals(extra):
                load_ckpt_safely(lambda *args, **kwargs: None, ckpt_path)
            allow = mock_ctx.call_args[0][0]
            assert (
                extra[0],
                extra[0].__module__ + "." + extra[0].__qualname__,
            ) in allow

    def test_rejects_weights_only_kwarg(self, tmp_path):
        ckpt_path = tmp_path / "dummy.ckpt"
        torch.save({"x": 1}, ckpt_path)

        with pytest.raises(ValueError, match="weights_only"):
            load_ckpt_safely(
                lambda *args, **kwargs: None,
                ckpt_path,
                weights_only=False,
            )


class TestDartsCheckpointIO:
    def test_safe_default_passes_weights_only_true_to_pl(self, tmp_path):
        ckpt_path = tmp_path / "test.ckpt"
        torch.save({"state": 0}, ckpt_path)

        io = _DartsCheckpointIO()
        with patch.object(
            TorchCheckpointIO, "load_checkpoint", return_value={"ok": True}
        ) as mock_super:
            result = io.load_checkpoint(str(ckpt_path), map_location="cpu")

        assert result == {"ok": True}
        mock_super.assert_called_once()
        assert mock_super.call_args.kwargs["weights_only"] is True

    def test_pl_weights_only_false_skips_safe_wrapper(self, tmp_path):
        """Lightning may pass ``weights_only=False`` on CheckpointIO; maps to full unpickle."""
        ckpt_path = tmp_path / "test.ckpt"
        torch.save({"state": 0}, ckpt_path)

        io = _DartsCheckpointIO()
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


def _save_reduce(tmp_path, name, callable_, args=()):
    class Evil:
        def __reduce__(self):
            return callable_, args

    path = tmp_path / name
    torch.save(Evil(), path)
    return path


class TestBlockedCheckpointGlobals:
    @pytest.mark.parametrize(
        "case",
        [
            (getattr, (torch._utils._rebuild_tensor_v2, "__globals__")),
            (execute_and_suppress_output, ()),
            (cpp_load, (torch._utils._rebuild_tensor_v2, "__globals__")),
            (os.system, ("echo pwned",)),
        ],
    )
    def test_blocks_malicious_case(self, tmp_path, case: tuple):
        path = _save_reduce(tmp_path, "getattr.pt", *case)
        with pytest.raises(UnpicklingError, match=case[0].__qualname__):
            load_wrapper_safely(path)
        with pytest.raises(UnpicklingError, match=case[0].__qualname__):
            _safe_globals_for_torch_file(path)

    def test_safe_globals_allows_one_class(self, tmp_path):
        path = tmp_path / "box.pt"
        torch.save(_UntrustedCheckpointClass(), path)
        with pytest.raises(UnpicklingError):
            load_wrapper_safely(path)
        with safe_globals([_UntrustedCheckpointClass]):
            loaded = load_wrapper_safely(path)
        assert isinstance(loaded, _UntrustedCheckpointClass)

        with pytest.raises(UnpicklingError):
            _safe_globals_for_torch_file(path)
        qualname = (
            f"{_UntrustedCheckpointClass.__module__}."
            f"{_UntrustedCheckpointClass.__qualname__}"
        )
        allowed = _safe_globals_for_torch_file(
            path,
            extra_user_globals={qualname: _UntrustedCheckpointClass},
        )
        assert _UntrustedCheckpointClass in allowed.values()
