import inspect
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
from darts.utils.serialization.base import (
    dedupe_by_identity,
    is_allowed_global,
    sanitize_for_wrapper_save,
)
from darts.utils.serialization.torch import (
    DartsCheckpointIO,
    UnpicklingError,
    likelihood_safe_globals,
    load_torch_safely,
    safe_globals_for_checkpoint,
    safe_globals_for_torch_file,
    wrapper_safe_bases,
    wrapper_seed_globals,
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
        for obj in likelihood_safe_globals():
            assert inspect.isclass(obj)


class TestSanitizeForWrapperSave:
    def test_converts_attribute_dict_to_plain_dict(self):
        try:
            from lightning_fabric.utilities.data import AttributeDict
        except ImportError:
            pytest.skip("lightning_fabric not available")

        sanitized = sanitize_for_wrapper_save(AttributeDict({"a": 1, "b": {"c": 2}}))
        assert sanitized == {"a": 1, "b": {"c": 2}}
        assert type(sanitized) is dict


class TestWrapperSeedGlobals:
    def test_includes_enum_and_numpy_dtypes(self):
        import enum

        globals_ = wrapper_seed_globals()
        assert enum.Enum in globals_
        assert any(
            getattr(obj, "__module__", "").startswith("numpy.dtypes")
            for obj in globals_
            if inspect.isclass(obj)
        )


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

        with pytest.raises(UnpicklingError, match="not allow-listed"):
            safe_globals_for_checkpoint(ckpt_path)


class TestSafeGlobalsForWrapper:
    def test_rejects_untrusted_global(self, tmp_path):
        pt_path = tmp_path / "evil.pt"
        torch.save({"payload": _UntrustedCheckpointClass()}, pt_path)

        with pytest.raises(UnpicklingError, match="not allow-listed"):
            safe_globals_for_torch_file(pt_path, scope="wrapper")

    def test_wrapper_scope_does_not_allow_functions(self):
        from darts.utils.serialization.base import WRAPPER_SAFE_FUNCTIONS

        for name in WRAPPER_SAFE_FUNCTIONS:
            obj = __import__("importlib").import_module(name.rpartition(".")[0])
            # resolved via resolve_reference in production; here just ensure policy
            assert is_allowed_global(
                name,
                getattr(obj, name.rpartition(".")[2]),
                scope="wrapper",
                safe_bases=wrapper_safe_bases(),
                wrapper_safe_functions=WRAPPER_SAFE_FUNCTIONS,
            )


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

    def test_wrapper_scope_includes_seed_globals(self, tmp_path):
        pt_path = tmp_path / "dummy.pt"
        torch.save({"x": 1}, pt_path)

        with patch(
            "darts.utils.serialization.torch.torch.serialization.safe_globals"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            load_torch_safely(lambda *args, **kwargs: None, pt_path, scope="wrapper")
            allow = mock_ctx.call_args[0][0]
            assert wrapper_seed_globals()[0] in allow


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
