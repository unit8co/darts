"""Tests for darts.utils.serialization.registry."""

import io
import pickle
from unittest.mock import patch

import pytest

from darts.utils.serialization.base import UnpicklingError, restricted_pickle_load
from darts.utils.serialization.registry import (
    _get_user_safe_globals,
    _parse_safe_global,
    add_safe_globals,
    clear_safe_globals,
    get_safe_globals,
    safe_globals,
)


class _UserClass:
    pass


def _user_fn():
    return 0


class TestParseSafeGlobal:
    def test_callable_default_qualname(self):
        obj, qn = _parse_safe_global(_UserClass)
        assert obj is _UserClass
        assert qn.endswith("_UserClass")
        assert qn == _UserClass.__module__ + "." + _UserClass.__qualname__

    def test_tuple_explicit_qualname(self):
        obj, qn = _parse_safe_global((_UserClass, "legacy.module.OldName"))
        assert obj is _UserClass
        assert qn == "legacy.module.OldName"

    def test_rejections(self):
        # not a callable
        with pytest.raises(ValueError, match="callables"):
            _parse_safe_global(42)

        # tuple wrong length
        with pytest.raises(ValueError, match="(callable, qualname_str)"):
            _parse_safe_global((42,))

        # tuple first element not a callable
        with pytest.raises(ValueError, match="callables"):
            _parse_safe_global((42, "foo"))

        # tuple second element not a string
        def func():
            pass

        with pytest.raises(ValueError, match="str"):
            _parse_safe_global((func, 42))


class TestRegistryLifecycle:
    def setup_method(self):
        clear_safe_globals()

    def teardown_method(self):
        clear_safe_globals()

    def test_add_get_clear(self):
        add_safe_globals([_UserClass, _user_fn])

        user_globals = get_safe_globals()
        assert _UserClass in user_globals
        assert _user_fn in user_globals

        clear_safe_globals()
        assert get_safe_globals() == []

    def test_clear(self):
        add_safe_globals([_UserClass, _user_fn])
        user_globals = get_safe_globals()
        assert _UserClass in user_globals
        assert _user_fn in user_globals

    def test_context_restores(self):
        assert get_safe_globals() == []
        assert _get_user_safe_globals() == {}

        add_safe_globals([_UserClass])
        with safe_globals([_user_fn]):
            merged = _get_user_safe_globals()
            _, fn_qn = _parse_safe_global(_user_fn)
            assert merged[fn_qn] is _user_fn

            user_globals = get_safe_globals()
            assert _UserClass in user_globals
            assert _user_fn in user_globals

        merged = _get_user_safe_globals()
        _, fn_qn = _parse_safe_global(_user_fn)
        assert fn_qn not in merged

        user_globals = get_safe_globals()
        assert _user_fn not in user_globals


class TestRestrictedLoadWithRegistry:
    def setup_method(self):
        clear_safe_globals()

    def teardown_method(self):
        clear_safe_globals()

    def test_add_safe_globals_allows_pickle_class(self):
        buf = io.BytesIO()
        pickle.dump(_UserClass(), buf)
        buf.seek(0)

        with pytest.raises(UnpicklingError):
            restricted_pickle_load(buf)

        buf.seek(0)
        add_safe_globals([_UserClass])
        assert isinstance(restricted_pickle_load(buf), _UserClass)

    def test_torch_load_uses_merged_registry(self, tmp_path):
        torch = pytest.importorskip("torch")
        from darts.utils.serialization.torch import load_ckpt_safely

        path = tmp_path / "custom.ckpt"
        torch.save(_UserClass(), path)
        add_safe_globals([_UserClass])
        try:
            with patch("torch.serialization.safe_globals") as mock_ctx:
                mock_ctx.return_value.__enter__ = lambda self: None
                mock_ctx.return_value.__exit__ = lambda *a: None
                with pytest.raises(Exception):
                    load_ckpt_safely(torch.load, path)
                assert mock_ctx.called
        finally:
            clear_safe_globals()
