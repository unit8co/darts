"""Tests for darts.utils.serialization.base (allowlist logic + RestrictedUnpickler)."""

import importlib
import io
import os
import pickle
import subprocess

import numpy as np
import pytest

from darts.logging import execute_and_suppress_output
from darts.utils.serialization.base import (
    RestrictedUnpickler,
    UnpicklingError,
    is_allowed_global,
    is_blocked_global,
    restricted_pickle_load,
)


class _UntrustedClass:
    """Module-level stand-in for an attacker-chosen class outside trusted packages."""


class TestSafeGlobals:
    @pytest.mark.parametrize(
        "global_", ["os.system", "subprocess.Popen", "builtins.eval", "pickle.loads"]
    )
    def test_blocks_global(self, global_):
        assert is_blocked_global(global_)

    @pytest.mark.parametrize(
        "global_", ["os.system", "subprocess.Popen", "builtins.eval", "pickle.loads"]
    )
    def test_blocks_and_raises_global(self, global_):
        with pytest.raises(UnpicklingError):
            is_allowed_global(global_)

    @pytest.mark.parametrize(
        "global_",
        ["darts.models.forecasting.DLinearModel", "numpy.float64", "pd.DatetimeIndex"],
    )
    def test_allows_global(self, global_):
        assert not is_blocked_global(global_)

    def test_allows_pandas_classes(self):
        assert is_allowed_global("pandas.DatetimeIndex")
        assert is_allowed_global("pandas.RangeIndex")
        assert is_allowed_global("pandas.DataFrame")

    def test_allows_pandas_offset_hierarchy(self):
        pandas = pytest.importorskip("pandas")
        from pandas._libs.tslibs.offsets import SemiMonthBegin

        for cls in (pandas.DateOffset, SemiMonthBegin):
            qualname = f"{cls.__module__}.{cls.__qualname__}"
            assert is_allowed_global(qualname)

    def test_rejects_pandas_user_offset_subclass(self):
        pandas = pytest.importorskip("pandas")

        class _UserOffset(pandas.offsets.MonthEnd):
            pass

        qualname = f"{_UserOffset.__module__}.{_UserOffset.__qualname__}"
        assert not is_allowed_global(qualname)

    def test_allows_numpy_dtype(self):
        assert is_allowed_global("numpy.dtype")
        assert is_allowed_global("numpy.int8")
        assert is_allowed_global("numpy.bool_")

    def test_rejects_numpy_memmap(self):
        assert not is_allowed_global("numpy.memmap")

    def test_allows_enum_under_trusted_prefix(self):
        from darts.utils.likelihood_models.base import LikelihoodType

        name = f"{LikelihoodType.__module__}.{LikelihoodType.__qualname__}"
        assert is_allowed_global(name)

    def test_rejects_untrusted_class(self):
        qn = f"{_UntrustedClass.__module__}.{_UntrustedClass.__qualname__}"
        assert not is_allowed_global(qn)

    def test_allows_user_opt_in_class(self):
        qn = f"{_UntrustedClass.__module__}.{_UntrustedClass.__qualname__}"
        assert is_allowed_global(
            qn,
            extra_user_globals={qn: _UntrustedClass},
        )

    def test_rejects_getattr(self):
        assert not is_allowed_global("builtins.getattr")

    def test_rejects_execute_and_suppress_output(self):
        assert not is_allowed_global(
            "darts.logging.execute_and_suppress_output",
        )

    def test_rejects_torch_callable_under_former_prefix(self):
        pytest.importorskip("torch")
        from torch.utils.cpp_extension import load as cpp_load

        qualname = f"{cpp_load.__module__}.{cpp_load.__qualname__}"
        assert not is_allowed_global(qualname)

    def test_allows_torch_dtype_singletons(self):
        pytest.importorskip("torch")
        assert is_allowed_global("torch.float32")
        assert is_allowed_global("torch.bfloat16")

    def test_allows_torch_size(self):
        pytest.importorskip("torch")
        assert is_allowed_global("torch.Size")


class TestRestrictedUnpickler:
    def test_find_class_skips_import_for_unknown_qualname(self, monkeypatch):
        imports: list[str] = []
        real_import = importlib.import_module

        def tracking_import(name):
            imports.append(name)
            return real_import(name)

        monkeypatch.setattr(importlib, "import_module", tracking_import)
        unpickler = RestrictedUnpickler(io.BytesIO())
        with pytest.raises(UnpicklingError, match="not allow-listed"):
            unpickler.find_class("evil_attacker_pkg", "Malware")
        assert imports == []

    def test_loads_safe_objects(self):
        buf = io.BytesIO()
        pickle.dump({"x": np.float64(1.0), "y": [1, 2, 3]}, buf)
        buf.seek(0)
        result = RestrictedUnpickler(buf).load()
        assert result["x"] == 1.0
        assert result["y"] == [1, 2, 3]

    def test_loads_pandas_offset(self):
        pandas = pytest.importorskip("pandas")
        offset = pandas.offsets.SemiMonthBegin(2)

        buf = io.BytesIO()
        pickle.dump(offset, buf)
        buf.seek(0)
        assert RestrictedUnpickler(buf).load() == offset

    def test_loads_torch_dtype(self):
        torch = pytest.importorskip("torch")

        buf = io.BytesIO()
        pickle.dump(torch.float32, buf)
        buf.seek(0)
        assert RestrictedUnpickler(buf).load() is torch.float32

    @pytest.mark.parametrize(
        "case",
        [
            (os.system, ("echo pwned",)),
            (subprocess.call, (["echo", "pwned"],)),
            (getattr, (str, "join")),
            (execute_and_suppress_output, (str, None, 0)),
        ],
    )
    def test_blocks_malicious(self, case):
        class Evil:
            def __reduce__(self):
                return case

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(UnpicklingError, match=case[0].__qualname__):
            RestrictedUnpickler(buf).load()

    def test_user_mapping_resolves_without_import(self, monkeypatch):
        qn = f"{_UntrustedClass.__module__}.{_UntrustedClass.__qualname__}"
        mod_name = _UntrustedClass.__module__
        real_import = importlib.import_module

        def fail_import(name):
            if name == mod_name:
                raise ImportError("simulated missing module")
            return real_import(name)

        monkeypatch.setattr(importlib, "import_module", fail_import)
        buf = io.BytesIO()
        pickle.dump(_UntrustedClass(), buf)
        buf.seek(0)
        unpickler = RestrictedUnpickler(
            buf,
            extra_user_globals={qn: _UntrustedClass},
        )
        assert isinstance(unpickler.load(), _UntrustedClass)


class TestRestrictedPickleLoad:
    def test_loads_safe_data(self):
        buf = io.BytesIO()
        pickle.dump({"hello": "world"}, buf)
        buf.seek(0)
        result = restricted_pickle_load(buf)
        assert result == {"hello": "world"}

    def test_trusted_loads_anything(self):
        buf = io.BytesIO()
        pickle.dump({"x": 1}, buf)
        buf.seek(0)
        result = restricted_pickle_load(buf, trusted=True)
        assert result == {"x": 1}

    def test_blocks_malicious_by_default(self):
        class Evil:
            def __reduce__(self):
                return os.system, ("echo pwned",)

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(UnpicklingError):
            restricted_pickle_load(buf)

    def test_trusted_allows_malicious(self, tmp_path):
        marker = tmp_path / "pwned.txt"

        class Evil:
            def __reduce__(self):
                return os.system, (f'echo pwned > "{marker}"',)

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        try:
            restricted_pickle_load(buf, trusted=True)
        except Exception:
            pass
        assert marker.exists()
