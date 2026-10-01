"""Tests for darts.utils.serialization.base (allowlist logic + RestrictedUnpickler)."""

import io
import os
import pickle

import numpy as np
import pytest

from darts.logging import execute_and_suppress_output
from darts.utils.serialization.base import (
    RestrictedUnpickler,
    is_allowed_global,
    is_blocked_global,
    restricted_pickle_load,
    safe_base_classes,
)


class _UntrustedClass:
    """Module-level stand-in for an attacker-chosen class outside trusted packages."""


class TestIsBlockedGlobal:
    def test_blocks_os(self):
        assert is_blocked_global("os.system")

    def test_blocks_subprocess(self):
        assert is_blocked_global("subprocess.Popen")

    def test_blocks_builtins_eval(self):
        assert is_blocked_global("builtins.eval")

    def test_blocks_pickle(self):
        assert is_blocked_global("pickle.loads")

    def test_allows_darts(self):
        assert not is_blocked_global("darts.models.forecasting.DLinearModel")

    def test_allows_numpy(self):
        assert not is_blocked_global("numpy.float64")


class TestIsAllowedGlobal:
    @pytest.fixture()
    def bases(self):
        return safe_base_classes()

    def test_blocks_os_system(self, bases):
        assert not is_allowed_global("os.system", os.system, safe_bases=bases)

    def test_allows_exact_class(self, bases):
        assert is_allowed_global("numpy.float64", np.float64, safe_bases=bases)

    def test_allows_numpy_dtype(self, bases):
        assert is_allowed_global("numpy.dtype", np.dtype, safe_bases=bases)

    def test_allows_numpy_scalar_hierarchy(self, bases):
        assert is_allowed_global("numpy.int8", np.int8, safe_bases=bases)
        assert is_allowed_global("numpy.bool_", np.bool_, safe_bases=bases)

    def test_rejects_numpy_memmap(self, bases):
        assert not is_allowed_global("numpy.memmap", np.memmap, safe_bases=bases)

    def test_allows_pandas_offset_hierarchy(self, bases):
        pandas = pytest.importorskip("pandas")
        from pandas._libs.tslibs.offsets import SemiMonthBegin

        for cls in (pandas.DateOffset, SemiMonthBegin):
            qualname = f"{cls.__module__}.{cls.__qualname__}"
            assert is_allowed_global(qualname, cls, safe_bases=bases)

    def test_rejects_user_offset_subclass(self, bases):
        pandas = pytest.importorskip("pandas")

        class _UserOffset(pandas.offsets.MonthEnd):
            pass

        qualname = f"{_UserOffset.__module__}.{_UserOffset.__qualname__}"
        assert not is_allowed_global(qualname, _UserOffset, safe_bases=bases)

    def test_allows_enum_under_trusted_prefix(self, bases):
        from darts.utils.likelihood_models.base import LikelihoodType

        name = f"{LikelihoodType.__module__}.{LikelihoodType.__qualname__}"
        assert is_allowed_global(name, LikelihoodType, safe_bases=bases)

    def test_rejects_untrusted_class(self, bases):
        qn = f"{_UntrustedClass.__module__}.{_UntrustedClass.__qualname__}"
        assert not is_allowed_global(qn, _UntrustedClass, safe_bases=bases)

    def test_rejects_none_obj(self, bases):
        assert not is_allowed_global("numpy.float64", None, safe_bases=bases)

    def test_rejects_getattr(self, bases):
        assert not is_allowed_global("builtins.getattr", getattr, safe_bases=bases)

    def test_rejects_execute_and_suppress_output(self, bases):
        assert not is_allowed_global(
            "darts.logging.execute_and_suppress_output",
            execute_and_suppress_output,
            safe_bases=bases,
        )

    def test_rejects_torch_callable_under_former_prefix(self, bases):
        pytest.importorskip("torch")
        from torch.utils.cpp_extension import load as cpp_load

        qualname = f"{cpp_load.__module__}.{cpp_load.__qualname__}"
        assert not is_allowed_global(qualname, cpp_load, safe_bases=bases)

    def test_allows_torch_dtype_singletons(self, bases):
        torch = pytest.importorskip("torch")

        assert is_allowed_global("torch.float32", torch.float32, safe_bases=bases)
        assert is_allowed_global("torch.bfloat16", torch.bfloat16, safe_bases=bases)

    def test_allows_torch_size(self, bases):
        torch = pytest.importorskip("torch")

        assert is_allowed_global("torch.Size", torch.Size, safe_bases=bases)


class TestRestrictedUnpickler:
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

    def test_blocks_os_system(self):
        class Evil:
            def __reduce__(self):
                return os.system, ("echo pwned",)

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError, match="os.system|Blocked"):
            RestrictedUnpickler(buf).load()

    def test_blocks_subprocess(self):
        import subprocess

        class Evil:
            def __reduce__(self):
                return subprocess.call, (["echo", "pwned"],)

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError, match="subprocess"):
            RestrictedUnpickler(buf).load()

    def test_blocks_getattr(self):
        class Evil:
            def __reduce__(self):
                return getattr, (str, "join")

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError, match="getattr"):
            RestrictedUnpickler(buf).load()

    def test_blocks_execute_and_suppress_output(self):
        class Evil:
            def __reduce__(self):
                return execute_and_suppress_output, (str, None, 0)

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError, match="execute_and_suppress_output"):
            RestrictedUnpickler(buf).load()

    def test_blocks_cmdstan_model(self):
        cmdstanpy = pytest.importorskip("cmdstanpy")

        class Evil:
            def __reduce__(self):
                return cmdstanpy.CmdStanModel, ()

        buf = io.BytesIO()
        pickle.dump(Evil(), buf)
        buf.seek(0)
        with pytest.raises(pickle.UnpicklingError, match="CmdStanModel"):
            RestrictedUnpickler(buf).load()


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
        with pytest.raises(pickle.UnpicklingError):
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
