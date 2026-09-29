"""Tests for darts.utils.serialization.base (allowlist logic + RestrictedUnpickler)."""

import io
import os
import pickle

import numpy as np
import pytest

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

    def test_allows_enum_under_trusted_prefix(self, bases):
        from darts.utils.likelihood_models.base import LikelihoodType

        name = f"{LikelihoodType.__module__}.{LikelihoodType.__qualname__}"
        assert is_allowed_global(name, LikelihoodType, safe_bases=bases)

    def test_rejects_untrusted_class(self, bases):
        qn = f"{_UntrustedClass.__module__}.{_UntrustedClass.__qualname__}"
        assert not is_allowed_global(qn, _UntrustedClass, safe_bases=bases)

    def test_rejects_none_obj(self, bases):
        assert not is_allowed_global("numpy.float64", None, safe_bases=bases)


class TestRestrictedUnpickler:
    def test_loads_safe_objects(self):
        buf = io.BytesIO()
        pickle.dump({"x": np.float64(1.0), "y": [1, 2, 3]}, buf)
        buf.seek(0)
        result = RestrictedUnpickler(buf).load()
        assert result["x"] == 1.0
        assert result["y"] == [1, 2, 3]

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
