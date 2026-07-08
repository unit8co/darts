import types

import pytest

from darts.utils._lazy import setup_lazy_imports


@pytest.fixture()
def fake_package():
    """Create a throwaway module that mimics a package ``__init__.py``."""
    mod = types.ModuleType("fake_pkg")
    mod.__name__ = "fake_pkg"
    return mod


class TestSetupLazyImports:
    def test_required_dep_loads(self, fake_package):
        lazy = {"join": "os.path"}
        __all__, __getattr__, __dir__ = setup_lazy_imports(
            lazy, fake_package.__name__, vars(fake_package)
        )

        import os.path

        value = __getattr__("join")

        assert value is os.path.join
        assert "join" in __all__
        assert "join" in __dir__()

        # check caching
        assert "join" in vars(fake_package)

        assert value("path_a", "suffiy.py") == os.path.join("path_a", "suffiy.py")

    def test_unknown_attr_raises(self, fake_package):
        lazy = {"join": "os.path"}
        _, __getattr__, _ = setup_lazy_imports(
            lazy, fake_package.__name__, vars(fake_package)
        )

        with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
            __getattr__("does_not_exist")

    def test_required_dep_missing_raises(self, fake_package):
        lazy = {"Foo": "nonexistent_module"}
        _, __getattr__, _ = setup_lazy_imports(
            lazy, fake_package.__name__, vars(fake_package)
        )

        with pytest.raises(ModuleNotFoundError):
            __getattr__("Foo")

    def test_optional_dep_missing(self, fake_package):
        lazy = {"Foo": ("nonexistent_module", "SomeLib")}
        _, __getattr__, _ = setup_lazy_imports(
            lazy, fake_package.__name__, vars(fake_package)
        )

        result = __getattr__("Foo")
        from darts.utils.utils import NotImportedModule

        assert isinstance(result, NotImportedModule)
        # check caching
        assert isinstance(vars(fake_package)["Foo"], NotImportedModule)

        # the missing module exception is raised upon first call
        with pytest.raises(
            ImportError, match="The `SomeLib` module could not be imported."
        ):
            _ = result()

    def test_optional_dep_none_propagates_error(self, fake_package):
        """When optional_dep is explicitly None in a tuple, error should propagate."""
        lazy = {"Foo": ("nonexistent_module", None)}
        _, __getattr__, _ = setup_lazy_imports(
            lazy, fake_package.__name__, vars(fake_package)
        )

        with pytest.raises(ModuleNotFoundError):
            __getattr__("Foo")

    def test_tuple_with_available_dep_loads(self, fake_package):
        lazy = {"join": ("os.path", None)}
        _, __getattr__, _ = setup_lazy_imports(
            lazy, fake_package.__name__, vars(fake_package)
        )

        import os.path

        assert __getattr__("join") is os.path.join

    def test_extra_attrs_resolved(self, fake_package):
        lazy = {"join": "os.path"}
        call_count = 0

        def build_extra():
            nonlocal call_count
            call_count += 1
            return 42

        __all__, __getattr__, __dir__ = setup_lazy_imports(
            lazy,
            fake_package.__name__,
            vars(fake_package),
            extra_attrs={"MAGIC": build_extra},
        )

        assert "MAGIC" in __all__
        assert "MAGIC" in __dir__()
        assert __getattr__("MAGIC") == 42
        assert call_count == 1

        # check caching
        assert vars(fake_package)["MAGIC"] == 42
        # still the same value
        assert call_count == 1

    def test_dir_returns_all_names(self, fake_package):
        lazy = {"a": "os", "b": ("os", None)}
        __all__, _, __dir__ = setup_lazy_imports(
            lazy,
            fake_package.__name__,
            vars(fake_package),
            extra_attrs={"c": lambda: None},
        )

        assert set(__dir__()) == {"a", "b", "c"}
        assert set(__all__) == {"a", "b", "c"}

    def test_all_contains_only_lazy_and_extra_names(self, fake_package):
        lazy = {"x": "os", "y": ("os.path", "opt")}
        __all__, _, _ = setup_lazy_imports(
            lazy,
            fake_package.__name__,
            vars(fake_package),
            extra_attrs={"z": lambda: 1},
        )

        assert __all__ == ["x", "y", "z"]


class TestLazyImportsIntegration:
    """Verify the real ``__init__.py`` packages expose the expected API."""

    def test_models_dir_contains_all_models(self):
        import darts.models

        names = dir(darts.models)
        assert "NaiveSeasonal" in names
        assert "NBEATSModel" in names
        assert "KalmanFilter" in names

    def test_metrics_dir_and_extra_attrs(self):
        import darts.metrics

        names = dir(darts.metrics)
        assert "mape" in names
        assert "TIME_DEPENDENT_METRICS" in names
        assert "CLASSIFICATION_METRICS" in names

        td = darts.metrics.TIME_DEPENDENT_METRICS
        assert isinstance(td, set) and len(td) > 0

        cl = darts.metrics.CLASSIFICATION_METRICS
        assert isinstance(cl, set) and len(cl) > 0

    def test_ad_dir(self):
        import darts.ad

        assert "AndAggregator" in dir(darts.ad)

    def test_datasets_dir(self):
        import darts.datasets

        assert "AirPassengersDataset" in dir(darts.datasets)

    def test_transformers_dir(self):
        import darts.dataprocessing.transformers

        assert "Scaler" in dir(darts.dataprocessing.transformers)

    def test_unknown_attribute_raises_for_real_packages(self):
        import darts.models

        with pytest.raises(AttributeError):
            _ = darts.models.CompletelyFakeModel12345
