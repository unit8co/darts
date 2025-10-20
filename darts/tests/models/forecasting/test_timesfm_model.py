"""
Comprehensive tests for TimesFM model wrapper

Test-Driven Development: These tests define the expected behavior before implementation.
"""
import pytest
import torch

from darts import TimeSeries
from darts.utils import timeseries_generation as tg


class TestTimesFMModelConstruction:
    """Test suite for TimesFM model construction and configuration"""

    def test_model_construction_default(self):
        """Test model can be constructed with default parameters"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel()
        assert model.model_version == "2.5"
        assert model.model_size == "200m"
        assert model.max_context_length == 1024
        assert model.zero_shot is True

    def test_model_construction_custom(self):
        """Test model construction with custom parameters"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel(
            model_version="2.5",
            model_size="200m",
            max_context_length=512,
            zero_shot=True,
            device="cpu"
        )
        assert model.max_context_length == 512
        assert model.device == "cpu"

    def test_invalid_model_version(self):
        """Test that invalid model version raises error"""
        from darts.models.forecasting.foundation import TimesFMModel

        with pytest.raises(ValueError, match="model_version"):
            TimesFMModel(model_version="3.0")

    def test_invalid_model_size(self):
        """Test that invalid model size raises error"""
        from darts.models.forecasting.foundation import TimesFMModel

        with pytest.raises(ValueError, match="model_size"):
            TimesFMModel(model_size="100m")

    def test_invalid_context_length_not_divisible_by_32(self):
        """Test that context length not divisible by 32 raises error"""
        from darts.models.forecasting.foundation import TimesFMModel

        with pytest.raises(ValueError, match="divisible by 32"):
            TimesFMModel(max_context_length=100)

    def test_invalid_context_length_negative(self):
        """Test that negative context length raises error"""
        from darts.models.forecasting.foundation import TimesFMModel

        with pytest.raises(ValueError, match="positive"):
            TimesFMModel(max_context_length=-32)

    def test_device_detection_auto(self):
        """Test automatic device detection"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel(device=None)  # Auto-detect

        # Should detect one of: cpu, cuda, mps
        assert model.device in ["cpu", "cuda", "mps"]

        # Verify it matches actual PyTorch capabilities
        if torch.cuda.is_available():
            assert model.device == "cuda"
        elif torch.backends.mps.is_available():
            assert model.device == "mps"
        else:
            assert model.device == "cpu"

    def test_device_detection_manual_cpu(self):
        """Test manual CPU device selection"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel(device="cpu")
        assert model.device == "cpu"

    def test_model_properties(self):
        """Test model capability properties"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel()

        # TimesFM supports only univariate series
        assert model.supports_multivariate is False

        # Minimum training length is TimesFM's patch size
        assert model.min_train_series_length == 32

        # Context window and lags (7-tuple format for historical_forecasts compatibility)
        assert model.extreme_lags == (-1024, 0, 0, 0, 0, 0, 0)  # Default context length

    def test_model_properties_custom_context(self):
        """Test model properties with custom context length"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel(max_context_length=512)
        assert model.extreme_lags == (-512, 0, 0, 0, 0, 0, 0)

    def test_string_representation(self):
        """Test string representation of model"""
        from darts.models.forecasting.foundation import TimesFMModel

        model = TimesFMModel(
            model_version="2.5",
            model_size="200m",
            max_context_length=512
        )

        str_repr = str(model)
        assert "TimesFM" in str_repr
        assert "2.5" in str_repr
        assert "512" in str_repr


class TestTimesFMModelFit:
    """Test suite for TimesFM model fit method"""

    def test_zero_shot_fit_validates_univariate(self):
        """Test that fit validates univariate series requirement"""
        from darts.models.forecasting.foundation import TimesFMModel

        # Create multivariate series
        series = tg.sine_timeseries(length=100, value_frequency=1)
        series = series.stack(tg.sine_timeseries(length=100, value_frequency=2))

        model = TimesFMModel(zero_shot=True)

        # Should raise error for multivariate
        with pytest.raises(ValueError, match="univariate"):
            model.fit(series)

    def test_zero_shot_fit_accepts_univariate(self):
        """Test that fit accepts univariate series"""
        from darts.models.forecasting.foundation import TimesFMModel

        series = tg.sine_timeseries(length=100)
        model = TimesFMModel(zero_shot=True)

        # Should not raise error
        result = model.fit(series)
        assert result is model  # Should return self

    def test_fit_warns_on_short_series(self):
        """Test that fit warns when series is shorter than recommended"""
        from darts.models.forecasting.foundation import TimesFMModel

        short_series = tg.sine_timeseries(length=20)  # Less than min_train_series_length
        model = TimesFMModel(zero_shot=True)

        # Should fit but log warning (we'll just verify it doesn't crash)
        model.fit(short_series)

    def test_fit_loads_model(self):
        """Test that fit loads the TimesFM model"""
        from darts.models.forecasting.foundation import TimesFMModel

        series = tg.sine_timeseries(length=100)
        model = TimesFMModel(zero_shot=True)

        # Model should be None before fit
        assert model._model is None

        # Fit should load the model
        model.fit(series)

        # Model should be loaded after fit
        assert model._model is not None

    def test_fit_rejects_multivariate_in_list(self):
        """Test that fit rejects multivariate series even in a list"""
        from darts.models.forecasting.foundation import TimesFMModel

        # Create list with one multivariate series
        series_list = [
            tg.sine_timeseries(length=100),
            tg.sine_timeseries(length=100).stack(tg.sine_timeseries(length=100, value_frequency=2))
        ]

        model = TimesFMModel(zero_shot=True)

        with pytest.raises(ValueError, match="univariate"):
            model.fit(series_list)

    def test_fit_accepts_list_of_univariate(self):
        """Test that fit accepts list of univariate series"""
        from darts.models.forecasting.foundation import TimesFMModel

        series_list = [
            tg.sine_timeseries(length=100),
            tg.sine_timeseries(length=120, value_frequency=2),
            tg.linear_timeseries(length=100)
        ]

        model = TimesFMModel(zero_shot=True)
        result = model.fit(series_list)
        assert result is model


class TestTimesFMModelPredict:
    """Test suite for TimesFM model predict method"""

    def test_predict_without_fit_zero_shot(self):
        """Test true zero-shot: predict() without calling fit() first"""
        from darts.models.forecasting.foundation import TimesFMModel
        from darts.utils import timeseries_generation as tg

        series = tg.sine_timeseries(length=100)
        model = TimesFMModel(zero_shot=True, max_context_length=512)

        # Predict WITHOUT calling fit() - true zero-shot!
        forecast = model.predict(n=12, series=series)

        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == 12
        # Model should be loaded automatically
        assert model._model is not None

    def test_predict_univariate(self):
        """Test basic prediction on univariate series"""
        from darts.models.forecasting.foundation import TimesFMModel
        from darts.datasets import AirPassengersDataset

        series = AirPassengersDataset().load()
        train, val = series.split_before(0.8)

        model = TimesFMModel(zero_shot=True, max_context_length=512)
        model.fit(train)

        forecast = model.predict(n=12, series=train)

        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == 12
        assert forecast.is_univariate
        # Check no NaN values
        import numpy as np
        assert not np.any(np.isnan(forecast.values()))

    def test_predict_returns_correct_horizon(self):
        """Test that predict returns exactly n points"""
        from darts.models.forecasting.foundation import TimesFMModel

        series = tg.sine_timeseries(length=100)
        model = TimesFMModel(zero_shot=True)
        model.fit(series)

        for horizon in [5, 12, 24, 50]:
            forecast = model.predict(n=horizon, series=series)
            assert len(forecast) == horizon

    def test_predict_multiple_series(self):
        """Test batch prediction on multiple series"""
        from darts.models.forecasting.foundation import TimesFMModel

        series_list = [
            tg.sine_timeseries(length=100),
            tg.linear_timeseries(length=100),
            tg.gaussian_timeseries(length=100)
        ]

        model = TimesFMModel(zero_shot=True, max_context_length=512)
        model.fit(series_list[0])

        forecasts = model.predict(n=10, series=series_list)

        assert isinstance(forecasts, list)
        assert len(forecasts) == 3
        for f in forecasts:
            assert len(f) == 10
            assert f.is_univariate

    def test_predict_multiple_series_with_multivariate_component_selection(self):
        """Test batch forecasting with multivariate series component selection"""
        from darts.models.forecasting.foundation import TimesFMModel

        # Create multivariate series
        multivariate = tg.sine_timeseries(length=100, value_frequency=0.1, column_name="A")
        multivariate = multivariate.stack(
            tg.sine_timeseries(length=100, value_frequency=0.2, column_name="B")
        )

        # Test component selection syntax used in notebooks
        series_list = [
            tg.sine_timeseries(length=100),
            multivariate["A"][:100],  # Component first, then time slice
            multivariate["B"][:50]
        ]

        model = TimesFMModel(zero_shot=True, max_context_length=512)

        # Should work without errors
        forecasts = model.predict(n=10, series=series_list)

        assert isinstance(forecasts, list)
        assert len(forecasts) == 3
        for f in forecasts:
            assert len(f) == 10
            assert f.is_univariate

    def test_predict_with_different_series_than_fit(self):
        """Test zero-shot: predict on different series than fit"""
        from darts.models.forecasting.foundation import TimesFMModel

        train_series = tg.sine_timeseries(length=100, value_frequency=1)
        test_series = tg.sine_timeseries(length=80, value_frequency=2)

        model = TimesFMModel(zero_shot=True)
        model.fit(train_series)

        # Should work on completely different series (zero-shot)
        forecast = model.predict(n=12, series=test_series)
        assert len(forecast) == 12

    def test_predict_requires_series_parameter(self):
        """Test that predict requires series parameter in zero-shot mode"""
        from darts.models.forecasting.foundation import TimesFMModel

        series = tg.sine_timeseries(length=100)
        model = TimesFMModel(zero_shot=True)
        model.fit(series)

        # Should raise error if series not provided
        with pytest.raises((ValueError, TypeError)):
            model.predict(n=12)  # No series parameter

    def test_predict_time_index_continuation(self):
        """Test that forecast continues the time index correctly"""
        from darts.models.forecasting.foundation import TimesFMModel
        import pandas as pd

        # Create series with datetime index
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        series = TimeSeries.from_times_and_values(dates, tg.sine_timeseries(length=100).values())

        model = TimesFMModel(zero_shot=True)
        model.fit(series)

        forecast = model.predict(n=10, series=series)

        # Check that forecast time index starts after series ends
        assert forecast.start_time() == series.end_time() + series.freq
        assert len(forecast) == 10
