"""
Comprehensive tests for Chronos model wrapper

Test-Driven Development: These tests define the expected behavior before implementation.
"""
import pandas as pd
import pytest

from darts import TimeSeries
from darts.utils import timeseries_generation as tg

# Check if chronos-forecasting is available
chronos_available = True
try:
    import chronos  # noqa: F401
except ImportError:
    chronos_available = False


class TestChronosConverters:
    """Test suite for Chronos DataFrame <-> TimeSeries converters"""

    def test_timeseries_to_chronos_df_univariate(self):
        """Test converting univariate TimeSeries to Chronos DataFrame format"""
        # Create simple univariate series
        series = tg.sine_timeseries(length=10, value_frequency=0.1)

        # Import converter (will trigger the bug if pd_dataframe() is used)
        from darts.models.forecasting.foundation.chronos import _timeseries_to_chronos_df

        # Convert to Chronos format
        result_df = _timeseries_to_chronos_df(series)

        # Verify DataFrame structure
        assert isinstance(result_df, pd.DataFrame)
        assert "id" in result_df.columns
        assert "timestamp" in result_df.columns
        assert "target" in result_df.columns

        # Verify correct number of rows
        assert len(result_df) == 10

        # Verify id is consistent
        assert result_df["id"].nunique() == 1
        assert result_df["id"].iloc[0] == "series_0"

    def test_timeseries_to_chronos_df_multivariate(self):
        """Test converting multivariate TimeSeries to Chronos DataFrame format"""
        # Create multivariate series
        series = tg.sine_timeseries(
            length=10,
            value_frequency=0.1,
            column_name="component"
        ).stack(
            tg.gaussian_timeseries(length=10, column_name="noise")
        )

        from darts.models.forecasting.foundation.chronos import _timeseries_to_chronos_df

        result_df = _timeseries_to_chronos_df(series)

        # Verify DataFrame structure for multivariate
        assert isinstance(result_df, pd.DataFrame)
        assert "id" in result_df.columns
        assert "timestamp" in result_df.columns
        assert "target_0" in result_df.columns
        assert "target_1" in result_df.columns
        assert len(result_df) == 10

    def test_timeseries_to_chronos_df_multiple_series(self):
        """Test converting list of TimeSeries to Chronos DataFrame format"""
        # Create list of series
        series_list = [
            tg.sine_timeseries(length=5, value_frequency=0.1),
            tg.gaussian_timeseries(length=5),
        ]

        from darts.models.forecasting.foundation.chronos import _timeseries_to_chronos_df

        result_df = _timeseries_to_chronos_df(series_list)

        # Verify combined DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 10  # 5 + 5
        assert result_df["id"].nunique() == 2
        assert "series_0" in result_df["id"].values
        assert "series_1" in result_df["id"].values


class TestChronosReverseConverters:
    """Test suite for Chronos DataFrame -> TimeSeries converters"""

    def test_chronos_df_to_timeseries_univariate(self):
        """Test converting Chronos prediction DataFrame back to univariate TimeSeries"""
        # Create original series for metadata
        original_series = tg.sine_timeseries(length=10, value_frequency=0.1)

        # Create a mock Chronos prediction DataFrame (with median quantile)
        import pandas as pd
        pred_data = {
            "id": ["series_0"] * 5,
            "timestamp": pd.date_range("2000-01-11", periods=5, freq="D"),
            "0.5": [0.1, 0.2, 0.3, 0.4, 0.5],  # Median quantile
        }
        pred_df = pd.DataFrame(pred_data)

        from darts.models.forecasting.foundation.chronos import _chronos_df_to_timeseries

        # Convert back to TimeSeries
        result = _chronos_df_to_timeseries(pred_df, original_series, n=5)

        # Verify result is TimeSeries
        assert isinstance(result, TimeSeries)
        assert len(result) == 5
        assert result.freq == original_series.freq

    def test_chronos_df_to_timeseries_multiple_series(self):
        """Test converting predictions for multiple series back to TimeSeries list"""
        # Create original series list
        original_series = [
            tg.sine_timeseries(length=10, value_frequency=0.1),
            tg.gaussian_timeseries(length=10),
        ]

        # Create mock predictions for both series
        pred_data = {
            "id": ["series_0"] * 3 + ["series_1"] * 3,
            "timestamp": list(pd.date_range("2000-01-11", periods=3, freq="D")) * 2,
            "0.5": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        pred_df = pd.DataFrame(pred_data)

        from darts.models.forecasting.foundation.chronos import _chronos_df_to_timeseries

        result = _chronos_df_to_timeseries(pred_df, original_series, n=3)

        # Verify result is list of TimeSeries
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(ts, TimeSeries) for ts in result)
        assert all(len(ts) == 3 for ts in result)


@pytest.mark.skipif(not chronos_available, reason="chronos-forecasting not installed")
class TestChronosModelIntegration:
    """Integration tests for full ChronosModel predict() workflow"""

    def test_predict_end_to_end_univariate(self):
        """Test complete predict workflow with univariate series"""
        from darts.models.forecasting.foundation import ChronosModel

        # Create training series
        series = tg.sine_timeseries(length=50, value_frequency=0.1)

        # Create model (use S3-hosted Chronos-2 model)
        model = ChronosModel(model_id="s3://autogluon/chronos-2")

        # Predict (zero-shot, no fit required)
        forecast = model.predict(n=10, series=series, num_samples=1)

        # Verify forecast structure
        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == 10
        assert forecast.freq == series.freq
        # Forecast should start right after training series
        assert forecast.start_time() == series.end_time() + series.freq

    def test_predict_multiple_series(self):
        """Test predict with multiple series"""
        from darts.models.forecasting.foundation import ChronosModel

        # Create list of series
        series_list = [
            tg.sine_timeseries(length=30, value_frequency=0.1),
            tg.gaussian_timeseries(length=30),
        ]

        model = ChronosModel(model_id="s3://autogluon/chronos-2")

        # Predict for multiple series
        forecasts = model.predict(n=5, series=series_list, num_samples=1)

        # Verify forecasts
        assert isinstance(forecasts, list)
        assert len(forecasts) == 2
        assert all(isinstance(f, TimeSeries) for f in forecasts)
        assert all(len(f) == 5 for f in forecasts)

    def test_quantile_forecast_naming_univariate(self):
        """Test that quantile forecasts use proper Darts naming convention (q{value:.3f})"""
        from darts.models.forecasting.foundation import ChronosModel

        # Create univariate series
        series = tg.sine_timeseries(length=30, value_frequency=0.1)

        model = ChronosModel(model_id="s3://autogluon/chronos-2")

        # Predict with specific quantiles
        forecast = model.predict(n=5, series=series, quantiles=[0.1, 0.5, 0.9])

        # Verify proper quantile naming convention
        expected_columns = ["q0.100", "q0.500", "q0.900"]
        assert forecast.components.tolist() == expected_columns

        # Verify shape: (horizon, n_quantiles, 1 sample) for deterministic
        # Note: Use all_values() to preserve all dimensions
        assert forecast.all_values().shape == (5, 3, 1)

    def test_quantile_forecast_naming_multivariate(self):
        """Test that multivariate quantile forecasts include component names"""
        from darts.models.forecasting.foundation import ChronosModel

        # Create multivariate series
        series = tg.sine_timeseries(length=30, value_frequency=0.1, column_name="sales")
        series = series.stack(tg.gaussian_timeseries(length=30, column_name="volume"))

        model = ChronosModel(model_id="s3://autogluon/chronos-2")

        # Predict with quantiles
        forecast = model.predict(n=5, series=series, quantiles=[0.1, 0.9])

        # Verify proper component_quantile naming
        expected_columns = ["sales_q0.100", "sales_q0.900", "volume_q0.100", "volume_q0.900"]
        assert forecast.components.tolist() == expected_columns

        # Verify shape: (horizon, n_components * n_quantiles, 1 sample)
        # Note: Use all_values() to preserve all dimensions
        assert forecast.all_values().shape == (5, 4, 1)
