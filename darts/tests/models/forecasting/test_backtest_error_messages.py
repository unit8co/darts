"""Tests for improved error messages in backtest() and historical_forecasts()."""

import pytest
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel, NBEATSModel
from darts.metrics import mape


class TestBacktestErrorMessages:
    """Test that backtest() and historical_forecasts() provide helpful error messages."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        target_data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        future_cov_data = np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

        series = TimeSeries.from_times_and_values(dates, target_data.astype(np.float32))
        future_covariates = TimeSeries.from_times_and_values(
            dates,
            future_cov_data.astype(np.float32)
        )

        return series, future_covariates

    def test_backtest_missing_future_covariates_error(self, sample_data):
        """
        Test that calling backtest() without future_covariates on a model
        that requires them produces an error message.

        Note: We only validate coverage (too short/late start), not None.
        When covariates are None, the model's own validation handles it.
        This test verifies that SOME error is raised (could be from model or our validation).
        """
        series, future_covariates = sample_data

        # Create and train model that requires future covariates
        model = TFTModel(
            input_chunk_length=12,
            output_chunk_length=6,
            n_epochs=1,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        # Train with future covariates
        model.fit(series[:70], future_covariates=future_covariates[:70])

        # Test that calling backtest without future_covariates raises an error
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            model.backtest(
                series=series,
                # Intentionally NOT providing future_covariates
                forecast_horizon=6,
                metric=mape
            )

        error_message = str(exc_info.value)

        # Verify error mentions future covariates (could be from model or our validation)
        assert ("future" in error_message.lower() and "covariate" in error_message.lower()) or \
               "future_covariates" in error_message.lower(), \
            "Error should mention future covariates"

        # The error can come from either:
        # 1. The model's own validation (mentions "future covariates", "encoders")
        # 2. Our validation (mentions "extend", "missing steps") if covariates were partially provided
        assert any(keyword in error_message.lower() for keyword in
                   ["covariate", "encoder", "extend", "missing"]), \
            "Error should provide actionable information"
    def test_backtest_future_covariates_too_short(self, sample_data):
        """
        Test that backtest() with future_covariates that don't extend far enough
        produces a clear error message.

        Addresses issue #2846 - Scenario 2: IndexError when covariates are too short.
        """
        series, future_covariates = sample_data

        # Create covariates that are too short
        future_covariates_short = future_covariates[:75]  # Only 75 days, need 100 + 6

        model = TFTModel(
            input_chunk_length=12,
            output_chunk_length=6,
            n_epochs=1,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        model.fit(series[:70], future_covariates=future_covariates_short[:70])

        # Should raise helpful error instead of IndexError
        with pytest.raises(ValueError) as exc_info:
            model.backtest(
                series=series,
                future_covariates=future_covariates_short,
                forecast_horizon=6,
                metric=mape
            )

        error_message = str(exc_info.value)

        # Verify error message is helpful
        assert "extend" in error_message.lower() or "missing" in error_message.lower(), \
            "Error should mention that covariates don't extend far enough"

        assert "Series ends at" in error_message, \
            "Error should show series end time"

        assert "Future covariates end at" in error_message, \
            "Error should show covariates end time"

        assert "31" in error_message or "missing" in error_message.lower(), \
            "Error should indicate how many steps are missing"

    def test_backtest_future_covariates_start_too_late(self, sample_data):
        """
        Test that backtest() with future_covariates that start after series start
        produces a clear error message.

        Addresses issue #2846 - Scenario 3: TypeError when covariates start too late.
        """
        series, _ = sample_data

        # Create covariates that start AFTER series starts
        cov_dates = pd.date_range('2020-03-01', periods=80, freq='D')  # Starts March 1
        cov_data = np.random.randn(80).astype(np.float32)
        future_covariates_late = TimeSeries.from_times_and_values(cov_dates, cov_data)

        model = TFTModel(
            input_chunk_length=12,
            output_chunk_length=6,
            n_epochs=1,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        # Train with shorter series that fits within covariates
        model.fit(series[60:80], future_covariates=future_covariates_late[:20])

        # Should raise helpful error when series starts before covariates
        with pytest.raises(ValueError) as exc_info:
            model.backtest(
                series=series,  # Starts Jan 1
                future_covariates=future_covariates_late,  # Starts March 1
                forecast_horizon=6,
                metric=mape
            )

        error_message = str(exc_info.value)

        # Verify error message is helpful
        assert "Series starts at" in error_message, \
            "Error should show series start time"

        assert "Future covariates start at" in error_message, \
            "Error should show covariates start time"

        assert "gap" in error_message.lower() or "after" in error_message.lower(), \
            "Error should mention the gap/timing issue"

    def test_model_without_future_covariates_works_normally(self):
        """
        Test that models not requiring future_covariates work normally.

        Ensures our validation doesn't break models that don't use future covariates.
        """
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        target_data = np.random.randn(100).astype(np.float32)
        series = TimeSeries.from_times_and_values(dates, target_data)

        # NBEATSModel doesn't require future covariates
        model = NBEATSModel(
            input_chunk_length=12,
            output_chunk_length=6,
            n_epochs=1,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        model.fit(series[:70])

        # Should work fine without future_covariates
        result = model.backtest(
            series=series,
            forecast_horizon=6,
            metric=mape
        )

        assert result is not None, "Backtest should succeed for models without future covariates"

    def test_exact_boundary_case(self, sample_data):
        """
        Test that covariates extending exactly to required length work correctly.

        Edge case: covariates are exactly long enough (not too short, not extra long).
        """
        series, _ = sample_data

        forecast_horizon = 6
        # Create covariates that are exactly the right length
        exact_length = len(series) + forecast_horizon
        cov_dates = pd.date_range('2020-01-01', periods=exact_length, freq='D')
        cov_data = np.random.randn(exact_length).astype(np.float32)
        future_covariates_exact = TimeSeries.from_times_and_values(cov_dates, cov_data)

        model = TFTModel(
            input_chunk_length=12,
            output_chunk_length=6,
            n_epochs=1,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )

        model.fit(series[:70], future_covariates=future_covariates_exact[:70])

        # Should work - covariates are exactly long enough
        result = model.backtest(
            series=series,
            future_covariates=future_covariates_exact,
            forecast_horizon=forecast_horizon,
            metric=mape
        )

        assert result is not None, "Backtest should succeed when covariates are exactly long enough"