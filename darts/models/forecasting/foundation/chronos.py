"""
Chronos Foundation Model Implementation

Amazon Chronos family of time series foundation models with lazy imports.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from darts.utils.likelihood_models.base import quantile_names, likelihood_component_names

from .base import FoundationForecastingModel

logger = get_logger(__name__)


def _check_chronos_available():
    """
    Check if chronos-forecasting is available and provide helpful error if not.

    Raises
    ------
    ImportError
        If chronos-forecasting package is not installed, with instructions.
    """
    try:
        import chronos  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'chronos-forecasting' package is required for ChronosModel but is not installed.\n"
            "\n"
            "Install it with:\n"
            "  uv pip install 'darts[chronos]'\n"
            "\n"
            "Or with pip:\n"
            "  pip install 'darts[chronos]'\n"
            "\n"
            "This will install chronos-forecasting>=2.0.0 from PyPI.\n"
            "See INSTALL.md for more details."
        ) from e


def _timeseries_to_chronos_df(
    series: Union[TimeSeries, List[TimeSeries]],
    past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    series_id_prefix: str = "series",
    is_future_df: bool = False
) -> pd.DataFrame:
    """
    Convert Darts TimeSeries to Chronos DataFrame format with covariate support.

    Chronos expects:
    - id column: identifies different time series
    - timestamp column: datetime information
    - target column(s): values to predict (univariate or multivariate)
    - covariate columns: past/future external variables

    Parameters
    ----------
    series : TimeSeries or List[TimeSeries]
        Input time series (omit for future_df)
    past_covariates : TimeSeries or List[TimeSeries], optional
        Past covariates to include in context_df
    future_covariates : TimeSeries or List[TimeSeries], optional
        Future covariates to include in both context_df and future_df
    series_id_prefix : str, default="series"
        Prefix for series IDs
    is_future_df : bool, default=False
        If True, create future_df format (no target columns, only covariates)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [id, timestamp, target(s), covariate columns]
    """
    # Normalize to lists
    series_list = [series] if isinstance(series, TimeSeries) else series if series else []
    past_cov_list = [past_covariates] if isinstance(past_covariates, TimeSeries) else past_covariates if past_covariates else []
    future_cov_list = [future_covariates] if isinstance(future_covariates, TimeSeries) else future_covariates if future_covariates else []

    # Determine number of series
    n_series = max(len(series_list), len(past_cov_list), len(future_cov_list))

    # Broadcast single-element lists
    if series_list and len(series_list) == 1 and n_series > 1:
        series_list = series_list * n_series
    if past_cov_list and len(past_cov_list) == 1 and n_series > 1:
        past_cov_list = past_cov_list * n_series
    if future_cov_list and len(future_cov_list) == 1 and n_series > 1:
        future_cov_list = future_cov_list * n_series

    dfs = []
    for idx in range(n_series):
        # Start with ID
        row_data = {"id": f"{series_id_prefix}_{idx}"}

        # Add target series if not future_df
        if not is_future_df and series_list:
            ts = series_list[idx]
            ts_df = ts.to_dataframe(time_as_index=False)

            # Extract timestamp
            time_col_name = ts_df.columns[0]
            ts_df = ts_df.rename(columns={time_col_name: "timestamp"})

            # Rename value columns to target/target_0/target_1...
            value_cols = [col for col in ts_df.columns if col != "timestamp"]
            if len(value_cols) == 1:
                ts_df = ts_df.rename(columns={value_cols[0]: "target"})
            else:
                rename_map = {old: f"target_{i}" for i, old in enumerate(value_cols)}
                ts_df = ts_df.rename(columns=rename_map)

            # Merge into result
            result_df = ts_df.copy()
            result_df.insert(0, "id", row_data["id"])
        else:
            # For future_df, create empty timestamp placeholder
            result_df = pd.DataFrame(row_data, index=[0])

        # Add past covariates (only in context_df, not future_df)
        if not is_future_df and past_cov_list and idx < len(past_cov_list):
            past_cov = past_cov_list[idx]
            past_df = past_cov.to_dataframe(time_as_index=True)
            # Prefix columns to avoid conflicts
            past_df.columns = [f"past_cov_{col}" for col in past_df.columns]
            # Merge on timestamp
            if "timestamp" in result_df.columns:
                result_df = result_df.merge(past_df, left_on="timestamp", right_index=True, how="left")

        # Add future covariates (in both context_df and future_df)
        if future_cov_list and idx < len(future_cov_list):
            future_cov = future_cov_list[idx]
            future_df = future_cov.to_dataframe(time_as_index=True)
            # Keep original column names (shared between context and future)
            if "timestamp" in result_df.columns:
                result_df = result_df.merge(future_df, left_on="timestamp", right_index=True, how="left")

        dfs.append(result_df)

    # Concatenate all series
    return pd.concat(dfs, ignore_index=True)


def _create_future_df(
    series: Union[TimeSeries, List[TimeSeries]],
    future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
    n: int,
    series_id_prefix: str = "series"
) -> Optional[pd.DataFrame]:
    """
    Create future DataFrame for Chronos with future covariate values.

    Generates future timestamps based on series frequency and extracts
    the corresponding future covariate values for the forecast horizon.

    Parameters
    ----------
    series : TimeSeries or List[TimeSeries]
        Target series (used for timestamp generation and frequency)
    future_covariates : TimeSeries, List[TimeSeries], or None
        Future covariates to include in the future DataFrame
    n : int
        Forecast horizon (number of future steps)
    series_id_prefix : str, default="series"
        Prefix for series IDs in the DataFrame

    Returns
    -------
    pd.DataFrame or None
        Future DataFrame with columns [id, timestamp, covariate_columns]
        Returns None if future_covariates is None
    """
    if future_covariates is None:
        return None

    # Normalize inputs to lists
    was_single_series = not isinstance(series, list)
    series_list = [series] if was_single_series else series

    was_single_cov = not isinstance(future_covariates, list)
    cov_list = [future_covariates] if was_single_cov else future_covariates

    # Broadcast single covariate to all series
    if len(cov_list) == 1 and len(series_list) > 1:
        cov_list = cov_list * len(series_list)

    dfs = []
    for idx, (ts, future_cov) in enumerate(zip(series_list, cov_list)):
        series_id = f"{series_id_prefix}_{idx}"

        # Generate future timestamps based on series frequency
        last_timestamp = ts.end_time()
        freq = ts.freq
        future_timestamps = pd.date_range(
            start=last_timestamp + freq,
            periods=n,
            freq=freq
        )

        # Create base DataFrame with id and timestamp
        future_df = pd.DataFrame({
            "id": series_id,
            "timestamp": future_timestamps
        })

        # Extract future covariate values for forecast horizon
        future_cov_df = future_cov.to_dataframe(time_as_index=True)

        # Filter to the forecast horizon timeframe
        future_cov_df = future_cov_df.loc[future_timestamps]

        # Merge covariates with future DataFrame
        future_df = future_df.merge(
            future_cov_df,
            left_on="timestamp",
            right_index=True,
            how="left"
        )

        dfs.append(future_df)

    return pd.concat(dfs, ignore_index=True)


def _chronos_df_to_timeseries(
    pred_df: pd.DataFrame,
    original_series: Union[TimeSeries, List[TimeSeries]],
    n: int,
    quantiles: Optional[List[float]] = None
) -> Union[TimeSeries, List[TimeSeries]]:
    """
    Convert Chronos prediction DataFrame back to Darts TimeSeries.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Chronos prediction output with columns: [id, timestamp, quantile columns]
    original_series : TimeSeries or List[TimeSeries]
        Original series for metadata (freq, component names)
    n : int
        Forecast horizon
    quantiles : List[float], optional
        Quantile levels that were predicted

    Returns
    -------
    TimeSeries or List[TimeSeries]
        Predicted time series in Darts format with proper quantile naming
    """
    # Normalize original series to list
    was_single = not isinstance(original_series, list)
    if was_single:
        series_list = [original_series]
    else:
        series_list = original_series

    results = []
    unique_ids = pred_df["id"].unique()

    for idx, series_id in enumerate(unique_ids):
        # Get predictions for this series
        series_df = pred_df[pred_df["id"] == series_id].copy()

        # Get original series for metadata
        orig_ts = series_list[idx]

        # Set timestamp as index
        series_df = series_df.set_index("timestamp")

        # Remove id column
        series_df = series_df.drop(columns=["id"], errors="ignore")

        # Convert to TimeSeries
        # Check if we have quantile columns (probabilistic forecast)
        quantile_cols = [col for col in series_df.columns if col.replace(".", "").replace("-", "").isdigit()]

        if len(quantile_cols) > 1 and quantiles is not None:
            # Probabilistic: Create stochastic TimeSeries with quantiles as components
            # Use Darts quantile naming convention

            # Check if this is multivariate (has target_name column from Chronos long format)
            if 'target_name' in series_df.columns and orig_ts.n_components > 1:
                # Multivariate case: Chronos returns data in long format
                # Need to pivot to wide format: (time, component×quantile)

                # Map target_0, target_1, ... back to original component names
                target_names = series_df['target_name'].unique()
                target_to_component = {f"target_{i}": comp for i, comp in enumerate(orig_ts.components)}

                # Create a wide dataframe: one row per timestamp, columns for each component×quantile
                # First, reset index to have timestamp as column
                wide_df = series_df.reset_index() if 'timestamp' not in series_df.columns else series_df.copy()

                # Pivot: index=timestamp, columns=target_name, values=each quantile
                pivoted_dfs = []
                for q_col in quantile_cols:
                    pivot = wide_df.pivot(index='timestamp' if 'timestamp' in wide_df.columns else wide_df.index.name,
                                         columns='target_name',
                                         values=q_col)
                    # Rename columns from target_0, target_1 to component names
                    pivot.columns = [target_to_component.get(col, col) for col in pivot.columns]
                    pivoted_dfs.append(pivot)

                # Now build the final structure: (time_steps, n_components * n_quantiles)
                # Order: comp0_q0, comp0_q1, ..., comp1_q0, comp1_q1, ...
                param_names = quantile_names(quantiles)
                column_names = likelihood_component_names(
                    components=orig_ts.components,
                    parameter_names=param_names
                )

                # Stack the values in the correct order
                all_values = []
                for comp in orig_ts.components:
                    for q_col in quantile_cols:
                        # Find the corresponding pivoted df for this quantile
                        q_idx = quantile_cols.index(q_col)
                        all_values.append(pivoted_dfs[q_idx][comp].values)

                # Shape: (time_steps, n_components * n_quantiles)
                quantile_values = np.column_stack(all_values)

                # Use the first pivoted df's index as timestamps
                timestamps = pivoted_dfs[0].index
            else:
                # Univariate case: standard processing
                quantile_values = series_df[quantile_cols].values  # (time_steps, n_quantiles)

                # Generate proper quantile names using Darts convention
                param_names = quantile_names(quantiles)
                column_names = param_names
                timestamps = series_df.index

            # Reshape to (time_steps, n_components * n_quantiles, 1 sample)
            forecast_values = quantile_values[:, :, np.newaxis]

            forecast_ts = TimeSeries.from_times_and_values(
                times=timestamps,
                values=forecast_values,
                freq=orig_ts.freq,
                columns=column_names
            )
        elif "0.5" in series_df.columns:
            # Deterministic: Use median (0.5 quantile) as point forecast
            forecast_values = series_df["0.5"].values
            forecast_ts = TimeSeries.from_times_and_values(
                times=series_df.index,
                values=forecast_values.reshape(-1, 1),
                freq=orig_ts.freq,
                columns=orig_ts.components if orig_ts.n_components == 1 else None
            )
        else:
            # Fallback: Use first value column
            forecast_values = series_df.iloc[:, 0].values
            forecast_ts = TimeSeries.from_times_and_values(
                times=series_df.index,
                values=forecast_values.reshape(-1, 1),
                freq=orig_ts.freq,
                columns=orig_ts.components if orig_ts.n_components == 1 else None
            )

        results.append(forecast_ts)

    # Return in same format as input
    return results[0] if was_single else results


class ChronosModel(FoundationForecastingModel):
    """Chronos 2 foundation model for time series forecasting.

    Amazon's Chronos 2 (120M parameters) supports multivariate forecasting,
    covariates, and fine-tuning with PEFT/LoRA.

    Parameters
    ----------
    context_length : int, optional
        Context window (default: 2048, max: 8192). Must be multiple of 16.
    max_forecast_horizon : int, optional
        Maximum forecast horizon (default: 1024).
    lora_config : dict, optional
        LoRA config for fine-tuning (e.g., {"r": 8, "lora_alpha": 16}).

    Examples
    --------
    >>> from darts.models import ChronosModel
    >>> model = ChronosModel()
    >>> forecast = model.predict(n=24, series=train)
    >>>
    >>> # With covariates
    >>> forecast = model.predict(n=24, series=sales,
    ...                          past_covariates=weather,
    ...                          future_covariates=promotions)

    Notes
    -----
    - Supports multivariate, past/future covariates
    - 21 quantile levels [0.01, 0.05, ..., 0.95, 0.99]
    - Install: pip install 'darts[chronos]'

    References
    ----------
    .. [1] Ansari et al. (2024). Chronos: Learning the Language of Time Series.
    """

    # Model constraints (from chronos-2-base specification)
    _HARD_MAX_CONTEXT = 8192
    _HARD_MAX_HORIZON = 1024
    _PATCH_SIZE = 16
    _DEFAULT_CONTEXT_LENGTH = 2048

    # Supported quantile levels
    SUPPORTED_QUANTILES = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    ]

    def __init__(
        self,
        model_id: str = "s3://autogluon/chronos-2",
        device: str = "auto",
        context_length: Optional[int] = None,
        max_forecast_horizon: Optional[int] = None,
        lora_config: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize Chronos model.

        Parameters
        ----------
        model_id : str, default="s3://autogluon/chronos-2"
            Model identifier for Chronos 2. Can be:
            - S3 path: "s3://autogluon/chronos-2" (120M params, default)
            - HuggingFace: Not currently available for Chronos 2
            - Local path: path to downloaded model
        device : str, default="auto"
            Device to use ("auto", "cuda", "mps", "cpu")
        context_length : int, optional
            MINIMUM context length enforced by this model instance for validation.
            If None, uses the model's default (2048).
            Must be divisible by patch_size (16) and ≤ hard max (8192).
            Example: For 144-point series with start=0.75 backtesting, use 96
            (0.75 × 144 ≈ 108, rounded down to nearest multiple of 16).
        max_forecast_horizon : int, optional
            MAXIMUM forecast horizon enforced by this model instance.
            If None, uses the model's hard limit (1024).
            Must be divisible by patch_size (16) and ≤ hard max (1024).
            Useful for constraining forecasts to your use case requirements.
        lora_config : dict, optional
            LoRA configuration for fine-tuning.
        **kwargs
            Additional arguments passed to FoundationForecastingModel.
        """
        # Check chronos-forecasting is available
        _check_chronos_available()

        # Pass device to base class for unified device management
        super().__init__(device=device, lora_config=lora_config, **kwargs)

        # Set architectural limits from class constants
        self._hard_max_context = self._HARD_MAX_CONTEXT
        self._hard_max_horizon = self._HARD_MAX_HORIZON
        self._patch_size = self._PATCH_SIZE
        self._default_context_length = self._DEFAULT_CONTEXT_LENGTH

        # Validate and set user's minimum context_length preference
        if context_length is None:
            self.context_length = self._default_context_length
        else:
            raise_if_not(
                context_length <= self._hard_max_context,
                f"context_length={context_length} exceeds model maximum of {self._hard_max_context}",
                logger
            )
            raise_if_not(
                context_length % self._patch_size == 0,
                f"context_length={context_length} must be divisible by patch_size={self._patch_size}",
                logger
            )
            raise_if_not(
                context_length >= self._patch_size,
                f"context_length={context_length} must be at least patch_size={self._patch_size}",
                logger
            )
            self.context_length = context_length

        # Validate and set user's maximum forecast_horizon preference
        if max_forecast_horizon is None:
            self.max_forecast_horizon = self._hard_max_horizon
        else:
            raise_if_not(
                max_forecast_horizon <= self._hard_max_horizon,
                f"max_forecast_horizon={max_forecast_horizon} exceeds model maximum of {self._hard_max_horizon}",
                logger
            )
            raise_if_not(
                max_forecast_horizon % self._patch_size == 0,
                f"max_forecast_horizon={max_forecast_horizon} must be divisible by patch_size={self._patch_size}",
                logger
            )
            raise_if_not(
                max_forecast_horizon >= self._patch_size,
                f"max_forecast_horizon={max_forecast_horizon} must be at least patch_size={self._patch_size}",
                logger
            )
            self.max_forecast_horizon = max_forecast_horizon

        self.model_id = model_id
        # Note: self.device now set by base class
        # Note: self._model now managed by base class (not self._pipeline)

    def load_model(self):
        """
        Load Chronos2Pipeline from pretrained source.

        Returns
        -------
        pipeline
            Loaded Chronos2Pipeline ready for inference.
        """
        from chronos import Chronos2Pipeline

        logger.info(f"Loading Chronos2Pipeline from '{self.model_id}'...")

        pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device if self.device != "auto" else None
        )

        logger.info("✓ Chronos2Pipeline loaded successfully")
        return pipeline

    @property
    def min_train_samples(self) -> int:
        """Minimum number of samples required for training."""
        return 1  # Chronos 2 can work with minimal context

    @property
    def supports_multivariate(self) -> bool:
        """Chronos 2 supports multivariate forecasting."""
        return True

    @property
    def extreme_lags(self) -> tuple:
        """
        Returns the extreme lags used by the model.

        For Chronos 2:
        - Can use up to context_length historical points
        - Supports both past and future covariates
        - Past covariates share the same context window as the target series
        - Future covariates must extend at least to the forecast horizon (validated at predict() time)
        - No output chunk shift

        Returns
        -------
        tuple
            (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag,
             min_future_cov_lag, max_future_cov_lag, output_chunk_shift)
        """
        return (
            -self.context_length,  # min_target_lag: lookback window
            0,                      # max_target_lag: no future target values
            -self.context_length,  # min_past_cov_lag: same lookback as target
            0,                      # max_past_cov_lag: up to present
            0,                      # min_future_cov_lag: from present onward
            self.context_length,   # max_future_cov_lag: at least context_length (extendable to forecast horizon at predict() time)
            0,                      # output_chunk_shift: no shift
        )

    def _target_window_lengths(self) -> tuple:
        """
        Returns the input and output window lengths for the model.
        For Chronos 2: (context_length, 0) - configurable context and arbitrary forecast horizon
        """
        return self.context_length, 0  # 0 means arbitrary forecast horizon

    def _model_encoder_settings(self) -> tuple:
        """
        Returns encoder settings.
        Chronos 2 supports both past and future covariates.
        """
        return -1, -1, True, True  # -1 = unlimited, both covariate types supported

    def _apply_peft(self) -> None:
        """
        Apply PEFT configuration to Chronos 2 model.

        This method applies LoRA adapters to the T5 encoder-decoder model
        within the Chronos2Pipeline, enabling parameter-efficient fine-tuning.

        Raises
        ------
        RuntimeError
            If model is not loaded or PEFT library is not available.
        """
        # Ensure base model is loaded
        if not self._is_loaded:
            _ = self.model  # Trigger lazy loading

        # Extract T5 model from Chronos2Pipeline
        # Chronos2Pipeline.model is T5ForConditionalGeneration
        base_model = self.model.model

        # Apply LoRA adapters using our utility
        from darts.models.forecasting.foundation.peft_utils import create_lora_model

        logger.info("Applying PEFT/LoRA adapters to Chronos 2 model")
        self._peft_model = create_lora_model(base_model, self.lora_config)

        # Replace the T5 model in the pipeline with PEFT-enhanced version
        self.model.model = self._peft_model

        self._is_peft_applied = True
        logger.info("PEFT adapters applied successfully")

    def _train_with_peft(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "ChronosModel":
        """
        Fine-tune PEFT adapters on provided time series data.

        This method trains LoRA adapters using the Chronos 2 upstream fit()
        implementation, which handles the training loop, loss computation,
        and optimizer configuration.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Training time series data.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Historical covariates for training.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates for training.
        **kwargs
            Training parameters:
                - prediction_length (int): Forecast horizon for tuning (default: 24)
                - learning_rate (float): Optimizer learning rate (default: 1e-5)
                - num_steps (int): Training iterations (default: 1000)
                - batch_size (int): Training batch size (default: 32)
                - output_dir (str): Checkpoint directory (default: './chronos_lora_adapters')
                - validation_series (TimeSeries or List[TimeSeries]): Validation data
                - **extra_trainer_kwargs: Additional TrainingArguments parameters

        Returns
        -------
        self
            ChronosModel with fine-tuned PEFT adapters.

        Raises
        ------
        RuntimeError
            If PEFT has not been applied via _apply_peft().
        """
        # Validate PEFT has been applied
        if not self._is_peft_applied:
            raise RuntimeError(
                "PEFT adapters must be applied before training. "
                "This should happen automatically when lora_config is provided."
            )

        logger.info("Fine-tuning PEFT adapters on provided data")

        # Convert Darts TimeSeries to Chronos DataFrame format
        training_df = _timeseries_to_chronos_df(series, past_covariates, future_covariates)

        # Extract training parameters from kwargs
        prediction_length = kwargs.get('prediction_length', 24)
        learning_rate = kwargs.get('learning_rate', 1e-5)
        num_steps = kwargs.get('num_steps', 1000)
        batch_size = kwargs.get('batch_size', 32)
        output_dir = kwargs.get('output_dir', './chronos_lora_adapters')

        # Handle validation data if provided
        validation_df = None
        if 'validation_series' in kwargs:
            val_series = kwargs['validation_series']
            val_past_cov = kwargs.get('validation_past_covariates', None)
            val_future_cov = kwargs.get('validation_future_covariates', None)
            validation_df = _timeseries_to_chronos_df(val_series, val_past_cov, val_future_cov)

        # Call upstream Chronos fit() - handles entire training loop
        logger.info(
            f"Starting fine-tuning: {num_steps} steps, "
            f"batch_size={batch_size}, lr={learning_rate}"
        )

        # Chronos2Pipeline.fit() returns a NEW pipeline with fine-tuned model
        self.model = self.model.fit(
            inputs=training_df,
            prediction_length=prediction_length,
            validation_inputs=validation_df,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=output_dir,
            **{k: v for k, v in kwargs.items()
               if k not in ['prediction_length', 'learning_rate', 'num_steps',
                           'batch_size', 'output_dir', 'validation_series',
                           'validation_past_covariates', 'validation_future_covariates']}
        )

        logger.info(f"Fine-tuning complete: {num_steps} training steps finished")
        return self

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        num_samples: int = 1,
        quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Generate forecasts using Chronos 2 with optional covariate support.

        Chronos 2 natively supports both past and future covariates, enabling
        exogenous variable integration for improved forecasting accuracy.

        Parameters
        ----------
        n : int
            Number of time steps to forecast.
        series : TimeSeries or List[TimeSeries], optional
            Input series for context. Required for zero-shot usage.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates to condition the forecast on.
            These are exogenous variables observed in the historical context.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates known in advance for the forecast horizon.
            These must extend at least n steps beyond the end of series.
        num_samples : int, default=1
            Number of probabilistic samples to generate.
            When num_samples=1, returns point forecast (median).
            When num_samples>1, returns probabilistic forecast.
        quantiles : List[float], optional
            Specific quantile levels to predict. If None, defaults to 9 quantiles
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for probabilistic forecasts
            or [0.5] for point forecasts. Chronos 2 supports up to 21 quantiles.
        **kwargs
            Additional prediction parameters.

        Returns
        -------
        TimeSeries or List[TimeSeries]
            Forecasted time series.

        Raises
        ------
        ValueError
            If series is None or incompatible with model capabilities.

        Examples
        --------
        >>> model = ChronosModel()
        >>> # Basic forecast
        >>> forecast = model.predict(n=24, series=train_series)
        >>> # Probabilistic forecast
        >>> prob_forecast = model.predict(n=24, series=train_series, num_samples=100)
        >>> # Forecast with covariates
        >>> forecast = model.predict(
        ...     n=24,
        ...     series=train_series,
        ...     past_covariates=past_cov,
        ...     future_covariates=future_cov
        ... )
        >>> # Custom quantile levels
        >>> forecast = model.predict(
        ...     n=24,
        ...     series=train_series,
        ...     quantiles=[0.1, 0.5, 0.9]
        ... )
        """
        # Validate inputs
        raise_if_not(
            series is not None,
            "series is required for zero-shot forecasting with ChronosModel"
        )

        # Truncate series to context_length (use last N points)
        # This ensures consistent behavior with TimesFM and respects user's context_length setting
        if isinstance(series, list):
            truncated_series = [s[-self.context_length:] if len(s) > self.context_length else s for s in series]
        else:
            truncated_series = series[-self.context_length:] if len(series) > self.context_length else series

        # Truncate past_covariates to align with truncated series
        if past_covariates is not None:
            if isinstance(past_covariates, list):
                truncated_past_cov = [pc[-self.context_length:] if len(pc) > self.context_length else pc for pc in past_covariates]
            else:
                truncated_past_cov = past_covariates[-self.context_length:] if len(past_covariates) > self.context_length else past_covariates
        else:
            truncated_past_cov = None

        # Convert Darts TimeSeries to Chronos DataFrame format with covariates
        logger.debug(f"Converting TimeSeries to Chronos DataFrame format (context_length={self.context_length})...")
        context_df = _timeseries_to_chronos_df(truncated_series, truncated_past_cov, future_covariates)

        # Create future DataFrame with future covariates if provided
        future_df = _create_future_df(series, future_covariates, n)

        # Determine quantile levels for probabilistic forecasting
        if quantiles is None:
            if num_samples > 1:
                # Generate multiple quantiles for probabilistic forecast
                quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            else:
                # Just get the median for point forecast
                quantiles = [0.5]

        # Call Chronos2Pipeline.predict_df()
        logger.debug(f"Calling Chronos2Pipeline.predict_df(prediction_length={n}, quantiles={quantiles})...")

        # Extract target columns (handles both univariate and multivariate)
        target_cols = [col for col in context_df.columns if col.startswith("target")]
        target = target_cols[0] if len(target_cols) == 1 else target_cols if target_cols else None

        pred_df = self.model.predict_df(
            context_df,
            prediction_length=n,
            quantile_levels=quantiles,  # Note: Chronos library still uses quantile_levels internally
            future_df=future_df,
            id_column="id",
            timestamp_column="timestamp",
            target=target,  # Pass list for multivariate, string for univariate
            **kwargs
        )

        # Convert predictions back to Darts TimeSeries
        logger.debug("Converting predictions back to TimeSeries format...")
        forecast = _chronos_df_to_timeseries(pred_df, series, n, quantiles=quantiles)

        logger.info(f"Generated {n}-step forecast successfully")
        return forecast
