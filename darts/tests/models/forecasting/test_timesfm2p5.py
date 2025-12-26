from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TimesFM2p5Model
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils.likelihood_models import QuantileRegression

# quantiles used during TimesFM 2.5 pre-training
all_quantiles = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]


def load_validation_inputs():
    """Load validation inputs for TimesFM2p5Model fidelity tests. The data imports
    here are adapted from the `20-SKLearnModel-examples` notebook.
    """
    # convert to float32 due to MPS not supporting float64
    ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

    # extract households energy consumption
    ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

    # create train and validation splits
    validation_cutoff = pd.Timestamp("2022-01-01")
    ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
    return ts_energy_train, ts_energy_val


class TestTimesFM2p5Model:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val = load_validation_inputs()
    # maximum prediction length w/o triggering auto-regression where the results
    # would diverge from the original implementation due to different sampling methods
    max_prediction_length = 128

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test TimesFM2p5Model predictions against original implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd
        import timesfm
        import torch

        from darts.datasets import ElectricityConsumptionZurichDataset

        # adapted from `20-SKLearnModel-examples` notebook
        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

        # extract households energy consumption
        ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

        # create train and validation splits
        validation_cutoff = pd.Timestamp("2022-01-01")
        ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)

        # set torch precision
        torch.set_float32_matmul_precision("high")

        # load TimesFM 2.5 model
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

        # compile the model with forecast configuration
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=False,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=False,
                fix_quantile_crossing=False,
            )
        )
        point_forecast, quantile_forecast = model.forecast(
            horizon=128, # use output_patch_len here to prevent auto-regressive forecasting
            inputs=[
                series
                for series in ts_energy_train.values().T
            ], # Two energy consumption series
        )

        # convert to numpy array with shape (time, variables, quantiles)
        quantile_forecast = quantile_forecast.transpose(1, 0, 2)
        # exclude the mean forecast (first quantile)
        quantile_forecast = quantile_forecast[:, :, 1:]

        # save quantiles to a npz file
        np.savez_compressed("timesfm2p5.npz", pred=quantile_forecast)
        ```

        Code accessed from https://github.com/google-research/timesfm/commit/6bd8044275f8b76cdc9554f2fecccac5f31a156c
        on 26th December 2025.

        """
        # load model
        model = TimesFM2p5Model(
            input_chunk_length=1024,  # maximum context length
            output_chunk_length=self.max_prediction_length,  # maximum prediction length w/o AR
            likelihood=(
                QuantileRegression(quantiles=all_quantiles) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(
            series=self.ts_energy_train,
        )

        # predict on the validation inputs w/ covariates
        pred = model.predict(
            n=self.max_prediction_length,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            self.max_prediction_length, self.ts_energy_train.n_components, -1
        )

        # load the original predictions
        path = (
            Path(__file__).parent
            / "artefacts"
            / "timesfm2p5"
            / "timesfm2p5_prediction"
            / "timesfm2p5.npz"
        )
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [4]]  # median quantile

        # compare predictions to original
        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)
