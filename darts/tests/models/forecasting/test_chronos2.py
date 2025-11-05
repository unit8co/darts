from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts.models import Chronos2Model


def load_validation_inputs():
    """Load validation inputs for Chronos2Model fidelity tests. The data imports
    here are adapted from the `20-SKLearnModel-examples` notebook.
    """
    # convert to float32 due to MPS not supporting float64
    ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

    # extract temperature, solar irradiation and rain duration
    ts_weather = ts_energy[["T [°C]", "StrGlo [W/m2]", "RainDur [min]"]]

    # extract households energy consumption
    ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

    # create train and validation splits
    validation_cutoff = pd.Timestamp("2022-01-01")
    ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
    return ts_energy_train, ts_energy_val, ts_weather


class TestChronos2Model:
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val, ts_weather = load_validation_inputs()
    # maximum prediction length w/o triggering auto-regression where the results
    # would diverge from the original implementation due to different sampling methods
    max_prediction_length = 1024

    @pytest.mark.slow
    def test_fidelity(self):
        """Test Chronos2Model predictions against original implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        """
        # load model
        model = Chronos2Model(
            input_chunk_length=8192,  # maximum context length
            output_chunk_length=self.max_prediction_length,  # maximum prediction length w/o AR
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(self.ts_energy_train, future_covariates=self.ts_weather)

        # predict on the validation inputs w/ covariates
        pred = model.predict(
            n=self.max_prediction_length,
            future_covariates=self.ts_weather,
            predict_likelihood_parameters=True,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            self.max_prediction_length, self.ts_energy_train.n_components, -1
        )

        # load the original predictions
        path = Path(__file__).parent / "fidelity" / "chronos2.npz"
        original = np.load(path)["pred"]

        # compare predictions to original
        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)
