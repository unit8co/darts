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
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test Chronos2Model predictions against original implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd
        from chronos import BaseChronosPipeline, Chronos2Pipeline
        from darts.datasets import ElectricityConsumptionZurichDataset

        # adapted from `20-SKLearnModel-examples` notebook
        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

        # extract temperature, solar irradiation and rain duration
        ts_weather = ts_energy[["T [°C]", "StrGlo [W/m2]", "RainDur [min]"]]

        # extract households energy consumption
        ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

        # create train and validation splits
        validation_cutoff = pd.Timestamp("2022-01-01")
        ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
        ts_weather_train, ts_weather_val = ts_weather.split_after(validation_cutoff)

        prediction_length = 1024

        # load the Chronos-2 pipeline
        pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2")

        # make predictions
        quantiles, mean = pipeline.predict_quantiles(
            inputs=[
                {
                    "target": ts_energy_train.values().T,
                    "past_covariates": {
                        c: ts_weather_train[c].values().flatten()
                        for c in ts_weather.components
                    },
                    "future_covariates": {
                        c: ts_weather_val[c].values().flatten()[:prediction_length]
                        for c in ts_weather.components
                    }
                }
            ],
            prediction_length=prediction_length,
            quantile_levels=pipeline.quantiles,
        )

        # convert to numpy array with shape (time, quantile, variables)
        quantiles_np = quantiles[0].cpu().numpy()
        quantiles_np = quantiles_np.transpose(1, 0, 2)

        # save quantiles to a npz file
        np.savez_compressed("chronos2.npz", pred=quantiles_np)
        ```

        Code accessed from https://github.com/amazon-science/chronos-forecasting/commit/93419cfe9fd678b06503b3ce22113f4482c44b6f
        on 5 November 2025.

        """
        # load model
        model = Chronos2Model(
            input_chunk_length=8192,  # maximum context length
            output_chunk_length=self.max_prediction_length,  # maximum prediction length w/o AR
            probabilistic=probabilistic,
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(self.ts_energy_train, future_covariates=self.ts_weather)

        # predict on the validation inputs w/ covariates
        pred = model.predict(
            n=self.max_prediction_length,
            future_covariates=self.ts_weather,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            self.max_prediction_length, self.ts_energy_train.n_components, -1
        )

        # load the original predictions
        path = Path(__file__).parent / "fidelity" / "chronos2.npz"
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [10]]  # median quantile

        # compare predictions to original
        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)
