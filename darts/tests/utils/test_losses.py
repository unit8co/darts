import numpy as np

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import NBEATSModel
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.losses import MapeLoss, SmapeLoss


class LossesTestCase(DartsBaseTestClass):
    air = AirPassengersDataset().load().astype(np.float32)
    scaler = Scaler()
    air_s = scaler.fit_transform(air)
    air_train, air_val = air_s[:-36], air_s[-36:]

    def test_smape_loss(self):
        model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            num_stacks=4,
            num_blocks=1,
            layer_widths=64,
            loss_fn=SmapeLoss(),
            random_state=42,
        )

        model.fit(self.air_train, epochs=150)

        pred = model.predict(n=36)

        self.assertLess(smape(pred, self.air_s), 16.5)

    def test_mape_loss(self):
        model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            num_stacks=4,
            num_blocks=1,
            layer_widths=64,
            loss_fn=MapeLoss(),
            random_state=42,
        )

        model.fit(self.air_train, epochs=100)

        pred = model.predict(n=36)

        self.assertLess(smape(pred, self.air_s), 13.5)
