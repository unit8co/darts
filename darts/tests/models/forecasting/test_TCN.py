import pytest

from darts.metrics import mae
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import torch

from darts.models.forecasting.tcn_model import TCNModel


class TestTCNModel:
    def test_creation(self):
        with pytest.raises(ValueError):
            # cannot choose a kernel size larger than the input length
            TCNModel(input_chunk_length=20, output_chunk_length=1, kernel_size=100)
        TCNModel(input_chunk_length=12, output_chunk_length=1)

    def test_fit(self):
        large_ts = tg.constant_timeseries(length=100, value=1000)
        small_ts = tg.constant_timeseries(length=100, value=10)

        # Test basic fit and predict
        model = TCNModel(
            input_chunk_length=12,
            output_chunk_length=1,
            n_epochs=10,
            num_layers=1,
            **tfm_kwargs,
        )
        model.fit(large_ts[:98])
        pred = model.predict(n=2).values()[0]

        # Test whether model trained on one series is better than one trained on another
        model2 = TCNModel(
            input_chunk_length=12,
            output_chunk_length=1,
            n_epochs=10,
            num_layers=1,
            **tfm_kwargs,
        )
        model2.fit(small_ts[:98])
        pred2 = model2.predict(n=2).values()[0]
        assert abs(pred2 - 10) < abs(pred - 10)

        # test short predict
        pred3 = model2.predict(n=1)
        assert len(pred3) == 1

    def test_performance(self):
        # test TCN performance on dummy time series
        ts = tg.sine_timeseries(length=100) + tg.linear_timeseries(
            length=100, end_value=2
        )
        train, test = ts[:90], ts[90:]
        model = TCNModel(
            input_chunk_length=12,
            output_chunk_length=10,
            n_epochs=300,
            random_state=0,
            **tfm_kwargs,
        )
        model.fit(train)
        pred = model.predict(n=10)

        assert mae(pred, test) < 0.3

    @pytest.mark.slow
    def test_coverage(self):
        torch.manual_seed(0)
        input_chunk_lengths = range(20, 50)
        kernel_sizes = range(2, 5)
        dilation_bases = range(2, 5)

        for kernel_size in kernel_sizes:
            for dilation_base in dilation_bases:
                if dilation_base > kernel_size:
                    continue
                for input_chunk_length in input_chunk_lengths:
                    # create model with all weights set to one
                    model = TCNModel(
                        input_chunk_length=input_chunk_length,
                        output_chunk_length=1,
                        kernel_size=kernel_size,
                        dilation_base=dilation_base,
                        weight_norm=False,
                        n_epochs=1,
                        **tfm_kwargs,
                    )

                    # we have to fit the model on a dummy series in order to create the internal nn.Module
                    model.fit(tg.gaussian_timeseries(length=100))

                    for res_block in model.model.res_blocks:
                        res_block.conv1.weight = torch.nn.Parameter(
                            torch.ones(
                                res_block.conv1.weight.shape, dtype=torch.float64
                            )
                        )
                        res_block.conv2.weight = torch.nn.Parameter(
                            torch.ones(
                                res_block.conv2.weight.shape, dtype=torch.float64
                            )
                        )

                    model.model.eval()

                    # also disable MC Dropout:
                    model.model.set_mc_dropout(False)

                    input_tensor = torch.zeros(
                        [1, input_chunk_length, 1], dtype=torch.float64
                    )
                    zero_output = model.model.forward((input_tensor, None))[0, -1, 0]

                    # test for full coverage
                    for i in range(input_chunk_length):
                        input_tensor[0, i, 0] = 1
                        curr_output = model.model.forward((input_tensor, None))[
                            0, -1, 0
                        ]
                        assert zero_output != curr_output
                        input_tensor[0, i, 0] = 0

                    # create model with all weights set to one and one layer less than is automatically detected
                    model_2 = TCNModel(
                        input_chunk_length=input_chunk_length,
                        output_chunk_length=1,
                        kernel_size=kernel_size,
                        dilation_base=dilation_base,
                        weight_norm=False,
                        num_layers=model.model.num_layers - 1,
                        n_epochs=1,
                        **tfm_kwargs,
                    )

                    # we have to fit the model on a dummy series in order to create the internal nn.Module
                    model_2.fit(tg.gaussian_timeseries(length=100))

                    for res_block in model_2.model.res_blocks:
                        res_block.conv1.weight = torch.nn.Parameter(
                            torch.ones(
                                res_block.conv1.weight.shape, dtype=torch.float64
                            )
                        )
                        res_block.conv2.weight = torch.nn.Parameter(
                            torch.ones(
                                res_block.conv2.weight.shape, dtype=torch.float64
                            )
                        )

                    model_2.model.eval()

                    # also disable MC Dropout:
                    model_2.model.set_mc_dropout(False)

                    input_tensor = torch.zeros(
                        [1, input_chunk_length, 1], dtype=torch.float64
                    )
                    zero_output = model_2.model.forward((input_tensor, None))[0, -1, 0]

                    # test for incomplete coverage
                    uncovered_input_found = False
                    if model_2.model.num_layers == 1:
                        continue
                    for i in range(input_chunk_length):
                        input_tensor[0, i, 0] = 1
                        curr_output = model_2.model.forward((input_tensor, None))[
                            0, -1, 0
                        ]
                        if zero_output == curr_output:
                            uncovered_input_found = True
                            break
                        input_tensor[0, i, 0] = 0
                    assert uncovered_input_found

    def helper_test_pred_length(self, pytorch_model, series):
        model = pytorch_model(
            input_chunk_length=12, output_chunk_length=3, n_epochs=1, **tfm_kwargs
        )
        model.fit(series)
        pred = model.predict(7)
        assert len(pred) == 7
        pred = model.predict(2)
        assert len(pred) == 2
        assert pred.width == 1
        pred = model.predict(4)
        assert len(pred) == 4
        assert pred.width == 1

    def test_pred_length(self):
        series = tg.linear_timeseries(length=100)
        self.helper_test_pred_length(TCNModel, series)
