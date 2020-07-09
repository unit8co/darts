import shutil
import unittest
import logging
import torch

from ..models.tcn_model import TCNModel
from ..utils import timeseries_generation as tg
from .test_RNN import RNNModelTestCase


class TCNModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('.darts')

    def test_creation(self):
        with self.assertRaises(ValueError):
            # cannot choose a kernel size larger than the input length
            TCNModel(kernel_size=100, input_length=20)
        TCNModel()

    def test_fit(self):
        large_ts = tg.constant_timeseries(length=100, value=1000)
        small_ts = tg.constant_timeseries(length=100, value=10)

        # Test basic fit and predict
        model = TCNModel(n_epochs=20, num_layers=1)
        model.fit(large_ts[:98])
        pred = model.predict(n=2).values()[0]

        # Test whether model trained on one series is better than one trained on another
        model2 = TCNModel(n_epochs=20, num_layers=1)
        model2.fit(small_ts[:98])
        pred2 = model2.predict(n=2).values()[0]
        self.assertTrue(abs(pred2 - 10) < abs(pred - 10))

        # test short predict
        pred3 = model2.predict(n=1)
        self.assertEqual(len(pred3), 1)

    def test_coverage(self):
        torch.manual_seed(0)
        input_lengths = range(20, 50)
        kernel_sizes = range(2, 5)
        dilation_bases = range(2, 5)

        for kernel_size in kernel_sizes:
            for dilation_base in dilation_bases:
                if dilation_base > kernel_size:
                    continue
                for input_length in input_lengths:

                    # create model with all weights set to one
                    model = TCNModel(kernel_size=kernel_size, dilation_base=dilation_base, input_length=input_length,
                                     weight_norm=False)
                    for res_block in model.model.res_blocks:
                        res_block.conv1.weight = torch.nn.Parameter(torch.ones(res_block.conv1.weight.shape))
                        res_block.conv2.weight = torch.nn.Parameter(torch.ones(res_block.conv2.weight.shape))

                    model.model.eval()
                    input_tensor = torch.zeros([1, input_length, 1], dtype=torch.float)
                    zero_output = model.model.forward(input_tensor).float()[0, -1, 0]

                    # test for full coverage
                    for i in range(input_length):
                        input_tensor[0, i, 0] = 1
                        curr_output = model.model.forward(input_tensor).float()[0, -1, 0]
                        self.assertNotEqual(zero_output, curr_output)
                        input_tensor[0, i, 0] = 0

                    # create model with all weights set to one and one layer less than is automatically detected
                    model_2 = TCNModel(kernel_size=kernel_size, dilation_base=dilation_base, input_length=input_length,
                                       weight_norm=False, num_layers=model.model.num_layers - 1)
                    for res_block in model_2.model.res_blocks:
                        res_block.conv1.weight = torch.nn.Parameter(torch.ones(res_block.conv1.weight.shape))
                        res_block.conv2.weight = torch.nn.Parameter(torch.ones(res_block.conv2.weight.shape))

                    model_2.model.eval()
                    input_tensor = torch.zeros([1, input_length, 1], dtype=torch.float)
                    zero_output = model_2.model.forward(input_tensor).float()[0, -1, 0]

                    # test for incomplete coverage
                    uncovered_input_found = False
                    if model_2.model.num_layers == 1:
                        continue
                    for i in range(input_length):
                        input_tensor[0, i, 0] = 1
                        curr_output = model_2.model.forward(input_tensor).float()[0, -1, 0]
                        if (zero_output == curr_output):
                            uncovered_input_found = True
                            break
                        input_tensor[0, i, 0] = 0
                    self.assertTrue(uncovered_input_found)

    def test_use_full_output_length(self):
        series = tg.linear_timeseries(length=100)
        RNNModelTestCase.helper_test_use_full_output_length(self, TCNModel, series)

    def test_multivariate(self):
        series_multivariate = tg.linear_timeseries(length=100).stack(tg.linear_timeseries(length=100))
        RNNModelTestCase.helper_test_multivariate(self, TCNModel, series_multivariate)
