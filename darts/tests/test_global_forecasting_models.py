import pandas as pd
import numpy as np

from .base_test_class import DartsBaseTestClass
from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape
from ..logging import get_logger
from ..dataprocessing.transformers import Scaler

logger = get_logger(__name__)

try:
    from ..models import RNNModel, TCNModel, TransformerModel, NBEATSModel
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not installed - will be skipping Torch models tests')
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12
    models_cls_kwargs_errs = [
        (RNNModel, {'model': 'RNN', 'hidden_size': 10, 'n_rnn_layers': 1, 'batch_size': 32, 'n_epochs': 10}, 180.),
        (TCNModel, {'n_epochs': 10, 'batch_size': 32}, 240.),
        (TransformerModel, {'d_model': 16, 'nhead': 2, 'num_encoder_layers': 2, 'num_decoder_layers': 2,
                            'dim_feedforward': 16, 'batch_size': 32, 'n_epochs': 10}, 180.),
        (NBEATSModel, {'num_stacks': 4, 'num_blocks': 1, 'num_layers': 2, 'layer_widths': 12, 'n_epochs': 10}, 180.)
    ]

    class GlobalForecastingModelsTestCase(DartsBaseTestClass):
        # forecasting horizon used in runnability tests
        forecasting_horizon = 12

        np.random.seed(42)
        torch.manual_seed(42)

        # real timeseries for functionality tests
        df = pd.read_csv('examples/AirPassengers.csv', delimiter=",")
        ts_passengers = TimeSeries.from_dataframe(df, 'Month', ['#Passengers'])
        scaler = Scaler()
        ts_passengers = scaler.fit_transform(ts_passengers)
        ts_pass_train, ts_pass_val = ts_passengers[:-36], ts_passengers[-36:]

        # an additional noisy series
        ts_pass_train_1 = ts_pass_train + 0.01 * tg.gaussian_timeseries(length=len(ts_pass_train),
                                                                        freq=ts_pass_train.freq_str(),
                                                                        start_ts=ts_pass_train.start_time())

        # an additional time series serving as covariates
        year_series = tg.datetime_attribute_timeseries(ts_passengers, attribute='year')
        month_series = tg.datetime_attribute_timeseries(ts_passengers, attribute='month')
        scaler_dt = Scaler()
        time_covariates = scaler_dt.fit_transform(year_series.stack(month_series))
        time_covariates_train, time_covariates_val = time_covariates[:-36], time_covariates[-36:]

        def test_single_ts(self):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit(self.ts_pass_train)
                pred = model.predict(n=36)
                mape_err = mape(self.ts_pass_val, pred)
                self.assertTrue(mape_err < err, 'Model {} produces errors too high (one time '
                                                'series). Error = {}'.format(model_cls, mape_err))

        def test_multi_ts(self):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit([self.ts_pass_train, self.ts_pass_train_1])
                with self.assertRaises(ValueError):
                    # when model is fit from >1 series, one must provide a series in argument
                    model.predict(n=1)
                pred = model.predict(n=36, series=self.ts_pass_train)
                mape_err = mape(self.ts_pass_val, pred)
                self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time '
                                'series). Error = {}'.format(model_cls, mape_err))

                # check prediction for several time series
                pred_list = model.predict(n=36, series=[self.ts_pass_train, self.ts_pass_train_1])
                self.assertTrue(len(pred_list) == 2, 'Model {} did not return a list of prediction'.format(model_cls))
                for pred in pred_list:
                    mape_err = mape(self.ts_pass_val, pred)
                    self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time series 2). '
                                                    'Error = {}'.format(model_cls, mape_err))

        def test_covariates(self):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                if model_cls == NBEATSModel:
                    # N-BEATS does not support multivariate
                    continue

                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit(series=[self.ts_pass_train, self.ts_pass_train_1],
                          covariates=[self.time_covariates_train, self.time_covariates_train])
                with self.assertRaises(ValueError):
                    # when model is fit from >1 series, one must provide a series in argument
                    model.predict(n=1)

                with self.assertRaises(ValueError):
                    # when model is fit using covariates, covariates are required at prediction time
                    model.predict(n=1, series=self.ts_pass_train)

                with self.assertRaises(ValueError):
                    # when model is fit using covariates, n cannot be greater than output_chunk_length
                    model.predict(n=13, series=self.ts_pass_train)

                pred = model.predict(n=12, series=self.ts_pass_train, covariates=self.time_covariates_train)
                mape_err = mape(self.ts_pass_val, pred)
                self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time '
                                                'series with covariates). Error = {}'.format(model_cls, mape_err))

        def test_predict_from_dataset_unsupported_input(self):
            # an exception should be thrown if an unsupported type is passed
            unsupported_type = 'unsupported_type'
            # just need to test this with one model
            model_cls, kwargs, err = models_cls_kwargs_errs[0]
            model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
            model.fit([self.ts_pass_train, self.ts_pass_train_1])

            with self.assertRaises(AttributeError):
                model.predict_from_dataset(n=1, input_series_dataset=unsupported_type)

        def test_multi_ts_numpy_and_tensor(self):
            # testing the predictions with np.ndarray/torch.Tensor as input. The function should return a
            # np.ndarray/torch.Tensor as well, containing all the predictions since the test for np.ndarray
            # and torch were pretty identical, using module, type and custom functions for avoiding too much
            # code repetition

            def _np_to_tensor(array: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(array).float()

            def _identity(x):
                return x

            def _tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
                return tensor.cpu().detach().numpy()

            prediction_types = [np.ndarray, torch.Tensor]
            modules = [np, torch]
            adjust_type_fns = [_identity, _np_to_tensor]  # convert numpy to tensor when needed
            to_np_fns = [_identity, _tensor_to_np]  # converting the tensor back for building TS

            for prediction_type, module, adjust_type_fn, to_np_fn in zip(prediction_types,
                                                                         modules,
                                                                         adjust_type_fns,
                                                                         to_np_fns):

                for model_cls, kwargs, err in models_cls_kwargs_errs:
                    model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                    model.fit([self.ts_pass_train, self.ts_pass_train_1])

                    # the np.array should have 3 dimensions, dim 0 for samples, dim 1 for time points and dim 2 for
                    # covariates

                    with self.assertRaises(ValueError):
                        dummy_input = module.ones((1, 1))
                        model.predict_from_dataset(n=1, input_series_dataset=dummy_input)

                    # chek input not long enough
                    with self.assertRaises(ValueError):
                        dummy_input = adjust_type_fn(self.ts_pass_train.values()[-1:])  # making the input too short

                        dummy_input = module.stack([dummy_input], axis=0)  # but right dimension
                        model.predict_from_dataset(n=1, input_series_dataset=dummy_input)

                    ts_pass_train = adjust_type_fn(self.ts_pass_train.values())

                    # stacking creating the new 0 axis for multiple samples
                    dummy_input = module.stack([ts_pass_train], axis=0)

                    PRED_LEN = len(self.ts_pass_val)
                    pred = model.predict_from_dataset(n=PRED_LEN, input_series_dataset=dummy_input)

                    # checking return type
                    self.assertIsInstance(pred, prediction_type, 'Return type not as expected.'
                                          'Expected type: {}, current type {}'.format(prediction_type, type(pred)))

                    # checking the overall returned shape
                    # (n_samples, prediction_length, prediction_width)
                    expected_shape = (dummy_input.shape[0], PRED_LEN, 1)
                    self.assertEqual(expected_shape, pred.shape, 'Model prediction size not as expected.'
                                     'Expected shape: {}, current shape: {}'.format(expected_shape, pred.shape))

                    # converting the pred into TS for checking the metric
                    pred_ts = TimeSeries.from_times_and_values(self.ts_pass_val.time_index(), to_np_fn(pred[0]))
                    mape_err = mape(self.ts_pass_val, pred_ts)
                    self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time '
                                    'series). Error = {}'.format(model_cls, mape_err))

                    # checking the output dimensions and error with multiple TS

                    # getting the numpy view of the second TS
                    ts_pass_train_1 = adjust_type_fn(self.ts_pass_train_1.values())

                    dummy_input = module.stack([ts_pass_train, ts_pass_train_1], axis=0)

                    # (samples, TSlen, 1)
                    pred_list = model.predict_from_dataset(n=PRED_LEN, input_series_dataset=dummy_input)

                    self.assertTrue(pred_list.shape[0] == 2,
                                    'Model {} did not return a list of prediction'.format(model_cls))

                    for pred in pred_list:
                        # turn into a TS
                        pred_ts = TimeSeries.from_times_and_values(self.ts_pass_val.time_index(), to_np_fn(pred))

                        mape_err = mape(self.ts_pass_val, pred_ts)
                        self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time series 2). '
                                                        'Error = {}'.format(model_cls, mape_err))

        def test_multivariates_and_covariates_numpy(self):

            for model_cls, kwargs, err in models_cls_kwargs_errs:
                if model_cls == NBEATSModel:
                    # N-BEATS does not support multivariate
                    continue

                pred_len = len(self.ts_pass_val)
                ts_pass_train_array = self.ts_pass_train.values()
                ts_pass_train_array_1 = self.ts_pass_train_1.values()

                multivariate_array = np.concatenate([ts_pass_train_array, ts_pass_train_array_1], axis=1)  # (108, 2)
                covariate_array = self.time_covariates_train.values()  # (108, 2)

                input_array = np.concatenate([multivariate_array, covariate_array], axis=1)  # (108, 4)
                input_array = np.expand_dims(input_array, axis=0)  # (1, 108, 4) 1 (multivariate TS) with 2 covariates

                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit(series=[self.ts_pass_train.stack(self.ts_pass_train_1)],
                          covariates=[self.time_covariates_train])

                pred = model.predict_from_dataset(n=pred_len, input_series_dataset=input_array)

                # checking the overall returned shape
                # (n_samples, prediction_length, prediction_width)
                expected_shape = (input_array.shape[0], pred_len, multivariate_array.shape[1])
                self.assertEqual(expected_shape, pred.shape, 'Model prediction size not as expected.'
                                 'Expected shape: {}, current shape: {}'.format(expected_shape, pred.shape))

        def test_multivariates_and_covariates_torch(self):

            for model_cls, kwargs, _ in models_cls_kwargs_errs:
                if model_cls == NBEATSModel:
                    # N-BEATS does not support multivariate
                    continue

                pred_len = len(self.ts_pass_val)
                ts_pass_train_array = torch.from_numpy(self.ts_pass_train.values()).float()
                ts_pass_train_array_1 = torch.from_numpy(self.ts_pass_train_1.values()).float()

                multivariate_array = torch.cat([ts_pass_train_array, ts_pass_train_array_1], axis=1)  # (108, 2)
                covariate_array = torch.from_numpy(self.time_covariates_train.values()).float()  # (108, 2)

                input_array = torch.cat([multivariate_array, covariate_array], axis=1)  # (108, 4)
                input_array = torch.unsqueeze(input_array, axis=0)  # (1, 108, 4) 1 (multivariate TS) with 2 covariates

                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit(series=[self.ts_pass_train.stack(self.ts_pass_train_1)],
                          covariates=[self.time_covariates_train])

                pred = model.predict_from_dataset(n=pred_len, input_series_dataset=input_array)

                # checking the overall returned shape
                # (n_samples, prediction_length, prediction_width)
                expected_shape = (input_array.shape[0], pred_len, multivariate_array.shape[1])
                self.assertEqual(expected_shape, pred.shape, 'Model prediction size not as expected.'
                                 'Expected shape: {}, current shape: {}'.format(expected_shape, pred.shape))
