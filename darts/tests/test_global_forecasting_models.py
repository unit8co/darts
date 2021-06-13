import numpy as np

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..metrics import mape
from ..logging import get_logger
from ..dataprocessing.transformers import Scaler
from ..datasets import AirPassengersDataset

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
        ts_passengers = AirPassengersDataset().load()
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
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, random_state=0, **kwargs)
                model.fit(self.ts_pass_train)
                pred = model.predict(n=36)
                mape_err = mape(self.ts_pass_val, pred)
                self.assertTrue(mape_err < err, 'Model {} produces errors too high (one time '
                                                'series). Error = {}'.format(model_cls, mape_err))

        def test_multi_ts(self):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, random_state=0, **kwargs)
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

                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, random_state=0, **kwargs)
                model.fit(series=[self.ts_pass_train, self.ts_pass_train_1],
                          covariates=[self.time_covariates_train, self.time_covariates_train])
                with self.assertRaises(ValueError):
                    # when model is fit from >1 series, one must provide a series in argument
                    model.predict(n=1)

                with self.assertRaises(ValueError):
                    # when model is fit using multiple covariates, covariates are required at prediction time
                    model.predict(n=1, series=self.ts_pass_train)

                with self.assertRaises(ValueError):
                    # when model is fit using covariates, n cannot be greater than output_chunk_length...
                    model.predict(n=13, series=self.ts_pass_train, covariates=self.time_covariates_train)

                # ... unless future covariates are provided
                model.predict(n=13, series=self.ts_pass_train, covariates=self.time_covariates)

                pred = model.predict(n=12, series=self.ts_pass_train, covariates=self.time_covariates_train)
                mape_err = mape(self.ts_pass_val, pred)
                self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time '
                                                'series with covariates). Error = {}'.format(model_cls, mape_err))

                
                # when model is fit using 1 training and 1 covariate series, time series args are optional
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit(series=self.ts_pass_train, covariates=self.time_covariates_train)
                pred1 = model.predict(1)
                pred2 = model.predict(1, series=self.ts_pass_train)
                pred3 = model.predict(1, covariates=self.time_covariates_train)
                pred4 = model.predict(1, covariates=self.time_covariates_train, series=self.ts_pass_train)
                self.assertEqual(pred1, pred2)
                self.assertEqual(pred1, pred3)
                self.assertEqual(pred1, pred4)

        def test_future_covariates(self):
            # models with future covariates should produce better predictions over a long forecasting horizon
            # than a model trained with no covariates
            ts_length = 400
            split_ratio = 0.6
            sine_1_ts = tg.sine_timeseries(length=ts_length)
            sine_2_ts = tg.sine_timeseries(length=ts_length, value_frequency=0.05)
            sine_3_ts = tg.sine_timeseries(length=ts_length, value_frequency=0.003, value_amplitude=5)
            linear_ts = tg.linear_timeseries(length=ts_length, start_value=3, end_value=8)


            covariates = sine_3_ts.stack(sine_2_ts).stack(linear_ts)
            covariates_past, covariates_future = covariates.split_after(split_ratio)

            target = sine_1_ts + sine_2_ts + linear_ts + sine_3_ts
            target_past, target_future = target.split_after(split_ratio)
            model = TCNModel(input_chunk_length=50, output_chunk_length=5, n_epochs=20, random_state=0)

            model.fit(series=target_past)
            long_pred_no_cov = model.predict(n=160)

            model = TCNModel(input_chunk_length=50, output_chunk_length=5, n_epochs=20, random_state=0)
            model.fit(series=target_past, covariates=covariates_past)
            long_pred_with_cov = model.predict(n=160, covariates=covariates)
            self.assertTrue(mape(target_future, long_pred_no_cov) > mape(target_future, long_pred_with_cov),
                            'Models with future covariates should produce better predictions.')

            # models can predict up to self.output_chunk_length points beyond the last future covariate...
            model.predict(n=165, covariates=covariates)

            # ... not more
            with self.assertRaises(ValueError):
                model.predict(n=166, series=self.ts_pass_train)

        def test_predict_from_dataset_unsupported_input(self):
            # an exception should be thrown if an unsupported type is passed
            unsupported_type = 'unsupported_type'
            # just need to test this with one model
            model_cls, kwargs, err = models_cls_kwargs_errs[0]
            model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
            model.fit([self.ts_pass_train, self.ts_pass_train_1])

            with self.assertRaises(ValueError):
                model.predict_from_dataset(n=1, input_series_dataset=unsupported_type)

        def test_same_result_with_different_n_jobs(self):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                multiple_ts = [self.ts_pass_train] * 10

                model.fit(multiple_ts)

                pred1 = model.predict(n=36, series=multiple_ts, n_jobs=1)
                pred2 = model.predict(n=36, series=multiple_ts, n_jobs=-1)  # assuming > 1 core available in the machine

                self.assertEqual(pred1, pred2, 'Model {} produces different predictions with different number of jobs')
