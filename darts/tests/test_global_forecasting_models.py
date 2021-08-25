import numpy as np
from unittest.mock import MagicMock, patch, ANY

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..metrics import mape
from ..logging import get_logger
from ..dataprocessing.transformers import Scaler
from ..datasets import AirPassengersDataset
from ..models.torch_forecasting_model import DualCovariatesTorchModel
from darts.utils.timeseries_generation import linear_timeseries

logger = get_logger(__name__)

try:
    from ..models import BlockRNNModel, TCNModel, TransformerModel, NBEATSModel, RNNModel
    from darts.utils.likelihood_models import GaussianLikelihoodModel
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not installed - will be skipping Torch models tests')
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12
    models_cls_kwargs_errs = [
        (BlockRNNModel, {'model': 'RNN', 'hidden_size': 10, 'n_rnn_layers': 1, 'batch_size': 32, 'n_epochs': 10}, 180.),
        (RNNModel, {'model': 'RNN', 'hidden_dim': 10, 'batch_size': 32, 'n_epochs': 10}, 180.),
        (RNNModel, {'training_length': 12, 'n_epochs': 10, 'likelihood': GaussianLikelihoodModel()}, 80),
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
                                                                        freq=ts_pass_train.freq_str,
                                                                        start_ts=ts_pass_train.start_time())

        # an additional time series serving as covariates
        year_series = tg.datetime_attribute_timeseries(ts_passengers, attribute='year')
        month_series = tg.datetime_attribute_timeseries(ts_passengers, attribute='month')
        scaler_dt = Scaler()
        time_covariates = scaler_dt.fit_transform(year_series.stack(month_series))
        time_covariates_train, time_covariates_val = time_covariates[:-36], time_covariates[-36:]

        # an artificial time series that is highly dependent on covariates
        ts_length = 400
        split_ratio = 0.6
        sine_1_ts = tg.sine_timeseries(length=ts_length)
        sine_2_ts = tg.sine_timeseries(length=ts_length, value_frequency=0.05)
        sine_3_ts = tg.sine_timeseries(length=ts_length, value_frequency=0.003, value_amplitude=5)
        linear_ts = tg.linear_timeseries(length=ts_length, start_value=3, end_value=8)

        covariates = sine_3_ts.stack(sine_2_ts).stack(linear_ts)
        covariates_past, _ = covariates.split_after(split_ratio)

        target = sine_1_ts + sine_2_ts + linear_ts + sine_3_ts
        target_past, target_future = target.split_after(split_ratio)

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

                # Here we rely on the fact that all non-Dual models currently are Past models
                cov_name = 'future_covariates' if isinstance(model, DualCovariatesTorchModel) else 'past_covariates'
                cov_kwargs = {cov_name: [self.time_covariates_train, self.time_covariates_train]}

                model.fit(series=[self.ts_pass_train, self.ts_pass_train_1], **cov_kwargs)
                with self.assertRaises(ValueError):
                    # when model is fit from >1 series, one must provide a series in argument
                    model.predict(n=1)

                with self.assertRaises(ValueError):
                    # when model is fit using multiple covariates, covariates are required at prediction time
                    model.predict(n=1, series=self.ts_pass_train)

                cov_kwargs_train = {cov_name: self.time_covariates_train}
                cov_kwargs_notrain = {cov_name: self.time_covariates}
                with self.assertRaises(ValueError):
                    # when model is fit using covariates, n cannot be greater than output_chunk_length...
                    model.predict(n=13, series=self.ts_pass_train, **cov_kwargs_train)

                # ... unless future covariates are provided
                pred = model.predict(n=13, series=self.ts_pass_train, **cov_kwargs_notrain)

                pred = model.predict(n=12, series=self.ts_pass_train, **cov_kwargs_notrain)
                mape_err = mape(self.ts_pass_val, pred)
                self.assertTrue(mape_err < err, 'Model {} produces errors too high (several time '
                                                'series with covariates). Error = {}'.format(model_cls, mape_err))

                # when model is fit using 1 training and 1 covariate series, time series args are optional
                if model._is_probabilistic:
                    continue
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                model.fit(series=self.ts_pass_train, **cov_kwargs_train)
                pred1 = model.predict(1)
                pred2 = model.predict(1, series=self.ts_pass_train)
                pred3 = model.predict(1, **cov_kwargs_train)
                pred4 = model.predict(1, **cov_kwargs_train, series=self.ts_pass_train)
                self.assertEqual(pred1, pred2)
                self.assertEqual(pred1, pred3)
                self.assertEqual(pred1, pred4)

        def test_future_covariates(self):
            # models with future covariates should produce better predictions over a long forecasting horizon
            # than a model trained with no covariates
            model = TCNModel(input_chunk_length=50, output_chunk_length=5, n_epochs=20, random_state=0)

            model.fit(series=self.target_past)
            long_pred_no_cov = model.predict(n=160)

            model = TCNModel(input_chunk_length=50, output_chunk_length=5, n_epochs=20, random_state=0)
            model.fit(series=self.target_past, past_covariates=self.covariates_past)
            long_pred_with_cov = model.predict(n=160, past_covariates=self.covariates)
            self.assertTrue(mape(self.target_future, long_pred_no_cov) > mape(self.target_future, long_pred_with_cov),
                            'Models with future covariates should produce better predictions.')

            # block models can predict up to self.output_chunk_length points beyond the last future covariate...
            model.predict(n=165, past_covariates=self.covariates)

            # ... not more
            with self.assertRaises(ValueError):
                model.predict(n=166, series=self.ts_pass_train)

            # recurrent models can only predict data points for time steps where future covariates are available
            model = RNNModel(n_epochs=1)
            model.fit(series=self.target_past, future_covariates=self.covariates_past)
            model.predict(n=160, future_covariates=self.covariates)
            with self.assertRaises(ValueError):
                model.predict(n=161, future_covariates=self.covariates)

        def test_batch_predictions(self):
            # predicting multiple time series at once needs to work for arbitrary batch sizes
            # univariate case
            targets_univar = [self.target_past, self.target_past[:60], self.target_past[:80]]
            self._batch_prediction_test_helper_function(targets_univar)

            # multivariate case
            targets_multivar = [tgt.stack(tgt) for tgt in targets_univar]
            self._batch_prediction_test_helper_function(targets_multivar)

        def _batch_prediction_test_helper_function(self, targets):
            epsilon = 1e-4
            model = TCNModel(input_chunk_length=50, output_chunk_length=10, n_epochs=10, random_state=0)
            model.fit(series=targets[0], past_covariates=self.covariates_past)
            preds_default = model.predict(n=160, series=targets,
                                          past_covariates=[self.covariates] * len(targets), batch_size=None)

            # make batch size large enough to test stacking samples
            for batch_size in range(1, 4 * len(targets)):
                preds = model.predict(n=160, series=targets,
                                      past_covariates=[self.covariates] * len(targets), batch_size=batch_size)

                for i in range(len(targets)):
                    self.assertLess(sum(sum((preds[i] - preds_default[i]).values())), epsilon)

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
                if model._is_probabilistic():
                    continue
                multiple_ts = [self.ts_pass_train] * 10

                model.fit(multiple_ts)

                pred1 = model.predict(n=36, series=multiple_ts, n_jobs=1)
                pred2 = model.predict(n=36, series=multiple_ts, n_jobs=-1)  # assuming > 1 core available in the machine

                self.assertEqual(pred1, pred2, 'Model {} produces different predictions with different number of jobs')

        @patch('darts.models.torch_forecasting_model.torch.save')
        @patch('darts.models.torch_forecasting_model.TorchForecastingModel._train')
        @patch('darts.models.torch_forecasting_model.shutil.rmtree')
        def test_fit_with_constr_epochs(self, rmtree_patch, train_patch, save_patch):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                multiple_ts = [self.ts_pass_train] * 10
                model.fit(multiple_ts)

                train_patch.assert_called_with(ANY, ANY, ANY, ANY, kwargs['n_epochs'])

        @patch('darts.models.torch_forecasting_model.torch.save')
        @patch('darts.models.torch_forecasting_model.TorchForecastingModel._train')
        @patch('darts.models.torch_forecasting_model.shutil.rmtree')
        def test_fit_with_fit_epochs(self, rmtree_patch, train_patch, save_patch):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                multiple_ts = [self.ts_pass_train] * 10
                epochs = 42

                model.fit(multiple_ts, epochs=epochs)

                train_patch.assert_called_with(ANY, ANY, ANY, ANY, epochs)

                model.total_epochs = epochs
                # continue training
                model.fit(multiple_ts, epochs=epochs)

                train_patch.assert_called_with(ANY, ANY, ANY, ANY, epochs)

        @patch('darts.models.torch_forecasting_model.torch.save')
        @patch('darts.models.torch_forecasting_model.TorchForecastingModel._train')
        @patch('darts.models.torch_forecasting_model.shutil.rmtree')
        def test_fit_from_dataset_with_epochs(self, rmtree_patch, train_patch, save_patch):
            for model_cls, kwargs, err in models_cls_kwargs_errs:
                model = model_cls(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, **kwargs)
                multiple_ts = [self.ts_pass_train] * 10
                train_dataset = model._build_train_dataset(multiple_ts, past_covariates=None, future_covariates=None)
                epochs = 42

                model.fit_from_dataset(train_dataset, epochs=epochs)

                train_patch.assert_called_with(ANY, ANY, ANY, ANY, epochs)

                model.total_epochs = epochs
                # continue training
                model.fit_from_dataset(train_dataset, epochs=epochs)

                train_patch.assert_called_with(ANY, ANY, ANY, ANY, epochs)

        def test_sample_smaller_than_batch_size(self):
            """
            Checking that the TorchForecastingModels do not crash even if the number of available samples for training
            is strictly lower than the selected batch_size
            """
            # TS with 50 timestamps. TorchForecastingModels will use the SequentialDataset for producing training
            # samples, which means we will have 50 - 22 - 2 + 1 = 27 samples, which is < 32 (batch_size). The model
            # should still train on those samples and not crash in any way
            ts = linear_timeseries(0, 1, 50)

            model = RNNModel(input_chunk_length=20,
                             output_chunk_length=2,
                             n_epochs=2,
                             batch_size=32)
            model.fit(ts)
