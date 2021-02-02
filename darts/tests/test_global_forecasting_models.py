import logging
import shutil

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
        (TCNModel, {'n_epochs': 10, 'batch_size': 32}, 180.),
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
