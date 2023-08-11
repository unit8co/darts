import os
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.encoders import SequentialEncoder
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.logging import get_logger
from darts.metrics import mape
from darts.tests.base_test_class import DartsBaseTestClass, tfm_kwargs
from darts.utils.timeseries_generation import linear_timeseries

logger = get_logger(__name__)

try:
    import torch
    from pytorch_lightning.loggers.logger import DummyLogger
    from pytorch_lightning.tuner.lr_finder import _LRFinder
    from torchmetrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MetricCollection,
    )

    from darts.models import DLinearModel, RNNModel
    from darts.utils.likelihood_models import (
        GaussianLikelihood,
        LaplaceLikelihood,
        Likelihood,
    )

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. RNN tests will be skipped.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class TestTorchForecastingModel(DartsBaseTestClass):
        def setUp(self):
            self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

            times = pd.date_range("20130101", "20130410")
            pd_series = pd.Series(range(100), index=times)
            self.series = TimeSeries.from_series(pd_series)

            df = pd.DataFrame({"var1": range(100), "var2": range(100)}, index=times)
            self.multivariate_series = TimeSeries.from_dataframe(df)

        def tearDown(self):
            shutil.rmtree(self.temp_work_dir)

        def test_save_model_parameters(self):
            # check if re-created model has same params as original
            model = RNNModel(12, "RNN", 10, 10, **tfm_kwargs)
            self.assertTrue(model._model_params, model.untrained_model()._model_params)

        @patch(
            "darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.save"
        )
        def test_suppress_automatic_save(self, patch_save_model):
            model_name = "test_model"
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=model_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=False,
                **tfm_kwargs,
            )
            model2 = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=model_name,
                work_dir=self.temp_work_dir,
                force_reset=True,
                save_checkpoints=False,
                **tfm_kwargs,
            )

            model1.fit(self.series, epochs=1)
            model2.fit(self.series, epochs=1)

            model1.predict(n=1)
            model2.predict(n=2)

            patch_save_model.assert_not_called()

            model1.save(path=os.path.join(self.temp_work_dir, model_name))
            patch_save_model.assert_called()

        def test_manual_save_and_load(self):
            """validate manual save with automatic save files by comparing output between the two"""

            model_dir = os.path.join(self.temp_work_dir)
            manual_name = "test_save_manual"
            auto_name = "test_save_automatic"
            model_manual_save = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=manual_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=False,
                random_state=42,
                **tfm_kwargs,
            )
            model_auto_save = RNNModel(
                12,
                "RNN",
                10,
                10,
                model_name=auto_name,
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                random_state=42,
                **tfm_kwargs,
            )

            # save model without training
            no_training_ckpt = "no_training.pth.tar"
            no_training_ckpt_path = os.path.join(model_dir, no_training_ckpt)
            model_manual_save.save(no_training_ckpt_path)
            # check that model object file was created
            self.assertTrue(os.path.exists(no_training_ckpt_path))
            # check that the PyTorch Ligthning ckpt does not exist
            self.assertFalse(os.path.exists(no_training_ckpt_path + ".ckpt"))
            # informative exception about `fit()` not called
            with self.assertRaises(
                ValueError,
                msg="The model must be fit before calling predict(). "
                "For global models, if predict() is called without specifying a series, "
                "the model must have been fit on a single training series.",
            ):
                no_train_model = RNNModel.load(no_training_ckpt_path)
                no_train_model.predict(n=4)

            model_manual_save.fit(self.series, epochs=1)
            model_auto_save.fit(self.series, epochs=1)

            # check that file was not created with manual save
            self.assertFalse(
                os.path.exists(os.path.join(model_dir, manual_name, "checkpoints"))
            )
            # check that file was created with automatic save
            self.assertTrue(
                os.path.exists(os.path.join(model_dir, auto_name, "checkpoints"))
            )

            # create manually saved model checkpoints folder
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)

            checkpoint_file_name = "checkpoint_0.pth.tar"
            model_path_manual = os.path.join(
                checkpoint_path_manual, checkpoint_file_name
            )
            checkpoint_file_name_cpkt = "checkpoint_0.pth.tar.ckpt"
            model_path_manual_ckpt = os.path.join(
                checkpoint_path_manual, checkpoint_file_name_cpkt
            )

            # save manually saved model
            model_manual_save.save(model_path_manual)
            self.assertTrue(os.path.exists(model_path_manual))

            # check that the PTL checkpoint path is also there
            self.assertTrue(os.path.exists(model_path_manual_ckpt))

            # load manual save model and compare with automatic model results
            model_manual_save = RNNModel.load(model_path_manual, map_location="cpu")
            model_manual_save.to_cpu()
            self.assertEqual(
                model_manual_save.predict(n=4), model_auto_save.predict(n=4)
            )

            # load automatically saved model with manual load() and load_from_checkpoint()
            model_auto_save1 = RNNModel.load_from_checkpoint(
                model_name=auto_name,
                work_dir=self.temp_work_dir,
                best=False,
                map_location="cpu",
            )
            model_auto_save1.to_cpu()
            # compare loaded checkpoint with manual save
            self.assertEqual(
                model_manual_save.predict(n=4), model_auto_save1.predict(n=4)
            )

            # save() model directly after load_from_checkpoint()
            checkpoint_file_name_2 = "checkpoint_1.pth.tar"
            checkpoint_file_name_cpkt_2 = checkpoint_file_name_2 + ".ckpt"

            model_path_manual_2 = os.path.join(
                checkpoint_path_manual, checkpoint_file_name_2
            )
            model_path_manual_ckpt_2 = os.path.join(
                checkpoint_path_manual, checkpoint_file_name_cpkt_2
            )
            model_auto_save2 = RNNModel.load_from_checkpoint(
                model_name=auto_name,
                work_dir=self.temp_work_dir,
                best=False,
                map_location="cpu",
            )
            # save model directly after loading, model has no trainer
            model_auto_save2.save(model_path_manual_2)

            # assert original .ckpt checkpoint was correctly copied
            self.assertTrue(os.path.exists(model_path_manual_ckpt_2))

            model_chained_load_save = RNNModel.load(
                model_path_manual_2, map_location="cpu"
            )

            # compare chained load_from_checkpoint() save() with manual save
            self.assertEqual(
                model_chained_load_save.predict(n=4), model_manual_save.predict(n=4)
            )

        def test_valid_save_and_load_weights_with_different_params(self):
            """
            Verify that save/load does not break encoders.

            Note: since load_weights() calls load_weights_from_checkpoint(), it will be used
            for all but one test.
            Note: Using DLinear since it supports both past and future covariates
            """

            def create_model(**kwargs):
                return DLinearModel(
                    input_chunk_length=4,
                    output_chunk_length=1,
                    **kwargs,
                    **tfm_kwargs,
                )

            model_dir = os.path.join(self.temp_work_dir)
            manual_name = "save_manual"
            # create manually saved model checkpoints folder
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = "checkpoint_0.pth.tar"
            model_path_manual = os.path.join(
                checkpoint_path_manual, checkpoint_file_name
            )
            model = create_model()
            model.fit(self.series, epochs=1)
            model.save(model_path_manual)

            kwargs_valid = [
                {"optimizer_cls": torch.optim.SGD},
                {"optimizer_kwargs": {"lr": 0.1}},
            ]
            # check that all models can be created with different valid kwargs
            for kwargs_ in kwargs_valid:
                model_new = create_model(**kwargs_)
                model_new.load_weights(model_path_manual)

        def test_save_and_load_weights_w_encoders(self):
            """
            Verify that save/load does not break encoders.

            Note: since load_weights() calls load_weights_from_checkpoint(), it will be used
            for all but one test.
            Note: Using DLinear since it supports both past and future covariates
            """

            def create_DLinearModel(
                model_name: str,
                save_checkpoints: bool = False,
                add_encoders: Dict = None,
            ):
                return DLinearModel(
                    input_chunk_length=4,
                    output_chunk_length=1,
                    kernel_size=5,
                    model_name=model_name,
                    add_encoders=add_encoders,
                    work_dir=self.temp_work_dir,
                    save_checkpoints=save_checkpoints,
                    random_state=42,
                    force_reset=True,
                    **tfm_kwargs,
                )

            model_dir = os.path.join(self.temp_work_dir)
            manual_name = "save_manual"
            auto_name = "save_auto"
            auto_name_other = "save_auto_other"
            # create manually saved model checkpoints folder
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = "checkpoint_0.pth.tar"
            model_path_manual = os.path.join(
                checkpoint_path_manual, checkpoint_file_name
            )

            # define encoders sets
            encoders_past = {
                "datetime_attribute": {"past": ["day"]},
                "transformer": Scaler(),
            }
            encoders_other_past = {
                "datetime_attribute": {"past": ["hour"]},
                "transformer": Scaler(),
            }
            encoders_past_noscaler = {
                "datetime_attribute": {"past": ["day"]},
            }
            encoders_past_other_transformer = {
                "datetime_attribute": {"past": ["day"]},
                "transformer": BoxCox(),
            }
            encoders_2_past = {
                "datetime_attribute": {"past": ["hour", "day"]},
                "transformer": Scaler(),
            }
            encoders_past_n_future = {
                "datetime_attribute": {"past": ["day"], "future": ["dayofweek"]},
                "transformer": Scaler(),
            }

            model_auto_save = create_DLinearModel(
                auto_name, save_checkpoints=True, add_encoders=encoders_past
            )
            model_auto_save.fit(self.series, epochs=1)

            model_manual_save = create_DLinearModel(
                manual_name, save_checkpoints=False, add_encoders=encoders_past
            )
            model_manual_save.fit(self.series, epochs=1)
            model_manual_save.save(model_path_manual)

            model_auto_save_other = create_DLinearModel(
                auto_name_other, save_checkpoints=True, add_encoders=encoders_other_past
            )
            model_auto_save_other.fit(self.series, epochs=1)

            # prediction are different when using different encoders
            self.assertNotEqual(
                model_auto_save.predict(n=4),
                model_auto_save_other.predict(n=4),
            )

            # model with undeclared encoders
            model_no_enc = create_DLinearModel("no_encoder", add_encoders=None)
            # weights were trained with encoders, new model must be instantiated with encoders
            with self.assertRaises(ValueError):
                model_no_enc.load_weights_from_checkpoint(
                    auto_name,
                    work_dir=self.temp_work_dir,
                    best=False,
                    load_encoders=False,
                    map_location="cpu",
                )
            # overwritte undeclared encoders
            model_no_enc.load_weights_from_checkpoint(
                auto_name,
                work_dir=self.temp_work_dir,
                best=False,
                load_encoders=True,
                map_location="cpu",
            )
            self.helper_equality_encoders(
                model_auto_save.add_encoders, model_no_enc.add_encoders
            )
            self.helper_equality_encoders_transfo(
                model_auto_save.add_encoders, model_no_enc.add_encoders
            )
            # cannot directly verify equality between encoders, using predict as proxy
            self.assertEqual(
                model_auto_save.predict(n=4),
                model_no_enc.predict(n=4, series=self.series),
            )

            # model with identical encoders (fittable)
            model_same_enc_noload = create_DLinearModel(
                "same_encoder_noload", add_encoders=encoders_past
            )
            model_same_enc_noload.load_weights(
                model_path_manual,
                load_encoders=False,
                map_location="cpu",
            )
            # cannot predict because of un-fitted encoder
            with self.assertRaises(ValueError):
                model_same_enc_noload.predict(n=4, series=self.series)

            model_same_enc_load = create_DLinearModel(
                "same_encoder_load", add_encoders=encoders_past
            )
            model_same_enc_load.load_weights(
                model_path_manual,
                load_encoders=True,
                map_location="cpu",
            )
            self.assertEqual(
                model_manual_save.predict(n=4),
                model_same_enc_load.predict(n=4, series=self.series),
            )

            # model with different encoders (fittable)
            model_other_enc_load = create_DLinearModel(
                "other_encoder_load", add_encoders=encoders_other_past
            )
            # cannot overwritte different declared encoders
            with self.assertRaises(ValueError):
                model_other_enc_load.load_weights(
                    model_path_manual,
                    load_encoders=True,
                    map_location="cpu",
                )

            # model with different encoders but same dimensions (fittable)
            model_other_enc_noload = create_DLinearModel(
                "other_encoder_noload", add_encoders=encoders_other_past
            )
            model_other_enc_noload.load_weights(
                model_path_manual,
                load_encoders=False,
                map_location="cpu",
            )
            self.helper_equality_encoders(
                model_other_enc_noload.add_encoders, encoders_other_past
            )
            self.helper_equality_encoders_transfo(
                model_other_enc_noload.add_encoders, encoders_other_past
            )
            # new encoders were instantiated
            self.assertTrue(
                isinstance(model_other_enc_noload.encoders, SequentialEncoder)
            )
            # since fit() was not called, new fittable encoders were not trained
            with self.assertRaises(ValueError):
                model_other_enc_noload.predict(n=4, series=self.series)

            # predict() can be called after fit()
            model_other_enc_noload.fit(self.series, epochs=1)
            model_other_enc_noload.predict(n=4, series=self.series)

            # model with same encoders but no scaler (non-fittable)
            model_new_enc_noscaler_noload = create_DLinearModel(
                "same_encoder_noscaler", add_encoders=encoders_past_noscaler
            )
            model_new_enc_noscaler_noload.load_weights(
                model_path_manual,
                load_encoders=False,
                map_location="cpu",
            )

            self.helper_equality_encoders(
                model_new_enc_noscaler_noload.add_encoders, encoders_past_noscaler
            )
            self.helper_equality_encoders_transfo(
                model_new_enc_noscaler_noload.add_encoders, encoders_past_noscaler
            )
            # predict() can be called directly since new encoders don't contain scaler
            model_new_enc_noscaler_noload.predict(n=4, series=self.series)

            # model with same encoders but different transformer (fittable)
            model_new_enc_other_transformer = create_DLinearModel(
                "same_encoder_other_transform",
                add_encoders=encoders_past_other_transformer,
            )
            # cannot overwritte different declared encoders
            with self.assertRaises(ValueError):
                model_new_enc_other_transformer.load_weights(
                    model_path_manual,
                    load_encoders=True,
                    map_location="cpu",
                )

            model_new_enc_other_transformer.load_weights(
                model_path_manual,
                load_encoders=False,
                map_location="cpu",
            )
            # since fit() was not called, new fittable encoders were not trained
            with self.assertRaises(ValueError):
                model_new_enc_other_transformer.predict(n=4, series=self.series)

            # predict() can be called after fit()
            model_new_enc_other_transformer.fit(self.series, epochs=1)
            model_new_enc_other_transformer.predict(n=4, series=self.series)

            # model with encoders containing more components (fittable)
            model_new_enc_2_past = create_DLinearModel(
                "encoder_2_components_past", add_encoders=encoders_2_past
            )
            # cannot overwritte different declared encoders
            with self.assertRaises(ValueError):
                model_new_enc_2_past.load_weights(
                    model_path_manual,
                    load_encoders=True,
                    map_location="cpu",
                )
            # new encoders have one additional past component
            with self.assertRaises(ValueError):
                model_new_enc_2_past.load_weights(
                    model_path_manual,
                    load_encoders=False,
                    map_location="cpu",
                )

            # model with encoders containing past and future covs (fittable)
            model_new_enc_past_n_future = create_DLinearModel(
                "encoder_past_n_future", add_encoders=encoders_past_n_future
            )
            # cannot overwritte different declared encoders
            with self.assertRaises(ValueError):
                model_new_enc_past_n_future.load_weights(
                    model_path_manual,
                    load_encoders=True,
                    map_location="cpu",
                )
            # identical past components, but different future components
            with self.assertRaises(ValueError):
                model_new_enc_past_n_future.load_weights(
                    model_path_manual,
                    load_encoders=False,
                    map_location="cpu",
                )

        def test_save_and_load_weights_w_likelihood(self):
            """
            Verify that save/load does not break likelihood.

            Note: since load_weights() calls load_weights_from_checkpoint(), it will be used
            for all but one test.
            Note: Using DLinear since it supports both past and future covariates
            """

            def create_DLinearModel(
                model_name: str,
                save_checkpoints: bool = False,
                likelihood: Likelihood = None,
            ):
                return DLinearModel(
                    input_chunk_length=4,
                    output_chunk_length=1,
                    kernel_size=5,
                    model_name=model_name,
                    work_dir=self.temp_work_dir,
                    save_checkpoints=save_checkpoints,
                    likelihood=likelihood,
                    random_state=42,
                    force_reset=True,
                    **tfm_kwargs,
                )

            model_dir = os.path.join(self.temp_work_dir)
            manual_name = "save_manual"
            auto_name = "save_auto"
            # create manually saved model checkpoints folder
            checkpoint_path_manual = os.path.join(model_dir, manual_name)
            os.mkdir(checkpoint_path_manual)
            checkpoint_file_name = "checkpoint_0.pth.tar"
            model_path_manual = os.path.join(
                checkpoint_path_manual, checkpoint_file_name
            )

            model_auto_save = create_DLinearModel(
                auto_name,
                save_checkpoints=True,
                likelihood=GaussianLikelihood(prior_mu=0.5),
            )
            model_auto_save.fit(self.series, epochs=1)
            pred_auto = model_auto_save.predict(n=4, series=self.series)

            model_manual_save = create_DLinearModel(
                manual_name,
                save_checkpoints=False,
                likelihood=GaussianLikelihood(prior_mu=0.5),
            )
            model_manual_save.fit(self.series, epochs=1)
            model_manual_save.save(model_path_manual)
            pred_manual = model_manual_save.predict(n=4, series=self.series)

            # predictions are identical when using the same likelihood
            self.assertTrue(np.array_equal(pred_auto.values(), pred_manual.values()))

            # model with identical likelihood
            model_same_likelihood = create_DLinearModel(
                "same_likelihood", likelihood=GaussianLikelihood(prior_mu=0.5)
            )
            model_same_likelihood.load_weights(model_path_manual, map_location="cpu")
            model_same_likelihood.predict(n=4, series=self.series)
            # cannot check predictions since this model is not fitted, random state is different

            # loading models weights with respective methods
            model_manual_same_likelihood = create_DLinearModel(
                "same_likelihood", likelihood=GaussianLikelihood(prior_mu=0.5)
            )
            model_manual_same_likelihood.load_weights(
                model_path_manual, map_location="cpu"
            )
            preds_manual_from_weights = model_manual_same_likelihood.predict(
                n=4, series=self.series
            )

            model_auto_same_likelihood = create_DLinearModel(
                "same_likelihood", likelihood=GaussianLikelihood(prior_mu=0.5)
            )
            model_auto_same_likelihood.load_weights_from_checkpoint(
                auto_name, work_dir=self.temp_work_dir, best=False, map_location="cpu"
            )
            preds_auto_from_weights = model_auto_same_likelihood.predict(
                n=4, series=self.series
            )
            # check that weights from checkpoint give identical predictions as weights from manual save
            self.assertTrue(preds_manual_from_weights == preds_auto_from_weights)

            # model with no likelihood
            model_no_likelihood = create_DLinearModel("no_likelihood", likelihood=None)
            with self.assertRaises(ValueError):
                model_no_likelihood.load_weights_from_checkpoint(
                    auto_name,
                    work_dir=self.temp_work_dir,
                    best=False,
                    map_location="cpu",
                )

            # model with a different likelihood
            model_other_likelihood = create_DLinearModel(
                "other_likelihood", likelihood=LaplaceLikelihood()
            )
            with self.assertRaises(ValueError):
                model_other_likelihood.load_weights(
                    model_path_manual, map_location="cpu"
                )

            # model with the same likelihood but different parameters
            model_same_likelihood_other_prior = create_DLinearModel(
                "same_likelihood_other_prior", likelihood=GaussianLikelihood()
            )
            with self.assertRaises(ValueError):
                model_same_likelihood_other_prior.load_weights(
                    model_path_manual, map_location="cpu"
                )

        def test_create_instance_new_model_no_name_set(self):
            RNNModel(12, "RNN", 10, 10, work_dir=self.temp_work_dir, **tfm_kwargs)
            # no exception is raised
            RNNModel(12, "RNN", 10, 10, work_dir=self.temp_work_dir, **tfm_kwargs)
            # no exception is raised

        def test_create_instance_existing_model_with_name_no_fit(self):
            model_name = "test_model"
            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                **tfm_kwargs,
            )
            # no exception is raised

            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                **tfm_kwargs,
            )
            # no exception is raised

        @patch(
            "darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model"
        )
        def test_create_instance_existing_model_with_name_force(
            self, patch_reset_model
        ):
            model_name = "test_model"
            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                **tfm_kwargs,
            )
            # no exception is raised
            # since no fit, there is no data stored for the model, hence `force_reset` does noting

            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                force_reset=True,
                **tfm_kwargs,
            )
            patch_reset_model.assert_not_called()

        @patch(
            "darts.models.forecasting.torch_forecasting_model.TorchForecastingModel.reset_model"
        )
        def test_create_instance_existing_model_with_name_force_fit_with_reset(
            self, patch_reset_model
        ):
            model_name = "test_model"
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                save_checkpoints=True,
                **tfm_kwargs,
            )
            # no exception is raised

            model1.fit(self.series, epochs=1)

            RNNModel(
                12,
                "RNN",
                10,
                10,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                save_checkpoints=True,
                force_reset=True,
                **tfm_kwargs,
            )
            patch_reset_model.assert_called_once()

        # TODO for PTL: currently we (have to (?)) create a mew PTL trainer object every time fit() is called which
        #  resets some of the model's attributes such as epoch and step counts. We have check whether there is another
        #  way of doing this.

        # n_epochs=20, fit|epochs=None, epochs_trained=0 - train for 20 epochs
        def test_train_from_0_n_epochs_20_no_fit_epochs(self):
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=20,
                work_dir=self.temp_work_dir,
                **tfm_kwargs,
            )

            model1.fit(self.series)

            self.assertEqual(20, model1.epochs_trained)

        # n_epochs = 20, fit|epochs=None, epochs_trained=20 - train for another 20 epochs
        def test_train_from_20_n_epochs_40_no_fit_epochs(self):
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=20,
                work_dir=self.temp_work_dir,
                **tfm_kwargs,
            )

            model1.fit(self.series)
            self.assertEqual(20, model1.epochs_trained)

            model1.fit(self.series)
            self.assertEqual(20, model1.epochs_trained)

        # n_epochs = 20, fit|epochs=None, epochs_trained=10 - train for another 20 epochs
        def test_train_from_10_n_epochs_20_no_fit_epochs(self):
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=20,
                work_dir=self.temp_work_dir,
                **tfm_kwargs,
            )

            # simulate the case that user interrupted training with Ctrl-C after 10 epochs
            model1.fit(self.series, epochs=10)
            self.assertEqual(10, model1.epochs_trained)

            model1.fit(self.series)
            self.assertEqual(20, model1.epochs_trained)

        # n_epochs = 20, fit|epochs=15, epochs_trained=10 - train for 15 epochs
        def test_train_from_10_n_epochs_20_fit_15_epochs(self):
            model1 = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=20,
                work_dir=self.temp_work_dir,
                **tfm_kwargs,
            )

            # simulate the case that user interrupted training with Ctrl-C after 10 epochs
            model1.fit(self.series, epochs=10)
            self.assertEqual(10, model1.epochs_trained)

            model1.fit(self.series, epochs=15)
            self.assertEqual(15, model1.epochs_trained)

        def test_load_weights_from_checkpoint(self):
            ts_training, ts_test = self.series.split_before(90)
            original_model_name = "original"
            retrained_model_name = "retrained"
            # original model, checkpoints are saved
            model = RNNModel(
                12,
                "RNN",
                5,
                1,
                n_epochs=5,
                work_dir=self.temp_work_dir,
                save_checkpoints=True,
                model_name=original_model_name,
                random_state=1,
                **tfm_kwargs,
            )
            model.fit(ts_training)
            original_preds = model.predict(10)
            original_mape = mape(original_preds, ts_test)

            # load last checkpoint of original model, train it for 2 additional epochs
            model_rt = RNNModel(
                12,
                "RNN",
                5,
                1,
                n_epochs=5,
                work_dir=self.temp_work_dir,
                model_name=retrained_model_name,
                random_state=1,
                **tfm_kwargs,
            )
            model_rt.load_weights_from_checkpoint(
                model_name=original_model_name,
                work_dir=self.temp_work_dir,
                best=False,
                map_location="cpu",
            )

            # must indicate series otherwise self.training_series must be saved in checkpoint
            loaded_preds = model_rt.predict(10, ts_training)
            # save/load checkpoint should produce identical predictions
            self.assertEqual(original_preds, loaded_preds)

            model_rt.fit(ts_training)
            retrained_preds = model_rt.predict(10)
            retrained_mape = mape(retrained_preds, ts_test)
            self.assertTrue(
                retrained_mape < original_mape,
                f"Retrained model has a greater error (mape) than the original model, "
                f"respectively {retrained_mape} and {original_mape}",
            )

            # raise Exception when trying to load ckpt weights in different architecture
            with self.assertRaises(ValueError):
                model_rt = RNNModel(
                    12,
                    "RNN",
                    10,  # loaded model has only 5 hidden_layers
                    5,
                )
                model_rt.load_weights_from_checkpoint(
                    model_name=original_model_name,
                    work_dir=self.temp_work_dir,
                    best=False,
                    map_location="cpu",
                )

            # raise Exception when trying to pass `weights_only`=True to `torch.load()`
            with self.assertRaises(ValueError):
                model_rt = RNNModel(12, "RNN", 5, 5, **tfm_kwargs)
                model_rt.load_weights_from_checkpoint(
                    model_name=original_model_name,
                    work_dir=self.temp_work_dir,
                    best=False,
                    weights_only=True,
                    map_location="cpu",
                )

        def test_load_weights(self):
            ts_training, ts_test = self.series.split_before(90)
            original_model_name = "original"
            retrained_model_name = "retrained"
            # original model, checkpoints are saved
            model = RNNModel(
                12,
                "RNN",
                5,
                1,
                n_epochs=5,
                work_dir=self.temp_work_dir,
                save_checkpoints=False,
                model_name=original_model_name,
                random_state=1,
                **tfm_kwargs,
            )
            model.fit(ts_training)
            path_manual_save = os.path.join(self.temp_work_dir, "RNN_manual_save.pt")
            model.save(path_manual_save)
            original_preds = model.predict(10)
            original_mape = mape(original_preds, ts_test)

            # load last checkpoint of original model, train it for 2 additional epochs
            model_rt = RNNModel(
                12,
                "RNN",
                5,
                1,
                n_epochs=5,
                work_dir=self.temp_work_dir,
                model_name=retrained_model_name,
                random_state=1,
                **tfm_kwargs,
            )
            model_rt.load_weights(path=path_manual_save, map_location="cpu")

            # must indicate series otherwise self.training_series must be saved in checkpoint
            loaded_preds = model_rt.predict(10, ts_training)
            # save/load checkpoint should produce identical predictions
            self.assertEqual(original_preds, loaded_preds)

            model_rt.fit(ts_training)
            retrained_preds = model_rt.predict(10)
            retrained_mape = mape(retrained_preds, ts_test)
            self.assertTrue(
                retrained_mape < original_mape,
                f"Retrained model has a greater mape error than the original model, "
                f"respectively {retrained_mape} and {original_mape}",
            )

        def test_multi_steps_pipeline(self):
            ts_training, ts_val = self.series.split_before(75)
            pretrain_model_name = "pre-train"
            retrained_model_name = "re-train"

            # pretraining
            model = self.helper_create_RNNModel(pretrain_model_name)
            model.fit(
                ts_training,
                val_series=ts_val,
            )

            # finetuning
            model = self.helper_create_RNNModel(retrained_model_name)
            model.load_weights_from_checkpoint(
                model_name=pretrain_model_name,
                work_dir=self.temp_work_dir,
                best=True,
                map_location="cpu",
            )
            model.fit(
                ts_training,
                val_series=ts_val,
            )

            # prediction
            model = model.load_from_checkpoint(
                model_name=retrained_model_name,
                work_dir=self.temp_work_dir,
                best=True,
                map_location="cpu",
            )
            model.predict(4, series=ts_training)

        def test_load_from_checkpoint_w_custom_loss(self):
            model_name = "pretraining_custom_loss"
            # model with a custom loss
            model = RNNModel(
                12,
                "RNN",
                5,
                1,
                n_epochs=1,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                save_checkpoints=True,
                force_reset=True,
                loss_fn=torch.nn.L1Loss(),
                **tfm_kwargs,
            )
            model.fit(self.series)

            loaded_model = RNNModel.load_from_checkpoint(
                model_name, self.temp_work_dir, best=False, map_location="cpu"
            )
            # custom loss function should be properly restored from ckpt
            self.assertTrue(isinstance(loaded_model.model.criterion, torch.nn.L1Loss))

            loaded_model.fit(self.series, epochs=2)
            # calling fit() should not impact the loss function
            self.assertTrue(isinstance(loaded_model.model.criterion, torch.nn.L1Loss))

        def test_load_from_checkpoint_w_metrics(self):
            model_name = "pretraining_metrics"
            # model with one torch_metrics
            pl_trainer_kwargs = dict(
                {"logger": DummyLogger(), "log_every_n_steps": 1},
                **tfm_kwargs["pl_trainer_kwargs"],
            )
            model = RNNModel(
                12,
                "RNN",
                5,
                1,
                n_epochs=1,
                work_dir=self.temp_work_dir,
                model_name=model_name,
                save_checkpoints=True,
                force_reset=True,
                torch_metrics=MeanAbsolutePercentageError(),
                pl_trainer_kwargs=pl_trainer_kwargs,
            )
            model.fit(self.series)
            # check train_metrics before loading
            self.assertTrue(isinstance(model.model.train_metrics, MetricCollection))
            self.assertEqual(len(model.model.train_metrics), 1)

            loaded_model = RNNModel.load_from_checkpoint(
                model_name,
                self.temp_work_dir,
                best=False,
                map_location="cpu",
            )
            # custom loss function should be properly restored from ckpt torchmetrics.Metric
            self.assertTrue(
                isinstance(loaded_model.model.train_metrics, MetricCollection)
            )
            self.assertEqual(len(loaded_model.model.train_metrics), 1)

        def test_optimizers(self):

            optimizers = [
                (torch.optim.Adam, {"lr": 0.001}),
                (torch.optim.SGD, {"lr": 0.001}),
            ]

            for optim_cls, optim_kwargs in optimizers:
                model = RNNModel(
                    12,
                    "RNN",
                    10,
                    10,
                    optimizer_cls=optim_cls,
                    optimizer_kwargs=optim_kwargs,
                    **tfm_kwargs,
                )
                # should not raise an error
                model.fit(self.series, epochs=1)

        def test_lr_schedulers(self):

            lr_schedulers = [
                (torch.optim.lr_scheduler.StepLR, {"step_size": 10}),
                (
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    {"threshold": 0.001, "monitor": "train_loss"},
                ),
                (torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.09}),
            ]

            for lr_scheduler_cls, lr_scheduler_kwargs in lr_schedulers:
                model = RNNModel(
                    12,
                    "RNN",
                    10,
                    10,
                    lr_scheduler_cls=lr_scheduler_cls,
                    lr_scheduler_kwargs=lr_scheduler_kwargs,
                    **tfm_kwargs,
                )
                # should not raise an error
                model.fit(self.series, epochs=1)

        def test_wrong_model_creation_params(self):
            valid_kwarg = {"pl_trainer_kwargs": {}}
            invalid_kwarg = {"some_invalid_kwarg": None}

            # valid params should not raise an error
            _ = RNNModel(12, "RNN", 10, 10, **valid_kwarg)

            # invalid params should raise an error
            with self.assertRaises(ValueError):
                _ = RNNModel(12, "RNN", 10, 10, **invalid_kwarg)

        def test_metrics(self):
            metric = MeanAbsolutePercentageError()
            metric_collection = MetricCollection(
                [MeanAbsolutePercentageError(), MeanAbsoluteError()]
            )

            model_kwargs = {
                "logger": DummyLogger(),
                "log_every_n_steps": 1,
                **tfm_kwargs["pl_trainer_kwargs"],
            }
            # test single metric
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=1,
                torch_metrics=metric,
                pl_trainer_kwargs=model_kwargs,
            )
            model.fit(self.series)

            # test metric collection
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=1,
                torch_metrics=metric_collection,
                pl_trainer_kwargs=model_kwargs,
            )
            model.fit(self.series)

            # test multivariate series
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=1,
                torch_metrics=metric,
                pl_trainer_kwargs=model_kwargs,
            )
            model.fit(self.multivariate_series)

        def test_metrics_w_likelihood(self):
            metric = MeanAbsolutePercentageError()
            metric_collection = MetricCollection(
                [MeanAbsolutePercentageError(), MeanAbsoluteError()]
            )
            model_kwargs = {
                "logger": DummyLogger(),
                "log_every_n_steps": 1,
                **tfm_kwargs["pl_trainer_kwargs"],
            }
            # test single metric
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=1,
                likelihood=GaussianLikelihood(),
                torch_metrics=metric,
                pl_trainer_kwargs=model_kwargs,
            )
            model.fit(self.series)

            # test metric collection
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=1,
                likelihood=GaussianLikelihood(),
                torch_metrics=metric_collection,
                pl_trainer_kwargs=model_kwargs,
            )
            model.fit(self.series)

            # test multivariate series
            model = RNNModel(
                12,
                "RNN",
                10,
                10,
                n_epochs=1,
                likelihood=GaussianLikelihood(),
                torch_metrics=metric_collection,
                pl_trainer_kwargs=model_kwargs,
            )
            model.fit(self.multivariate_series)

        def test_invalid_metrics(self):
            torch_metrics = ["invalid"]
            with self.assertRaises(AttributeError):
                model = RNNModel(
                    12,
                    "RNN",
                    10,
                    10,
                    n_epochs=1,
                    torch_metrics=torch_metrics,
                    **tfm_kwargs,
                )
                model.fit(self.series)

        @pytest.mark.slow
        def test_lr_find(self):
            train_series, val_series = self.series[:-40], self.series[-40:]
            model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)
            # find the learning rate
            res = model.lr_find(series=train_series, val_series=val_series, epochs=50)
            assert isinstance(res, _LRFinder)
            assert res.suggestion() is not None
            # verify that learning rate finder bypasses the `fit` logic
            assert model.model is None
            assert not model._fit_called
            # cannot predict with an untrained model
            with pytest.raises(ValueError):
                model.predict(n=3, series=self.series)

            # check that results are reproducible
            model = RNNModel(12, "RNN", 10, 10, random_state=42, **tfm_kwargs)
            res2 = model.lr_find(series=train_series, val_series=val_series, epochs=50)
            assert res.suggestion() == res2.suggestion()

            # check that suggested learning rate is better than the worst
            lr_worst = res.results["lr"][np.argmax(res.results["loss"])]
            lr_suggested = res.suggestion()
            scores = {}
            for lr, lr_name in zip([lr_worst, lr_suggested], ["worst", "suggested"]):
                model = RNNModel(
                    12,
                    "RNN",
                    10,
                    10,
                    n_epochs=10,
                    random_state=42,
                    optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={"lr": lr},
                    **tfm_kwargs,
                )
                model.fit(train_series)
                scores[lr_name] = mape(
                    val_series, model.predict(len(val_series), series=train_series)
                )
            assert scores["worst"] > scores["suggested"]

        def test_encoders(self):
            series = linear_timeseries(length=10)
            pc = linear_timeseries(length=12)
            fc = linear_timeseries(length=13)
            # 1 == output_chunk_length, 3 > output_chunk_length
            ns = [1, 3]

            model = self.helper_create_DLinearModel()
            model.fit(series)
            for n in ns:
                _ = model.predict(n=n)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, future_covariates=fc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

            model = self.helper_create_DLinearModel()
            for n in ns:
                model.fit(series, past_covariates=pc)
                _ = model.predict(n=n)
                _ = model.predict(n=n, past_covariates=pc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, future_covariates=fc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

            model = self.helper_create_DLinearModel()
            for n in ns:
                model.fit(series, future_covariates=fc)
                _ = model.predict(n=n)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc)
                _ = model.predict(n=n, future_covariates=fc)
                with pytest.raises(ValueError):
                    _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

            model = self.helper_create_DLinearModel()
            for n in ns:
                model.fit(series, past_covariates=pc, future_covariates=fc)
                _ = model.predict(n=n)
                _ = model.predict(n=n, past_covariates=pc)
                _ = model.predict(n=n, future_covariates=fc)
                _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

        def helper_equality_encoders(
            self, first_encoders: Dict[str, Any], second_encoders: Dict[str, Any]
        ):
            if first_encoders is None:
                first_encoders = {}
            if second_encoders is None:
                second_encoders = {}
            self.assertEqual(
                {k: v for k, v in first_encoders.items() if k != "transformer"},
                {k: v for k, v in second_encoders.items() if k != "transformer"},
            )

        def helper_equality_encoders_transfo(
            self, first_encoders: Dict[str, Any], second_encoders: Dict[str, Any]
        ):
            if first_encoders is None:
                first_encoders = {}
            if second_encoders is None:
                second_encoders = {}
            self.assertEqual(
                type(first_encoders.get("transformer", None)),
                type(second_encoders.get("transformer", None)),
            )

        def helper_create_RNNModel(self, model_name: str):
            return RNNModel(
                input_chunk_length=4,
                hidden_dim=3,
                add_encoders={
                    "cyclic": {"past": ["month"]},
                    "datetime_attribute": {
                        "past": ["hour"],
                    },
                    "transformer": Scaler(),
                },
                n_epochs=2,
                model_name=model_name,
                work_dir=self.temp_work_dir,
                force_reset=True,
                save_checkpoints=True,
                **tfm_kwargs,
            )

        def helper_create_DLinearModel(self):
            return DLinearModel(
                input_chunk_length=4,
                output_chunk_length=1,
                add_encoders={
                    "datetime_attribute": {"past": ["hour"], "future": ["month"]}
                },
                n_epochs=1,
                **tfm_kwargs,
            )
