from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import IceCreamHeaterDataset
from darts.explainability import TFTExplainer
from darts.explainability.explainability import ExplainabilityResult
from darts.models import TFTModel
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import datetime_attribute_timeseries


class TFTExplainerTestCase(DartsBaseTestClass):

    np.random.seed(342)

    # Ice Example from the TFT tutorial
    series_ice_heater = IceCreamHeaterDataset().load()
    # convert monthly sales to average daily sales per month
    converted_series = []
    for col in ["ice cream", "heater"]:
        converted_series.append(
            series_ice_heater[col]
            / TimeSeries.from_series(series_ice_heater.time_index.days_in_month)
        )
    converted_series = concatenate(converted_series, axis=1)
    converted_series = converted_series[pd.Timestamp("20100101") :]

    # define train/validation cutoff time
    forecast_horizon_ice = 12
    training_cutoff_ice = converted_series.time_index[-(2 * forecast_horizon_ice)]

    # use ice cream sales as target, create train and validation sets and transform data
    series_ice = converted_series["ice cream"]
    train_ice, val_ice = series_ice.split_before(training_cutoff_ice)
    transformer_ice = Scaler()
    train_ice_transformed = transformer_ice.fit_transform(train_ice)
    val_ice_transformed = transformer_ice.transform(val_ice)
    series_ice_transformed = transformer_ice.transform(series_ice)

    # use heater sales as past covariates and transform data
    covariates_heat = converted_series["heater"]
    cov_heat_train, cov_heat_val = covariates_heat.split_before(training_cutoff_ice)
    transformer_heat = Scaler()
    transformer_heat.fit(cov_heat_train)
    covariates_heat_transformed = transformer_heat.transform(covariates_heat)

    # create input with multiple past covariates
    multiple_covariates = covariates_heat.stack(
        datetime_attribute_timeseries(covariates_heat, attribute="year", one_hot=False)
    ).stack(
        datetime_attribute_timeseries(covariates_heat, attribute="month", one_hot=False)
    )
    multi_cov_train, multi_cov_val = multiple_covariates.split_before(
        training_cutoff_ice
    )
    transformer_multi_cov = Scaler()
    transformer_multi_cov.fit(multi_cov_train)
    multiple_covariates_transformed = transformer_multi_cov.transform(
        multiple_covariates
    )

    # use the last 3 years as past input data
    input_chunk_length_ice = 36

    models = []
    models.append(
        TFTModel(
            input_chunk_length=input_chunk_length_ice,
            output_chunk_length=forecast_horizon_ice,
            hidden_size=32,
            lstm_layers=1,
            batch_size=16,
            n_epochs=10,
            dropout=0.1,
            add_encoders={"cyclic": {"future": ["month"]}},
            add_relative_index=False,
            optimizer_kwargs={"lr": 1e-3},
            random_state=42,
        )
    )

    def test_class_init_not_fitted_model_raises_error(self):
        """The TFTExplainer class should raise an error if the model we want to explain is not fitted."""
        # arrange
        model = deepcopy(self.models[0])

        # act / assert
        with self.assertRaises(ValueError):
            TFTExplainer(model)

    def test_class_init_with_fitted_model_works(self):
        """The TFTExplainer class should work if the model we want to explain is fitted."""
        # arrange
        model = deepcopy(self.models[0])

        model.fit(
            self.train_ice_transformed,
            past_covariates=self.covariates_heat_transformed,
            verbose=True,
        )

        # act
        res = TFTExplainer(model)

        # assert
        self.assertTrue(isinstance(res, TFTExplainer))
        self.assertTrue(hasattr(res, "model"))

    def test_get_variable_selection_weight(self):
        """The get_variable_selection_weight method returns the feature importance."""
        # arrange
        model = deepcopy(self.models[0])

        # fit the model with past covariates
        np.random.seed(342)
        _ = model.fit(
            self.train_ice_transformed,
            past_covariates=self.covariates_heat_transformed,
            verbose=False,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        # expected results
        expected_encoder_importance = pd.DataFrame(
            [
                {
                    "darts_enc_fc_cyc_month_cos": 68.4,
                    "heater": 16.5,
                    "ice cream": 10.6,
                    "darts_enc_fc_cyc_month_sin": 4.5,
                },
            ],
        )
        expected_decoder_importance = pd.DataFrame(
            [
                {
                    "darts_enc_fc_cyc_month_cos": 87.8,
                    "darts_enc_fc_cyc_month_sin": 12.2,
                },
            ]
        )

        # act
        res = explainer.get_variable_selection_weight(plot=False)

        # assert
        self.assertTrue(isinstance(res, dict))
        self.assertTrue(res.keys() == {"encoder_importance", "decoder_importance"})
        pd.testing.assert_frame_equal(
            res["encoder_importance"], expected_encoder_importance
        )
        pd.testing.assert_frame_equal(
            res["decoder_importance"], expected_decoder_importance
        )

    def test_get_variable_selection_weight_plot(self):
        """The get_variable_selection_weight method returns the feature importance."""
        # arrange
        model = deepcopy(self.models[0])

        # fit the model with past covariates
        np.random.seed(342)
        _ = model.fit(
            self.train_ice_transformed,
            past_covariates=self.covariates_heat_transformed,
            verbose=False,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        # expected results
        expected_encoder_importance = pd.DataFrame(
            [
                {
                    "darts_enc_fc_cyc_month_cos": 68.4,
                    "heater": 16.5,
                    "ice cream": 10.6,
                    "darts_enc_fc_cyc_month_sin": 4.5,
                },
            ],
        )
        expected_decoder_importance = pd.DataFrame(
            [
                {
                    "darts_enc_fc_cyc_month_cos": 87.8,
                    "darts_enc_fc_cyc_month_sin": 12.2,
                },
            ]
        )

        # act
        with patch("matplotlib.pyplot.show") as _:
            res = explainer.get_variable_selection_weight(plot=True)

        # assert
        self.assertTrue(isinstance(res, dict))
        self.assertTrue(res.keys() == {"encoder_importance", "decoder_importance"})
        pd.testing.assert_frame_equal(
            res["encoder_importance"], expected_encoder_importance
        )
        pd.testing.assert_frame_equal(
            res["decoder_importance"], expected_decoder_importance
        )

    def test_get_variable_selection_weight_multiple_covariates(self):
        """The get_variable_selection_weight method returns the feature importance for multiple covariates as input."""
        # arrange
        model = deepcopy(self.models[0])

        # fit the model with past covariates
        np.random.seed(342)
        _ = model.fit(
            self.train_ice_transformed,
            past_covariates=self.multiple_covariates_transformed,
            verbose=False,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        # expected results
        expected_encoder_importance = pd.DataFrame(
            [
                {
                    "month": 49.1,
                    "year": 18.9,
                    "darts_enc_fc_cyc_month_cos": 14.7,
                    "darts_enc_fc_cyc_month_sin": 9.5,
                    "ice cream": 5.4,
                    "heater": 2.4,
                }
            ],
        )
        expected_decoder_importance = pd.DataFrame(
            [
                {
                    "darts_enc_fc_cyc_month_cos": 80.2,
                    "darts_enc_fc_cyc_month_sin": 19.8,
                },
            ]
        )

        # act
        res = explainer.get_variable_selection_weight(plot=False)

        # assert
        self.assertTrue(isinstance(res, dict))
        self.assertTrue(res.keys() == {"encoder_importance", "decoder_importance"})

        # Test specific variable selection weights
        # Because of numerical differences between architectures (mac vs linux) we allow a difference of 1
        pd.testing.assert_frame_equal(
            res["encoder_importance"],
            expected_encoder_importance,
            atol=1,
        )
        pd.testing.assert_frame_equal(
            res["decoder_importance"],
            expected_decoder_importance,
            atol=1,
        )

    def test_explain(self):
        """The get_variable_selection_weight method returns the feature importance."""
        # arrange
        model = deepcopy(self.models[0])
        # fit the model with past covariates
        np.random.seed(342)
        model.fit(
            self.train_ice_transformed,
            past_covariates=self.covariates_heat_transformed,
            verbose=True,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        expected_average_attention = [
            [0.1186],
            [0.1015],
            [0.094],
            [0.0955],
            [0.0996],
            [0.102],
            [0.1016],
            [0.0975],
            [0.0935],
            [0.0943],
            [0.0988],
            [0.1019],
            [0.0997],
            [0.0928],
            [0.0908],
            [0.0944],
            [0.0999],
            [0.1041],
            [0.1018],
            [0.0981],
            [0.0935],
            [0.0951],
            [0.0997],
            [0.1033],
            [0.1013],
            [0.094],
            [0.0913],
            [0.0945],
            [0.0993],
            [0.1023],
            [0.102],
            [0.0981],
            [0.094],
            [0.0951],
            [0.1009],
            [0.104],
            [0.0737],
            [0.0611],
            [0.0533],
            [0.0513],
            [0.0492],
            [0.0449],
            [0.0382],
            [0.0308],
            [0.0239],
            [0.0165],
            [0.0081],
            [0.0],
        ]

        # act
        res = explainer.explain()

        # assert
        self.assertTrue(isinstance(res, ExplainabilityResult))
        res_attention_heads = res.get_explanation(
            component="attention_heads",
            horizon=0,
        )
        self.assertTrue(len(res_attention_heads) == 48)
        self.assertTrue(
            (
                res_attention_heads.time_index
                == pd.RangeIndex(start=0, stop=48, step=1, name="time")
            ).all()
        )

        self.assertTrue(
            res_attention_heads.mean(1).values().round(4).tolist()
            == expected_average_attention
        )

    def test_get_explanation(self):
        """The get_variable_selection_weight method returns the feature importance."""
        # arrange
        model = deepcopy(self.models[0])
        # fit the model with past covariates
        np.random.seed(342)
        model.fit(
            self.train_ice_transformed,
            past_covariates=self.covariates_heat_transformed,
            verbose=True,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        expl_result = explainer.explain()

        # act
        res = expl_result.get_explanation(component="attention_heads", horizon=0)

        # assert
        self.assertTrue(isinstance(res, TimeSeries))

    def test_plot_attention_heads(self):
        """The get_variable_selection_weight method returns the feature importance."""
        # arrange
        model = deepcopy(self.models[0])
        # fit the model with past covariates
        model.fit(
            self.train_ice_transformed,
            past_covariates=self.covariates_heat_transformed,
            verbose=True,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        expl_result = explainer.explain()

        # act / assert
        #
        with patch("matplotlib.pyplot.show") as _:
            _ = explainer.plot_attention_heads(expl_result, plot_type="all")
            _ = explainer.plot_attention_heads(expl_result, plot_type="time")
            _ = explainer.plot_attention_heads(expl_result, plot_type="heatmap")

    def test_plot_attention_heads_multiple_covariates(self):
        """The get_variable_selection_weight method returns the feature importance."""
        # arrange
        model = deepcopy(self.models[0])
        # fit the model with past covariates
        model.fit(
            self.train_ice_transformed,
            past_covariates=self.multiple_covariates_transformed,
            verbose=True,
        )

        # call methods for debugging / development
        explainer = TFTExplainer(model)

        expl_result = explainer.explain()

        # act / assert
        #
        with patch("matplotlib.pyplot.show") as _:
            _ = explainer.plot_attention_heads(expl_result, plot_type="all")
            _ = explainer.plot_attention_heads(expl_result, plot_type="time")
            _ = explainer.plot_attention_heads(expl_result, plot_type="heatmap")
