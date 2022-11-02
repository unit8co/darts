from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import IceCreamHeaterDataset
from darts.explainability.explainability import (
    ExplainabilityResult,
    ForecastingModelExplainer,
)
from darts.models import TFTModel


class TFTExplainer(ForecastingModelExplainer):

    def __init__(
            self,
            model: TFTModel,
            background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            background_past_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            background_future_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
    ):
        """
        Explainer class for the TFT model.

        Parameters
        ----------
        model
            The fitted TFT model to be explained.
        background_series
            The background series to be used for the TFT predict method.
        background_past_covariates
            The past covariates to be used for the TFT predict method.
        background_future_covariates
            The future covariates to be used for the TFT predict method.

        """
        super().__init__(
            model,
            background_series,
            background_past_covariates,
            background_future_covariates,
        )

        self._model = model
        self.background_series = background_series
        self.background_past_covariates = background_past_covariates
        self.background_future_covariates = background_future_covariates
        self._explain_results = None

    def explain(
            self,
            foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            foreground_past_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            foreground_future_covariates: Optional[
                Union[TimeSeries, Sequence[TimeSeries]]
            ] = None,
            horizons: Optional[Sequence[int]] = None,
            target_components: Optional[Sequence[str]] = None,
    ) -> ExplainabilityResult:
        super().explain(
            foreground_series, foreground_past_covariates, foreground_future_covariates
        )
        if self._explain_results is None:
            # without the predict call, the weights will still bet set to the last iteration of the forward() method
            # of the _TFTModule class
            _ = self._model.predict(n=self._model.model.output_chunk_length)

            # get the weights and the attention head from the trained model for the prediction
            encoder_weights = self._model.model._encoder_sparse_weights.mean(axis=1)
            decoder_weights = self._model.model._decoder_sparse_weights.mean(axis=1)
            attention_heads = self._model.model._attn_out_weights.squeeze().sum(axis=1).detach()

            # format the weights as the feature importance scaled 0-100%
            encoder_weights_percentage = encoder_weights.detach().numpy().mean(axis=0).round(3) * 100
            decoder_weights_percentage = decoder_weights.detach().numpy().mean(axis=0).round(3) * 100

            # get the feature names
            # TODO: This are not the correct feature names
            encoder_names = self._model.model.encoder_variables
            decoder_names = self._model.model.decoder_variables

            # return the explainer result to be used in other methods
            expl_res = {
                "decoder_weights_percentage": decoder_weights_percentage,
                "decoder_names": decoder_names,
                "encoder_weights_percentage": encoder_weights_percentage,
                "encoder_names": encoder_names,
                "attention_heads": attention_heads,
            }
            self._explain_results = ExplainabilityResult({"tft": expl_res})

        return self._explain_results

    def feature_importance(self, plot=True):
        if self._explain_results is None:
            self.explain()
        expl_res = self._explain_results.explained_forecasts["tft"]
        encoder_importance = dict(
            zip(
                expl_res["encoder_names"],
                expl_res["encoder_weights_percentage"][0],
            ),
        )
        decoder_importance = dict(
            zip(
                expl_res["decoder_names"],
                expl_res["decoder_weights_percentage"][0],
            ),
        )
        if plot:
            plt.figure(figsize=(12, 6))
            plt.barh(*zip(*encoder_importance.items()))
            plt.title("Encoder feature importance")
            plt.show()
            plt.figure(figsize=(12, 6))
            plt.barh(*zip(*decoder_importance.items()))
            plt.title("Decoder feature importance")
            plt.show()

        return {"encoder_importance": encoder_importance, "decoder_importance": decoder_importance}

    def time_plots(self, plot_type="time"):
        if self._explain_results is None:
            self.explain()
        expl_res = self._explain_results.explained_forecasts["tft"]
        attention_heads = expl_res["attention_heads"]

        if plot_type == "time":
            attention_matrix = attention_heads.mean(axis=0)
            plt.plot(attention_matrix)
            plt.xlabel("Time steps in past")
            plt.ylabel("Attention")
            plt.show()
        if plot_type == "heatmap":
            plt.imshow(attention_heads, cmap='hot', interpolation='nearest')
            # plt.legend()
            # plt.xticks(range(0, attention_matrix_avarege.shape[1], attention_matrix_avarege.shape[0]))
            plt.xlabel("Time steps in past")
            plt.ylabel("Horizon")
            plt.show()
        else:
            raise ValueError("`plot_type` must be either 'time' or 'heatmap'")


### Debug Code: Ice Example from the TFT turotial ############################
series_ice_heater = IceCreamHeaterDataset().load()

# convert monthly sales to average daily sales per month
converted_series = []
for col in ["ice cream", "heater"]:
    converted_series.append(
        series_ice_heater[col]
        / TimeSeries.from_series(series_ice_heater.time_index.days_in_month)
    )
converted_series = concatenate(converted_series, axis=1)
converted_series = converted_series[pd.Timestamp("20100101"):]

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

# use the last 3 years as past input data
input_chunk_length_ice = 36

# use `add_encoders` as we don't have future covariates
my_model_ice = TFTModel(
    input_chunk_length=input_chunk_length_ice,
    output_chunk_length=forecast_horizon_ice,
    hidden_size=32,
    lstm_layers=1,
    batch_size=16,
    n_epochs=3,
    dropout=0.1,
    add_encoders={"cyclic": {"future": ["month"]}},
    add_relative_index=False,
    optimizer_kwargs={"lr": 1e-3},
    random_state=42,
)

# fit the model with past covariates
my_model_ice.fit(
    train_ice_transformed, past_covariates=covariates_heat_transformed, verbose=True
)

# call methods for debugging / development
tft_explainer = TFTExplainer(
    my_model_ice,
    background_series=series_ice_transformed,
    background_past_covariates=covariates_heat_transformed,
)
tft_explainer.feature_importance()
