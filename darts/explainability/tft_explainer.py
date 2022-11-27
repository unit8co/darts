from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries
from darts.explainability.explainability import (
    ExplainabilityResult,
    ForecastingModelExplainer,
)
from darts.models import TFTModel


class TFTExplainer(ForecastingModelExplainer):
    def __init__(
        self,
        model: TFTModel,
    ):
        """
        Explainer class for the TFT model.

        Parameters
        ----------
        model
            The fitted TFT model to be explained.
        """
        super().__init__(model)

        if not model._fit_called:
            raise ValueError("The model needs to be trained before explaining it.")

        self._model = model

    def get_variable_selection_weight(self, plot=False) -> Dict[str, pd.DataFrame]:
        """Returns the variable selection weight of the TFT model.

        Parameters
        ----------
        plot
            Whether to plot the variable selection weight.

        Returns
        -------
        TimeSeries
            The variable selection weight.

        """
        encoder_weights = self._model.model._encoder_sparse_weights.mean(axis=1)
        decoder_weights = self._model.model._decoder_sparse_weights.mean(axis=1)

        # format the weights as the feature importance scaled 0-100%
        encoder_weights_percentage = (
            encoder_weights.detach().numpy().mean(axis=0).round(3) * 100
        )
        decoder_weights_percentage = (
            decoder_weights.detach().numpy().mean(axis=0).round(3) * 100
        )

        # get the feature names
        # TODO: These are not the correct feature names
        encoder_names = self._model.model.encoder_variables
        decoder_names = self._model.model.decoder_variables

        encoder_importance = pd.DataFrame(
            encoder_weights_percentage, columns=encoder_names
        )
        decoder_importance = pd.DataFrame(
            decoder_weights_percentage, columns=decoder_names
        )

        # sort importance from high to low
        encoder_importance = (
            encoder_importance.transpose().sort_values(0, ascending=False).transpose()
        )
        decoder_importance = (
            decoder_importance.transpose().sort_values(0, ascending=False).transpose()
        )

        if plot:
            # plot the encoder and decoder weights sorted descending
            encoder_importance.plot(kind="bar", title="Encoder weights")
            decoder_importance.plot(kind="bar", title="Decoder weights")

        return {
            "encoder_importance": encoder_importance,
            "decoder_importance": decoder_importance,
        }

    def explain(self, **kwargs) -> ExplainabilityResult:
        """Returns the explainability result of the TFT model.

        The explainability result contains the attention heads of the TFT model.
        The attention heads determine the contribution of time-varying inputs.

        Parameters
        ----------
        kwargs
            Arguments passed to the `predict` method of the TFT model.

        Returns
        -------
        ExplainabilityResult
            The explainability result containing the attention heads.

        """
        super().explain()
        # without the predict call, the weights will still bet set to the last iteration of the forward() method
        # of the _TFTModule class
        if "n" not in kwargs:
            kwargs["n"] = self._model.model.output_chunk_length

        _ = self._model.predict(**kwargs)

        # get the weights and the attention head from the trained model for the prediction
        attention_heads = (
            self._model.model._attn_out_weights.squeeze().sum(axis=1).detach()
        )

        # return the explainer result to be used in other methods
        return ExplainabilityResult(
            {
                0: {
                    "attention_heads": TimeSeries.from_dataframe(
                        pd.DataFrame(attention_heads).transpose()
                    ),
                }
            },
        )

    @staticmethod
    def plot_attention_heads(
        expl_result: ExplainabilityResult,
        plot_type: Optional[Literal["all", "time", "heatmap"]] = "time",
    ):
        """Plots the attention heads of the TFT model."""
        attention_heads = expl_result.get_explanation(
            component="attention_heads", horizon=0
        )
        if plot_type == "all":
            fig = plt.figure()
            attention_heads.plot(
                label="Attention Head", plot_all_components=True, figure=fig
            )
            plt.xlabel("Time steps in past")
            plt.ylabel("Attention")
        elif plot_type == "time":
            fig = plt.figure()
            attention_heads.mean(1).plot(label="Mean Attention Head", figure=fig)
            plt.xlabel("Time steps in past")
            plt.ylabel("Attention")
        elif plot_type == "heatmap":
            avg_attention = attention_heads.values().transpose()
            fig = plt.figure()
            plt.imshow(avg_attention, cmap="hot", interpolation="nearest", figure=fig)
            plt.xlabel("Time steps in past")
            plt.ylabel("Horizon")
        else:
            raise ValueError("`plot_type` must be either 'all', 'time' or 'heatmap'")
