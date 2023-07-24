"""
TFT Explainer for Temporal Fusion Transformer models.
------------------------------------

This module contains the implementation of the TFT Explainer class. The TFTExplainer uses a trained TFT model
and extracts the explainability information from the model.

The .get_variable_selection_weights() method returns the variable selection weights for each of the input features.
This reflects the feature importance for the model. The weights of the encoder and decoder matrix are returned.
An optional plot parameter can be used to plot the variable selection weights.

The .plot_attention_heads() method shows the transformer attention heads learned by the TFT model.
The attention heads reflect the effect of past values of the dependent variable onto the prediction and
what autoregressive pattern the model has learned.

The values of the attention heads can also be extracted using the .get_attention_heads() method.
explain_result = .explain()
res_attention_heads = explain_result.get_explanation(component="attention_heads", horizon=0)

For an examples on how to use the TFT explainer, please have a look at the TFT notebook in the /examples directory
 <https://github.com/unit8co/darts/blob/master/examples/13-TFT-examples.ipynb>`_.

"""

from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor

from darts import TimeSeries
from darts.explainability.explainability import (
    ExplainabilityResult,
    ForecastingModelExplainer,
)
from darts.logging import get_logger, raise_log
from darts.models import TFTModel
from darts.utils.timeseries_generation import generate_index

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


logger = get_logger(__name__)


class TFTExplainer(ForecastingModelExplainer):
    model: TFTModel

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
            The fitted `TFTModel` to be explained.
        background_series
            Optionally, a series or list of series to use as a default target series for the explanations.
            If `None`, some `TFTExplainer` methods will require a `foreground_series`.
        background_past_covariates
            Optionally, a past covariates series or list of series to use as a default past covariates series
            for the explanations. If `None`, some `TFTExplainer` methods will require `foreground_past_covariates`.
        background_future_covariates
            Optionally, a future covariates series or list of series to use as a default future covariates series
            for the explanations. If `None`, some `TFTExplainer` methods will require `foreground_future_covariates`.
        """
        super().__init__(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            requires_background=True,
            requires_covariates_encoding=False,
            check_component_names=False,
            test_stationarity=False,
        )
        # add the relative index that generated inside the model (not in the input data)
        if model.add_relative_index:
            self.future_covariates_components.append("add_relative_index")

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
        # """Returns the explainability result of the TFT model.
        #
        # The explainability result contains the attention heads of the TFT model.
        # The attention heads determine the contribution of time-varying inputs.
        #
        # Parameters
        # ----------
        # kwargs
        #     Arguments passed to the `predict` method of the TFT model.
        #
        # Returns
        # -------
        # ExplainabilityResult
        #     The explainability result containing the attention heads.
        #
        # """
        super().explain(
            foreground_series, foreground_past_covariates, foreground_future_covariates
        )
        (
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        horizons, _ = self._process_horizons_and_targets(
            horizons,
            None,
        )
        preds = self.model.predict(
            n=self.n,
            series=foreground_series,
            past_covariates=foreground_past_covariates,
            future_covariates=foreground_future_covariates,
        )
        # get the weights and the attention head from the trained model for the prediction
        # aggregate over attention heads
        attention_heads = (
            self.model.model._attn_out_weights.detach().numpy().sum(axis=-2)
        )
        index = [h - 1 for h in horizons]
        results = []

        icl = self.model.input_chunk_length
        for idx, (series, pred_series) in enumerate(zip(foreground_series, preds)):
            results.append(
                ExplainabilityResult(
                    {
                        "attention_heads": TimeSeries.from_times_and_values(
                            values=np.take(attention_heads[idx], index, axis=0).T,
                            times=series.time_index[-icl:].union(
                                pred_series.time_index
                            ),
                            columns=[f"horizon {str(i)}" for i in horizons],
                        ),
                    }
                )
            )
        # if "n" not in kwargs:
        #     kwargs["n"] = self.model.model.output_chunk_length
        #
        # _ = self.model.predict(**kwargs)
        #
        # # get the weights and the attention head from the trained model for the prediction
        # attention_heads = (
        #     self.model.model._attn_out_weights.squeeze().sum(axis=1).detach()
        # )
        #
        # # return the explainer result to be used in other methods
        # return ExplainabilityResult(
        #     {
        #         "attention_heads": TimeSeries.from_values(attention_heads.T),
        #     }
        # )
        return results

    @property
    def encoder_importance(self):
        """Returns the encoder variable importance of the TFT model.

        The encoder_weights are calculated for the past inputs of the model.
        The encoder_importance contains the weights of the encoder variable selection network.
        The encoder variable selection network is used to select the most important static and time dependent
        covariates. It provides insights which variable are most significant for the prediction problem.
        See section 4.2 of the paper for more details.

        Returns
        -------
        pd.DataFrame
            The encoder variable importance.
        """
        return self._get_importance(
            weight=self.model.model._encoder_sparse_weights,
            names=self.model.model.encoder_variables,
        )

    @property
    def static_covariates_importance(self):
        """Returns the static covariates importance of the TFT model.

        The static covariate importances are calculated for the static inputs of the model (numeric and / or
        categorical). The static variable selection network is used to select the most important static covariates.
        It provides insights which variable are most significant for the prediction problem.
        See section 4.2, and 4.3 of the paper for more details.

        Returns
        -------
        pd.DataFrame
            The static covariates importance.
        """
        return self._get_importance(
            weight=self.model.model._static_covariate_var,
            names=self.model.model.static_variables,
        )

    @property
    def decoder_importance(self):
        """Returns the decoder variable importance of the TFT model.

        The decoder_weights are calculated for the known future inputs of the model.
        The decoder_importance contains the weights of the decoder variable selection network.
        The decoder variable selection network is used to select the most important static and time dependent
        covariates. It provides insights which variable are most significant for the prediction problem.
        See section 4.2 of the paper for more details.

        Returns
        -------
        pd.DataFrame
            The importance of the decoder variables.
        """
        return self._get_importance(
            weight=self.model.model._decoder_sparse_weights,
            names=self.model.model.decoder_variables,
        )

    def get_variable_selection_weight(
        self, plot=False, figsize=None
    ) -> Dict[str, pd.DataFrame]:
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

        encoder_importance = self.encoder_importance
        decoder_importance = self.decoder_importance
        static_covariates_importance = self.static_covariates_importance
        uses_static_covariates = not static_covariates_importance.empty
        if plot:
            # plot the encoder and decoder weights
            fig, axes = plt.subplots(
                nrows=3 if uses_static_covariates else 2, sharex=True, figsize=figsize
            )
            self._plot_cov_selection(
                encoder_importance, title="Encoder variable importance", ax=axes[0]
            )
            axes[0].set_xlabel("")
            self._plot_cov_selection(
                decoder_importance, title="Decoder variable importance", ax=axes[1]
            )
            if uses_static_covariates:
                axes[1].set_xlabel("")
                self._plot_cov_selection(
                    static_covariates_importance,
                    title="Static variable importance",
                    ax=axes[2],
                )
            fig.tight_layout()
            plt.show()

        return {
            "encoder_importance": encoder_importance,
            "decoder_importance": decoder_importance,
            "static_covariates_importance": static_covariates_importance,
        }

    def plot_attention_heads(
        self,
        expl_result: ExplainabilityResult,
        plot_type: Optional[Literal["all", "time", "heatmap"]] = "all",
        show_index_as: Literal["relative", "time"] = "relative",
        ax=None,
        max_nr_series: int = 5,
    ):
        """Plots the attention heads of the TFT model.

        Parameters
        ----------
        expl_result
            One or a list of `TFTExplainabilityResult`. Corresponds to the output of `TFTExplainer.explain()`
        plot_type
            The type of attention head plot. One of ("all", "time", "heatmap").
            If "all", will plot the attention per horizon (given the horizons in the ExplainabilityResult).
            The maximum horizon corresponds to the `output_chunk_length` of the trained `TFTModel`.
            If "time", will plot the mean attention over all horizons.
            If "heatmap", will plot the attention per horizon on a heat map. The horizons are shown on the y-axis,
            and times / relative indices on the x-axis.
        show_index_as
            The type of index to be shown. One of ("relative", "time").
            If "relative", will plot the x-axis from `(-input_chunk_length, output_chunk_length - 1)`. `0` corresponds
            to the first prediction point.
            If "time", will plot the x-axis with the actual time index (or range index) of the corresponding
            ExplainabilityResult.
        ax
            Optionally, an axis to plot on. Only effective on a single `expl_result`.
        max_nr_series
            The maximum number of plots to show in case of a list of `expl_result`.
        """
        single_series = False
        if isinstance(expl_result, ExplainabilityResult):
            expl_result = [expl_result]
            single_series = True

        for idx, res in enumerate(expl_result):
            attention_heads = res.get_explanation(component="attention_heads")
            if ax is None or not single_series:
                fig, ax = plt.subplots()
            if show_index_as == "relative":
                x_ticks = generate_index(
                    start=-self.model.input_chunk_length, end=self.n - 1
                )
                attention_heads = TimeSeries.from_times_and_values(
                    times=generate_index(
                        start=-self.model.input_chunk_length, end=self.n - 1
                    ),
                    values=attention_heads.values(copy=False),
                )
                x_label = "Index relative to first prediction point"
            elif show_index_as == "time":
                x_ticks = attention_heads.time_index
                x_label = "Time index"
            else:
                x_label, x_ticks = None, None
                raise_log(
                    ValueError("`show_index_as` must either be 'relative', or 'time'.")
                )

            prediction_start_color = "red"
            if plot_type == "all":
                ax_title = "Mean Attention"
                y_label = "Attention"
                attention_heads.plot(max_nr_components=-1, ax=ax)
            elif plot_type == "time":
                ax_title = "Attention per Horizon"
                y_label = "Attention"
                attention_heads.mean(1).plot(label="Mean Attention Head", ax=ax)
            elif plot_type == "heatmap":
                ax_title = "Attention Heat Map"
                y_label = "Horizon"

                # generate a heat map
                x, y = np.meshgrid(x_ticks, np.arange(1, self.n + 1))
                c = ax.pcolormesh(
                    x, y, attention_heads.values().transpose(), cmap="hot"
                )
                ax.axis([x.min(), x.max(), y.max(), y.min()])
                prediction_start_color = "lightblue"
                fig.colorbar(c, ax=ax, orientation="horizontal")
            else:
                raise raise_log(
                    ValueError("`plot_type` must be either 'all', 'time' or 'heatmap'"),
                    logger=logger,
                )

            # draw the prediction start point
            y_min, y_max = ax.get_ylim()
            ax.vlines(
                x=x_ticks[-12],
                ymin=y_min,
                ymax=y_max,
                label="prediction start",
                ls="dashed",
                lw=2,
                colors=prediction_start_color,
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            title_suffix = "" if single_series else f": series index {idx}"
            ax.set_title(ax_title + title_suffix)
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
            plt.show()

            if idx + 1 == max_nr_series:
                break

    def _get_importance(
        self,
        weight: Tensor,
        names: List[str],
        n_decimals=3,
    ) -> pd.DataFrame:
        """Returns the encoder or decoder variable of the TFT model.

        Parameters
        ----------
        weights
            The weights of the encoder or decoder of the trained TFT model.
        names
            The encoder or decoder names saved in the TFT model class.
        n_decimals
            The number of decimals to round the importance to.

        Returns
        -------
        pd.DataFrame
            The importance of the variables.
        """
        if weight is None:
            return pd.DataFrame()

        # transform the encoder/decoder weights to percentages, rounded to n_decimals
        weights_percentage = (
            weight.mean(axis=1).detach().numpy().mean(axis=0).round(n_decimals) * 100
        )
        # static covariates aggregation needs expansion in first dimension
        if len(weights_percentage.shape) == 1:
            weights_percentage = np.expand_dims(weights_percentage, 0)
        # create a dataframe with the variable names and the weights
        name_mapping = self._name_mapping
        importance = pd.DataFrame(
            weights_percentage,
            columns=[name_mapping[name] for name in names],
        )

        # return the importance sorted descending
        return importance.transpose().sort_values(0, ascending=True).transpose()

    @property
    def _name_mapping(self) -> Dict[str, str]:
        """Returns the feature name mapping of the TFT model.

        Returns
        -------
        Dict[str, str]
            The feature name mapping. For example
            {
                'target_0': 'ice cream',
                'past_covariate_0': 'heater',
                'past_covariate_1': 'year',
                'past_covariate_2': 'month',
                'future_covariate_0': 'darts_enc_fc_cyc_month_sin',
                'future_covariate_1': 'darts_enc_fc_cyc_month_cos',
             }
        """

        def map_cols(comps, name):
            comps = comps if comps is not None else []
            return {f"{name}_{i}": colname for i, colname in enumerate(comps)}

        return {
            **map_cols(self.target_components, "target"),
            **map_cols(self.static_covariates_components, "static_covariate"),
            **map_cols(self.past_covariates_components, "past_covariate"),
            **map_cols(self.future_covariates_components, "future_covariate"),
        }

    @staticmethod
    def _plot_cov_selection(
        importance: pd.DataFrame,
        title: str = "Variable importance",
        ax=None,
    ):
        """Plots the variable importance of the TFT model.

        Parameters
        ----------
        importance
            The encoder / decoder importance.
        title
            The title of the plot.

        """
        if ax is None:
            _, ax = plt.subplots()
        ax.barh(importance.columns.tolist(), importance.values[0].tolist())
        ax.set_title(title)
        ax.set_ylabel("Variable", fontsize=12)
        ax.set_xlabel("Variable importance in %")
        return ax
