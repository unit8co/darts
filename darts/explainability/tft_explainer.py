"""
TFT Explainer for Temporal Fusion Transformer (TFTModel)
------------------------------------

The `TFTExplainer` uses a trained :class:`TFTModel <darts.models.forecasting.tft_model.TFTModel>` and extracts the
explainability information from the model.

- :func:`plot_variable_selection() <TFTExplainer.plot_variable_selection>` plots the variable selection weights for
  each of the input features.
  - encoder importance: historic part of target, past covariates and historic part of future covariates
  - decoder importance: future part of future covariates
  - static covariates importance: the numeric and catageorical static covariates importance

- :func:`plot_attention() <TFTExplainer.plot_attention>` plots the transformer attention that the `TFTModel` applies
  on the given past and future input. The attention is aggregated over all attention heads.

The attention and feature importance values can be extracted using the :class:`TFTExplainabilityResult
<darts.explainability.explainability_result.TFTExplainabilityResult>` returned by
:func:`explain() <TFTExplainer.explain>`. An example of this is shown in the method description.

We also show how to use the `TFTExplainer` in the example notebook of the `TFTModel` `here
<https://unit8co.github.io/darts/examples/13-TFT-examples.html#Explainability>`_.
"""

from collections.abc import Sequence
from typing import Literal, Optional, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor

from darts import TimeSeries
from darts.explainability import TFTExplainabilityResult
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.logging import get_logger, raise_log
from darts.models import TFTModel
from darts.utils.utils import generate_index

logger = get_logger(__name__)


class TFTExplainer(_ForecastingModelExplainer):
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
        Explainer class for the `TFTModel`.

        **Definitions**

        - A background series is a `TimeSeries` that is used as a default for generating the explainability result
          (if no `foreground` is passed to :func:`explain() <TFTExplainer.explain>`).
        - A foreground series is a `TimeSeries` that can be passed to :func:`explain() <TFTExplainer.explain>` to use
          instead of the background for generating the explainability result.

        Parameters
        ----------
        model
            The fitted `TFTModel` to be explained.
        background_series
            Optionally, a series or list of series to use as a default target series for the explanations.
            Optional if `model` was trained on a single target series. By default, it is the `series` used at fitting
            time.
            Mandatory if `model` was trained on multiple (sequence of) target series.
        background_past_covariates
            Optionally, a past covariates series or list of series to use as a default past covariates series
            for the explanations. The same requirements apply as for `background_series` .
        background_future_covariates
            Optionally, a future covariates series or list of series to use as a default future covariates series
            for the explanations. The same requirements apply as for `background_series`.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.explainability.tft_explainer import TFTExplainer
        >>> from darts.models import TFTModel
        >>> series = AirPassengersDataset().load()
        >>> model = TFTModel(
        >>>     input_chunk_length=12,
        >>>     output_chunk_length=6,
        >>>     add_encoders={"cyclic": {"future": ["hour"]}}
        >>> )
        >>> model.fit(series)
        >>> # create the explainer and generate explanations
        >>> explainer = TFTExplainer(model)
        >>> results = explainer.explain()
        >>> # plot the results
        >>> explainer.plot_attention(results, plot_type="all")
        >>> explainer.plot_variable_selection(results)
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
            if self.future_covariates_components is not None:
                self.future_covariates_components.append("add_relative_index")
            else:
                self.future_covariates_components = ["add_relative_index"]

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
    ) -> TFTExplainabilityResult:
        """Returns the :class:`TFTExplainabilityResult
        <darts.explainability.explainability_result.TFTExplainabilityResult>` result for all series in
        `foreground_series`. If `foreground_series` is `None`, will use the `background` input
        from `TFTExplainer` creation (either the `background` passed to creation, or the series stored in the
        `TFTModel` in case it was only trained on a single series).
        For each series, the results contain the attention heads, encoder variable importances, decoder variable
        importances, and static covariates importances.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates `TimeSeries` if required by the forecasting model.
        horizons
            This parameter is not used by the `TFTExplainer`.
        target_components
            This parameter is not used by the `TFTExplainer`.

        Returns
        -------
        TFTExplainabilityResult
            The explainability result containing the attention heads, encoder variable importances, decoder variable
            importances, and static covariates importances.

        Examples
        --------
        >>> explainer = TFTExplainer(model)  # requires `background` if model was trained on multiple series

        Optionally, give a foreground input to generate the explanation on a new input.
        Otherwise, leave it empty to compute the explanation on the background from `TFTExplainer` creation

        >>> explain_results = explainer.explain(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates,
        >>> )
        >>> attn = explain_results.get_attention()
        >>> importances = explain_results.get_feature_importances()
        """
        if target_components is not None or horizons is not None:
            logger.warning(
                "`horizons`, and `target_components` are not supported by `TFTExplainer` and will be ignored."
            )
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
        horizons, _ = self._process_horizons_and_targets(None, None)
        preds = self.model.predict(
            n=self.n,
            series=foreground_series,
            past_covariates=foreground_past_covariates,
            future_covariates=foreground_future_covariates,
        )
        # get the weights and the attention head from the trained model for the prediction
        # aggregate over attention heads
        attention_heads = (
            self.model.model._attn_out_weights.detach().cpu().numpy().sum(axis=-2)
        )
        # get the variable importances (pd.DataFrame with rows corresponding to the number of input series)
        encoder_importance = self._encoder_importance
        decoder_importance = self._decoder_importance
        static_covariates_importance = self._static_covariates_importance

        horizon_idx = [h - 1 for h in horizons]

        results = []
        icl = self.model.input_chunk_length
        for idx, (series, pred_series) in enumerate(zip(foreground_series, preds)):
            times = series.time_index[-icl:].union(pred_series.time_index)
            attention = TimeSeries.from_times_and_values(
                values=np.take(attention_heads[idx], horizon_idx, axis=0).T,
                times=times,
                columns=[f"horizon {str(i)}" for i in horizons],
            )
            results.append({
                "attention": attention,
                "encoder_importance": encoder_importance.iloc[idx : idx + 1],
                "decoder_importance": decoder_importance.iloc[idx : idx + 1],
                "static_covariates_importance": static_covariates_importance.iloc[
                    idx : idx + 1
                ],
            })
        return TFTExplainabilityResult(
            explanations=results[0] if len(results) == 1 else results
        )

    def plot_variable_selection(
        self,
        expl_result: TFTExplainabilityResult,
        fig_size=None,
        max_nr_series: int = 5,
    ):
        """Plots the variable selection / feature importances of the `TFTModel` based on the input.
        The figure includes three subplots:

        - encoder importances: contains the past target, past covariates, and historic future covariates importance
          on the encoder (input chunk)
        - decoder importances: contains the future covariates importance on the decoder (output chunk)
        - static covariates importances: contains the numeric and / or categorical static covariates importance

        Parameters
        ----------
        expl_result
            A `TFTExplainabilityResult` object. Corresponds to the output of :func:`explain() <TFTExplainer.explain>`.
        fig_size
            The size of the figure to be plotted.
        max_nr_series
            The maximum number of plots to show in case `expl_result` was computed on multiple series.
        """
        encoder_importance = expl_result.get_encoder_importance()
        decoder_importance = expl_result.get_decoder_importance()
        static_covariates_importance = expl_result.get_static_covariates_importance()
        if not isinstance(encoder_importance, list):
            encoder_importance = [encoder_importance]
            decoder_importance = [decoder_importance]
            static_covariates_importance = [static_covariates_importance]

        uses_static_covariates = not static_covariates_importance[0].empty
        for idx, (enc_imp, dec_imp, stc_imp) in enumerate(
            zip(encoder_importance, decoder_importance, static_covariates_importance)
        ):
            # plot the encoder and decoder weights
            fig, axes = plt.subplots(
                nrows=3 if uses_static_covariates else 2, sharex=True, figsize=fig_size
            )
            self._plot_cov_selection(
                enc_imp, title="Encoder variable importance", ax=axes[0]
            )
            axes[0].set_xlabel("")
            self._plot_cov_selection(
                dec_imp, title="Decoder variable importance", ax=axes[1]
            )
            if uses_static_covariates:
                axes[1].set_xlabel("")
                self._plot_cov_selection(
                    stc_imp,
                    title="Static variable importance",
                    ax=axes[2],
                )
            fig.tight_layout()
            plt.show()

            if idx + 1 == max_nr_series:
                break

    def plot_attention(
        self,
        expl_result: TFTExplainabilityResult,
        plot_type: Optional[Literal["all", "time", "heatmap"]] = "all",
        show_index_as: Literal["relative", "time"] = "relative",
        ax: Optional[matplotlib.axes.Axes] = None,
        max_nr_series: int = 5,
        show_plot: bool = True,
    ) -> matplotlib.axes.Axes:
        """Plots the attention heads of the `TFTModel`.

        Parameters
        ----------
        expl_result
            A `TFTExplainabilityResult` object. Corresponds to the output of :func:`explain() <TFTExplainer.explain>`.
        plot_type
            The type of attention head plot. One of ("all", "time", "heatmap").
            If "all", will plot the attention per horizon (given the horizons in the `TFTExplainabilityResult`).
            The maximum horizon corresponds to the `output_chunk_length` of the trained `TFTModel`.
            If "time", will plot the mean attention over all horizons.
            If "heatmap", will plot the attention per horizon on a heat map. The horizons are shown on the y-axis,
            and times / relative indices on the x-axis.
        show_index_as
            The type of index to be shown. One of ("relative", "time").
            If "relative", will plot the x-axis from `(-input_chunk_length, output_chunk_length - 1)`. `0` corresponds
            to the first prediction point.
            If "time", will plot the x-axis with the actual time index (or range index) of the corresponding
            `TFTExplainabilityResult`.
        ax
            Optionally, an axis to plot on. Only effective on a single `expl_result`.
        max_nr_series
            The maximum number of plots to show in case `expl_result` was computed on multiple series.
        show_plot
            Whether to show the plot.
        """
        single_series = False
        attentions = expl_result.get_explanation(component="attention")
        if isinstance(attentions, TimeSeries):
            attentions = [attentions]
            single_series = True

        for idx, attention in enumerate(attentions):
            if ax is None or not single_series:
                fig, ax = plt.subplots()
            if show_index_as == "relative":
                x_ticks = generate_index(
                    start=-self.model.input_chunk_length, end=self.n - 1
                )
                attention = TimeSeries.from_times_and_values(
                    times=generate_index(
                        start=-self.model.input_chunk_length, end=self.n - 1
                    ),
                    values=attention.values(copy=False),
                    columns=attention.components,
                )
                x_label = "Index relative to first prediction point"
            elif show_index_as == "time":
                x_ticks = attention.time_index
                x_label = "Time index"
            else:
                x_label, x_ticks = None, None
                raise_log(
                    ValueError("`show_index_as` must either be 'relative', or 'time'.")
                )

            prediction_start_color = "red"
            if plot_type == "all":
                ax_title = "Attention per Horizon"
                y_label = "Attention"
                attention.plot(max_nr_components=-1, ax=ax)
            elif plot_type == "time":
                ax_title = "Mean Attention"
                y_label = "Attention"
                attention.mean(1).plot(label="Mean Attention Head", ax=ax)
            elif plot_type == "heatmap":
                ax_title = "Attention Heat Map"
                y_label = "Horizon"

                # generate a heat map
                x, y = np.meshgrid(x_ticks, np.arange(1, self.n + 1))
                c = ax.pcolormesh(x, y, attention.values().transpose(), cmap="hot")
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
                x=x_ticks[-self.n],
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
            if show_plot:
                plt.show()

            if idx + 1 == max_nr_series:
                break
        return ax

    @property
    def _encoder_importance(self) -> pd.DataFrame:
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
    def _decoder_importance(self) -> pd.DataFrame:
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

    @property
    def _static_covariates_importance(self) -> pd.DataFrame:
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

    def _get_importance(
        self,
        weight: Tensor,
        names: list[str],
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

        # static covariates aggregation needs expansion in first dimension
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)

        # transform the encoder/decoder weights to percentages, rounded to n_decimals
        weights_percentage = (
            weight.detach().cpu().numpy().mean(axis=1).squeeze(axis=1).round(n_decimals)
            * 100
        )
        # create a dataframe with the variable names and the weights
        name_mapping = self._name_mapping
        importance = pd.DataFrame(
            weights_percentage,
            columns=[name_mapping[name] for name in names],
        )

        # return the importance sorted descending
        return importance.transpose().sort_values(0, ascending=True).transpose()

    @property
    def _name_mapping(self) -> dict[str, str]:
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

        def map_cols(comps, name, suffix):
            comps = comps if comps is not None else []
            return {
                f"{name}_{i}": colname + f"_{suffix}" for i, colname in enumerate(comps)
            }

        return {
            **map_cols(self.target_components, "target", "target"),
            **map_cols(
                self.static_covariates_components, "static_covariate", "statcov"
            ),
            **map_cols(self.past_covariates_components, "past_covariate", "pastcov"),
            **map_cols(self.future_covariates_components, "future_covariate", "futcov"),
        }

    @staticmethod
    def _plot_cov_selection(
        importance: pd.DataFrame,
        title: str = "Variable importance",
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        """Plots the variable importance of the TFT model.

        Parameters
        ----------
        importance
            The encoder / decoder importance.
        title
            The title of the plot.
        ax
            Optionally, an axis to plot on. Otherwise, will create and plot on a new axis.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.barh(importance.columns.tolist(), importance.values[0].tolist())
        ax.set_title(title)
        ax.set_ylabel("Variable", fontsize=12)
        ax.set_xlabel("Variable importance in %")
        return ax
