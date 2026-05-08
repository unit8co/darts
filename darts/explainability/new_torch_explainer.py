from collections.abc import Sequence

import pandas as pd

from darts import TimeSeries
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.explainability.explainability_result import (
    ShapExplainabilityResult,
    ShapSingleExplainabilityResult,
)
from darts.logging import get_logger
from darts.typing import TimeSeriesLike

logger = get_logger(__name__)


class TorchExplainer(_ForecastingModelExplainer):
    def explain(
        self,
        foreground_series: TimeSeriesLike | None = None,
        foreground_past_covariates: TimeSeriesLike | None = None,
        foreground_future_covariates: TimeSeriesLike | None = None,
        horizons: int | Sequence[int] | None = None,
        target_components: Sequence[str] | None = None,
        **kwargs,
    ) -> ShapExplainabilityResult:
        """
        Explains foreground time series forecasts and returns a :class:`ShapExplainabilityResult
        <darts.explainability.explainability_result.ShapExplainabilityResult>` of SHAP values.

        The results can then be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.ShapExplainabilityResult.get_explanation>`,
        which returns a multivariate ``TimeSeries`` instance containing the SHAP values for the
        ``(horizon, target_component)`` forecasts at all timestamps forecastable in the foreground series.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce
        the forecasts. See above for the naming convention.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target ``TimeSeries`` to be explained. Can be multivariate.
            If not provided, the background ``TimeSeries`` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates ``TimeSeries`` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates ``TimeSeries`` if required by the forecasting model.
        horizons
            Optionally, an integer or sequence of integers representing the future time steps to be explained.
            ``1`` corresponds to the first timestamp being forecasted.
            All values must be no greater than ``output_chunk_length`` of the explained forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer when calling it for explanations, e.g.,
            `npermutations` for the default permutation explainer.

        Returns
        -------
        ShapExplainabilityResult
            The forecast explanations of the specified horizons and target components.

        Examples
        --------
        Say we have a ``TorchForecastingModel`` instance with:

          - 2 target components named ``"T_0"`` and ``"T_1"``,
          - 3 past covariates with default component names ``"P_0"``, ``"P_1"``, and ``"P_2"``,
          - 1 future covariate with default component name ``"F_0"``,
          - ``input_chunk_length=3``,
          - ``output_chunk_length=2``.

        We provide ``foreground_series``, ``foreground_past_covariates``, ``foreground_future_covariates`` (extending
        far enough into the future) each of length 5.

        >>> results = explainer.explain(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates)

        Calling the method returns a ``ShapExplainabilityResult`` object containing the SHAP values, feature values,
        and raw ``shap.Explanation`` objects for each horizon and target component. They can be accessed with:

        >>> # Get SHAP values for forecasting "T_1" at horizon 1 as a `TimeSeries`
        >>> output = results.get_explanation(horizon=1, component="T_1")
        >>> # Get feature values used for forecasting as a `TimeSeries`
        >>> feature_values = results.get_feature_values(horizon=1, component="T_1")
        >>> # Get the raw `shap.Explanation` object for further processing
        >>> shap_objects = results.get_shap_explanation_object(horizon=1, component="T_1")

        For SHAP and feature values, the components of the returned ``TimeSeries`` correspond to different lags of the
        target and covariates (see convention above). In our example, the component names would be:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - P_0_pastcov_lag-1
             - P_0_pastcov_lag-2
             - P_0_pastcov_lag-3
             - P_1_pastcov_lag-1
             - P_1_pastcov_lag-2
             - P_1_pastcov_lag-3
             - P_2_pastcov_lag-1
             - P_2_pastcov_lag-2
             - P_2_pastcov_lag-3
             - F_0_futcov_lag0
             - F_0_futcov_lag1

        Each series has length 3, as the model can explain 5-3+1 forecasts (timestamp indices 4, 5, and 6).
        """
        fallback = foreground_series is None
        (
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
            _,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        horizons, target_names = self._process_horizons_and_targets(
            horizons,
            target_components,
        )

        shap_values_list = []
        feature_values_list = []
        shap_explanation_object_list = []
        for idx, foreground_ts in enumerate(foreground_series):
            foreground_past_cov_ts = None
            foreground_future_cov_ts = None

            if foreground_past_covariates:
                foreground_past_cov_ts = foreground_past_covariates[idx]

            if foreground_future_covariates:
                foreground_future_cov_ts = foreground_future_covariates[idx]

            foreground_X, _, prediction_times = self.explainer.create_shap_array(
                foreground_ts,
                foreground_past_cov_ts,
                foreground_future_cov_ts,
                train=fallback,
            )

            shap_ = self.explainer.shap_explanations(
                foreground_X, horizons, target_names, **kwargs
            )

            shap_values_dict = {}
            feature_values_dict = {}
            shap_explanation_object_dict = {}
            for h in horizons:
                shap_values_dict_single_h = {}
                feature_values_dict_single_h = {}
                shap_explanation_object_dict_single_h = {}
                for t in target_names:
                    shap_values_dict_single_h[t] = TimeSeries(
                        times=prediction_times,
                        values=shap_[h][t].values,
                        components=shap_[h][t].feature_names,
                        copy=False,
                    )
                    feature_values_dict_single_h[t] = TimeSeries(
                        times=prediction_times,
                        values=shap_[h][t].data,
                        components=shap_[h][t].feature_names,
                        copy=False,
                    )
                    shap_explanation_object_dict_single_h[t] = shap_[h][t]
                shap_values_dict[h] = shap_values_dict_single_h
                feature_values_dict[h] = feature_values_dict_single_h
                shap_explanation_object_dict[h] = shap_explanation_object_dict_single_h

            shap_values_list.append(shap_values_dict)
            feature_values_list.append(feature_values_dict)
            shap_explanation_object_list.append(shap_explanation_object_dict)

        if len(shap_values_list) == 1:
            shap_values_list = shap_values_list[0]
            feature_values_list = feature_values_list[0]
            shap_explanation_object_list = shap_explanation_object_list[0]

        return ShapExplainabilityResult(
            shap_values_list, feature_values_list, shap_explanation_object_list
        )

    def explain_single(
        self,
        foreground_series: TimeSeries | None = None,
        foreground_past_covariates: TimeSeries | None = None,
        foreground_future_covariates: TimeSeries | None = None,
        target_components: Sequence[str] | None = None,
        **kwargs,
    ):
        """
        Explains a foreground time series forecast starting from one last forecastable timestamp and returns a
        :class:`ShapSingleExplainabilityResult
        <darts.explainability.explainability_result.ShapSingleExplainabilityResult>` of SHAP values.

        The results can then be retrieved with method :func:`get_explanation()
        <darts.explainability.explainability_result.ShapSingleExplainabilityResult.get_explanation>`,
        which returns a multivariate ``TimeSeries`` instance containing the SHAP values for ``target_component``
        starting from the last forecastable timestamp.

        The components of the ``TimeSeries`` correspond to the input features used by the model to produce
        the forecast. See above for the naming convention.

        Parameters
        ----------
        foreground_series
            Optionally, one or a sequence of target ``TimeSeries`` to be explained. Can be multivariate.
            If not provided, the background ``TimeSeries`` will be explained instead.
        foreground_past_covariates
            Optionally, one or a sequence of past covariates ``TimeSeries`` if required by the forecasting model.
        foreground_future_covariates
            Optionally, one or a sequence of future covariates ``TimeSeries`` if required by the forecasting model.
        target_components
            Optionally, a string or sequence of strings with the target components to explain.
        **kwargs
            Additional keyword arguments to be passed to the SHAP explainer when calling it for explanations, e.g.,
            `npermutations` for the default permutation explainer.

        Returns
        -------
        ShapSingleExplainabilityResult
            The forecast explanations of the specified target components for the single forecasted timestamp.

        Examples
        --------
        Say we have a ``TorchForecastingModel`` instance with:

          - 2 target components named ``"T_0"`` and ``"T_1"``,
          - 3 past covariates with default component names ``"P_0"``, ``"P_1"``, and ``"P_2"``,
          - 1 future covariate with default component name ``"F_0"``,
          - ``input_chunk_length=3``,
          - ``output_chunk_length=2``.

        We provide ``foreground_series``, ``foreground_past_covariates``, ``foreground_future_covariates`` (extending
        far enough into the future) each of length 5.

        >>> results = explainer.explain_single(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates)

        Calling the method returns a ``ShapSingleExplainabilityResult`` object containing the SHAP values,
        feature values, and raw ``shap.Explanation`` objects for each target component at the single forecasted
        timestamp (timestamp index 6 in our example, as it is the last forecastable).

        >>> # Get SHAP values for forecasting "T_1" as a `TimeSeries`
        >>> output = results.get_explanation(component="T_1")
        >>> # Get feature values used for forecasting as a `TimeSeries`
        >>> feature_values = results.get_feature_values(component="T_1")
        >>> # Get the raw `shap.Explanation` object for further processing
        >>> shap_objects = results.get_shap_explanation_object(component="T_1")

        For SHAP and feature values, the components of the returned ``TimeSeries`` correspond to different lags of the
        target and covariates (see convention above). In our example, the component names would be:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - P_0_pastcov_lag-1
             - P_0_pastcov_lag-2
             - P_0_pastcov_lag-3
             - P_1_pastcov_lag-1
             - P_1_pastcov_lag-2
             - P_1_pastcov_lag-3
             - P_2_pastcov_lag-1
             - P_2_pastcov_lag-2
             - P_2_pastcov_lag-3
             - F_0_futcov_lag0
             - F_0_futcov_lag1

        The SHAP value ``TimeSeries`` has length ``output_chunk_length=2``, as the model predicts that many timestamps
        in the future. The feature value ``TimeSeries`` has length 1, as it corresponds to the single forecasted
        timestamp explained.

        .. note::
            The single forecast explained by this method should be equivalent to the one obtained by calling
            ``model.predict(n=output_chunk_length)`` when foreground data is provided. However, the "equivalent"
            forecast is temporally backshifted by ``output_chunk_length`` when the model uses future covariates
            AND both foreground and background data are not provided. In this case, the explainer would use training
            data as reference, whose future covariates were trimmed to match the target series during training.
            As a result, the forecast explained would be backshifted to the last timestamp when future covariates are
            known.
        """
        fallback = foreground_series is None
        (
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            _,
            _,
            _,
            _,
            _,
        ) = self._process_foreground(
            foreground_series,
            foreground_past_covariates,
            foreground_future_covariates,
        )
        _, target_names = self._process_horizons_and_targets(None, target_components)

        foreground_X, schemas, prediction_times = self.explainer.create_shap_array(
            foreground_series_,
            foreground_past_covariates_,
            foreground_future_covariates_,
            train=fallback,
        )

        # explain only the last forecasted timestamp
        foreground_X = foreground_X[-1:]
        schema = schemas[-1]
        prediction_time = prediction_times[-1]

        shap_ = self.explainer.shap_explanations_single(
            foreground_X, target_names, **kwargs
        )

        shap_values_dict = {}
        feature_values_dict = {}
        shap_explanation_object_dict = {}
        for t in target_names:
            shap_values_dict[t] = TimeSeries(
                times=pd.date_range(
                    start=prediction_time,
                    freq=schema["time_freq"],
                    periods=shap_[t].values.shape[0],
                ),
                values=shap_[t].values,
                components=shap_[t].feature_names,
            )
            feature_values_dict[t] = TimeSeries(
                times=pd.date_range(
                    start=prediction_time,
                    freq=schema["time_freq"],
                    periods=1,
                ),
                values=shap_[t].data[:1],
                components=shap_[t].feature_names,
            )
            shap_explanation_object_dict[t] = shap_[t]

        return ShapSingleExplainabilityResult(
            shap_values_dict,
            feature_values_dict,
            shap_explanation_object_dict,
        )
