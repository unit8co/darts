import numpy as np
import pytest

from darts import TimeSeries
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
from darts.explainability import NBEATSExplainabilityResult, NBEATSExplainer
from darts.models import NBEATSModel, TFTModel


class TestNBEATSExplainer:
    icl = 12
    ocl = 6
    series = tg.sine_timeseries(length=60, freq="D").astype("float32")
    series_mv = series.stack(tg.linear_timeseries(length=60, freq="D")).astype(
        "float32"
    )

    def helper_fit(self, series, **kwargs):
        kwargs = {"generic_architecture": False, "n_epochs": 1, **kwargs}
        model = NBEATSModel(
            input_chunk_length=self.icl,
            output_chunk_length=self.ocl,
            random_state=0,
            **tfm_kwargs,
            **kwargs,
        )
        model.fit(series)
        return model

    def test_explain_univariate(self):
        model = self.helper_fit(self.series)
        result = NBEATSExplainer(model).explain()
        assert isinstance(result, NBEATSExplainabilityResult)
        trend, seasonality = result.get_trend(), result.get_seasonality()
        pred = model.predict(self.ocl)
        for res in (trend, seasonality):
            assert isinstance(res, TimeSeries)
            assert res.time_index.equals(pred.time_index)
        # the stack contributions add up to the forecast
        np.testing.assert_allclose(
            trend.values() + seasonality.values(), pred.values(), rtol=1e-4, atol=1e-5
        )

    def test_explain_multivariate_target_components(self):
        model = self.helper_fit(self.series_mv)
        explainer = NBEATSExplainer(model)
        result = explainer.explain()
        pred = model.predict(self.ocl)
        for comp in self.series_mv.components:
            np.testing.assert_allclose(
                result.get_trend(comp).values() + result.get_seasonality(comp).values(),
                pred[comp].values(),
                rtol=1e-4,
                atol=1e-5,
            )
        with pytest.raises(ValueError, match="component parameter is required"):
            result.get_trend()
        # restrict to a subset of components
        comp = self.series_mv.components[1]
        result = explainer.explain(target_components=[comp])
        assert result.available_components == [comp]

    def test_explain_foreground_multiple_series(self):
        model = self.helper_fit([self.series, self.series * 2])
        explainer = NBEATSExplainer(model, background_series=[self.series] * 2)
        foreground = [self.series, self.series * 2]
        result = explainer.explain(foreground_series=foreground)
        trends = result.get_trend()
        seasonalities = result.get_seasonality()
        assert len(trends) == len(seasonalities) == 2
        preds = model.predict(self.ocl, series=foreground)
        for t, s, p in zip(trends, seasonalities, preds):
            np.testing.assert_allclose(
                t.values() + s.values(), p.values(), rtol=1e-4, atol=1e-5
            )

    def test_invalid_models(self):
        with pytest.raises(ValueError, match="NBEATSModel"):
            NBEATSExplainer(
                TFTModel(
                    self.icl,
                    self.ocl,
                    add_relative_index=True,
                    n_epochs=1,
                    **tfm_kwargs,
                ).fit(self.series)
            )
        # not fitted
        with pytest.raises(ValueError, match="must be fitted"):
            NBEATSExplainer(NBEATSModel(self.icl, self.ocl, generic_architecture=False))
        # generic architecture
        model = self.helper_fit(self.series, generic_architecture=True, num_stacks=2)
        with pytest.raises(ValueError, match="generic_architecture=False"):
            NBEATSExplainer(model)
        # reversible instance norm
        model = self.helper_fit(self.series, use_reversible_instance_norm=True)
        with pytest.raises(ValueError, match="use_reversible_instance_norm"):
            NBEATSExplainer(model)
