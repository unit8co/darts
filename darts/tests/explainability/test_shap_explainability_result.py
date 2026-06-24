import pytest

from darts.explainability.explainability_result import HorizonBasedExplainabilityResult
from darts.utils.timeseries_generation import sine_timeseries


class TestExplainabilityResult:
    series = sine_timeseries(length=20, column_name="target")

    def test_invalid_explainability_result_input(self):
        with pytest.raises(
            ValueError,
            match="The `explained_forecasts` list must consist of dictionaries.",
        ):
            HorizonBasedExplainabilityResult([0])

        with pytest.raises(
            ValueError,
            match="The `explained_forecasts` dictionary list must have all integer keys.",
        ):
            HorizonBasedExplainabilityResult([{"invalid": 0}])

        with pytest.raises(
            ValueError,
            match="The `explained_forecasts` dictionary must have all integer keys.",
        ):
            HorizonBasedExplainabilityResult({"invalid": 0})

        with pytest.raises(
            ValueError,
            match="The `explained_forecasts` must be a dictionary or a list of dictionaries.",
        ):
            HorizonBasedExplainabilityResult("invalid")
