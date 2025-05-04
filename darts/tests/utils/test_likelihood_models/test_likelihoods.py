import numpy as np
import pytest

from darts import TimeSeries
from darts.utils.likelihood_models.base import Likelihood, LikelihoodType


class TestLikelihoodModel:
    def test_likelihood_component_names(self):
        lkl = Likelihood(
            likelihood_type=LikelihoodType.Gaussian, parameter_names=["mu", "sigma"]
        )

        components = ["a", "b"]
        series = TimeSeries.from_values(
            values=np.zeros((3, len(components))), columns=components
        )
        # cannot give series and components at the same time
        with pytest.raises(ValueError):
            _ = lkl.component_names(series=series, components=components)

        names_expected = ["a_mu", "a_sigma", "b_mu", "b_sigma"]
        names_1 = lkl.component_names(series=series)
        names_2 = lkl.component_names(components=components)
        assert names_1 == names_2 == names_expected
