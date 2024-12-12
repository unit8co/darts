"""
StatsForecastAutoTheta
-----------
"""

from statsforecast.models import AutoTheta as SFAutoTheta

from darts import TimeSeries
from darts.models.components.statsforecast_utils import (
    create_normal_samples,
    one_sigma_rule,
    unpack_sf_dict,
)
from darts.models.forecasting.forecasting_model import LocalForecastingModel


class StatsForecastAutoTheta(LocalForecastingModel):
    def __init__(self, *autotheta_args, **autotheta_kwargs):
        """Auto-Theta based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best Theta (Standard Theta Model (‘STM’), Optimized Theta Model (‘OTM’),
        Dynamic Standard Theta Model (‘DSTM’), Dynamic Optimized Theta Model (‘DOTM’)) model using mse.
        <https://www.sciencedirect.com/science/article/pii/S0169207016300243>

        It is probabilistic, whereas :class:`FourTheta` is not.

        We refer to the `statsforecast AutoTheta documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotheta>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        autotheta_args
            Positional arguments for ``statsforecasts.models.AutoTheta``.
        autotheta_kwargs
            Keyword arguments for ``statsforecasts.models.AutoTheta``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastAutoTheta
        >>> series = AirPassengersDataset().load()
        >>> # define StatsForecastAutoTheta parameters
        >>> model = StatsForecastAutoTheta(season_length=12)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[442.94078295],
               [432.22936898],
               [495.30609727],
               [482.30625563],
               [487.49312172],
               [555.57902659]])
        """
        super().__init__()
        self.model = SFAutoTheta(*autotheta_args, **autotheta_kwargs)

    def fit(self, series: TimeSeries):
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series
        self.model.fit(
            series.values(copy=False).flatten(),
        )
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
    ):
        super().predict(n, num_samples)
        forecast_dict = self.model.predict(
            h=n,
            level=(one_sigma_rule,),  # ask one std for the confidence interval.
        )

        mu, std = unpack_sf_dict(forecast_dict)
        if num_samples > 1:
            samples = create_normal_samples(mu, std, num_samples, n)
        else:
            samples = mu

        return self._build_forecast_series(samples)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def min_train_series_length(self) -> int:
        return 10

    @property
    def _supports_range_index(self) -> bool:
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True
