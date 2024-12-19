"""
StatsForecastAutoTBATS
-----------
"""

from statsforecast.models import AutoTBATS as SFAutoTBATS

from darts import TimeSeries
from darts.models.components.statsforecast_utils import (
    create_normal_samples,
    one_sigma_rule,
    unpack_sf_dict,
)
from darts.models.forecasting.forecasting_model import LocalForecastingModel


class StatsForecastAutoTBATS(LocalForecastingModel):
    def __init__(self, *autoTBATS_args, **autoTBATS_kwargs):
        """Auto-TBATS based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best TBATS model from all feasible combinations of the parameters `use_boxcox`,
        `use_trend`, `use_damped_trend`, and `use_arma_errors`. Selection is made using the AIC.
        Default value for `use_arma_errors` is True since this enables the evaluation of models with
        and without ARMA errors.
        <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f3de25596ab60ef0e886366826bf58a02b35a44f>
        <https://doi.org/10.4225/03/589299681de3d>

        We refer to the `statsforecast AutoTBATS documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        autoTBATS_args
            Positional arguments for ``statsforecasts.models.AutoTBATS``.
        autoTBATS_kwargs
            Keyword arguments for ``statsforecasts.models.AutoTBATS``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastAutoTBATS
        >>> series = AirPassengersDataset().load()
        >>> # define StatsForecastAutoTBATS parameters
        >>> model = StatsForecastAutoTBATS(season_length=12)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[450.79653684],
               [472.09265790],
               [497.76948306],
               [510.74927369],
               [520.92224557],
               [570.33881522]])
        """
        super().__init__()
        self.model = SFAutoTBATS(*autoTBATS_args, **autoTBATS_kwargs)

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
