"""
StatsForecastAutoCES
-----------
"""

from statsforecast.models import AutoCES as SFAutoCES

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import LocalForecastingModel


class StatsForecastAutoCES(LocalForecastingModel):
    def __init__(self, *autoces_args, **autoces_kwargs):
        """Auto-CES based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best Complex Exponential Smoothing model using an information criterion.
        <https://onlinelibrary.wiley.com/doi/full/10.1002/nav.22074>

        We refer to the `statsforecast AutoCES documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoces>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        autoces_args
            Positional arguments for ``statsforecasts.models.AutoCES``.
        autoces_kwargs
            Keyword arguments for ``statsforecasts.models.AutoCES``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import StatsForecastAutoCES
        >>> series = AirPassengersDataset().load()
        >>> # define StatsForecastAutoCES parameters
        >>> model = StatsForecastAutoCES(season_length=12, model="Z")
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[453.03417969],
               [429.34039307],
               [488.64471436],
               [500.28955078],
               [519.79962158],
               [586.47503662]])
        """
        super().__init__()
        self.model = SFAutoCES(*autoces_args, **autoces_kwargs)

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
        )

        mu = forecast_dict["mean"]

        return self._build_forecast_series(mu)

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def min_train_series_length(self) -> int:
        return 10

    @property
    def _supports_range_index(self) -> bool:
        return True
