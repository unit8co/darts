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
        <https://nixtla.github.io/statsforecast/src/core/models.html#autoces>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        season_length
            Number of observations per cycle. Default: 1.
        model
            Single-character string identifying kind of CES model:

            * "N" for simple CES without seasonality,
            * "S" for simple simple seasonality (lagged CES),
            * "P" for partial seasonality (without complex part),
            * "F" for full seasonality (lagged CES with both real and complex seasonality).

            Furthermore, the character "Z" is a placeholder telling statsforecast
            to search for the best parameter. Default: "Z".

        .. note::
            Positional and keyword arguments can be used to pass additional parameters to statsforecast's `AutoCES`
            constructor, such as ``prediction_intervals`` to compute the conformal prediction intervals.

        Examples
        --------
        >>> from darts.models import StatsForecastAutoCES
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> model = StatsForecastAutoCES(season_length=12)
        >>> model.fit(series[:-36])
        >>> pred = model.predict(36, num_samples=100)
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

    @property
    def _is_probabilistic(self) -> bool:
        return False
