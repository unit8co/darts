"""
AutoCES
-----------
"""

from typing import Optional

from statsforecast.models import AutoCES as SFAutoCES

from darts.models.components.statsforecast_utils import (
    StatsForecastFutureCovariatesLocalModel,
)
from darts.utils.likelihood_models.statsforecast import QuantileRegression


class AutoCES(StatsForecastFutureCovariatesLocalModel):
    def __init__(
        self, *autoces_args, add_encoders: Optional[dict] = None, **autoces_kwargs
    ):
        """Auto-CES based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best Complex Exponential Smoothing model using an information criterion.
        <https://onlinelibrary.wiley.com/doi/full/10.1002/nav.22074>

        We refer to the `statsforecast AutoCES documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoces>`_
        for the exhaustive documentation of the arguments.

        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first
        regressing the series against the future covariates using the :class:'LinearRegressionModel' model and then
        running StatsForecast's AutoETS on the in-sample residuals from this original regression. This approach was
        inspired by 'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>'_.

        Parameters
        ----------
        autoces_args
            Positional arguments for ``statsforecasts.models.AutoCES``.
        add_encoders
            A large number of future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        autoces_kwargs
            Keyword arguments for ``statsforecasts.models.AutoCES``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoCES
        >>> series = AirPassengersDataset().load()
        >>> # define AutoCES parameters
        >>> model = AutoCES(season_length=12, model="Z")
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
        super().__init__(
            model=SFAutoCES(*autoces_args, **autoces_kwargs),
            likelihood=QuantileRegression(
                quantiles=[0.05, 0.15865, 0.5, 0.84135, 0.95]
            ),
            add_encoders=add_encoders,
        )

    @property
    def _supports_native_future_covariates(self) -> bool:
        return False
