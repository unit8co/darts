"""
AutoARIMA
---------
"""

from typing import Optional

from statsforecast.models import AutoARIMA as SFAutoARIMA

from darts.models.forecasting.sf_model import StatsForecastModel
from darts.utils.likelihood_models.statsforecast import QuantileRegression


class AutoARIMA(StatsForecastModel):
    def __init__(
        self, *autoarima_args, add_encoders: Optional[dict] = None, **autoarima_kwargs
    ):
        """Auto-ARIMA based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the :class:`AutoARIMA` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        It is probabilistic, whereas :class:`AutoARIMA` is not.

        We refer to the `statsforecast AutoARIMA documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima>`_
        for the exhaustive documentation of the arguments.

        Parameters
        ----------
        autoarima_args
            Positional arguments for ``statsforecasts.models.AutoARIMA``.
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
        autoarima_kwargs
            Keyword arguments for ``statsforecasts.models.AutoARIMA``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoARIMA
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define AutoARIMA parameters
        >>> model = AutoARIMA(season_length=12)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[450.55179949],
               [415.00597806],
               [454.61353249],
               [486.51218795],
               [504.09229632],
               [555.06463942]])
        """
        super().__init__(
            model=SFAutoARIMA(*autoarima_args, **autoarima_kwargs),
            likelihood=QuantileRegression(
                quantiles=[0.05, 0.15865, 0.5, 0.84135, 0.95]
            ),
            add_encoders=add_encoders,
        )
