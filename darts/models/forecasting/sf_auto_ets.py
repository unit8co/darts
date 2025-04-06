"""
AutoETS
-----------
"""

from typing import Optional

from statsforecast.models import AutoETS as SFAutoETS

from darts.models.components.statsforecast_utils import (
    StatsForecastFutureCovariatesLocalModel,
)
from darts.utils.likelihood_models.statsforecast import QuantileRegression


class AutoETS(StatsForecastFutureCovariatesLocalModel):
    def __init__(
        self, *autoets_args, add_encoders: Optional[dict] = None, **autoets_kwargs
    ):
        """ETS based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        This implementation can perform faster than the :class:`ExponentialSmoothing` model,
        but typically requires more time on the first call, because it relies
        on Numba and jit compilation.

        We refer to the `statsforecast AutoETS documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoets>`_
        for the exhaustive documentation of the arguments.

        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first
        regressing the series against the future covariates using the :class:'LinearRegressionModel' model and then
        running StatsForecast's AutoETS on the in-sample residuals from this original regression. This approach was
        inspired by 'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>'_.

        Parameters
        ----------
        autoets_args
            Positional arguments for ``statsforecasts.models.AutoETS``.
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
        autoets_kwargs
            Keyword arguments for ``statsforecasts.models.AutoETS``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoETS
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define AutoETS parameters
        >>> model = AutoETS(season_length=12, model="AZZ")
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[441.40323676],
               [415.09871431],
               [448.90785391],
               [491.38584654],
               [493.11817462],
               [549.88974472]])
        """
        super().__init__(
            model=SFAutoETS(*autoets_args, **autoets_kwargs),
            likelihood=QuantileRegression(
                quantiles=[0.05, 0.15865, 0.5, 0.84135, 0.95]
            ),
            add_encoders=add_encoders,
        )

    @property
    def _supports_native_future_covariates(self) -> bool:
        return False
