"""
AutoMFLES
-----------
"""

from typing import Optional

from statsforecast.models import AutoMFLES as SFAutoMFLES

from darts.logging import get_logger
from darts.models.forecasting.sf_model import StatsForecastModel

logger = get_logger(__name__)


class AutoMFLES(StatsForecastModel):
    def __init__(
        self,
        *args,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Auto-MFLES based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best MFLES model from all feasible combinations of the parameters
        `seasonality_weights`, `smoother`, `ma`, and `seasonal_period`. Selection is made using the sMAPE by default.

        We refer to the `statsforecast AutoMFLES documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#mfles>`_
        for the exhaustive documentation of the arguments.

        This model comes with transferrable `series` support (applying the fitted model to a new input `series` at
        prediction time). It adds support by re-fitting a copy of the model on the new series and then generating the
        forecast for it using the StatsForecast model's `forecast()` method.

        Parameters
        ----------
        args
            Positional arguments for ``statsforecasts.models.AutoMFLES``.
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
        quantiles
            Optionally, produce quantile predictions at `quantiles` levels when performing probabilistic forecasting
            with `num_samples > 1` or `predict_likelihood_parameters=True`.
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.
        kwargs
            Keyword arguments for ``statsforecasts.models.AutoMFLES``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoMFLES
        >>> from darts.utils.timeseries_generation import datetime_attribute_timeseries
        >>> series = AirPassengersDataset().load()
        >>> # optionally, use some future covariates; e.g. the value of the month encoded as a sine and cosine series
        >>> future_cov = datetime_attribute_timeseries(series, "month", cyclic=True, add_length=6)
        >>> # define AutoMFLES parameters
        >>> model = AutoMFLES(season_length=12, test_size=12)
        >>> model.fit(series, future_covariates=future_cov)
        >>> pred = model.predict(6, future_covariates=future_cov)
        >>> pred.values()
        array([[466.03298745],
               [450.76192105],
               [517.6342497 ],
               [511.62988828],
               [520.15305998],
               [593.38690019]])
        """
        super().__init__(
            model=SFAutoMFLES(*args, **kwargs),
            quantiles=quantiles,
            add_encoders=add_encoders,
            random_state=random_state,
        )

    @property
    def _supports_native_future_covariates(self) -> bool:
        # StatsForecast didn't set the `use_exog=True` flag for AutoMFLES even though it supports it.
        return True
