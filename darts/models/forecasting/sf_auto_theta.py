"""
AutoTheta
-----------
"""

from typing import Optional

from statsforecast.models import AutoTheta as SFAutoTheta

from darts.models.forecasting.sf_model import StatsForecastModel


class AutoTheta(StatsForecastModel):
    def __init__(
        self,
        *args,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        **kwargs,
    ):
        """Auto-Theta based on `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best Theta (Standard Theta Model (‘STM’), Optimized Theta Model (‘OTM’),
        Dynamic Standard Theta Model (‘DSTM’), Dynamic Optimized Theta Model (‘DOTM’)) model using mse.
        <https://www.sciencedirect.com/science/article/pii/S0169207016300243>

        It is probabilistic, whereas :class:`FourTheta` is not.

        We refer to the `statsforecast AutoTheta documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotheta>`_
        for the exhaustive documentation of the arguments.

        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first
        regressing the series against the future covariates using the :class:'LinearRegressionModel' model and then
        running StatsForecast's AutoETS on the in-sample residuals from this original regression. This approach was
        inspired by 'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>'_.

        Parameters
        ----------
        args
            Positional arguments for ``statsforecasts.models.AutoTheta``.
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
        kwargs
            Keyword arguments for ``statsforecasts.models.AutoTheta``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoTheta
        >>> series = AirPassengersDataset().load()
        >>> # define AutoTheta parameters
        >>> model = AutoTheta(season_length=12)
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
        super().__init__(
            model=SFAutoTheta(*args, **kwargs),
            quantiles=quantiles,
            add_encoders=add_encoders,
        )
