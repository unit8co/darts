"""
AutoTBATS
---------
"""

from typing import Optional

from statsforecast.models import AutoTBATS as SFAutoTBATS

from darts.models.forecasting.sf_model import StatsForecastModel


class AutoTBATS(StatsForecastModel):
    def __init__(
        self,
        *args,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        **kwargs,
    ):
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

        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first
        regressing the series against the future covariates using the :class:'LinearRegressionModel' model and then
        running StatsForecast's AutoETS on the in-sample residuals from this original regression. This approach was
        inspired by 'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>'_.

        This model comes with transferrable `series` support (applying the fitted model to a new input `series` at
        prediction time). It adds support by re-fitting a copy of the model on the new series and then generating the
        forecast for it using the StatsForecast model's `forecast()` method.

        Parameters
        ----------
        args
            Positional arguments for ``statsforecasts.models.AutoTBATS``.
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
            Keyword arguments for ``statsforecasts.models.AutoTBATS``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import AutoTBATS
        >>> series = AirPassengersDataset().load()
        >>> # define AutoTBATS parameters
        >>> model = AutoTBATS(season_length=12)
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
        super().__init__(
            model=SFAutoTBATS(*args, **kwargs),
            quantiles=quantiles,
            add_encoders=add_encoders,
        )
