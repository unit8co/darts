"""
TBATS
-----
"""

from typing import Optional

from statsforecast.models import TBATS as SF_TBATS

from darts.models.forecasting.sf_model import StatsForecastModel


class TBATS(StatsForecastModel):
    def __init__(
        self,
        *args,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """TBATS based on the `Statsforecasts package <https://github.com/Nixtla/statsforecast>`_.

        Trigonometric Box-Cox transform, ARMA errors, Trend and Seasonal components (TBATS) model. It is an innovations
        state space model framework used for forecasting time series with multiple seasonalities. It uses a Box-Cox
        tranformation, ARMA errors, and a trigonometric representation of the seasonal patterns based on Fourier series.

        We refer to the `StatsForecast documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#tbats>`_ for the exhaustive documentation of
        the arguments.

        In addition to univariate deterministic forecasting, it comes with additional support:

        - **Future covariates:** Use exogenous features to potentially improve predictive accuracy.
          Darts adds support by first regressing the series against the future covariates using a
          :class:`~darts.models.forecasting.linear_regression_model.LinearRegressionModel` model and then running the
          StatsForecast model on the in-sample residuals from this original regression. This approach was inspired by
          `this post of Stephan Kolassa <https://stats.stackexchange.com/q/220885>`_.

        - **Probabilstic forecasting:** To generate probabilistic forecasts, you can set the following
          parameters when calling :meth:`~darts.models.forecasting.sf_model.StatsForecastModel.predict`:

          - Forecast quantile values directly by setting `predict_likelihood_parameters=True`.

          - Generate sampled forecasts from these quantiles by setting `num_samples >> 1`.

        - **Transferable series forecasting:** Apply the fitted model to a new input `series` at prediction time.
          Darts adds support by first fitting a copy of the model on the new series, and then using that model to
          generate the corresponding forecast.

        .. note::
            Future covariates are not supported when the input series contain missing values.

        .. note::
            The first model call might take more time than all subsequent calls as the model relies on Numba and jit
            compilation.

        Parameters
        ----------
        args
            Positional arguments for ``statsforecasts.models.TBATS``.
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
            Keyword arguments for ``statsforecasts.models.TBATS``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import TBATS
        >>> series = AirPassengersDataset().load()
        >>> # define TBATS parameters
        >>> model = TBATS(season_length=12)
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
            model=SF_TBATS(*args, **kwargs),
            quantiles=quantiles,
            add_encoders=add_encoders,
            random_state=random_state,
        )
