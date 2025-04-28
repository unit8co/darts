"""
AutoETS
-------
"""

from typing import Optional

from statsforecast.models import AutoETS as SFAutoETS

from darts.models.forecasting.sf_model import StatsForecastModel


class AutoETS(StatsForecastModel):
    def __init__(
        self,
        *args,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Auto-ETS based on the `Statsforecasts package <https://github.com/Nixtla/statsforecast>`_.

        Automatically selects the best Exponential Smoothing model (ETS) model using an information criterion.
        We refer to the `StatsForecast documentation
        <https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoets>`_ for the exhaustive documentation
        of the arguments.

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

        - **Conformal prediction:** In addition to the native probabilistic support, you can perform conformal
          prediction / forecasting by setting `prediction_intervals` at model creation. Then predict the in the same
          way as described above.

        - **Transferable series forecasting:** Apply the fitted model to a new input `series` at prediction time.

        .. note::
            Future covariates are not supported when the input series contain missing values.

        .. note::
            The first model call might take more time than all subsequent calls as the model relies on Numba and jit
            compilation.

        Parameters
        ----------
        args
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
        quantiles
            Optionally, produce quantile predictions at `quantiles` levels when performing probabilistic forecasting
            with `num_samples > 1` or `predict_likelihood_parameters=True`.
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.
        kwargs
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
            model=SFAutoETS(*args, **kwargs),
            quantiles=quantiles,
            add_encoders=add_encoders,
            random_state=random_state,
        )
