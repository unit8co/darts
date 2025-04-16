"""
Croston method
--------------
"""

import copy
from typing import Optional

from statsforecast.models import TSB as CrostonTSB
from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA

from darts.logging import get_logger, raise_log
from darts.models.forecasting.sf_model import StatsForecastModel

logger = get_logger(__name__)


class Croston(StatsForecastModel):
    def __init__(
        self,
        version: str = "classic",
        alpha_d: float = None,
        alpha_p: float = None,
        add_encoders: Optional[dict] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """An implementation of the `Croston method
        <https://otexts.com/fpp3/counts.html>`_ for intermittent
        count series.

        Relying on the implementation of `Statsforecasts package
        <https://github.com/Nixtla/statsforecast>`_.

        In addition to the StatsForecast implementation, this model can handle future covariates. It does so by first
        regressing the series against the future covariates using the :class:'LinearRegressionModel' model and then
        running StatsForecast's AutoETS on the in-sample residuals from this original regression. This approach was
        inspired by 'this post of Stephan Kolassa< https://stats.stackexchange.com/q/220885>'_.

        This model comes with transferrable `series` support (applying the fitted model to a new input `series` at
        prediction time). It adds support by re-fitting a copy of the model on the new series and then generating the
        forecast for it using the StatsForecast model's `forecast()` method.

        .. note::
            Future covariates are not supported when the input series contain missing values.

        Parameters
        ----------
        version
            - "classic" corresponds to classic Croston.
            - "optimized" corresponds to optimized classic Croston, which searches
              for the optimal ``alpha`` smoothing parameter and can take longer
              to run. Otherwise, a fixed value of ``alpha=0.1`` is used.
            - "sba" corresponds to the adjustment of the Croston method known as
              the Syntetos-Boylan Approximation [1]_.
            - "tsb" corresponds to the adjustment of the Croston method proposed by
              Teunter, Syntetos and Babai [2]_. In this case, `alpha_d` and `alpha_p` must
              be set.
        alpha_d
            For the "tsb" version, the alpha smoothing parameter to apply on demand.
        alpha_p
            For the "tsb" version, the alpha smoothing parameter to apply on probability.
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
            Keyword arguments for ``statsforecasts.models.Croston*``.

        References
        ----------
        .. [1] Aris A. Syntetos and John E. Boylan. The accuracy of intermittent demand estimates.
               International Journal of Forecasting, 21(2):303 – 314, 2005.
        .. [2] Ruud H. Teunter, Aris A. Syntetos, and M. Zied Babai.
               Intermittent demand: Linking forecasting to inventory obsolescence.
               European Journal of Operational Research, 214(3):606 – 615, 2011.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import Croston
        >>> series = AirPassengersDataset().load()
        >>> # use the optimized version to automatically select best alpha parameter
        >>> model = Croston(version="optimized")
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[461.7666],
               [461.7666],
               [461.7666],
               [461.7666],
               [461.7666],
               [461.7666]])
        """
        if version.lower() not in ["classic", "optimized", "sba", "tsb"]:
            raise_log(
                ValueError(
                    'The provided "version" parameter must be set to "classic", "optimized", "sba" or "tsb".'
                ),
                logger=logger,
            )

        kwargs = copy.deepcopy(kwargs)
        if version == "classic":
            model_cls = CrostonClassic
        elif version == "optimized":
            model_cls = CrostonOptimized
        elif version == "sba":
            model_cls = CrostonSBA
        else:
            if alpha_d is None or alpha_p is None:
                raise_log(
                    ValueError(
                        'alpha_d and alpha_p must be specified when using "tsb".'
                    ),
                    logger=logger,
                )
            kwargs["alpha_d"] = alpha_d
            kwargs["alpha_p"] = alpha_p
            model_cls = CrostonTSB

        model = model_cls(**kwargs)
        super().__init__(
            model=model,
            quantiles=quantiles,
            add_encoders=add_encoders,
            random_state=random_state,
        )
