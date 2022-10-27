"""
Hierarchical Reconciliation
---------------------------

A set of posthoc hierarchical reconciliation transformers. These transformers
work on any ``TimeSeries`` (e.g., a forecast) that contain a ``hierarchy``.

A ``hierarchy`` is a dict that maps each component to their parent(s) in the hierarchy.
It can be added to a ``TimeSeries`` using e.g., the :meth:`TimeSeries.with_hierarchy` method.
"""


from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np

from darts.dataprocessing.transformers import (
    BaseDataTransformer,
    FittableDataTransformer,
)
from darts.timeseries import TimeSeries
from darts.utils.utils import raise_if_not


def _get_summation_matrix(series: TimeSeries):
    """
    Returns the matrix S for a series, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.

    The dimension of the matrix is `(n, m)`, where `n` is the number of components and `m` the number
    of base components (components that are not the sum of any other components).
    S[i, j] contains 1 if component i is "made up" of base component j, and 0 otherwise.
    The order of the `n` and `m` components in the matrix match the order of the components in the `series`.

    The matrix is built using the ``hierarchy`` property of the ``series``. ``hierarchy`` must be a
    dictionary mapping each (non top-level) component to its parent(s) in the aggregation.
    """

    raise_if_not(
        series.has_hierarchy,
        "The provided series must have a hierarchy defined for reconciliation to be performed.",
    )
    hierarchy = series.hierarchy
    components_seq = list(series.components)
    leaves_seq = series.bottom_level_components
    m = len(leaves_seq)
    n = len(components_seq)
    S = np.zeros((n, m))

    components_indexes = {c: i for i, c in enumerate(components_seq)}
    leaves_indexes = {l: i for i, l in enumerate(leaves_seq)}

    def increment(cur_node, leaf_idx):
        """
        Recursive function filling S for a given base component and all its ancestors
        """
        S[components_indexes[cur_node], leaf_idx] = 1.0
        if cur_node in hierarchy:
            for parent in hierarchy[cur_node]:
                increment(parent, leaf_idx)

    for leaf in leaves_seq:
        leaf_idx = leaves_indexes[leaf]
        increment(leaf, leaf_idx)

    return S.astype(series.dtype)


def _reconcile_from_S_and_G(
    series: TimeSeries, S: np.ndarray, G: np.ndarray
) -> TimeSeries:
    """
    Returns the TimeSeries linearly reconciled from the projection matrix G and the summation matrix S.
    """
    y_hat = series.all_values(copy=False)
    reconciled_values = S @ G @ y_hat  # (n, m) * (m, n) * (time, n, samples)
    return series.with_values(reconciled_values)


class BottomUpReconciliator(BaseDataTransformer):
    """
    Performs bottom up reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.
    """

    @staticmethod
    def get_projection_matrix(series):
        n, m = series.n_components, len(series.bottom_level_components)
        return np.concatenate([np.zeros((m, n - m)), np.eye(m)], axis=1)

    @staticmethod
    def ts_transform(series: TimeSeries, *args, **kwargs) -> TimeSeries:
        S = _get_summation_matrix(series)
        G = BottomUpReconciliator.get_projection_matrix(series)
        return _reconcile_from_S_and_G(series, S, G)


class TopDownReconciliator(FittableDataTransformer):
    """
    Performs top down reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.

    This estimator computes the proportions (of the base components w.r.t. the top component)
    based on the TimeSeries provided to the method :meth:`fit()`. If the historical series
    is provided, then the historical proportions will be used.
    """

    @staticmethod
    def ts_fit(series: TimeSeries, *args, **kwargs) -> np.ndarray:
        G = TopDownReconciliator.get_projection_matrix(series)
        return G

    @staticmethod
    def ts_transform(series: TimeSeries, G: np.ndarray, *args, **kwargs) -> TimeSeries:
        S = _get_summation_matrix(series)
        return _reconcile_from_S_and_G(series, S, G)

    @staticmethod
    def get_projection_matrix(series):
        n, m = series.n_components, len(series.bottom_level_components)

        # compute sum of total component
        sum_total = (
            series[series.top_level_component].all_values(copy=False).flatten().sum()
        )

        base_forecasts = series.bottom_level_series

        # compute sum of base components
        sum_base = base_forecasts.all_values(copy=False).sum(axis=2).sum(axis=0)

        # compute proportions for each base component
        proportions = sum_base / sum_total

        G = np.zeros((m, n))
        G[:, 0] = proportions

        return G

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any]]:
        # since '_ts_fit()' returns the G matrices, the 'fit()' call will save matrix instance into
        # self._fitted_params
        return zip(series, self._fitted_params)


class MinTReconciliator(FittableDataTransformer):
    def __init__(self, method="ols"):
        """
        MinT Reconcilator.

        This implements the MinT reconciliation approach presented in [1]_ and
        summarised in [2]_.

        Parameters
        ----------
        method
            This parameter can take four different values, determining how the covariance
            matrix ``W`` of the forecast errors is estimated (corresponding to ``Wh`` in [2]_):

            * ``ols`` uses ``W = I``. This option looks only at the hierarchy but ignores the
              values of the series provided to ``fit()``.
            * ``wls_struct`` uses ``W = diag(S1)``, where ``S1`` is a vector of size `n` with values
              between 0 and `m`, representing the number of base components composing each
              of the `n` components. This options looks only at the hierarchy but ignores
              the values of the series provided to ``fit()``.
            * ``wls_var`` uses ``W = diag(W1)``, where ``W1`` is the temporal average of the
              variance of the forecasting residuals. This method assumes that the series
              provided to ``fit()`` contain the forecast residuals (deterministic series).
            * ``mint_cov`` computes ``W`` as the empirical covariance matrix of the residuals
              for each component, with residuals samples taken over time. This method assumes
              that the series provided to ``fit()`` contain the forecast residuals
              (deterministic series), and it requires the residuals to be linearly independent.
            * ``wls_val`` uses ``W = diag(V1)``, where ``V1`` is the temporal average of the
              component values. This method assumes that the series provided to ``fit()`` contains
              an example of the actual values (e.g., either the training series or the forecasts).
              This method is not presented in [2]_.

        References
        ----------
        .. [1] `Optimal forecast reconciliation for hierarchical and grouped time series through
                trace minimization <https://robjhyndman.com/papers/MinT.pdf>`_
        .. [2] https://otexts.com/fpp3/reconciliation.html#the-mint-optimal-reconciliation-approach
        """
        super().__init__()
        known_methods = ["ols", "wls", "wls_var", "wls_struct", "wls_val", "mint_cov"]
        raise_if_not(
            method in known_methods,
            f"The method must be one of {known_methods}",
        )
        self.method = method

    @staticmethod
    def ts_fit(series: TimeSeries, method: str, *args, **kwargs) -> np.ndarray:
        S, G = MinTReconciliator.get_matrices(series, method)
        return S, G

    @staticmethod
    def ts_transform(series: TimeSeries, S_and_G, *args, **kwargs) -> TimeSeries:
        S, G = S_and_G
        return _reconcile_from_S_and_G(series, S, G)

    @staticmethod
    def _assert_deterministic(series: TimeSeries):
        raise_if_not(
            series.is_deterministic,
            "When used with method wls_var or mint_cov, the MinT reconciliator "
            + "has to be fit on a deterministic series "
            + "containing residuals. This series is stochastic.",
        )

    @staticmethod
    def get_matrices(series: Optional[TimeSeries], method: str):
        """Returns the G matrix given a specified reconciliation method."""
        S = _get_summation_matrix(series)
        if method == "ols":
            # G = inv(S'*S)*S'
            G = np.linalg.inv(S.T @ S) @ S.T
            return S, G
        elif method == "wls_struct":
            # Wh is a diagonal matrix with entry i,i being the sum of row i of S_mat
            Wh = np.diag(np.sum(S, axis=1))
        elif method == "wls_var":
            # In this case we assume that series contains the residuals of some forecasts
            MinTReconciliator._assert_deterministic(series)
            et2 = series.values(copy=False) ** 2  # squared residuals
            # Wh diagonal is mean squared residual over time:
            Wh = np.diag(et2.mean(axis=0))
        elif method == "wls_val":
            # Wh is a diagonal matrix with entry i,i being the average value of the corresponding time series
            quantities = series.all_values(copy=False).mean(axis=2).mean(axis=0)
            Wh = np.diag(np.array(quantities))
        elif method == "mint_cov":
            MinTReconciliator._assert_deterministic(series)
            Wh = np.cov(
                series.values(copy=False).T
            )  # + 1e-3 * np.eye(len(series.components))
        else:
            raise_if_not(False, f"Unknown method: {method}")

        Wh_inv = np.linalg.inv(Wh)
        G = np.linalg.inv(S.T @ Wh_inv @ S) @ S.T @ Wh_inv
        return S, G

    def _fit_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any]]:
        # generator which also contains the method to use
        return zip(series, (self.method for _ in range(len(series))))

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any]]:
        # since '_ts_fit()' returns the G matrices, the 'fit()' call will save matrix instance into
        # self._fitted_params
        return zip(series, self._fitted_params)
