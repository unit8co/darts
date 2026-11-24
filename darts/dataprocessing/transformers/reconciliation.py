"""
Hierarchical Reconciliation
---------------------------

A set of posthoc hierarchical reconciliation transformers. These transformers
work on any ``TimeSeries`` (e.g., a forecast) that contain a ``hierarchy``.

A ``hierarchy`` is a dict that maps each component to their parent(s) in the hierarchy.
It can be added to a ``TimeSeries`` using e.g., the :meth:`TimeSeries.with_hierarchy` method.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import nnls

from darts import TimeSeries
from darts.dataprocessing.transformers import (
    BaseDataTransformer,
    FittableDataTransformer,
)
from darts.logging import raise_log
from darts.utils.utils import _maybe_cast_array_dtype


def _get_summation_matrix(series: TimeSeries):
    """
    Returns the matrix S for a series, as defined `here <https://otexts.com/fpp3/reconciliation.html>`__.

    The dimension of the matrix is `(n, m)`, where `n` is the number of components and `m` the number
    of base components (components that are not the sum of any other components).
    S[i, j] contains 1 if component i is "made up" of base component j, and 0 otherwise.
    The order of the `n` and `m` components in the matrix match the order of the components in the `series`.

    The matrix is built using the ``hierarchy`` property of the ``series``. ``hierarchy`` must be a
    dictionary mapping each (non top-level) component to its parent(s) in the aggregation.
    """

    if not series.has_hierarchy:
        raise_log(
            ValueError(
                "The provided series must have a hierarchy defined for reconciliation to be performed."
            ),
        )
    hierarchy = series.hierarchy
    components_seq = list(series.components)
    leaves_seq = series.bottom_level_components
    m = len(leaves_seq)
    n = len(components_seq)
    S = np.zeros((n, m), dtype=series.dtype)

    components_indexes = {comp: i for i, comp in enumerate(components_seq)}
    leaves_indexes = {leaf: i for i, leaf in enumerate(leaves_seq)}

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

    return S


def _get_whitening_matrix(W_inv: np.ndarray) -> np.ndarray:
    """
    Returns the matrix `L` with `L.T @ L = W_inv`, so that `r.T @ W_inv @ r = ||L @ r||^2` for any vector `r`.
    """
    W_inv = np.asarray(W_inv, dtype=np.float64)
    try:
        C = np.linalg.cholesky((W_inv + W_inv.T) / 2)
    except np.linalg.LinAlgError:
        raise_log(
            ValueError(
                "Non-negative MinT reconciliation requires the matrix `W` of the chosen method "
                "to be positive definite."
            ),
        )
    return C.T


def _reconcile_from_S_and_G(
    series: TimeSeries,
    S: np.ndarray,
    G: np.ndarray,
    L: np.ndarray | None = None,
) -> TimeSeries:
    """
    Returns the TimeSeries linearly reconciled from the projection matrix G and the summation matrix S.

    If the whitening matrix `L` (with `L.T @ L = inv(W)`) is given, every time step and sample with a negative
    reconciled value is reconciled again by solving `min_b ||L @ (y_hat - S @ b)||^2` subject to `b >= 0` with
    non-negative least squares. All other values are left unchanged.
    """
    y_hat = series.all_values(copy=False)
    reconciled_values = S @ G @ y_hat  # (n, m) * (m, n) * (time, n, samples)
    if L is not None:
        negative = (reconciled_values < 0).any(axis=1)  # (time, samples)
        if negative.any():
            L_S = L @ S
            for t, s in zip(*np.nonzero(negative)):
                b, _ = nnls(L_S, L @ y_hat[t, :, s])
                reconciled_values[t, :, s] = S @ b
    return TimeSeries(
        times=series.time_index,
        values=_maybe_cast_array_dtype(reconciled_values, series.dtype),
        components=series.components,
        copy=False,
        **series._attrs,
    )


class BottomUpReconciliator(BaseDataTransformer):
    """
    Performs bottom up reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`__.
    """

    @staticmethod
    def get_projection_matrix(series):
        leaves_seq = list(series.bottom_level_components)
        n, m = series.n_components, len(leaves_seq)
        leaves_indexes = {leaf: i for i, leaf in enumerate(leaves_seq)}
        G = np.zeros((m, n), dtype=series.dtype)
        for i, c in enumerate(series.components):
            if c in leaves_indexes:
                G[leaves_indexes[c], i] = 1.0
        return G

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ) -> TimeSeries:
        S = _get_summation_matrix(series)
        G = BottomUpReconciliator.get_projection_matrix(series)
        return _reconcile_from_S_and_G(series, S, G)


class TopDownReconciliator(FittableDataTransformer):
    """
    Performs top down reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`__.

    This estimator computes the proportions (of the base components w.r.t. the top component)
    based on the TimeSeries provided to the method :meth:`fit()`. If the historical series
    is provided, then the historical proportions will be used.
    """

    @staticmethod
    def ts_fit(
        series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ) -> np.ndarray:
        G = TopDownReconciliator.get_projection_matrix(series)
        return G

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ) -> TimeSeries:
        G = params["fitted"]
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
        top_level_index = list(series.components).index(series.top_level_component)
        G = np.zeros((m, n), dtype=series.dtype)
        G[:, top_level_index] = proportions

        return G


class MinTReconciliator(FittableDataTransformer):
    def __init__(self, method="ols", nonnegative: bool = False):
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
        nonnegative
            Whether to constrain the reconciled forecasts to be non-negative. MinT reconciliation can produce
            negative values even when all base forecasts are non-negative. If ``True``, at every time step and
            sample where the reconciled forecast has a negative value, the bottom-level forecasts are instead
            obtained by minimizing the same objective as MinT (the squared differences to the base forecasts,
            weighted by the inverse of ``W``) under the constraint that they are non-negative, as in [3]_. All
            other values are the same as with ``nonnegative=False``. Only use this if all components are
            non-negative by nature (e.g., sales). Requires ``W`` to be positive definite.

        References
        ----------
        .. [1] `Optimal forecast reconciliation for hierarchical and grouped time series through
                trace minimization <https://robjhyndman.com/papers/MinT.pdf>`__
        .. [2] https://otexts.com/fpp3/reconciliation.html#the-mint-optimal-reconciliation-approach
        .. [3] `Optimal non-negative forecast reconciliation
                <https://doi.org/10.1007/s11222-020-09930-0>`__, Wickramasuriya, Turlach and Hyndman (2020)
        """
        known_methods = ["ols", "wls", "wls_var", "wls_struct", "wls_val", "mint_cov"]
        if method not in known_methods:
            raise_log(ValueError(f"The method must be one of {known_methods}."))
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self.method = method
        self.nonnegative = nonnegative
        super().__init__()

    @staticmethod
    def ts_fit(
        series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ) -> np.ndarray:
        method = params["fixed"]["method"]
        if not params["fixed"]["nonnegative"]:
            return MinTReconciliator.get_matrices(series, method)
        S, G, W_inv = MinTReconciliator._get_matrices(series, method)
        return S, G, _get_whitening_matrix(W_inv)

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ) -> TimeSeries:
        return _reconcile_from_S_and_G(series, *params["fitted"])

    @staticmethod
    def _assert_deterministic(series: TimeSeries):
        if not series.is_deterministic:
            raise_log(
                ValueError(
                    "When used with method wls_var or mint_cov, the MinT reconciliator "
                    + "has to be fit on a deterministic series "
                    + "containing residuals. This series is stochastic."
                ),
            )

    @staticmethod
    def get_matrices(series: TimeSeries | None, method: str):
        """Returns the G matrix given a specified reconciliation method."""
        S, G, _ = MinTReconciliator._get_matrices(series, method)
        return S, G

    @staticmethod
    def _get_matrices(series: TimeSeries | None, method: str):
        """Returns the S, G and inverse W matrices given a specified reconciliation method."""
        S = _get_summation_matrix(series)
        if method == "ols":
            # G = inv(S'*S)*S'
            G = np.linalg.inv(S.T @ S) @ S.T
            return S, G, np.eye(len(S), dtype=S.dtype)
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
                series.values(copy=False).T,
                dtype=series.dtype,
            )  # + 1e-3 * np.eye(len(series.components))
        else:
            raise_log(ValueError(f"Unknown method: {method}."))

        Wh_inv = np.linalg.inv(Wh)
        G = np.linalg.inv(S.T @ Wh_inv @ S) @ S.T @ Wh_inv
        return S, G, Wh_inv
