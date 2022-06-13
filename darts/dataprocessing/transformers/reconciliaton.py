"""
Hierarchical Reconciliation
---------------------------
"""

# from abc import ABC, abstractmethod

from typing import Any, Iterator, Sequence, Tuple

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

    # leaves_seq = [c for c in components_seq if c in leaves]
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


class BottomUpReconciliatior(BaseDataTransformer):
    """
    Performs bottom up reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.
    """

    @staticmethod
    def get_projection_matrix(series):
        n, m = series.n_components, len(series.bottom_level_components)
        return np.concatenate([np.zeros((m, n - m)), np.eye(m)], axis=1)

    @staticmethod
    def ts_transform(series: TimeSeries) -> TimeSeries:
        S = _get_summation_matrix(series)
        G = BottomUpReconciliatior.get_projection_matrix(series)
        return _reconcile_from_S_and_G(series, S, G)


class TopDownReconciliatior(FittableDataTransformer):
    """
    Performs top down reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.

    This estimator computes the proportions (of the base components w.r.t. the top component)
    based on the TimeSeries provided to the method :meth:`fit()`. If the historical series
    is provided, then the historical proportions will be used.
    """

    @staticmethod
    def ts_fit(series: TimeSeries) -> np.ndarray:
        G = TopDownReconciliatior.get_projection_matrix(series)
        return G

    @staticmethod
    def ts_transform(series: TimeSeries, G: np.ndarray) -> TimeSeries:
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
