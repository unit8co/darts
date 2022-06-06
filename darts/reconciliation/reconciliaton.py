"""
Posthoc
"""

from abc import ABC, abstractmethod

import numpy as np

from darts.timeseries import TimeSeries
from darts.utils.utils import raise_if_not


def get_summation_matrix(series: TimeSeries):
    """
    Returns the matrix S for a series, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.

    The dimension of the matrix is `(n, m)`, where `n` is the number of components and `m` the number
    of base components (components that are not aggregated).
    S[i, j] contains 1 if component i is "made up" of base component j.
    The order of the `n` and `m` components in the matrix match the order of the components in the `series`.

    The matrix is built using the ``groups`` property of the ``series``. ``groups`` must be a
    dictionary mapping each "non maximimally aggregated" component to its parent(s) in the aggregation.
    """

    raise_if_not(
        series.has_grouping,
        "The provided series must have a grouping defined for reconciliation to be performed.",
    )
    groups = series.groups
    components = set(series.components)
    ancestors = set().union(
        *groups.values()
    )  # all components appearing as an ancestor in the tree (i.e., non-leaves)
    leaves = components - ancestors
    m = len(leaves)
    n = len(components)
    S = np.zeros((n, m))

    components_seq = list(series.components)
    leaves_seq = [c for c in components_seq if c in leaves]
    components_indexes = {c: i for i, c in enumerate(components_seq)}
    leaves_indexes = {l: i for i, l in enumerate(leaves_seq)}

    def increment(cur_node, leaf_idx):
        """
        Recursive function filling S for a given base component and all its ancestors
        """
        S[components_indexes[cur_node], leaf_idx] = 1.0
        if cur_node in groups:
            for parent in groups[cur_node]:
                increment(parent, leaf_idx)

    for leaf in leaves:
        leaf_idx = leaves_indexes[leaf]
        increment(leaf, leaf_idx)

    return S.astype(series.dtype)


class Reconciliation(ABC):
    """
    Super class for all forecast reconciliators.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reconcile(
        self,
        series: TimeSeries,
        # forecast_errors: Optional[
        #     np.array
        # ],  # Forecast errors for each of the components
    ) -> TimeSeries:
        pass

    # TODO:
    # def get_forecast_errors(forecasts: Union[TimeSeries, Sequence[TimeSeries]],)


class LinearReconciliaton(Reconciliation, ABC):
    """
    Super class for all linear forecast reconciliators (bottom-up, top-down, GLS, MinT, ...)

    TODO: Actually not a super class. A moedl where users can give a desired P or get_P method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _get_projection_matrix(series: TimeSeries, S: np.ndarray):
        """
        Defines the kind of reconciliation being applied
        """
        pass

    def reconcile(
        self,
        series: TimeSeries,
    ) -> TimeSeries:
        S = get_summation_matrix(series)
        G = self._get_projection_matrix(series, S)
        return self.reconcile_from_S_and_G(series, S, G)

    def reconcile_from_S_and_G(
        self, series: TimeSeries, S: np.ndarray, G: np.ndarray
    ) -> TimeSeries:

        y_hat = series.all_values(copy=False).transpose((1, 0, 2))  # (n, time, samples)
        reconciled_values = S @ G @ y_hat  # (n, m) * (m, n) * (n, time, samples)
        return series.with_values(reconciled_values)


class BottomUpReconciliation(LinearReconciliaton):
    """
    Performs bottom up reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.
    """

    def _get_projection_matrix(series, S):
        n, m = S.shape  # n = total nr of components, m = nr of bottom components
        return np.concatenate([np.zeros((m, n - m)), np.eye(m)], axis=1)


class TopDownForwardReconciliation(LinearReconciliaton):
    """
    Performs top down reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.

    In this estimator the proportions used for the de-aggregations are estimated from the
    forecasts themselves (and not from the historical proportions).
    """

    def _get_projection_matrix(series, S):
        n, m = S.shape
