"""
Save/Load Mixin for Anomaly Detection
--------------------------------------

Provides shared save() and load() methods for AD base classes to
avoid code duplication across AnomalyScorer, Detector, Aggregator,
and AnomalyModel.
"""

import datetime
import os
import pickle

from darts.logging import get_logger, raise_log

logger = get_logger(__name__)


class SaveableMixin:
    """Mixin that adds pickle-based save() and load() to anomaly detection classes.

    Subclasses inherit these methods and can override the default path pattern
    ``{ClassName}_{YYYY-mm-dd_HH_MM_SS}.pkl``.

    Example
    -------
    >>> from darts.ad.scorers import KMeansScorer
    >>> scorer = KMeansScorer(window=10, k=8)
    >>> scorer.fit(series)
    >>> scorer.save("my_scorer.pkl")
    >>> loaded = KMeansScorer.load("my_scorer.pkl")
    """

    def save(
        self,
        path: str | os.PathLike | None = None,
        **pkl_kwargs,
    ) -> None:
        """Saves the object under a given path or generates a default path.

        Parameters
        ----------
        path
            Path under which to save the object at its current state. If no path
            is specified, a default path ``"{ClassName}_{YYYY-mm-dd_HH_MM_SS}.pkl"``
            is generated automatically.
        pkl_kwargs
            Keyword arguments passed to ``pickle.dump()``.
        """
        if path is None:
            path = (
                f"{type(self).__name__}"
                f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.pkl"
            )
        if isinstance(path, str | os.PathLike):
            with open(path, "wb") as handle:
                pickle.dump(obj=self, file=handle, **pkl_kwargs)
        else:
            raise_log(
                ValueError(
                    "Argument 'path' has to be a filepath (str or PathLike), "
                    f"but was '{path.__class__}'."
                ),
                logger=logger,
            )

    @staticmethod
    def load(path: str | os.PathLike) -> "SaveableMixin":
        """Loads an object from a given path.

        Parameters
        ----------
        path
            Path from which to load the object.
        """
        if isinstance(path, str | os.PathLike):
            if not os.path.exists(path):
                raise_log(
                    FileNotFoundError(f"The file {path} doesn't exist"),
                    logger=logger,
                )
            with open(path, "rb") as handle:
                obj = pickle.load(file=handle)
        else:
            raise_log(
                ValueError(
                    "Argument 'path' has to be a filepath (str or PathLike), "
                    f"but was '{path.__class__}'."
                ),
                logger=logger,
            )
        return obj
