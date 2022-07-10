"""
Static Covariates Transformer
------
"""
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

from .fittable_data_transformer import FittableDataTransformer
from .invertible_data_transformer import InvertibleDataTransformer

logger = get_logger(__name__)


class StaticCovariatesTransformer(InvertibleDataTransformer, FittableDataTransformer):
    def __init__(
        self,
        scaler_numerical=None,
        scaler_categorical=None,
        name="StaticCovariatesTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Generic wrapper class for scalers/encoders/transformers of static covariates.

        The underlying `scaler_numerical` and `scaler_categorical` have to implement the ``fit()``, ``transform()``
        and ``inverse_transform()`` methods (typically from scikit-learn).

        `scaler_numerical` addresses numerical static covariate data of the underlying series.
        `scaler_categorical` addresses categorical static covariate data.

        Parameters
        ----------
        scaler_numerical
            The scaler to transform numeric static covariate data with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`; this will scale all
            the values of a time series between 0 and 1.
        scaler_categorical
            The scaler to transform categorical static covariate data with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.OrdinalEncoder(feature_range=(0, 1))`; this will convert categories
            into integer valued arrays where each integer stands for a specific category.
        name
            A specific name for the scaler
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from sklearn.preprocessing import MinMaxScaler, OrdicalEncoder
        >>> from darts.dataprocessing.transformers import StaticCovariatesTransformer
        >>> series = AirPassengersDataset().load()
        >>> scaler_num = MinMaxScaler(feature_range=(-1, 1))
        >>> scaler_cat = OrdinalEncoder()
        >>> transformer = StaticCovariatesTransformer(scaler_numerical=scaler_num, scaler_categorical=scaler_cat)
        >>> series_transformed = transformer.fit_transform(series)
        >>> print(series.static_covariates_values())
        [-1.]
        >>> print(series_transformed.static_covariates_values())
        [2.]
        """
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        self.scaler_numerical = (
            MinMaxScaler() if scaler_numerical is None else scaler_numerical
        )
        self.scaler_categorical = (
            OrdinalEncoder() if scaler_categorical is None else scaler_categorical
        )

        for scaler, scaler_name in zip(
            [self.scaler_numerical, self.scaler_categorical],
            ["scaler_numerical", "scaler_categorical"],
        ):
            if (
                not callable(getattr(scaler, "fit", None))
                or not callable(getattr(scaler, "transform", None))
                or not callable(getattr(scaler, "inverse_transform", None))
            ):
                raise_log(
                    ValueError(
                        f"The provided `{scaler_name}` object must have fit(), transform() and "
                        f"inverse_transform() methods"
                    ),
                    logger,
                )

        # categoricals might need a mapping from input features to output (i.e. OneHotEncoding)
        self._cat_feature_map = None
        self._numeric_col_mask = None

    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> "FittableDataTransformer":

        self._fit_called = True

        if isinstance(series, TimeSeries):
            data = series.static_covariates
        else:
            data = pd.concat([s.static_covariates for s in series], axis=0)

        self._numeric_col_mask = data.columns.isin(
            data.select_dtypes(include=np.number).columns
        )
        cat_cols = data.columns[~self._numeric_col_mask]

        data = data.to_numpy(copy=False)
        if sum(self._numeric_col_mask):
            self.scaler_numerical.fit(data[:, self._numeric_col_mask])
        if sum(~self._numeric_col_mask):
            self.scaler_categorical.fit(data[:, ~self._numeric_col_mask])
            if isinstance(self.scaler_categorical, OneHotEncoder):
                self._cat_feature_map = OrderedDict(
                    {
                        col: [f"{col}_{cat}" for cat in categories]
                        for col, categories in zip(
                            cat_cols, self.scaler_categorical.categories_
                        )
                    }
                )
            else:
                self._cat_feature_map = OrderedDict({col: [col] for col in cat_cols})
        return self

    def transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        kwargs = {key: val for key, val in kwargs.items()}
        kwargs["component_mask"] = self._numeric_col_mask
        kwargs["cat_feature_map"] = self._cat_feature_map
        return super().transform(series, *args, **kwargs)

    def inverse_transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:

        kwargs = {key: val for key, val in kwargs.items()}

        cat_features = [len(vals) for vals in self._cat_feature_map.values()]
        static_covs = (
            series.static_covariates
            if isinstance(series, TimeSeries)
            else series[0].static_covariates
        )

        component_mask = []
        cat_idx = 0
        for col, is_numeric in zip(static_covs.columns, self._numeric_col_mask):
            if is_numeric:
                component_mask.append(True)
            else:
                component_mask += [False] * cat_features[cat_idx]
                cat_idx += 1

        kwargs["component_mask"] = np.array(component_mask)
        kwargs["cat_feature_map"] = OrderedDict(
            {
                name: [col]
                for col, names in self._cat_feature_map.items()
                for name in names
            }
        )
        return super().inverse_transform(series, *args, **kwargs)

    @staticmethod
    def ts_fit(series: TimeSeries):
        raise NotImplementedError(
            "StaticCovariatesTransformer does not use method `ts_fit()`"
        )

    @staticmethod
    def ts_transform(
        series: TimeSeries, transformer_cont, transformer_cat, **kwargs
    ) -> TimeSeries:
        component_mask = kwargs.get("component_mask")
        cat_feature_map = kwargs.get("cat_feature_map")

        vals_cont, vals_cat = StaticCovariatesTransformer._reshape_in(
            series, component_mask=component_mask
        )

        tr_out_cont, tr_out_cat = None, None
        if sum(component_mask):
            tr_out_cont = transformer_cont.transform(vals_cont)
        if sum(~component_mask):
            tr_out_cat = transformer_cat.transform(vals_cat)

            # sparse one hot encoding to dense array
            if isinstance(tr_out_cat, csr_matrix):
                tr_out_cat = tr_out_cat.toarray()

        transformed_df = StaticCovariatesTransformer._reshape_out(
            series,
            (tr_out_cont, tr_out_cat),
            component_mask=component_mask,
            cat_feature_map=cat_feature_map,
        )

        return series.with_static_covariates(transformed_df)

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, transformer_cont, transformer_cat, **kwargs
    ) -> TimeSeries:
        component_mask = kwargs.get("component_mask")
        cat_feature_map = kwargs.get("cat_feature_map")

        vals_cont, vals_cat = StaticCovariatesTransformer._reshape_in(
            series, component_mask=component_mask
        )
        tr_out_cont, tr_out_cat = None, None
        if sum(component_mask):
            tr_out_cont = transformer_cont.inverse_transform(vals_cont)
        if sum(~component_mask):
            tr_out_cat = transformer_cat.inverse_transform(vals_cat)

        transformed_df = StaticCovariatesTransformer._reshape_out(
            series,
            (tr_out_cont, tr_out_cat),
            component_mask=component_mask,
            cat_feature_map=cat_feature_map,
        )

        return series.with_static_covariates(transformed_df)

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any, Any]]:
        # since '_ts_fit()' returns the scaler objects, the 'fit()' call will save transformers instances into
        # self.scaler_numerical and self.scaler_categorical
        return zip(
            series,
            [self.scaler_numerical] * len(series),
            [self.scaler_categorical] * len(series),
        )

    def _inverse_transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any, Any]]:
        # the same self.scaler_numerical and self.scaler_categorical will be used also for the 'ts_inverse_transform()'
        return zip(
            series,
            [self.scaler_numerical] * len(series),
            [self.scaler_categorical] * len(series),
        )

    @staticmethod
    def _reshape_in(
        series: TimeSeries, component_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.array, np.array]:
        assert component_mask is not None

        # component mask points at continuous variables
        vals = series.static_covariates_values(copy=False)

        # returns tuple of (continuous static covariates, categorical static covariates)
        return vals[:, component_mask], vals[:, ~component_mask]

    @staticmethod
    def _reshape_out(
        series: TimeSeries,
        vals: Tuple[np.ndarray, np.ndarray],
        component_mask: Optional[np.ndarray] = None,
        cat_feature_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        assert component_mask is not None
        assert cat_feature_map is not None

        vals_cont, vals_cat = vals
        assert (
            len(
                np.unique(
                    [name for names in cat_feature_map.values() for name in names]
                )
            )
            == vals_cat.shape[1]
        )

        data = {}
        idx_cont, idx_cat = 0, 0
        static_cov_columns = []
        for col, is_numeric in zip(series.static_covariates.columns, component_mask):
            if is_numeric:
                data[col] = vals_cont[:, idx_cont]
                static_cov_columns.append(col)
                idx_cont += 1
            else:
                # covers one to one feature map (ordinal/label encoding) and one to multi feature (one hot encoding)
                for col_name in cat_feature_map[col]:
                    if col_name not in static_cov_columns:
                        data[col_name] = vals_cat[:, idx_cat]
                        static_cov_columns.append(col_name)
                        idx_cat += 1
                    else:
                        pass
        return pd.DataFrame(
            data,
            columns=static_cov_columns,
            index=series.static_covariates.index,
        )
