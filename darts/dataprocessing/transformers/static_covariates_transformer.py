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
        scaler_num=None,
        scaler_cat=None,
        cols_num: Optional[List[str]] = None,
        cols_cat: Optional[List[str]] = None,
        name="StaticCovariatesTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Generic wrapper class for scalers/encoders/transformers of static covariates.

        The underlying ``scaler_num`` and ``scaler_cat`` have to implement the ``fit()``, ``transform()``
        and ``inverse_transform()`` methods (typically from scikit-learn).

        By default, numerical and categorical columns/features are inferred and allocated to ``scaler_num`` and
        ``scaler_cat``, respectively. Alternatively, specify which columns to scale/transform with ``cols_num`` and
        ``cols_cat``.

        Both ``scaler_num`` and ``scaler_cat`` are fit globally on static covariate data from all series passed
        to ``StaticCovariatesTransformer.fit()``

        Parameters
        ----------
        scaler_num
            The scaler to transform numeric static covariate data with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`; this will scale all
            the values of a time series between 0 and 1.
        scaler_cat
            The scaler to transform categorical static covariate data with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.OrdinalEncoder()`; this will convert categories
            into integer valued arrays where each integer stands for a specific category.
        cols_num
            Optionally, a list of column names which for which to apply the numeric transformer `scaler_num`.
            By default, the transformer will infer all numerical features and scale them with `scaler_num`.
            If an empty list, no column will be scaled.
        cols_cat
            Optionally, a list of column names which for which to apply the categorical transformer `scaler_cat`.
            By default, the transformer will infer all categorical features and transform them with `scaler_cat`.
            If an empty list, no column will be transformed.
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> from darts import TimeSeries
        >>> from darts.dataprocessing.transformers import StaticCovariatesTransformer
        >>> static_covs = pd.DataFrame(data={"cont": [0, 2, 1], "cat": ["a", "c", "b"]})
        >>> series = TimeSeries.from_values(
        >>>     values=np.random.random((10, 3)),
        >>>     columns=["comp1", "comp2", "comp3"],
        >>>     static_covariates=static_covs,
        >>> )
        >>> transformer = StaticCovariatesTransformer()
        >>> series_transformed = transformer.fit_transform(series)
        >>> print(series.static_covariates)
        static_covariates  cont cat
        component
        comp1               0.0   a
        comp2               2.0   c
        comp3               1.0   b
        >>> print(series_transformed.static_covariates)
        static_covariates  cont  cat
        component
        comp1               0.0  0.0
        comp2               1.0  2.0
        comp3               0.5  1.0
        """
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        self.scaler_num = MinMaxScaler() if scaler_num is None else scaler_num
        self.scaler_cat = OrdinalEncoder() if scaler_cat is None else scaler_cat

        for scaler, scaler_name in zip(
            [self.scaler_num, self.scaler_cat],
            ["scaler_num", "scaler_cat"],
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

        # numeric/categorical cols will be inferred at fitting time, if user did not set them
        self.cols = None
        self.cols_num = cols_num
        self.cols_cat = cols_cat
        self.mask_num = None
        self.mask_cat = None

        # categoricals might need a mapping from input features to output (i.e. OneHotEncoding)
        self.col_map_cat = None

    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> "FittableDataTransformer":

        self._fit_called = True

        if isinstance(series, TimeSeries):
            data = series.static_covariates
        else:
            data = pd.concat([s.static_covariates for s in series], axis=0)
        self.cols = data.columns

        # get all numeric and categorical columns
        mask_num = data.columns.isin(data.select_dtypes(include=np.number).columns)
        mask_cat = ~mask_num

        # infer numeric and categorical columns if user didn't supply them at transformer construction
        if self.cols_num is None:
            self.cols_num = data.columns[mask_num]
        if self.cols_cat is None:
            self.cols_cat = data.columns[mask_cat]

        self.mask_num = data.columns.isin(self.cols_num)
        self.mask_cat = data.columns.isin(self.cols_cat)

        data = data.to_numpy(copy=False)
        if sum(self.mask_num):
            self.scaler_num.fit(data[:, self.mask_num])
        if sum(self.mask_cat):
            self.scaler_cat.fit(data[:, self.mask_cat])
            if isinstance(self.scaler_cat, OneHotEncoder):
                self.col_map_cat = OrderedDict(
                    {
                        col: [f"{col}_{cat}" for cat in categories]
                        for col, categories in zip(
                            self.cols_cat, self.scaler_cat.categories_
                        )
                    }
                )
            else:
                self.col_map_cat = OrderedDict({col: [col] for col in self.cols_cat})
        else:
            self.col_map_cat = {}
        return self

    def transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        kwargs = {key: val for key, val in kwargs.items()}
        kwargs["component_mask"] = (self.mask_num, self.mask_cat)
        kwargs["col_map_cat"] = self.col_map_cat
        return super().transform(series, *args, **kwargs)

    def inverse_transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:

        kwargs = {key: val for key, val in kwargs.items()}

        cat_features = [len(vals) for vals in self.col_map_cat.values()]
        component_mask_num, component_mask_cat = [], []
        cat_idx = 0
        for col, is_num, is_cat in zip(self.cols, self.mask_num, self.mask_cat):
            if is_num:
                component_mask_num.append(True)
                component_mask_cat.append(False)
            elif is_cat:
                component_mask_num += [False] * cat_features[cat_idx]
                component_mask_cat += [True] * cat_features[cat_idx]
                cat_idx += 1
            else:  # don't scale this feature/column
                component_mask_num.append(False)
                component_mask_cat.append(False)

        kwargs["component_mask"] = (
            np.array(component_mask_num),
            np.array(component_mask_cat),
        )
        kwargs["col_map_cat"] = OrderedDict(
            {name: [col] for col, names in self.col_map_cat.items() for name in names}
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
        component_mask_num, component_mask_cat = kwargs.get("component_mask")
        col_map_cat = kwargs.get("col_map_cat")

        vals_cont, vals_cat = StaticCovariatesTransformer._reshape_in(
            series, component_mask=(component_mask_num, component_mask_cat)
        )

        tr_out_cont, tr_out_cat = None, None
        if sum(component_mask_num):
            tr_out_cont = transformer_cont.transform(vals_cont)
        if sum(component_mask_cat):
            tr_out_cat = transformer_cat.transform(vals_cat)

            # sparse one hot encoding to dense array
            if isinstance(tr_out_cat, csr_matrix):
                tr_out_cat = tr_out_cat.toarray()

        transformed_df = StaticCovariatesTransformer._reshape_out(
            series,
            (tr_out_cont, tr_out_cat),
            component_mask=(component_mask_num, component_mask_cat),
            col_map_cat=col_map_cat,
        )

        return series.with_static_covariates(transformed_df)

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, transformer_cont, transformer_cat, **kwargs
    ) -> TimeSeries:
        component_mask_num, component_mask_cat = kwargs.get("component_mask")
        col_map_cat = kwargs.get("col_map_cat")

        vals_cont, vals_cat = StaticCovariatesTransformer._reshape_in(
            series, component_mask=(component_mask_num, component_mask_cat)
        )
        tr_out_cont, tr_out_cat = None, None
        if sum(component_mask_num):
            tr_out_cont = transformer_cont.inverse_transform(vals_cont)
        if sum(component_mask_cat):
            tr_out_cat = transformer_cat.inverse_transform(vals_cat)

        transformed_df = StaticCovariatesTransformer._reshape_out(
            series,
            (tr_out_cont, tr_out_cat),
            component_mask=(component_mask_num, component_mask_cat),
            col_map_cat=col_map_cat,
        )

        return series.with_static_covariates(transformed_df)

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any, Any]]:
        # since '_ts_fit()' returns the scaler objects, the 'fit()' call will save transformers instances into
        # self.scaler_num and self.scaler_cat
        return zip(
            series,
            [self.scaler_num] * len(series),
            [self.scaler_cat] * len(series),
        )

    def _inverse_transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any, Any]]:
        # the same self.scaler_num and self.scaler_cat will be used also for the 'ts_inverse_transform()'
        return zip(
            series,
            [self.scaler_num] * len(series),
            [self.scaler_cat] * len(series),
        )

    @staticmethod
    def _reshape_in(
        series: TimeSeries,
        component_mask: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.array, np.array]:
        assert component_mask is not None
        component_mask_num, component_mask_cat = component_mask

        # component mask points at continuous variables
        vals = series.static_covariates_values(copy=False)

        # returns tuple of (continuous static covariates, categorical static covariates)
        return vals[:, component_mask_num], vals[:, component_mask_cat]

    @staticmethod
    def _reshape_out(
        series: TimeSeries,
        vals: Tuple[np.ndarray, np.ndarray],
        component_mask: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        col_map_cat: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        assert component_mask is not None
        assert col_map_cat is not None

        component_mask_num, component_mask_cat = component_mask
        vals_cont, vals_cat = vals

        n_cat_cols = len(
            np.unique([name for names in col_map_cat.values() for name in names])
        )
        if vals_cat is None:
            assert n_cat_cols == 0
        else:
            assert n_cat_cols == vals_cat.shape[1]

        data = {}
        idx_cont, idx_cat = 0, 0
        static_cov_columns = []
        for col, is_num, is_cat in zip(
            series.static_covariates.columns, component_mask_num, component_mask_cat
        ):
            if is_num:  # numeric scaled column
                data[col] = vals_cont[:, idx_cont]
                static_cov_columns.append(col)
                idx_cont += 1
            elif is_cat:  # categorical transformed column
                # covers one to one feature map (ordinal/label encoding) and one to multi feature (one hot encoding)
                for col_name in col_map_cat[col]:
                    if col_name not in static_cov_columns:
                        data[col_name] = vals_cat[:, idx_cat]
                        static_cov_columns.append(col_name)
                        idx_cat += 1
                    else:
                        pass
            else:  # is_num and is_cat are False -> feature not part of transformer, use original values
                data[col] = series.static_covariates[col]
                static_cov_columns.append(col)

        return pd.DataFrame(
            data,
            columns=static_cov_columns,
            index=series.static_covariates.index,
        )
