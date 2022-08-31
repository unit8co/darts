"""
Static Covariates Transformer
------
"""
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

from .fittable_data_transformer import FittableDataTransformer
from .invertible_data_transformer import InvertibleDataTransformer

logger = get_logger(__name__)


class StaticCovariatesTransformer(InvertibleDataTransformer, FittableDataTransformer):
    def __init__(
        self,
        transformer_num=None,
        transformer_cat=None,
        cols_num: Optional[List[str]] = None,
        cols_cat: Optional[List[str]] = None,
        name="StaticCovariatesTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Generic wrapper class for scalers/encoders/transformers of static covariates. This transformer acts
        only on static covariates of the series passed to ``fit()``, ``transform()``, ``fit_transform()``, and
        ``inverse_transform()``. It can both scale numerical features, as well as encode categorical features.

        The underlying ``transformer_num`` and ``transformer_cat`` have to implement the ``fit()``, ``transform()``,
        and ``inverse_transform()`` methods (typically from scikit-learn).

        By default, numerical and categorical columns/features are inferred and allocated to ``transformer_num`` and
        ``transformer_cat``, respectively. Alternatively, specify which columns to scale/transform with ``cols_num``
        and ``cols_cat``.

        Both ``transformer_num`` and ``transformer_cat`` are fit globally on static covariate data from all series
        passed to :class:`StaticCovariatesTransformer.fit()`

        Parameters
        ----------
        transformer_num
            The transformer to transform numeric static covariate columns with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`; this will scale all
            values between 0 and 1.
        transformer_cat
            The encoder to transform categorical static covariate columns with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.OrdinalEncoder()`; this will convert categories
            into integer valued arrays where each integer stands for a specific category.
        cols_num
            Optionally, a list of column names for which to apply the numeric transformer ``transformer_num``.
            By default, the transformer will infer all numerical features based on types, and scale them with
            `transformer_num`. If an empty list, no column will be scaled.
        cols_cat
            Optionally, a list of column names for which to apply the categorical transformer `transformer_cat`.
            By default, the transformer will infer all categorical features based on types, and transform them with
            `transformer_cat`. If an empty list, no column will be transformed.
        name
            A specific name for the :class:`StaticCovariatesTransformer`.
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
        >>> static_covs = pd.DataFrame(data={"num": [0, 2, 1], "cat": ["a", "c", "b"]})
        >>> series = TimeSeries.from_values(
        >>>     values=np.random.random((10, 3)),
        >>>     columns=["comp1", "comp2", "comp3"],
        >>>     static_covariates=static_covs,
        >>> )
        >>> transformer = StaticCovariatesTransformer()
        >>> series_transformed = transformer.fit_transform(series)
        >>> print(series.static_covariates)
        static_covariates  num cat
        component
        comp1               0.0   a
        comp2               2.0   c
        comp3               1.0   b
        >>> print(series_transformed.static_covariates)
        static_covariates  num  cat
        component
        comp1               0.0  0.0
        comp2               1.0  2.0
        comp3               0.5  1.0
        """
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        self.transformer_num = (
            MinMaxScaler() if transformer_num is None else transformer_num
        )
        self.transformer_cat = (
            OrdinalEncoder() if transformer_cat is None else transformer_cat
        )

        for transformer, transformer_name in zip(
            [self.transformer_num, self.transformer_cat],
            ["transformer_num", "transformer_cat"],
        ):
            if (
                not callable(getattr(transformer, "fit", None))
                or not callable(getattr(transformer, "transform", None))
                or not callable(getattr(transformer, "inverse_transform", None))
            ):
                raise_log(
                    ValueError(
                        f"The provided `{transformer_name}` object must have fit(), transform() and "
                        f"inverse_transform() methods"
                    ),
                    logger,
                )

        # numeric/categorical cols will be inferred at fitting time, if user did not set them
        self.cols = None
        self.cols_num, self.cols_cat = cols_num, cols_cat
        self.mask_num, self.mask_cat = None, None

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

        # infer numeric and categorical columns if user didn't supply them at transformer construction
        if self.cols_num is None:
            mask_num = data.columns.isin(data.select_dtypes(include=np.number).columns)
            self.cols_num = data.columns[mask_num]
        if self.cols_cat is None:
            mask_cat = data.columns.isin(data.select_dtypes(exclude=np.number).columns)
            self.cols_cat = data.columns[mask_cat]

        self.mask_num = data.columns.isin(self.cols_num)
        self.mask_cat = data.columns.isin(self.cols_cat)

        data = data.to_numpy(copy=False)
        if sum(self.mask_num):
            self.transformer_num.fit(data[:, self.mask_num])
        if sum(self.mask_cat):
            self.transformer_cat.fit(data[:, self.mask_cat])
            # check how many features the transformer generates
            n_cat_out = self.transformer_cat.transform(
                np.expand_dims(data[0, self.mask_cat], 0)
            ).shape[-1]
            if n_cat_out == sum(self.mask_cat):
                # transformer generates same number of features -> make a 1-1 column map
                self.col_map_cat = OrderedDict({col: [col] for col in self.cols_cat})
            else:
                # transformer generates more features (i.e. OneHotEncoder) -> create a 1-many column map
                self.col_map_cat = OrderedDict(
                    {
                        col: [f"{col}_{cat}" for cat in categories]
                        for col, categories in zip(
                            self.cols_cat, self.transformer_cat.categories_
                        )
                    }
                )
        else:
            self.col_map_cat = {}
        return self

    def transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        kwargs = {key: val for key, val in kwargs.items()}
        kwargs["component_mask"] = (self.mask_num, self.mask_cat)
        kwargs["col_map_cat"] = self.col_map_cat
        kwargs["method"] = "transform"
        return super().transform(series, *args, **kwargs)

    def inverse_transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:

        kwargs = {key: val for key, val in kwargs.items()}

        # check how many categorical features were generated per categorical column after transforming the data
        cat_features = [len(vals) for vals in self.col_map_cat.values()]
        component_mask_num, component_mask_cat = [], []
        cat_idx = 0
        for col, is_num, is_cat in zip(self.cols, self.mask_num, self.mask_cat):
            if is_num:
                component_mask_num.append(True)
                component_mask_cat.append(False)
            elif is_cat:
                # some categorical encoders (OneHotEncoder) generate more features and we need to keep track of that
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
        kwargs["method"] = "inverse_transform"
        return super().inverse_transform(series, *args, **kwargs)

    @staticmethod
    def ts_fit(series: TimeSeries):
        raise NotImplementedError(
            "StaticCovariatesTransformer does not use method `ts_fit()`"
        )

    @staticmethod
    def ts_transform(series: TimeSeries, *args, **kwargs) -> TimeSeries:
        transformer_num, transformer_cat = args
        component_mask_num, component_mask_cat = kwargs.get("component_mask")
        col_map_cat = kwargs.get("col_map_cat")
        method = kwargs.get("method")  # "transform" or "inverse_transform"

        vals_num, vals_cat = StaticCovariatesTransformer._reshape_in(
            series, component_mask=(component_mask_num, component_mask_cat)
        )
        tr_out_num, tr_out_cat = None, None
        if sum(component_mask_num):
            tr_out_num = getattr(transformer_num, method)(vals_num)
        if sum(component_mask_cat):
            tr_out_cat = getattr(transformer_cat, method)(vals_cat)

            # sparse one hot encoding to dense array
            if isinstance(tr_out_cat, csr_matrix):
                tr_out_cat = tr_out_cat.toarray()

        transformed_df = StaticCovariatesTransformer._reshape_out(
            series,
            (tr_out_num, tr_out_cat),
            component_mask=(component_mask_num, component_mask_cat),
            col_map_cat=col_map_cat,
        )

        return series.with_static_covariates(transformed_df)

    @staticmethod
    def ts_inverse_transform(series: TimeSeries, *args, **kwargs) -> TimeSeries:
        # inverse transform will be called with kwarg method="inverse_transform"
        return StaticCovariatesTransformer.ts_transform(series, *args, **kwargs)

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any, Any]]:
        # use numerical and categorical transformers for 'ts_transform()'
        return zip(
            series,
            [self.transformer_num] * len(series),
            [self.transformer_cat] * len(series),
        )

    def _inverse_transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Any, Any]]:
        # use numerical and categorical transformers for 'ts_inverse_transform()'
        return zip(
            series,
            [self.transformer_num] * len(series),
            [self.transformer_cat] * len(series),
        )

    @staticmethod
    def _reshape_in(
        series: TimeSeries,
        component_mask: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.array, np.array]:
        # we expect component mask to be (numeric component mask, categorical component mask)
        component_mask_num, component_mask_cat = component_mask

        # returns tuple of (numeric static covariates, categorical static covariates)
        vals = series.static_covariates_values(copy=False)
        return vals[:, component_mask_num], vals[:, component_mask_cat]

    @staticmethod
    def _reshape_out(
        series: TimeSeries,
        vals: Tuple[np.ndarray, np.ndarray],
        component_mask: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        col_map_cat: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        # we expect component mask to be (numeric component mask, categorical component mask)
        component_mask_num, component_mask_cat = component_mask
        vals_num, vals_cat = vals

        # infer the number of categorical output features
        # `col_map_cat` contains information to which features the original categorical feature were mapped
        # (i.e. 1-1 mapping for OrdinalEncoder, or 1-many mapping for OneHotEncoder).
        n_cat_cols = len({name for names in col_map_cat.values() for name in names})
        # quick check if everything is in order
        n_vals_cat_cols = 0 if vals_cat is None else vals_cat.shape[1]
        if n_vals_cat_cols != n_cat_cols:
            raise_log(
                ValueError(
                    f"Expected `{n_cat_cols}` categorical value columns but only encountered `{n_vals_cat_cols}`"
                ),
                logger,
            )

        data = {}
        idx_num, idx_cat = 0, 0
        static_cov_columns = []
        for col, is_num, is_cat in zip(
            series.static_covariates.columns, component_mask_num, component_mask_cat
        ):
            if is_num:  # numeric scaled column
                data[col] = vals_num[:, idx_num]
                static_cov_columns.append(col)
                idx_num += 1
            elif is_cat:  # categorical transformed column
                # covers one to one feature map (ordinal/label encoding) and one to multi feature (one hot encoding)
                for col_name in col_map_cat[col]:
                    if col_name not in static_cov_columns:
                        data[col_name] = vals_cat[:, idx_cat]
                        static_cov_columns.append(col_name)
                        idx_cat += 1
            else:  # is_num and is_cat are False -> feature is not part of transformer, use original values
                data[col] = series.static_covariates[col]
                static_cov_columns.append(col)

        # returns a pandas DataFrame of static covariates to be added to the series
        return pd.DataFrame(
            data,
            columns=static_cov_columns,
            index=series.static_covariates.index,
        )
