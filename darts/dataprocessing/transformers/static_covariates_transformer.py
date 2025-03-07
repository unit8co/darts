"""
Static Covariates Transformer
------
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from darts.dataprocessing.transformers.fittable_data_transformer import (
    FittableDataTransformer,
)
from darts.dataprocessing.transformers.invertible_data_transformer import (
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class StaticCovariatesTransformer(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        transformer_num=None,
        transformer_cat=None,
        cols_num: Optional[list[str]] = None,
        cols_cat: Optional[list[str]] = None,
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
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
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
        self.cols_num, self.cols_cat = cols_num, cols_cat

        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            mask_components=False,
            global_fit=True,
        )

    #
    #   Fitting Methods:
    #

    @staticmethod
    def ts_fit(
        series: Sequence[TimeSeries], params: dict[str, dict[str, Any]], *args, **kwargs
    ):
        """
        Collates static covariates of all provided `TimeSeries` and fits the following parameters:
            1. `transformer_num`, the fitted numerical static covariate transformer.
            2. `transformer_cat`, the fitted categorical static covariate transformer.
            3. `mask_num`, a dictionary containing two boolean arrays: one that indicates which
            components of the *untransformed* static covariates are numerical, and another that
            indicates which components of the *transformed* static covariates are numerical.
            4. `mask_cat`, a dictionary containing two boolean arrays: one that indicates which
            components of the *untransformed* static covariates are categorical, and another that
            indicates which components of the *transformed* static covariates are categorical.
            5. `n_cat_cols`, a dictionary that stores the number of categorical columns
            we should expect in the untransformed and in the transformed static covariates.
        """

        fixed_params = params["fixed"]
        transformer_num = fixed_params["transformer_num"]
        transformer_cat = fixed_params["transformer_cat"]
        cols_num = fixed_params["cols_num"]
        cols_cat = fixed_params["cols_cat"]

        # Collate static covariates of all `series`:
        stat_covs = pd.concat([s.static_covariates for s in series], axis=0)

        cols_num, cols_cat = StaticCovariatesTransformer._infer_static_cov_dtypes(
            stat_covs, cols_num, cols_cat
        )

        mask_num, mask_cat = StaticCovariatesTransformer._create_component_masks(
            stat_covs, cols_num, cols_cat
        )

        # Fit numerical and categorical static covariate transformers:
        stat_covs = stat_covs.to_numpy(copy=False)
        if mask_num.any():
            transformer_num = transformer_num.fit(stat_covs[:, mask_num])
        if mask_cat.any():
            transformer_cat = transformer_cat.fit(stat_covs[:, mask_cat])

        (
            cat_mapping,
            inv_cat_mapping,
        ) = StaticCovariatesTransformer._create_category_mappings(
            stat_covs, transformer_cat, mask_cat, cols_cat
        )

        (
            inv_mask_num,
            inv_mask_cat,
        ) = StaticCovariatesTransformer._create_inv_component_masks(
            mask_num, mask_cat, cat_mapping, cols_cat
        )

        # Store masks and category mappings for untransformed and transformed static covariates:
        mask_num_dict = {"transform": mask_num, "inverse_transform": inv_mask_num}
        mask_cat_dict = {"transform": mask_cat, "inverse_transform": inv_mask_cat}
        col_map_cat_dict = {
            "transform": cat_mapping,
            "inverse_transform": inv_cat_mapping,
        }
        # Count number of categorical features in untransformed and transformed static covariates:
        n_cat_cols = {
            method: len(col_map_cat_dict[method])
            for method in ("transform", "inverse_transform")
        }

        return {
            "transformer_num": transformer_num,
            "transformer_cat": transformer_cat,
            "mask_num": mask_num_dict,
            "mask_cat": mask_cat_dict,
            "col_map_cat": col_map_cat_dict,
            "n_cat_cols": n_cat_cols,
        }

    @staticmethod
    def _infer_static_cov_dtypes(
        stat_covs: pd.DataFrame,
        cols_num: Optional[Sequence[str]],
        cols_cat: Optional[Sequence[str]],
    ):
        """
        Returns a list of names of numerical static covariates and a list
        of names of categorical/ordinal static covariates.
        """
        if cols_num is None:
            mask_num = stat_covs.columns.isin(
                stat_covs.select_dtypes(include=np.number).columns
            )
            cols_num = stat_covs.columns[mask_num]
        if cols_cat is None:
            mask_cat = stat_covs.columns.isin(
                stat_covs.select_dtypes(exclude=np.number).columns
            )
            cols_cat = stat_covs.columns[mask_cat]
        return cols_num, cols_cat

    @staticmethod
    def _create_component_masks(
        untransformed_stat_covs: pd.DataFrame,
        cols_num: Sequence[str],
        cols_cat: Sequence[str],
    ):
        """
        Returns a boolean array indicating which components of the UNTRANSFORMED
        `stat_covs` are numerical and a boolean array indicating which components
        of the UNTRANSFORMED `stat_covs` are categorical.

        It's important to recognize that these masks only apply to the UNTRANSFORMED
        static covariates since some transformations can generate multiple new components
        from a single component (e.g. one-hot encoding).
        """
        mask_num = untransformed_stat_covs.columns.isin(cols_num)
        mask_cat = untransformed_stat_covs.columns.isin(cols_cat)
        return mask_num, mask_cat

    @staticmethod
    def _create_category_mappings(
        untransformed_stat_covs: np.ndarray,
        transformer_cat,
        mask_cat: np.ndarray,
        cols_cat: Sequence[str],
    ):
        """
        Returns mapping from names of untransformed categorical static covariates names
        and names of transformed categorical static covariate names (i.e. `col_map_cat`), as well
        as a mapping from the transformed categorical static covariate names to the untransformed
        ones (i.e. `inv_col_map_cat`).

        These mappings will be many-to-one/one-to-many if a transformation that generates
        multiple components from a single categorical variable is being used (e.g. one-hot
        encoding).
        """
        if mask_cat.any():
            # check how many features the transformer generates
            n_cat_out = transformer_cat.transform(
                np.expand_dims(untransformed_stat_covs[0, mask_cat], 0)
            ).shape[-1]
            # transformer generates same number of features -> make a 1-1 column map
            if n_cat_out == sum(mask_cat):
                col_map_cat = inv_col_map_cat = OrderedDict({
                    col: [col] for col in cols_cat
                })
            # transformer generates more features (i.e. OneHotEncoder) -> create a 1-many column map
            else:
                col_map_cat = OrderedDict()
                inv_col_map_cat = OrderedDict()
                for col, categories in zip(cols_cat, transformer_cat.categories_):
                    col_map_cat_i = []
                    for cat in categories:
                        col_map_cat_i.append(str(col) + "_" + str(cat))
                        inv_col_map_cat[str(col) + "_" + str(cat)] = [col]
                    col_map_cat[col] = col_map_cat_i
        # If we don't have any categorical static covariates, don't need to generate mapping:
        else:
            col_map_cat = {}
            inv_col_map_cat = {}
        return col_map_cat, inv_col_map_cat

    @staticmethod
    def _create_inv_component_masks(
        mask_num: np.ndarray,
        mask_cat: np.ndarray,
        cat_mapping: dict[str, str],
        cols_cat: Sequence[str],
    ):
        """
        Returns a boolean array indicating which components of the TRANSFORMED
        `stat_covs` are numerical and a boolean array indicating which components
        of the TRANSFORMED `stat_covs` are categorical.

        It's important to recognize that these masks only apply to the UNTRANSFORMED
        static covariates since some transformations can generate multiple new components
        from a single component (e.g. one-hot encoding).
        """
        # check how many categorical features were generated per categorical column after transforming the data
        cat_idx = 0
        inv_mask_num, inv_mask_cat = [], []
        for is_num, is_cat in zip(mask_num, mask_cat):
            if is_num:
                inv_mask_num.append(True)
                inv_mask_cat.append(False)
            elif is_cat:
                # some categorical encoders (OneHotEncoder) generate more features and we need to keep track of that
                cat_name = cols_cat[cat_idx]
                num_cat_outputs = len(cat_mapping[cat_name])
                inv_mask_num += num_cat_outputs * [False]
                inv_mask_cat += num_cat_outputs * [True]
                cat_idx += 1
            else:  # don't scale this feature/column
                inv_mask_num.append(False)
                inv_mask_cat.append(False)
        inv_mask_num = np.array(inv_mask_num, dtype=bool)
        inv_mask_cat = np.array(inv_mask_cat, dtype=bool)
        return inv_mask_num, inv_mask_cat

    #
    #   Transform and Inverse Transform Methods:
    #

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: dict[str, Any], *args, **kwargs
    ) -> TimeSeries:
        return StaticCovariatesTransformer._transform_static_covs(
            series, params["fitted"], method="transform"
        )

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, params: dict[str, Any], *args, **kwargs
    ) -> TimeSeries:
        return StaticCovariatesTransformer._transform_static_covs(
            series, params["fitted"], method="inverse_transform"
        )

    @staticmethod
    def _transform_static_covs(
        series: TimeSeries,
        fitted_params: dict[str, Any],
        method: Literal["transform", "inverse_transform"],
    ):
        """
        Transforms the static covariates of a `series` if `method = 'transform'`, and inverse
        transforms the static covariates of a `series` if `method = 'inverse_transform'`.
        """

        # Unpack parameters:
        transformer_num = fitted_params["transformer_num"]
        transformer_cat = fitted_params["transformer_cat"]
        mask_num = fitted_params["mask_num"][method]
        mask_cat = fitted_params["mask_cat"][method]
        col_map_cat = fitted_params["col_map_cat"][method]
        n_cat_cols = fitted_params["n_cat_cols"][method]

        vals_num, vals_cat = StaticCovariatesTransformer._extract_static_covs(
            series, mask_num, mask_cat
        )

        # quick check if everything is in order
        n_vals_cat_cols = 0 if vals_cat is None else vals_cat.shape[1]
        if (method == "inverse_transform") and (n_vals_cat_cols != n_cat_cols):
            raise_log(
                ValueError(
                    f"Expected `{n_cat_cols}` categorical value columns but only encountered `{n_vals_cat_cols}`"
                ),
                logger,
            )

        # Transform static covs:
        tr_out_num, tr_out_cat = None, None
        if mask_num.any():
            tr_out_num = getattr(transformer_num, method)(vals_num)
        if mask_cat.any():
            tr_out_cat = getattr(transformer_cat, method)(vals_cat)
            # sparse one hot encoding to dense array
            if isinstance(tr_out_cat, csr_matrix):
                tr_out_cat = tr_out_cat.toarray()

        series = StaticCovariatesTransformer._add_back_static_covs(
            series, tr_out_num, tr_out_cat, mask_num, mask_cat, col_map_cat
        )

        return series

    @staticmethod
    def _extract_static_covs(
        series: TimeSeries, mask_num: np.ndarray, mask_cat: np.ndarray
    ) -> tuple[np.array, np.array]:
        """
        Extracts all static covariates from a `TimeSeries`, and then extracts the numerical
        and categorical components to transform from these static covariates.
        """
        vals = series.static_covariates_values(copy=False)
        return vals[:, mask_num], vals[:, mask_cat]

    @staticmethod
    def _add_back_static_covs(
        series: TimeSeries,
        vals_num: np.ndarray,
        vals_cat: np.ndarray,
        mask_num: np.ndarray,
        mask_cat: np.ndarray,
        col_map_cat: dict[str, str],
    ) -> pd.DataFrame:
        """
        Adds transformed static covariates back to original `TimeSeries`. The categorical component
        mapping is used to correctly name categorical components with a one-to-many mapping
        between their untransformed and transformed versions (e.g. components generated using
        one-hot encoding).
        """
        data = {}
        idx_num, idx_cat = 0, 0
        static_cov_columns = []
        for col, is_num, is_cat in zip(
            series.static_covariates.columns, mask_num, mask_cat
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

        transformed_static_covs = pd.DataFrame(
            data,
            columns=static_cov_columns,
            index=series.static_covariates.index,
        )

        return series.with_static_covariates(transformed_static_covs)
