from darts.timeseries import TimeSeries
from darts.logging import (
    get_logger,
    raise_if_not,
    raise_log,
    raise_user_warning,
)
from typing import List, Sequence, Tuple, Union, Iterator
from darts.utils.utils import series2seq
import pandas as pd
import itertools
import copy

from darts.dataprocessing.transformers import BaseDataTransformer

logger = get_logger(__name__)


class ForecastingWindowTransformer(BaseDataTransformer):
    # functions based on pandas window transformations https://pandas.pydata.org/docs/reference/window.html
    # The dictionary can be reads as follows:
    # - key: name of the function to be provided by the user
    # - value: (function_group, function, ([mandatory arguments for the function_group],
    #                                       [mandatory arguments for the function]))
    BUILTIN_TRANSFORMS = {
        "count": (
            pd.DataFrame.rolling,
            "count",
            (["window"], []),
        ),
        "sum": (pd.DataFrame.rolling, "sum", (["window"], [])),
        "mean": (pd.DataFrame.rolling, "mean", (["window"], [])),  # moving average
        "median": (pd.DataFrame.rolling, "median", (["window"], [])),
        "std": (pd.DataFrame.rolling, "std", (["window"], [])),
        "var": (pd.DataFrame.rolling, "var", (["window"], [])),
        "kurt": (pd.DataFrame.rolling, "kurt", (["window"], [])),
        "min": (pd.DataFrame.rolling, "min", (["window"], [])),
        "max": (pd.DataFrame.rolling, "max", (["window"], [])),
        "corr": (pd.DataFrame.rolling, "corr", (["window"], [])),
        "cov": (pd.DataFrame.rolling, "cov", (["window"], [])),
        "skew": (pd.DataFrame.rolling, "skew", (["window"], [])),
        "quantile": (pd.DataFrame.rolling, "quantile", (["window"], ["quantile"])),
        "sem": (
            pd.DataFrame.rolling,
            "sem",
            (["window"], []),
        ),  # standard error of the mean
        "rank": (pd.DataFrame.rolling, "rank", (["window"], [])),  # rolling rank
        "ewmmean": (pd.DataFrame.ewm, "mean", ([], [])),  # exponential weighted mean
        "ewmstd": (
            pd.DataFrame.ewm,
            "std",
            ([], []),
        ),  # exponential weighted standard deviation
        "ewmvar": (pd.DataFrame.ewm, "var", ([], [])),  # exponential weighted variance
        "ewmcorr": (
            pd.DataFrame.ewm,
            "corr",
            ([], []),
        ),  # exponential weighted correlation
        "ewmcov": (
            pd.DataFrame.ewm,
            "cov",
            ([], []),
        ),  # exponential weighted covariance
        "ewmskew": (
            pd.DataFrame.ewm,
            "skew",
            ([], []),
        ),  # exponential weighted skewness
        "ewmsum": (pd.DataFrame.ewm, "sum", ([], [])),  # exponential weighted sum
    }

    # TODO: add atrribute to indicate if transformer is being called from pipeline, standalone or from model (useful to set column names for example)
    def __init__(
        self,
        window_transformations: Union[dict, List[dict]],
        name: str = "ForecastingWindowTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        A transformer that applies window transformation to a TimeSeries or a Sequence of TimeSeries. It expects a
        dictionary or a list of dictionaries.

        Parameters
        ----------
        window_transformations
            A dictionary or a list of dictionaries. Each dictionary should contain at least the 'function' key.
            The 'function' value should be a string with the name of one of the builtin transformations function to be
            applied from {BUILTIN_TRANSFORMS}, or a callable function that can be applied to the input series.

            Two option are available for callable functions:
             1) the function is provided along a 'rolling':True item and a 'window' value: in this case the function is
             applied to a rolling window of the input series.
             2) no 'rolling':True is provided, the function is applied as is to the input TimeSeries and should return a
             TimeSeries.

            When using the builtin functions, the 'window' key should be provided for the
            pandas.DataFrame.rolling functions group.
            The 'window' value should be a positive integer representing the size of the window to be used for the
            transformation. The 'window' value can be a list of integers, in which case the transformation will be
            applied multiple times, once with each window size value in the list.
            Two optional keys can be provided for more flexibility: 'series_id' and 'comp_id'.
            The 'series_id' key specifies the index of the series in the input sequence of series to which the
            transformation should be applied.
            The 'comp_id' key specifies the index of the component of the series to which the transformation
            should be applied.
            When 'series_id' and 'comp_id' are not provided, the transformation is applied to all the series
            and components.
            All other keys provided will be treated as keyword arguments for the function group
            (i.e., pandas.DataFrame.rolling or pandas.DataFrame.ewm) or for the specific function in that group
            (i.e., pandas.DataFrame.rolling.mean/std/max/min... or pandas.DataFrame.ewm.mean/std/sum).
            Example of use:

                    .. highlight:: python
                    .. code-block:: python
                        from darts.dataprocessing.transformers import ForecastingWindowTransformer
                        all_series = [series_1, series_2, series_3] # each series could have multiple components
                        window_transformations_1 = [{'function':'mean','window':[3],'series_id': 0,'comp_id':[0,1,2]},
                                                    {'function':'quantile', 'window':[3, 5, 30], 'quantile':0.5}]
                         window_transformer_1 = ForecastingWindowTransformer(window_transformations_1)
                        transformed_series_1 = window_transformer.transform(all_series)

                        zscore_fn = lambda x: (x[-1] - x.mean()) / x.std()
                        window_transformations_2 = {'function': zscore_fn ,'rolling':True,'window':[3],'series_id': 0,
                                                                                                    'comp_id':[0,1,2]}
                        window_transformer_2 = ForecastingWindowTransformer(window_transformations_2)
                        transformed_series_2 = window_transformer_2.transform(all_series)
                    ..
        name
            A specific name for the transformer.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries`. Defaults to `1`
        verbose
            Whether to print operations progress
        """
        super().__init__(name, n_jobs, verbose)

        builtin_keys_list = list(self.BUILTIN_TRANSFORMS.keys())

        # check transformation_dict format and set it

        if window_transformations is None:
            raise_log(
                ValueError(
                    "window_transformations argument should be provided and "
                    "must be a non-empty dictionary or a non-empty list of dictionaries."
                ),
                logger,
            )

        if window_transformations is not None:

            raise_if_not(
                (
                    isinstance(window_transformations, dict)
                    and len(window_transformations) > 0
                )
                or (
                    isinstance(window_transformations, list)
                    and len(window_transformations) > 0
                ),
                "`window_transformations` must be a non-empty dictionary or a non-empty list of dictionaries. ",
            )

            if isinstance(window_transformations, dict):
                window_transformations = [
                    window_transformations
                ]  # if ony one dictionary, make it a list

            for idx, transformation in enumerate(window_transformations):
                raise_if_not(
                    isinstance(transformation, dict),
                    f"`window_transformation` must contain dictionaries. Element at index {idx} is not a dictionary.",
                )

                raise_if_not(
                    "function" in transformation
                    and transformation["function"] is not None
                    and (
                        callable(transformation["function"])
                        or (transformation["function"] in builtin_keys_list)
                    ),
                    f"`window_transformation` at index {idx} must contain the 'function' key and be callable or one of"
                    f" {builtin_keys_list}. ",
                )

                raise_if_not(
                    # test that mandatory arguments are provided
                    # (tests that intersection between two sets is equal to the set of mandatory arguments)
                    transformation["function"] in builtin_keys_list
                    and set(
                        itertools.chain(
                            *self.BUILTIN_TRANSFORMS[transformation["function"]][2]
                        )
                    ).intersection(set(transformation.keys()))
                    == set(
                        itertools.chain(
                            *self.BUILTIN_TRANSFORMS[transformation["function"]][2]
                        )
                    ),
                    f"`window_transformation` dictionary at index {idx} must contain at least the following keys  "
                    f"{list(itertools.chain(*self.BUILTIN_TRANSFORMS[transformation['function']][2]))} for built-in "
                    f"{self.BUILTIN_TRANSFORMS[transformation['function']][0].__name__}.{transformation['function']}.",
                )

                if "window" in transformation:
                    raise_if_not(
                        transformation["window"] is not None,
                        f"`window_transformation` dictionary at index {idx} must contain a non-empty 'window' key.",
                    )

                    if isinstance(transformation["window"], int):
                        raise_if_not(
                            transformation["window"] > 0,
                            f"`window_transformation` at index {idx} must contain a positive integer for the 'window'.",
                        )
                        window_transformations[idx]["window"] = [
                            transformation["window"]
                        ]
                    elif isinstance(transformation["window"], list):
                        raise_if_not(
                            len(transformation["window"]) > 0,
                            f"`window_transformation` at index {idx} must contain a non-empty list for the 'window'. ",
                        )

                        for idws, window in enumerate(transformation["window"]):
                            raise_if_not(
                                isinstance(window, int) and window > 0,
                                f"`window_transformation` at index {idx} must contain only positive integers for the "
                                f"'window'. Found {window} at index {idws}.",
                            )
                    else:
                        raise_log(
                            ValueError(
                                f"`window_transformation` at index {idx} must contain a positive integer or a list "
                                f"of positive integers for the 'window'. "
                            ),
                            logger,
                        )

                if "step" in transformation:

                    raise_if_not(
                        (
                            isinstance(transformation["step"], int)
                            and transformation["step"] > 0
                        )
                        or transformation["step"] is None,
                        f"`window_transformation` at index {idx} must contain a positive integer for the 'step'. ",
                    )
                    raise_user_warning(
                        transformation["step"] > 1,
                        f"`window_transformation` at index {idx} has a step greater than 1. "
                        f"This may lead to a transformed series with a different "
                        f"frequency than the original input series.",
                        logger,
                    )

                if (
                    "series_id" in transformation
                    and transformation["series_id"] is None
                ):
                    window_transformations[idx].pop("series_id")

                if (
                    "series_id" in transformation
                    and transformation["series_id"] is not None
                ):
                    raise_if_not(
                        (
                            isinstance(transformation["series_id"], int)
                            and transformation["series_id"] >= 0
                        )
                        or (
                            isinstance(transformation["series_id"], list)
                            and all(
                                isinstance(x, int) and x >= 0
                                for x in transformation["series_id"]
                            )
                        ),
                        f"`window_transformation` at index {idx} must contain a positive integer or 0 for the "
                        f"'series_id', or a non-empty list containing positive integers/0. ",
                    )
                    if isinstance(transformation["series_id"], int):
                        window_transformations[idx]["series_id"] = [
                            transformation["series_id"]
                        ]

                if "comp_id" in transformation and transformation["comp_id"] is None:
                    window_transformations[idx].pop("comp_id")

                if (
                    "comp_id" in transformation
                    and transformation["comp_id"] is not None
                ):
                    raise_if_not(
                        (
                            isinstance(transformation["comp_id"], int)
                            and transformation["comp_id"] >= 0
                        )
                        or (
                            isinstance(transformation["comp_id"], list)
                            and (
                                all(
                                    isinstance(x, int) and x >= 0
                                    for x in transformation["comp_id"]
                                )
                                or (
                                    all(
                                        isinstance(x, list)
                                        and (
                                            all(
                                                isinstance(y, int) and y >= 0 for y in x
                                            )
                                        )
                                        for x in transformation["comp_id"]
                                    )
                                )
                            )
                        ),
                        f"`window_transformation` at index {idx} must contain a positive integer or 0 "
                        f"for the 'comp_id', or a non-empty list containing positive integers/0. ",
                    )
                    if isinstance(transformation["comp_id"], int):
                        window_transformations[idx]["comp_id"] = [
                            transformation["comp_id"]
                        ]

        self.window_transformations = window_transformations

    def _transform_iterator(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Iterator[Tuple[TimeSeries, dict]]:

        series = series2seq(series)
        builtins = self.BUILTIN_TRANSFORMS

        # check if series_id and comp_id exist in the series list
        # run through the transformations
        series_subset = []
        for idx, transformation in enumerate(self.window_transformations):
            # check whether different transformations are to be performed for different series and different components
            if "series_id" not in transformation and "comp_id" not in transformation:
                # apply the transformation to all series and all components
                series_subset += [
                    (s.univariate_component(c), transformation, builtins)
                    for s in series
                    for c in range(s.width)
                ]

            elif "series_id" in transformation and "comp_id" not in transformation:
                # apply the transformation to a specific series and all its components
                raise_if_not(
                    len(series) - 1 >= max(transformation["series_id"]),
                    f"`window_transformation` at index {idx} has a 'series_id' that is greater than "
                    f"the number of series. ",
                )
                series_subset += [
                    (series[s_idx].univariate_component(c), transformation, builtins)
                    for s_idx in transformation["series_id"]
                    for c in range(series[s_idx].width)
                ]

            elif "series_id" not in transformation and "comp_id" in transformation:
                # apply the transformation to all series on a specific component only in each series
                # test that component exists in each relevant series
                raise_if_not(
                    all(
                        c_idx <= series[s_idx].width - 1
                        for s_idx in range(len(series))
                        for c_idx in transformation["comp_id"]
                    ),
                    "Some components are not available in the provided series.",
                )

                series_subset += [
                    (s.univariate_component(c), transformation, builtins)
                    for (s, c) in itertools.product(series, transformation["comp_id"])
                ]

            else:
                # apply the transformation to a specific component in a specific series
                # if a different component is provided for each selected series
                if all(isinstance(x, list) for x in transformation["comp_id"]) and len(
                    transformation["comp_id"]
                ) == len(transformation["series_id"]):
                    # test that components exist in the corresponding series
                    raise_if_not(
                        all(
                            max(c_idxvec) <= series[s_idx].width - 1
                            for (s_idx, c_idxvec) in zip(
                                transformation["series_id"], transformation["comp_id"]
                            )
                        ),
                        "Some components are not available in the provided series.",
                    )
                    series_subset += [
                        (
                            series[s_idx].univariate_component(c_idx),
                            transformation,
                            builtins,
                        )
                        for (s_idx, c_idx) in itertools.chain(
                            *[
                                list(itertools.product([s_idvec], c_idvec))
                                for (s_idvec, c_idvec) in zip(
                                    transformation["series_id"],
                                    transformation["comp_id"],
                                )
                            ]
                        )
                    ]
                # if the same components are provided for all the selected series
                else:
                    # test that the components exist in each series
                    raise_if_not(
                        all(
                            c_idx <= series[s_idx].width - 1
                            for s_idx in transformation["series_id"]
                            for c_idx in transformation["comp_id"]
                        ),
                        "Some components are not available in the provided series.",
                    )
                    series_subset += [
                        (
                            series[s_id].univariate_component(c_id),
                            transformation,
                            builtins,
                        )
                        for (s_id, c_id) in itertools.product(
                            transformation["series_id"], transformation["comp_id"]
                        )
                    ]

        return iter(series_subset)  # the iterator object for ts_transform function

    def ts_transform(
        series: TimeSeries, transformation, builtins, **kwargs
    ) -> TimeSeries:
        """
        Applies the transformation to the given TimeSeries.
        This function is called by the `transform` method of the `BaseDataTransformer` class.
        It takes only one TimeSeries as input and returns the transformed TimeSeries.

        Parameters
        ----------
        series
            The TimeSeries to be transformed.
        transformation
            The transformation to be applied.
        builtins
            The built-in transformations read from the ForecastingWindowTransformer class.

        Returns
        -------
        TimeSeries
            A transformed copy of the input TimeSeries.
        """

        def _get_function_kwargs(transformation, builtins):
            """
            Builds the kwargs dictionary for the transformation function.

            Parameters
            ----------
            transformation
                The transformation dictionary.
            builtins
                The built-in transformations read from the ForecastingWindowTransformer class.

            Returns
            -------
            dict
                The kwargs dictionary.
            """
            BUILTIN_TRANSFORMS = builtins
            fn = transformation["function"]
            useless_keys = ["function", "series_id", "comp_id"]
            keys = list(transformation.keys() - useless_keys)

            if fn in BUILTIN_TRANSFORMS:
                # if builtin function, get the kwargs for the function group and the specific function

                function_group_expected_args = set(
                    BUILTIN_TRANSFORMS[fn][0].__code__.co_varnames
                )
                function_group_available_keys = list(
                    function_group_expected_args.intersection(set(keys))
                )
                function_group_available_kwargs = {
                    k: v
                    for k, v in transformation.items()
                    if k in function_group_available_keys
                }

                function_expected_args = set(
                    getattr(
                        getattr(pd.DataFrame(), BUILTIN_TRANSFORMS[fn][0].__name__)(
                            **function_group_available_kwargs
                        ),
                        fn,
                    ).__code__.co_varnames
                )
                function_available_keys = list(
                    function_expected_args.intersection(set(keys))
                )
                function_available_kwargs = {
                    k: v
                    for k, v in transformation.items()
                    if k in function_available_keys
                }

                return function_group_available_kwargs, function_available_kwargs
            elif callable(fn):
                # if user function, get the kwargs for the function
                function_available_kwargs = {
                    k: v for k, v in transformation.items() if k in keys
                }
                return function_available_kwargs
            else:
                raise_log(
                    Exception("The transformation function is not valid."), logger
                )

        fillna = kwargs.get("fillna", "bfill")

        BUILTIN_TRANSFORMS = builtins

        fn = transformation["function"]

        df_ts = series.pd_dataframe(
            copy=True
        )  # get the series values in a dataframe. TODO: check if copy is necessary

        if isinstance(fn, str):
            # verification of the value of the string should have been already validated in the constructor

            function_group = BUILTIN_TRANSFORMS[fn][0].__name__
            function_name = BUILTIN_TRANSFORMS[fn][1]

            # if rolling, we need to run through window list
            if function_group == "rolling":
                transf_ts = []  # list to get all windows resulting transformations
                for window in transformation["window"]:
                    copy_transformation = copy.deepcopy(
                        transformation
                    )  # to avoid writing the original dict
                    copy_transformation["window"] = window
                    # get function_group and function kwargs
                    function_group_kwargs, function_kwargs = _get_function_kwargs(
                        copy_transformation, BUILTIN_TRANSFORMS
                    )

                    if "closed" not in function_group_kwargs:
                        function_group_kwargs[
                            "closed"
                        ] = "left"  # to garantee that the latest value is not included in the window: forecasting safe

                    transf_df_ts = getattr(
                        getattr(df_ts, function_group)(**function_group_kwargs),
                        function_name,
                    )(**function_kwargs)

                    # TODO : set new feature column name by series_id, comp_id and function name, window_size?

                    transf_ts.append(transf_df_ts)
            else:
                function_group_kwargs, function_kwargs = _get_function_kwargs(
                    transformation, BUILTIN_TRANSFORMS
                )

                transf_ts = getattr(
                    getattr(df_ts, function_group)(**function_group_kwargs),
                    function_name,
                )(**function_kwargs)

                # TODO :set new feature column name by series_id, comp_id and function name, window_size?

        else:  # user provided function with "rolling" key
            function_kwargs = _get_function_kwargs(transformation, BUILTIN_TRANSFORMS)
            if "rolling" in function_kwargs:
                transf_ts = []
                window_list_copy = copy.deepcopy(function_kwargs["window"])
                function_kwargs.pop("rolling")
                function_kwargs.pop("window")
                for window in window_list_copy:  # run through window list
                    transf_ts.append(df_ts.rolling(window).apply(fn, **function_kwargs))

            else:  # if no rolling argument, apply function as provided by the user
                transf_ts = df_ts.apply(lambda x: fn(x))
            # TODO : set new column name ?

        # validate output and return pandas.DataFrame
        transf_ts = (
            pd.concat(transf_ts, axis=1) if isinstance(transf_ts, list) else transf_ts
        )

        # fill NAs
        transf_ts.fillna(method=fillna, inplace=True)  # managed by pandas

        return TimeSeries.from_dataframe(transf_ts)

    # TODO: move to forecasting model class
    def construct_transformation_dictionary(cls, series_configurations):
        """
        Function to help the user construct the window transformation dictionary by assigning the transformation
        function to the series and assigning the identification tags, when relevant, to the series and other arguments.
        Useful when many series are to be tranformed with different transformation functions.
        Typical example would be to transform the target series with one function and the past/future covariates with
        another function.
        Example of use :
        .. highlight:: python
        .. code-block:: python
            series_window_transf_config = [
                              [all_targets, [('mean', 10), ('std', 5)], 'target', [(0, [0,1]), (1, [0,1,2])]],
                              [all_past_covariates, [('mean', 10)], 'past'],
                              [all_future_covariates,[('std', 10), ('ewma', 3)], 'future', [(1, None), (0, None)]]
                              ]

            window_transformer = WindowTransformer.construct_transformation_dictionary(series_window_transf_config) ?
        ..

        :return:
        """
        pass
