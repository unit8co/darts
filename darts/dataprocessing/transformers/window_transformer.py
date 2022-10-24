from typing import Iterator, List, Sequence, Tuple, Union

from darts.dataprocessing.transformers import BaseDataTransformer
from darts.logging import get_logger, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.utils import series2seq

logger = get_logger(__name__)


class ForecastingWindowTransformer(BaseDataTransformer):
    def __init__(
        self,
        window_transformations: Union[dict, List[dict]],
        name: str = "ForecastingWindowTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        A transformer that applies window transformation to a TimeSeries or a Sequence of TimeSeries. It expects a
        dictionary or a list of dictionaries specifying the window transformation(s) to be applied.

        Parameters
        ----------
        window_transformations
            A dictionary or a list of dictionaries. Each dictionary should contain at least the 'function' key.

            The 'function' value should be a string with the name of one of the builtin transformation functions,
            or a callable function provided by the user that can be applied to the input series
            by using pandas.DatFrame.rolling object.

            The two following options are available for built-in transformation functions:
            1) Based on pandas.DataFrame.Rolling windows, the 'function' key should have one of
                {BUILTIN_TRANSFORMS_WINDOW}.
            2) Based on pandas.DataFrame.ewm (Exponentially-weighted window), the 'function' key should have one of
                {BUILTIN_TRANSFORMS_EWM} prefixed by 'ewm_'. For example, 'function': 'ewm_mean', 'ewm_sum'.

            The 'window' key should be provided for built-in and for user provided callable functions.
            The 'window' value should be a positive integer representing the size of the window to be used for the
            transformation.

            Two optional keys can be provided for more flexibility: 'components' and 'series_id':
            1) The 'components' key can be a string or a list of strings specifying the names of the components
            of the series on which the transformation should be applied.
            If not provided, the transformation will be applied to all components of the series.
            2) When the input to the transformer is a sequence of TimeSeries, the 'series_id' key can be a positive
            integer >= 0 or a list of postivie integers >= 0 specifying the indices of the series on which
            the transformation should be applied. Series indices in the sequence start at 0.
            If not provided, the transformation will be applied to all series in the sequence.
            The following are possbible combination scenarios for the 'components' and 'series_id' keys:
             - 'components' and 'series_id' are not provided: the transformation will be applied to all components in
                all series in the sequence.
             - 'components' is not provided and 'series_id' is provided: the transformation will be applied to all
                components in the series specified by 'series_id'.
             - 'components' is provided and 'series_id' is not provided: the transformation will be applied to the
                components specified by 'components' in all series in the sequence.
             - 'components' and 'series_id' are provided: the transformation will be applied to the components
                specified by 'components' in the series specified by 'series_id'.
                If particular components are to be transformed in particular series, the 'components' key should be a
                list of lists with the same size as the 'series_id' key. For example, if 'series_id' is [0, 1] and
                'components' is [['component_1'], ['component_2', 'component_3']], the transformation will be applied to
                'component_1' in series 0, 'component_2' and 'component_3' in series 1.

            All other dictionary items provided will be treated as keyword arguments for the function group
            (i.e., pandas.DataFrame.rolling or pandas.DataFrame.ewm) or for the specific function in that group
            (i.e., pandas.DataFrame.rolling.mean/std/max/min... or pandas.DataFrame.ewm.mean/std/sum).
            This allows for more flexibility in configuring how the window slides over the data, by providing for
            example:
            'center': True/False to set the observation at the current timestep at the center of the windows
            (default is False),
            'closed': 'right'/'left'/'both'/'neither' to specify whether the right, left or both ends of the window are
            excluded (Darts enforces default to 'left', to guarantee the outcome to be forecasting safe);
            'step':int slides the window of 'step' size between each window evaluation (Darts enforces default to 1
            to guarantee outcome to have same frequency as the input series).
            More information on the available options for builtin functions can be found in the pandas documentation:
            https://pandas.pydata.org/docs/reference/window.html

            For user provided functions, extra arguments in the transformation dictionary are passed to the function.
            Darts sets by default that the user provided function will receive numpy arrays as input. User can modify
            this behavior by adding item 'raw':False in the transformation dictionary.
            It is expected that the function returns a single value for each window.
            Other possible configurations can be found in the pandas.DataFrame.Rolling().apply() documentation:
            https://pandas.pydata.org/docs/reference/window.html

            When calling transform(), user can pass different keyword arguments to configure the transformed series
            output:
            1) treat_na
            String to specify how to treat missing values in the resulting transformed TimeSeries.
            Can be 'dropna' to truncate the TimeSeries and drop observations with missing values,
            'bfill' to specify that NAs should be filled with the last valid observation.
            Can also be a value, in which case NAs will be filled with this value.

            2) forecasting_safe
            If True, Darts enforces that the resulting TimeSeries is safe to be used in forecasting models as target
            or as features. This parameter guarantees that the window transformation will not include any future values
            in the current timestep and will not fill NAs with future values. Default is True.
            Only pandas.DataFrame.Rolling functions can be currently guaranteed to be forecasting-safe.

            3) target
            If forecasting_safe is True and the target TimeSeries is provided, then the target TimeSeries will be
            truncated to align it with the window transformed TimeSeries.

            4) keep_non_transformed
            If True, the resulting TimeSeries will contain the non-transformed components along the transformed
            ones. The non-transformed components maintain their original name while the transformed components are
            named with the transformation name as a prefix. Default is False.

        name
            A specific name for the transformer.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries`. Defaults to `1`
        verbose
            Whether to print operations progress
        """
        super().__init__(name, n_jobs, verbose)

        # dictionary checks are mostly implemented in TimeSeries.window_transform()
        # here we only need to verify that the input is not None
        # and that 'series_id', if provided, is a list of positive integers

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
                ]  # if only one dictionary, make it a list

            for idx, transformation in enumerate(window_transformations):
                raise_if_not(
                    isinstance(transformation, dict),
                    f"`window_transformations` must contain dictionaries. Element at index {idx} is not a dictionary.",
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
                        f"`window_transformation` at index {idx} must contain a positive integer >= 0 for the "
                        f"'series_id', or a non-empty list containing positive integers >= 0. ",
                    )
                    if isinstance(transformation["series_id"], int):
                        window_transformations[idx]["series_id"] = [
                            transformation["series_id"]
                        ]  # make list

                if (
                    "components" in transformation
                    and transformation["components"] is None
                ):
                    window_transformations[idx].pop("components")

        self.window_transformations = window_transformations

    def _transform_iterator(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Iterator[Tuple[TimeSeries, dict]]:

        series = series2seq(series)

        # run through the transformations
        series_subset = []
        for idx, transformation in enumerate(self.window_transformations):

            if "series_id" not in transformation and "components" not in transformation:
                # apply the transformation to all series and all components
                series_subset += [(s, transformation) for s in series]

            elif "series_id" in transformation and "components" not in transformation:
                # apply the transformation to a specific series and all its components
                raise_if_not(
                    len(series) - 1 >= max(transformation["series_id"]),
                    f"`window_transformation` at index {idx} has a 'series_id' that is greater than "
                    f"the number of series in the provided sequence. ",
                )
                series_subset += [
                    (series[s_idx], transformation)
                    for s_idx in transformation["series_id"]
                ]

            elif "series_id" not in transformation and "components" in transformation:
                # apply the transformation to all series on specific components only in each series
                # test that component exists in each relevant series is implemented in TimeSeries.window_transform()
                # This scenario does not make sense unless series in the sequence have the same components names !
                series_subset += [(s, transformation) for s in series]

            else:
                # apply the transformation to a specific component in a specific series
                # if a different component is provided for each selected series
                if all(
                    isinstance(x, list) for x in transformation["components"]
                ) and len(transformation["components"]) == len(
                    transformation["series_id"]
                ):
                    # testing that components exist in the corresponding series is implemented
                    # in TimeSeries.window_transform()
                    components_list = transformation["components"]
                    for (s_idx, c_idvec) in zip(
                        transformation["series_id"], components_list
                    ):
                        # pair each series with its components
                        transformation.update({"components": c_idvec})
                        series_subset += [(series[s_idx], transformation.copy())]
                        # copy to avoid having the same final iteration dictionary for all series
                else:
                    # if the same components are provided for all the selected series (series would need to have same
                    # components names)!
                    # testing that the components exist in each series is implemented in TimeSeries.window_transform()
                    series_subset += [
                        (series[s_idx], transformation)
                        for s_idx in transformation["series_id"]
                    ]
                    # copy to avoid having the same final iteration dictionary for all series

        return iter(series_subset)  # the iterator object for ts_transform function

    @staticmethod
    def ts_transform(series: TimeSeries, transformation, **kwargs) -> TimeSeries:
        return series.window_transform(transformation, **kwargs)
