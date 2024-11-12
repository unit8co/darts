"""
Window Transformer
------------------
"""

from collections.abc import Mapping
from typing import Any, Optional, Union

from darts.dataprocessing.transformers import BaseDataTransformer
from darts.logging import get_logger
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class WindowTransformer(BaseDataTransformer):
    def __init__(
        self,
        transforms: Union[dict, list[dict]],
        treat_na: Optional[Union[str, Union[int, float]]] = None,
        forecasting_safe: Optional[bool] = True,
        keep_non_transformed: Optional[bool] = False,
        include_current: Optional[bool] = True,
        keep_names: Optional[bool] = False,
        name: str = "WindowTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        A transformer that applies window transformation to a TimeSeries or a Sequence of TimeSeries. It expects a
        dictionary or a list of dictionaries specifying the window transformation(s) to be applied. All series in the
        sequence will be transformed with the same transformations.

        Parameters
        ----------
        transforms
            A dictionary or a list of dictionaries.
            Each dictionary specifies a different window transform.

            The dictionaries can contain the following keys:

            :``"function"``: Mandatory. The name of one of the pandas builtin transformation functions,
                            or a callable function that can be applied to the input series.
                            Pandas' functions can be found in the
                            `documentation <https://pandas.pydata.org/docs/reference/window.html>`_.

            :``"mode"``: Optional. The name of the pandas windowing mode on which the ``"function"`` is going to be
                        applied. The options are "rolling", "expanding" and "ewm".
                        If not provided, Darts defaults to "expanding".
                        User defined functions can use either "rolling" or "expanding" modes.
                        More information on pandas windowing operations can be found in the `documentation
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html>`_.

            :``"components"``: Optional. A string or list of strings specifying the TimeSeries components on which the
                               transformation should be applied. If not specified, the transformation will be
                               applied on all components.

            :``"function_name"``: Optional. A string specifying the function name referenced as part of
                                  the transformation output name. For example, given a user-provided function
                                  transformation on rolling window size of 5 on the component "comp", the
                                  default transformation output name is "rolling_udf_5_comp" whereby "udf"
                                  refers to "user defined function". If specified, the ``"function_name"`` will
                                  replace the default name "udf". Similarly, the ``"function_name"`` will replace
                                  the name of the pandas builtin transformation function name in the output name.

            All other dictionary items provided will be treated as keyword arguments for the windowing mode
            (i.e., ``rolling/ewm/expanding``) or for the specific function
            in that mode (i.e., ``pandas.DataFrame.rolling.mean/std/max/min...`` or
            ``pandas.DataFrame.ewm.mean/std/sum``).
            This allows for more flexibility in configuring the transformation, by providing for
            example:

            * :``"window"``: Size of the moving window for the "rolling" mode.
                            If an integer, the fixed number of observations used for each window.
                            If an offset, the time period of each window with data type :class:`pandas.Timedelta`
                            representing a fixed duration.
            * :``"min_periods"``: The minimum number of observations in the window required to have a value (otherwise
                NaN). Darts reuses pandas defaults of 1 for "rolling" and "expanding" modes and of 0 for "ewm" mode.
            * :``"win_type"``: The type of weigthing to apply to the window elements.
                If provided, it should be one of `scipy.signal.windows
                <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`_.
            * :``"center"``: ``True``/``False`` to set the observation at the current timestep at the center of the
                window (when ``forecasting_safe`` is `True`, Darts enforces ``"center"`` to ``False``).
            * :``"closed"``: ``"right"``/``"left"``/``"both"``/``"neither"`` to specify whether the right,
                left or both ends of the window are included in the window, or neither of them.
                Darts defaults to ``"both"``.

            More information on the available functions and their parameters can be found in the
            `Pandas documentation <https://pandas.pydata.org/docs/reference/window.html>`_.

            For user-provided functions, extra keyword arguments in the transformation dictionary are passed to the
            user-defined function.
            By default, Darts expects user-defined functions to receive numpy arrays as input.
            This can be modified by adding item ``"raw": False`` in the transformation dictionary.
            It is expected that the function returns a single
            value for each window. Other possible configurations can be found in the
            `pandas.DataFrame.rolling().apply() documentation
            <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`_
            and `pandas.DataFrame.expanding().apply() documentation
            <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.expanding.html>`_.

        treat_na
            Specifies how to treat missing values that were added by the window transformations
            at the beginning of the resulting TimeSeries. By default, Darts will leave NaNs in the resulting TimeSeries.
            This parameter can be one of the following:

            * :``"dropna"``: to truncate the TimeSeries and drop time steps containing missing values.
                If multiple columns contain different numbers of missing values, only the minimum number
                of rows is dropped. This operation might reduce the length of the resulting TimeSeries.

            * :``"bfill"`` or ``"backfill"``: to specify that NaNs should be filled with the last transformed
                and valid observation. If the original TimeSeries starts with NaNs, those are kept.
                When ``forecasting_safe`` is ``True``, this option returns an exception to avoid future observation
                contaminating the past.

            * :an integer or float: in which case NaNs will be filled with this value.
                All columns will be filled with the same provided value.

        forecasting_safe
            If True, Darts enforces that the resulting TimeSeries is safe to be used in forecasting models as target
            or as feature. The window transformation will not allow future values to be included in the computations
            at their corresponding current timestep. Default is ``True``.
            "ewm" and "expanding" modes are forecasting safe by default.
            "rolling" mode is forecasting safe if ``"center": False`` is guaranteed.

        keep_non_transformed
            ``False`` to return the transformed components only, ``True`` to return all original components along
            the transformed ones. Default is ``False``. If the series has a hierarchy, must be set to ``False``.

        include_current
            ``True`` to include the current time step in the window, ``False`` to exclude it. Default is ``True``.

        keep_names
            Whether the transformed components should keep the original component names or. Must be set to ``False``
            if `keep_non_transformed = True` or the number of transformation is greater than 1.

        name
            A specific name for the transformer.

        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`.

        verbose
            Whether to print operations progress.
        """

        # dictionary checks are implemented in TimeSeries.window_transform()

        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self.transforms = transforms
        self.keep_non_transformed = keep_non_transformed
        self.treat_na = treat_na
        self.forecasting_safe = forecasting_safe
        self.include_current = include_current
        self.keep_names = keep_names
        super().__init__(name, n_jobs, verbose)

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
        return series.window_transform(**params["fixed"])
