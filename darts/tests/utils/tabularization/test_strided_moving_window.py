from itertools import product

import numpy as np
import pytest

from darts.utils.data.tabularization import strided_moving_window


class TestStridedMovingWindow:
    """
    Tests `strided_moving_window` function defined in `darts.utils.data.tabularization`.
    """

    def test_strided_moving_windows_extracted_windows(self):
        """
        Tests that each of the windows extracted by `strided_moving_windows`
        is correct over a number of input parameter combinations.

        This is achieved by looping over each extracted window, and checking that the
        `i`th window corresponds to the the next `window_len` values found after
        `i * stride` (i.e. the index position at which the `i`th window should begin).
        """
        # Parameter combos to test:
        window_len_combos = (1, 2, 5)
        axis_combos = (0, 1, 2)
        stride_combos = (1, 2, 3)
        # Create a 'dummy input' with linearly increasing values:
        x_shape = (10, 8, 12)
        x = np.arange(np.prod(x_shape)).reshape(*x_shape)
        for axis, stride, window_len in product(
            axis_combos, stride_combos, window_len_combos
        ):
            windows = strided_moving_window(
                x=x, window_len=window_len, stride=stride, axis=axis
            )
            # Iterate over extracted windows:
            for i in range(windows.shape[axis]):
                # All of the extract windows are found along the `axis` dimension; shift
                # `axis` so that it now appears as the final dimension and then extract
                # `i`th window:
                window = np.moveaxis(windows, axis, -1)[:, :, :, i]
                # `i`th window should begin at `i * stride`:
                window_start_idx = i * stride
                # Window should include next `window_len` values after window start;
                # shift `axis` to last dimension then extract expected window:
                expected = np.moveaxis(x, axis, -1)[
                    :, :, window_start_idx : window_start_idx + window_len
                ]
                assert np.allclose(window, expected)

    def test_strided_moving_window_invalid_stride_error(self):
        """
        Checks that appropriate error is thrown when `stride` is set to
        a non-positive number and/or a non-`int` value.
        """
        x = np.arange(1)
        # `stride` isn't positive:
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=0)
        assert ("`stride` must be a positive `int`.") == str(err.value)
        # `stride` is `float`, not `int`:
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=1.1)
        assert ("`stride` must be a positive `int`.") == str(err.value)

    def test_strided_moving_window_negative_window_len_error(self):
        """
        Checks that appropriate error is thrown when `wendow_len`
        is set to a non-positive number and/or a non-`int` value.
        """
        x = np.arange(1)
        # `window_len` isn't positive:
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=0, stride=1)
        assert ("`window_len` must be a positive `int`.") == str(err.value)
        # `window_len` is `float`, not `int`:
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1.1, stride=1)
        assert ("`window_len` must be a positive `int`.") == str(err.value)

    def test_strided_moving_window_pass_invalid_axis_error(self):
        """
        Checks that appropriate error is thrown when `axis`
        is set to a non-`int` value, or a value not less than
        `x.ndim`.
        """
        x = np.arange(1)
        # `axis` NOT an int:
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=1, axis=0.1)
        assert ("`axis` must be an `int` that is less than `x.ndim`.") == str(err.value)
        # `axis` NOT < x.ndim:
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=1, stride=1, axis=1)
        assert ("`axis` must be an `int` that is less than `x.ndim`.") == str(err.value)

    def test_strided_moving_window_window_len_too_large_error(self):
        """
        Checks that appropriate error is thrown when `window_len`
        is set to a value larger than `x.shape[axis]`.
        """
        x = np.arange(1)
        with pytest.raises(ValueError) as err:
            strided_moving_window(x, window_len=2, stride=1)
        assert ("`window_len` must be less than or equal to x.shape[axis].") == str(
            err.value
        )
