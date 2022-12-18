from itertools import product

import numpy as np

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import strided_moving_window


class StridedMovingWindowTestCase(DartsBaseTestClass):
    def test_extracted_moving_windows(self):
        x_shape = (10, 8, 12)
        x = np.arange(np.prod(x_shape)).reshape(*x_shape)
        window_len_combos = (1, 2, 5)
        axis_combos = (0, 1, 2)
        stride_combos = (1, 2, 3)
        for (axis, stride, window_len) in product(
            axis_combos, stride_combos, window_len_combos
        ):
            windows = strided_moving_window(x, window_len, stride, axis)
            # Iterate over extracted windows:
            for i in range(windows.shape[axis]):
                # Take `i`th window:
                window = np.moveaxis(windows, axis, -1)[:, :, :, i]
                # `i`th window should correspond to taking the first `window_len` values
                # along `axis` in `x`, starting from the position `i*stride` (i.e. this is
                # the window of length `window_len` that 'begins' `i` strides from the start
                # of `axis`):
                expected = np.moveaxis(x, axis, -1)[
                    :, :, i * stride + np.arange(window_len)
                ]
                self.assertTrue(np.allclose(window, expected))

    def test_invalid_stride_error(self):
        x = np.arange(10)
        with self.assertRaises(ValueError) as e:
            strided_moving_window(x, window_len=1, stride=-1)
        self.assertEqual(
            ("`stride` must be positive."),
            str(e.exception),
        )

    def test_negative_window_len_error(self):
        x = np.arange(10)
        with self.assertRaises(ValueError) as e:
            strided_moving_window(x, window_len=-1, stride=1)
        self.assertEqual(
            ("`window_len` must be positive."),
            str(e.exception),
        )

    def test_pass_invalid_axis_error(self):
        x = np.arange(10)
        with self.assertRaises(ValueError) as e:
            strided_moving_window(x, window_len=1, stride=1, axis=1)
        self.assertEqual(
            ("`axis` must be less than `x.ndim`."),
            str(e.exception),
        )

    def test_window_too_large_error(self):
        x = np.arange(10)
        with self.assertRaises(ValueError) as e:
            strided_moving_window(x, window_len=11, stride=1)
        self.assertEqual(
            ("`window_len` must be less than or equal to x.shape[axis]."),
            str(e.exception),
        )
