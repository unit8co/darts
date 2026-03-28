#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace dtw {
    nb::ndarray<nb::numpy, double, nb::ndim<2>> dtw_cost_matrix_no_window_1d(nb::ndarray<double> x,
                                    nb::ndarray<double> y);
} // namespace dtw
