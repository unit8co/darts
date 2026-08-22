#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace dtw {
    nb::ndarray<nb::numpy, double, nb::ndim<2>> dtw_cost_matrix_no_window_1d(
        nb::ndarray<double, nb::ndim<1>> x,
        nb::ndarray<double, nb::ndim<1>> y);
    nb::ndarray<nb::numpy, double, nb::ndim<2>> dtw_cost_matrix_no_window_2d(
        nb::ndarray<double, nb::ndim<2>> x,
        nb::ndarray<double, nb::ndim<2>> y);
} // namespace dtw
