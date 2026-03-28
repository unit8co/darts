#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace dtw {
    void dtw_cost_matrix_no_window_1d(nb::ndarray<double, nb::ndim<1>> x,
                                    nb::ndarray<double, nb::ndim<1>> y,
                                    nb::ndarray<double, nb::ndim<2>> cost_matrix);
} // namespace dtw
