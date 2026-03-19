#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace dtw {
void dtw_cost_matrix_no_window_1d(
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> cost_matrix
);
} // namespace dtw
