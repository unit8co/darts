#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

#define ABS(x) ((x) < 0 ? -(x) : (x))

namespace dtw {

namespace py = pybind11;

void dtw_cost_matrix_no_window_1d(py::array_t<double> x, py::array_t<double> y, py::array_t<double> cost_matrix) {
    // use direct access to the underlying data of the numpy arrays
    auto x_ptr = x.unchecked<1>();  // 2D array
    auto y_ptr = y.unchecked<1>();  // 2D array
    auto cost_ptr = cost_matrix.mutable_unchecked<2>();  // 2D array
    // initialize the first values of the cost matrix
    cost_ptr(0, 0) = 0.0;
    double cost = 0.0, min_prev_cost = 0.0;
    ssize_t x_size = x_ptr.shape(0), y_size = y_ptr.shape(0);
    // iterate over the cost matrix and fill it in
    for (py::ssize_t i = 1; i <= x_size; ++i) {
        for (py::ssize_t j = 1; j <= y_size; ++j) {
            cost = ABS(x_ptr(i-1) - y_ptr(j-1));
            min_prev_cost = std::min({cost_ptr(i-1, j), cost_ptr(i, j-1), cost_ptr(i-1, j-1)});
            cost_ptr(i, j) = cost + min_prev_cost;
        }
    }
}

}  // namespace dtw
