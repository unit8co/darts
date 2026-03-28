#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>

#define ABS(x) ((x) < 0 ? -(x) : (x))

namespace dtw {

    namespace nb = nanobind;

    void dtw_cost_matrix_no_window_1d(nb::ndarray<double, nb::ndim<1>> x,
                                    nb::ndarray<double, nb::ndim<1>> y,
                                    nb::ndarray<double, nb::ndim<2>> cost_matrix) {
        auto x_view = x.view();
        auto y_view = y.view();
        auto cost_view = cost_matrix.view();
        // initialize the first values of the cost matrix
        cost_view(0, 0) = 0.0;
        double cost = 0.0, min_prev_cost = 0.0;
        for (size_t i = 1; i <= x_view.shape(0); ++i) {
            for (size_t j = 1; j <= y_view.shape(0); ++j) {
                cost = ABS(x_view(i-1) - y_view(j-1));
                min_prev_cost = std::min({cost_view(i-1, j), cost_view(i, j-1), cost_view(i-1, j-1)});
                cost_view(i, j) = cost + min_prev_cost;
            }
        }
    }

}  // namespace dtw
