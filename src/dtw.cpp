#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>

#define ABS(x) ((x) < 0 ? -(x) : (x))

namespace dtw {

    namespace nb = nanobind;

    nb::ndarray<nb::numpy, double, nb::ndim<2>> dtw_cost_matrix_no_window_1d(
        nb::ndarray<double, nb::ndim<1>> x,
        nb::ndarray<double, nb::ndim<1>> y) {
        auto x_view = x.view<double, nb::ndim<1>>();
        auto y_view = y.view<double, nb::ndim<1>>();
        size_t x_size = x_view.shape(0);
        size_t y_size = y_view.shape(0);
        size_t total_size = (x_size + 1) * (y_size + 1);

        // allocate the cost matrix and initialize it with infinity
        double *cost_matrix = new double[total_size];
        double inf = std::numeric_limits<double>::infinity();
        std::fill_n(cost_matrix, y_size + 1, inf);

        // initialize the first values of the cost matrix
        cost_matrix[0] = 0.0;
        double cost = 0.0, min_prev_cost = 0.0;
        for (size_t i = 1; i <= x_size; ++i) {
            cost_matrix[i * (y_size + 1)] = inf;
            for (size_t j = 1; j <= y_size; ++j) {
                cost = ABS(x_view(i-1) - y_view(j-1));
                min_prev_cost = std::min({
                    cost_matrix[(i-1) * (y_size + 1) + j],
                    cost_matrix[i * (y_size + 1) + (j-1)],
                    cost_matrix[(i-1) * (y_size + 1) + (j-1)]
                });
                cost_matrix[i * (y_size + 1) + j] = cost + min_prev_cost;
            }
        }

        // create a capsule to manage the memory of the cost matrix
        nb::capsule owner(cost_matrix, [](void *ptr) noexcept {
            delete[] (double *)ptr;
        });

        // return the cost matrix as a numpy array
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            cost_matrix,
            {x_size + 1, y_size + 1},
            owner
        );
    }


    nb::ndarray<nb::numpy, double, nb::ndim<2>> dtw_cost_matrix_no_window_2d(
        nb::ndarray<double, nb::ndim<2>> x,
        nb::ndarray<double, nb::ndim<2>> y) {
        auto x_view = x.view<double, nb::ndim<2>>();
        auto y_view = y.view<double, nb::ndim<2>>();
        size_t x_size = x_view.shape(0);
        size_t y_size = y_view.shape(0);
        size_t n_components = x_view.shape(1);
        size_t total_size = (x_size + 1) * (y_size + 1);

        // allocate the cost matrix and initialize it with infinity
        double *cost_matrix = new double[total_size];
        double inf = std::numeric_limits<double>::infinity();
        std::fill_n(cost_matrix, y_size + 1, inf);

        // initialize the first values of the cost matrix
        cost_matrix[0] = 0.0;
        double cost = 0.0, min_prev_cost = 0.0;
        for (size_t i = 1; i <= x_size; ++i) {
            cost_matrix[i * (y_size + 1)] = inf;
            for (size_t j = 1; j <= y_size; ++j) {
                cost = 0.0;
                for (size_t k = 0; k < n_components; ++k) {
                    cost += ABS(x_view(i-1, k) - y_view(j-1, k));
                }
                min_prev_cost = std::min({
                    cost_matrix[(i-1) * (y_size + 1) + j],
                    cost_matrix[i * (y_size + 1) + (j-1)],
                    cost_matrix[(i-1) * (y_size + 1) + (j-1)]
                });
                cost_matrix[i * (y_size + 1) + j] = cost + min_prev_cost;
            }
        }

        // create a capsule to manage the memory of the cost matrix
        nb::capsule owner(cost_matrix, [](void *ptr) noexcept {
            delete[] (double *)ptr;
        });

        // return the cost matrix as a numpy array
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            cost_matrix,
            {x_size + 1, y_size + 1},
            owner
        );
    }


}  // namespace dtw
