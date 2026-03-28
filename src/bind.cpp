#include "dtw.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "Internal C++ implementation of Darts algorithms.";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("dtw_cost_matrix_no_window_1d", &dtw::dtw_cost_matrix_no_window_1d, "x"_a, "y"_a,
        R"pbdoc(
        Compute the cost matrix for Dynamic Time Warping (DTW) without any window constraints.

        Parameters
        ----------
        x : numpy.ndarray
            First input sequence (1D array).
        y : numpy.ndarray
            Second input sequence (1D array).

        Returns
        -------
        numpy.ndarray
            A 2D array representing the cost matrix for DTW between the two input sequences.
            Shape: (len(x) + 1, len(y) + 1)
    )pbdoc");
}
