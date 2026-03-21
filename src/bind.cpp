#include <pybind11/pybind11.h>
#include "dtw.h"

namespace py = pybind11;


PYBIND11_MODULE(_internal, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        Internal C++ implementation of DARTS algorithms.
    )pbdoc";

    m.def("dtw_cost_matrix_no_window_1d", &dtw::dtw_cost_matrix_no_window_1d, R"pbdoc(
        Compute the cost matrix for Dynamic Time Warping (DTW) without any window constraints.

        Parameters
        ----------
        x : numpy.ndarray
            First input sequence (1D array).
        y : numpy.ndarray
            Second input sequence (1D array).
        cost_matrix : numpy.ndarray
            Pre-allocated 2D array to store the computed cost matrix. Should have shape (len(x)+1, len(y)+1).
    )pbdoc");
}
