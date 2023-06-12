#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include "typing.h"
#include "motif.h"
#include "miner.h"

namespace py = pybind11;

PYBIND11_MODULE(_frm_cpp, m) {
    py::class_<Motif>(m, "Motif")
        .def_property("pattern", &Motif::get_pattern, nullptr)
        .def_property("indexes", &Motif::get_indexes, nullptr)
        .def_property("representative", &Motif::get_representative, nullptr)
        .def_property("best_matches", &Motif::get_best_matches, nullptr)
        .def_property("rmse", &Motif::get_rmse, nullptr)
        .def("__repr__", [](Motif &m){
            std::string s;
            for (auto i : m.get_pattern())
                s += (s.empty() ? "" : ",") + std::to_string(i);
            return "Motif(" + s + ")"; })
        ;

    py::class_<Miner>(m, "Miner")
        .def(py::init<double, int, int, int, double, int>(), py::arg("minsup"), py::arg("seglen"), py::arg("alphabet"), py::arg("min_len") = 3, py::arg("max_overlap") = 0.9, py::arg("k")=0)
        .def("mine", &Miner::mine)
        .def_property("motifs", &Miner::get_motifs, nullptr)
        ;
}