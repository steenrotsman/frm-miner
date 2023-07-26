#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include "typing.h"
#include "motif.h"
#include "miner.h"
#include "patterns.h"
#include "sax.h"

namespace py = pybind11;

PYBIND11_MODULE(_frm_cpp, m) {
    py::class_<Motif>(m, "Motif")
        .def_property("pattern", [](Motif &m){
            std::string s;
            for (auto c : m.get_pattern())
                s += c;
            return s;
        }, nullptr)
        .def_property("indexes", &Motif::get_indexes, nullptr)
        .def_property("average_occurrences", &Motif::get_average_occurrences, nullptr)
        .def_property("representative", &Motif::get_representative, nullptr)
        .def_property("best_matches", &Motif::get_best_matches, nullptr)
        .def_property("length", &Motif::get_length, nullptr)
        .def_property("naed", &Motif::get_naed, nullptr)
        .def("__repr__", [](Motif &m){
            std::string s;
            for (auto c : m.get_pattern())
                s += c;
            return "Motif('" + s + "')";
        })
        ;

    py::class_<Miner>(m, "Miner")
        .def(py::init<double, int, int, int, int, double, int>(), py::arg("minsup"), py::arg("seglen"), py::arg("alphabet"), py::arg("min_len")=3, py::arg("max_len")=0, py::arg("max_overlap")=0.9, py::arg("k")=0)
        .def("mine", &Miner::mine)
        .def_property("motifs", &Miner::get_motifs, nullptr)
        ;

    py::class_<PatternMiner>(m, "PatternMiner")
        .def(py::init<double, int, int, double>(), py::arg("minsup"), py::arg("min_len")=3, py::arg("max_len")=0, py::arg("max_overlap")=0.9)
        .def("mine", &PatternMiner::mine)
        .def_property("frequent", &PatternMiner::get_frequent, nullptr)
        ;

    m.def("sax", &sax, py::arg("ts"), py::arg("seglen"), py::arg("alphabet"));
}