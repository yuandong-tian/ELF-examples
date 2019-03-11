#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "elf_adaptor.h"
#include "elf/options/OptionSpec.h"
#include "elf/options/reflection_option.h"
#include "elf/options/pybind_utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_mcts_demo, m) {
  auto ref = py::return_value_policy::reference_internal;

  elf::options::PyInterface<elf::ai::tree_search::TSOptions>(m, "TSOptions");

  py::class_<MyContext>(m, "MyContext")
    .def(py::init<const elf::ai::tree_search::TSOptions &, std::string>())
    .def("setGameContext", &MyContext::setGameContext, ref)
    .def("getBatchSpec", &MyContext::getBatchSpec)
    .def("getParams", &MyContext::getParams);
}
