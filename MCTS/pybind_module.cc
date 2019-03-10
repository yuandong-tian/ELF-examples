#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "elf_adaptor.h"
#include "elf/options/OptionSpec.h"
#include "elf/options/reflection_option.h"

namespace py = pybind11;

PYBIND11_MODULE(_mcts_demo, m) {
  elf::snippet::reg_pybind11(m);

  py::class_<MyContext>(m, "MyContext")
    .def(py::init<std::string>());
}
