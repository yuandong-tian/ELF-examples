#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "elf_adaptor.h"
#include "elf/options/OptionSpec.h"
#include "elf/options/reflection_option.h"
#include "elf/interface/snippets_pybind.h"

namespace py = pybind11;

PYBIND11_MODULE(_atari, m) {
  elf::snippet::reg_pybind11(m);

  py::class_<atari::Options>(m, "AtariOptions")
    .def(py::init<>());

  m.def("getGameFactory", &GameWrapper::getGameFactory);
}
