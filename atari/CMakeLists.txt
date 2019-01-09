CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(atari)

find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -Wno-register -fPIC -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # require at least gcc 7.1
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.1)
        message(FATAL_ERROR "GCC version must be at least 7.1!")
    endif()
endif()

set(ALE_INCLUDE_DIRS
    ./Arcade-Learning-Environment/src
)

set(ALE_LIBRARY_DIRS
    ./Arcade-Learning-Environment
)

include_directories(SYSTEM ${ALE_INCLUDE_DIRS})
link_directories(${ALE_LIBRARY_DIRS})

set(ELF_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../ELF)

include_directories(${ELF_PATH}/src_cpp/)
include_directories(${ELF_PATH}/third_party/)
include_directories(${ELF_PATH}/third_party/json/include)

add_subdirectory(
    ${ELF_PATH}/third_party/pybind11
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11)

# Source files

set(SOURCES
    atari_game.cc
)

add_library(atari ${SOURCES})

# Python bindings
pybind11_add_module(_atari pybind_module.cc)
target_link_libraries(_atari PRIVATE
    ale
    atari
)
