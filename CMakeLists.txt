cmake_minimum_required(VERSION 3.16)
project(microtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# keep compile_commands.json around, useful for clangd/editor tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# default to a sane build type if none given (Release would hide bugs via
# optimization, Debug is what you want while building+gradient-checking ops)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# ---- Eigen (header-only, pulled at configure time, no manual vendoring) ----
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
# Eigen ships its own tests/docs targets we don't want CMake to configure
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# ---- the engine itself (builds as a static library, see src/CMakeLists.txt) ----
add_subdirectory(src)

# ---- app/: thin entry points that link against libmicrotorch ----
add_subdirectory(app)

# ---- tests/: Catch2-based, mirrors src/ 1:1, op by op ----
option(MICROTORCH_BUILD_TESTS "Build the test suite" ON)
if(MICROTORCH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
