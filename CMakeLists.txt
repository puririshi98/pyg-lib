cmake_minimum_required(VERSION 3.12)
project(pyg)
set(CMAKE_CXX_STANDARD 14)
set(PYG_VERSION 0.1.0)

option(BUILD_TEST "Enable testing" OFF)
option(BUILD_BENCHMARK "Enable benchmarks" OFF)
option(WITH_COV "Enable code coverage" OFF)
option(USE_PYTHON "Link to Python when building" OFF)
option(WITH_CUDA "Enable CUDA support" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

if (USE_PYTHON)
  add_definitions(-DUSE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
endif()

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

  set(CUTLASS_DIR third_party/cutlass/include)
  include_directories(${CUTLASS_DIR})
  set(CUTLASS_UTIL_DIR third_party/cutlass/tools/util/include)
  include_directories(${CUTLASS_UTIL_DIR})
endif()

set(CSRC pyg_lib/csrc)
file(GLOB_RECURSE ALL_SOURCES ${CSRC}/*.cpp)
if (WITH_CUDA)
  file(GLOB_RECURSE ALL_SOURCES ${ALL_SOURCES} ${CSRC}/*.cu)
endif()
file(GLOB_RECURSE ALL_HEADERS ${CSRC}/*.h)
add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

set(PHMAP_DIR third_party/parallel-hashmap)
target_include_directories(${PROJECT_NAME} PRIVATE ${PHMAP_DIR})

find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if (USE_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

if(BUILD_TEST)
  if(WITH_COV)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
  endif()
  include(cmake/test.cmake)
endif()

message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message("-- CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_RELEASE}")
message("-- CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_DEBUG}")

if(BUILD_BENCHMARK)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "Disable benchmarking tool tests")
  set(COMPILE_HAVE_GNU_POSIX_REGEX OFF CACHE INTERNAL "Disable GNU POSIX regex compilation check")
  include(cmake/benchmark.cmake)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  EXPORT_NAME PyG
  INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)

# Cmake creates *.dylib by default, but python expects *.so by default
if (APPLE)
  set_property(TARGET ${PROJECT_NAME} PROPERTY SUFFIX .so)
endif()
