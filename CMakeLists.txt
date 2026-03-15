cmake_minimum_required(VERSION 3.20)
project(simd_engine VERSION 2.1.0 LANGUAGES CXX)

# ══════════════════════════════════════════════════════════════════════════════
# CMakeLists.txt — SIMD Image Processing Engine
#
# Tested configurations:
#   Windows:  MSVC 2022 (cl.exe), Clang-cl 16+, MinGW-w64 13+ (GCC 13)
#   Linux:    GCC 12+, Clang 15+
#
# Quick start (Windows, Visual Studio generator):
#   cmake -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#
# Quick start (Ninja, MSVC):
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Quick start (MinGW):
#   cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
# ══════════════════════════════════════════════════════════════════════════════

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ── Windows: set minimum Windows version to 10 (for NUMA APIs) ─────────────
if(WIN32)
  add_compile_definitions(
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00     # Windows 10
    NTDDI_VERSION=0x0A000000
    _CRT_SECURE_NO_WARNINGS
  )
endif()

# ── Find Python ──────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ── Find pybind11 ────────────────────────────────────────────────────────────
# Strategy: cmake config file → python -m pybind11 --cmakedir → FetchContent
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(PYBIND11_CMAKEDIR)
    find_package(pybind11 REQUIRED PATHS "${PYBIND11_CMAKEDIR}" NO_DEFAULT_PATH)
  else()
    include(FetchContent)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.12.0)
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# ── Compiler-specific flags ──────────────────────────────────────────────────
if(MSVC)
  # /GL + /LTCG → whole-program optimization (link-time)
  # /Gy         → function-level linking
  # /Oi         → intrinsic functions
  # /fp:fast    → aggressive FP optimization (reorder, contract)
  # /GS-        → disable stack-security check (perf in hot loops)
  set(MSVC_COMMON "/O2;/Oi;/Oy-;/Gy;/GS-;/fp:fast;/favor:INTEL64;/MP")
  set(MSVC_LTO    "/GL")

  # SIMD: detect highest available instruction set
  # /arch:AVX512 — available in VS 2019 16.3+
  # /arch:AVX2   — available in VS 2013+
  include(CheckCXXSourceCompiles)
  # AVX-512 disabled: compile-time detection does not guarantee runtime CPU support
  set(HAVE_AVX512 FALSE)
  if(HAVE_AVX512)
    set(SIMD_FLAG "/arch:AVX512")
    add_compile_definitions(SIMD_ENGINE_AVX512=1)
    message(STATUS "MSVC: enabling /arch:AVX512")
  else()
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
    check_cxx_source_compiles("
      #include <immintrin.h>
      int main(){__m256 v=_mm256_setzero_ps();return 0;}" HAVE_AVX2)
    if(HAVE_AVX2)
      set(SIMD_FLAG "/arch:AVX2")
      add_compile_definitions(SIMD_ENGINE_AVX2=1)
      message(STATUS "MSVC: enabling /arch:AVX2")
    else()
      set(SIMD_FLAG "/arch:SSE2")
      message(STATUS "MSVC: falling back to /arch:SSE2")
    endif()
  endif()
  set(CMAKE_REQUIRED_FLAGS "")

  add_compile_options(
    ${MSVC_COMMON} ${SIMD_FLAG}
    # C++20 conformance
    /permissive- /Zc:__cplusplus /Zc:preprocessor
    # Warnings: W3 + specific suppressions
    /W3
    /wd4244   # conversion, possible loss of data (intentional narrowing)
    /wd4267   # size_t → int (intentional)
    /wd4996   # deprecated POSIX names
  )

  # LTO for Release only
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR
     (CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug"))
    add_compile_options(${MSVC_LTO})
    add_link_options(/LTCG)
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # GCC 12+ / Clang 15+ / MinGW-w64 (uses GCC flags)
  add_compile_options(
    -O3
    -ffast-math
    -funroll-loops
    -fomit-frame-pointer
  )

  # Auto-detect best SIMD
  include(CheckCXXCompilerFlag)
  foreach(FLAG -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx2 -mfma -mbmi2 -mpopcnt)
    string(MAKE_C_IDENTIFIER "HAVE_${FLAG}" VAR)
    check_cxx_compiler_flag(${FLAG} ${VAR})
    if(${VAR})
      add_compile_options(${FLAG})
    endif()
  endforeach()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-march=native -fopenmp-simd)
  endif()

  # LTO
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-flto=auto)
    add_link_options(-flto=auto)
  endif()
endif()

# ── Optional dependencies ─────────────────────────────────────────────────────
# NUMA (Linux only — skip gracefully on Windows)
if(NOT WIN32)
  find_library(NUMA_LIB numa)
  if(NUMA_LIB)
    message(STATUS "Found libnuma: ${NUMA_LIB}")
    add_compile_definitions(SIMD_HAVE_NUMA=1)
  else()
    message(STATUS "libnuma not found — NUMA placement disabled on Linux")
  endif()
endif()

# Windows: we use Win32 NUMA APIs directly (no extra lib needed)
if(WIN32)
  message(STATUS "Windows NUMA: using GetLogicalProcessorInformationEx + VirtualAllocExNuma")
endif()

# ── Copy headers into build dir — each file quoted separately (spaces in path) ──
foreach(_hdr
    platform.hpp
    image_features.hpp
    simd_optimization.hpp
    multi_threading.hpp
    neural_rendering.hpp)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/${_hdr}"
       DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

# ── Source files ──────────────────────────────────────────────────────────────
set(ENGINE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/engine_impl.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
)

# ── pybind11 extension module ─────────────────────────────────────────────────
pybind11_add_module(simd_engine MODULE ${ENGINE_SOURCES})

# Both src dir AND binary dir (where copied headers land)
target_include_directories(simd_engine PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

# ── Status summary ───────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "══════════════════════════════════════════")
message(STATUS "  SIMD Engine ${PROJECT_VERSION} build config")
message(STATUS "  OS:       ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python:   ${Python3_VERSION} @ ${Python3_EXECUTABLE}")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "══════════════════════════════════════════")
message(STATUS "")