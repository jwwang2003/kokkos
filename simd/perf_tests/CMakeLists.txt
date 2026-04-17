if(NOT (Kokkos_ENABLE_CUDA AND WIN32))
  # Keep perf benchmark coverage aligned with the supported SIMD test matrix.
  kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/include)

  kokkos_add_benchmark(PerformanceTest_SIMD_Host MAIN_FILE BenchmarkMain.cpp)
  target_compile_definitions(Kokkos_PerformanceTest_SIMD_Host PRIVATE KOKKOS_IMPL_SIMD_HOST_PERFTEST)

  kokkos_add_benchmark(PerformanceTest_SIMD_Device MAIN_FILE BenchmarkMain.cpp)
  target_compile_definitions(Kokkos_PerformanceTest_SIMD_Device PRIVATE KOKKOS_IMPL_SIMD_DEVICE_PERFTEST)
endif()
