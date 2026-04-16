// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_BENCHMARK_MACA_CUDA_PARITY_COMMON_HPP
#define KOKKOS_BENCHMARK_MACA_CUDA_PARITY_COMMON_HPP

#include <cstdlib>

namespace MacaCudaParity {

inline int parse_int_arg(char* arg, int fallback) {
  return arg ? std::atoi(arg) : fallback;
}

}  // namespace MacaCudaParity

#endif
