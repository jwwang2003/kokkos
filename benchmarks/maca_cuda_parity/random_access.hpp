// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_BENCHMARK_MACA_CUDA_PARITY_RANDOM_ACCESS_HPP
#define KOKKOS_BENCHMARK_MACA_CUDA_PARITY_RANDOM_ACCESS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "common.hpp"

#include <algorithm>
#include <cstdio>

namespace MacaCudaParity {

inline int run_random_access_benchmark(int argc, char* argv[]) {
#if !defined(KOKKOS_ENABLE_MACA)
  (void)argc;
  (void)argv;
  std::fprintf(stderr, "random-access benchmark requires Kokkos_ENABLE_MACA\n");
  return 1;
#else
  using exec_space = Kokkos::Maca;
  using mem_space  = typename exec_space::memory_space;

  const int count      = argc > 0 ? parse_int_arg(argv[0], 1 << 20) : 1 << 20;
  const int span       = argc > 1 ? parse_int_arg(argv[1], 1 << 16) : 1 << 16;
  const int repeat     = argc > 2 ? parse_int_arg(argv[2], 20) : 20;
  const int value_size = std::max(count, span);

  Kokkos::View<double*, mem_space> values("values", value_size);
  Kokkos::View<int*, mem_space> indices("indices", count);
  Kokkos::parallel_for(
      "maca_cuda_parity_random_access_init",
      Kokkos::RangePolicy<exec_space>(0, value_size), KOKKOS_LAMBDA(int i) {
        values(i) = 0.25 * static_cast<double>((i % 251) + 1);
      });
  Kokkos::parallel_for(
      "maca_cuda_parity_random_access_indices",
      Kokkos::RangePolicy<exec_space>(0, count), KOKKOS_LAMBDA(int i) {
        indices(i) = (i * 17) % value_size;
      });

  Kokkos::View<const double*, mem_space, Kokkos::MemoryRandomAccess>
      random_access(values);

  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "maca_cuda_parity_random_access_warmup",
      Kokkos::RangePolicy<exec_space>(0, count),
      KOKKOS_LAMBDA(int i, double& sum) { sum += random_access(indices(i)); },
      checksum);

  Kokkos::fence();
  Kokkos::Timer timer;
  for (int iter = 0; iter < repeat; ++iter) {
    Kokkos::parallel_reduce(
        "maca_cuda_parity_random_access",
        Kokkos::RangePolicy<exec_space>(0, count),
        KOKKOS_LAMBDA(int i, double& sum) { sum += random_access(indices(i)); },
        checksum);
  }
  Kokkos::fence();

  const double seconds = timer.seconds();
  const double bytes =
      static_cast<double>(repeat) * static_cast<double>(count) *
      static_cast<double>(sizeof(double) + sizeof(int));

  std::printf(
      "subtest=random-access count=%d span=%d repeat=%d checksum=%.17g "
      "seconds=%.9f bandwidth_gib_s=%.6f\n",
      count, value_size, repeat, checksum, seconds,
      bytes / seconds / (1024.0 * 1024.0 * 1024.0));
  return 0;
#endif
}

}  // namespace MacaCudaParity

#endif
