// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include "launch_tuning.hpp"
#include "random_access.hpp"
#include "team_collectives.hpp"

#include <cstdio>
#include <string_view>

namespace {

void print_usage() {
  std::printf("Usage: Kokkos_maca_cuda_parity <subtest> [args...]\n");
  std::printf("  random-access [count span repeat]\n");
  std::printf("  team-reduce [league team vector repeat extent]\n");
  std::printf("  team-scan [league team vector repeat extent]\n");
  std::printf("  launch-tuning [league team scratch_words repeat]\n");
}

}  // namespace

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  Kokkos::initialize(argc, argv);

  int rc = 0;
  if (argc < 2) {
    print_usage();
  } else {
    const std::string_view subtest(argv[1]);
    if (subtest == "random-access") {
      rc = MacaCudaParity::run_random_access_benchmark(argc - 2, argv + 2);
    } else if (subtest == "team-reduce" || subtest == "team-scan") {
      rc = MacaCudaParity::run_team_collective_benchmark(subtest, argc - 2,
                                                         argv + 2);
    } else if (subtest == "launch-tuning") {
      rc = MacaCudaParity::run_launch_tuning_benchmark(argc - 2, argv + 2);
    } else {
      print_usage();
      rc = 1;
    }
  }

  Kokkos::finalize();
  return rc;
}
