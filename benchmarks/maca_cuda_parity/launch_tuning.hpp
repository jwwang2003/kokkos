// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_BENCHMARK_MACA_CUDA_PARITY_LAUNCH_TUNING_HPP
#define KOKKOS_BENCHMARK_MACA_CUDA_PARITY_LAUNCH_TUNING_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "common.hpp"

#include <algorithm>
#include <cstdio>

namespace MacaCudaParity {

#if defined(KOKKOS_ENABLE_MACA)
struct LaunchTuningBenchmarkFunctor {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using scratch_space   = typename execution_space::scratch_memory_space;
  using scratch_view    =
      Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>;

  Kokkos::View<int*, execution_space> sink;
  int scratch_words = 0;

  size_t team_shmem_size(int) const { return scratch_view::shmem_size(scratch_words); }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    scratch_view scratch(team.team_scratch(0), scratch_words);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, scratch_words),
                         [&](const int i) {
                           scratch(i) =
                               (team.league_rank() + 1) * (i % 17 + 1);
                         });
    team.team_barrier();

    int team_total = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, scratch_words),
        [&](const int i, int& val) { val += scratch(i); }, team_total);

    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { sink(team.league_rank()) = team_total; });
  }
};
#endif

inline int run_launch_tuning_benchmark(int argc, char* argv[]) {
#if !defined(KOKKOS_ENABLE_MACA)
  (void)argc;
  (void)argv;
  std::fprintf(stderr, "launch-tuning benchmark requires Kokkos_ENABLE_MACA\n");
  return 1;
#else
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  const int league_size =
      argc > 0 ? parse_int_arg(argv[0], 2048) : static_cast<int>(2048);
  const int requested_team =
      argc > 1 ? parse_int_arg(argv[1], 8) : static_cast<int>(8);
  const int scratch_words =
      argc > 2 ? parse_int_arg(argv[2], 512) : static_cast<int>(512);
  const int repeat =
      argc > 3 ? parse_int_arg(argv[3], 40) : static_cast<int>(40);

  Kokkos::View<int*, execution_space> sink("sink", league_size);
  LaunchTuningBenchmarkFunctor functor{sink, std::max(scratch_words, 1)};

  auto policy = policy_type(league_size, std::max(requested_team, 1));
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(functor.team_shmem_size(0)));

  Kokkos::parallel_for("maca_cuda_parity_launch_tuning_warmup", policy,
                       functor);

  Kokkos::fence();
  Kokkos::Timer timer;
  for (int iter = 0; iter < repeat; ++iter) {
    Kokkos::parallel_for("maca_cuda_parity_launch_tuning", policy, functor);
  }
  Kokkos::fence();

  long long checksum = 0;
  Kokkos::parallel_reduce(
      "maca_cuda_parity_launch_tuning_checksum",
      Kokkos::RangePolicy<execution_space>(0, league_size),
      KOKKOS_LAMBDA(int i, long long& val) { val += sink(i); }, checksum);

  std::printf(
      "subtest=launch-tuning league=%d team=%d scratch_words=%d repeat=%d "
      "checksum=%lld seconds=%.9f\n",
      league_size, requested_team, scratch_words, repeat, checksum,
      timer.seconds());
  return 0;
#endif
}

}  // namespace MacaCudaParity

#endif
