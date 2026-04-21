// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_BENCHMARK_MACA_CUDA_PARITY_TEAM_COLLECTIVES_HPP
#define KOKKOS_BENCHMARK_MACA_CUDA_PARITY_TEAM_COLLECTIVES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "common.hpp"

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace MacaCudaParity {

#if defined(KOKKOS_ENABLE_MACA)
template <class ViewType>
struct TeamCollectiveBenchmarkFunctor {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;

  ViewType values;
  Kokkos::View<double*, execution_space> sink;
  int extent = 0;
  bool do_scan = false;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const int league_rank = team.league_rank();

    if (do_scan) {
      Kokkos::parallel_scan(
          Kokkos::TeamThreadRange(team, extent),
          [&](const int i, double& val, const bool final) {
            val += values(league_rank, i);
            if (final && i == extent - 1) {
              sink(league_rank) = val + values(league_rank, i);
            }
          });
    } else {
      double team_total = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, extent),
          [&](const int i, double& val) { val += values(league_rank, i); },
          team_total);
      Kokkos::single(Kokkos::PerTeam(team),
                     [&]() { sink(league_rank) = team_total; });
    }
  }
};
#endif

inline int run_team_collective_benchmark(std::string_view subtest, int argc,
                                         char* argv[]) {
#if !defined(KOKKOS_ENABLE_MACA)
  (void)subtest;
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "team collective benchmarks require Kokkos_ENABLE_MACA\n");
  return 1;
#else
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  const int league_size =
      argc > 0 ? parse_int_arg(argv[0], 4096) : static_cast<int>(4096);
  const int requested_team =
      argc > 1 ? parse_int_arg(argv[1], 8) : static_cast<int>(8);
  const int requested_vector =
      argc > 2 ? parse_int_arg(argv[2], 1) : static_cast<int>(1);
  const int repeat =
      argc > 3 ? parse_int_arg(argv[3], 50) : static_cast<int>(50);
  const int extent =
      argc > 4 ? parse_int_arg(argv[4], 67) : static_cast<int>(67);

  const int vector_length =
      std::clamp(requested_vector, 1, policy_type::vector_length_max());
  const int team_size = std::max(requested_team, 1);

  Kokkos::View<double**, execution_space> values("values", league_size, extent);
  Kokkos::View<double*, execution_space> sink("sink", league_size);
  Kokkos::parallel_for(
      "maca_cuda_parity_team_collective_init",
      Kokkos::RangePolicy<execution_space>(0, league_size * extent),
      KOKKOS_LAMBDA(int i) {
        const int league_rank = i / extent;
        const int idx         = i % extent;
        values(league_rank, idx) =
            1.0 + static_cast<double>((league_rank + idx) % 29);
      });

  TeamCollectiveBenchmarkFunctor<decltype(values)> functor{
      values, sink, extent, subtest == "team-scan"};
  const policy_type policy(league_size, team_size, vector_length);

  Kokkos::parallel_for("maca_cuda_parity_team_collective_warmup", policy,
                       functor);

  Kokkos::fence();
  Kokkos::Timer timer;
  for (int iter = 0; iter < repeat; ++iter) {
    Kokkos::parallel_for("maca_cuda_parity_team_collective", policy, functor);
  }
  Kokkos::fence();

  double checksum = 0.0;
  Kokkos::parallel_reduce(
      "maca_cuda_parity_team_collective_checksum",
      Kokkos::RangePolicy<execution_space>(0, league_size),
      KOKKOS_LAMBDA(int i, double& val) { val += sink(i); }, checksum);

  std::printf(
      "subtest=%.*s league=%d team=%d vector=%d extent=%d repeat=%d "
      "checksum=%.17g seconds=%.9f\n",
      static_cast<int>(subtest.size()), subtest.data(), league_size, team_size,
      vector_length, extent, repeat, checksum, timer.seconds());
  return 0;
#endif
}

}  // namespace MacaCudaParity

#endif
