// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <Kokkos_Timer.hpp>
#include <TestMaca_Category.hpp>

#include <Maca/Kokkos_Maca_BlockSize_Deduction.hpp>
#include <Maca/Kokkos_Maca_ParallelFor_Range.hpp>
#include <Maca/Kokkos_Maca_ParallelReduce_Team.hpp>

#include <array>
#include <cstddef>

namespace Test {

struct MacaRangeBlocksizeProbeFunctor {
  using execution_space = Kokkos::Maca;
  KOKKOS_FUNCTION void operator()(int) const {}
};

template <class DriverType, class LaunchBounds,
          Kokkos::Impl::MacaLaunchMechanism LaunchMechanism =
              Kokkos::Impl::DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
int explicit_no_shmem_block_size_scan(
    Kokkos::Impl::MacaInternal const* maca_instance) {
  auto const& props =
      maca_instance->m_deviceProp;
  auto const attr =
      Kokkos::Impl::MacaParallelLaunch<DriverType, LaunchBounds,
                                       LaunchMechanism>::get_maca_func_attributes(
          maca_instance->m_macaDev);
  auto const kernel_func =
      Kokkos::Impl::MacaParallelLaunch<DriverType, LaunchBounds,
                                       LaunchMechanism>::get_kernel_func();

  maca_instance->set_maca_device();

  const int warp_size =
      std::max(1, int(props.warpSize > 0 ? props.warpSize
                                         : Kokkos::Impl::MacaTraits::WarpSize));
  int max_threads_per_block =
      std::min(attr.maxThreadsPerBlock,
               LaunchBounds::maxTperB == 0 ? int(props.maxThreadsPerBlock)
                                           : int(LaunchBounds::maxTperB));
  max_threads_per_block = (max_threads_per_block / warp_size) * warp_size;
  const int min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : int(LaunchBounds::minBperSM);

  int best_block_size     = 0;
  int best_threads_per_sm = 0;

  for (int block_size = max_threads_per_block; block_size >= warp_size;
       block_size -= warp_size) {
    int blocks_per_sm = 0;
    auto const status = macaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, reinterpret_cast<void const*>(kernel_func), block_size,
        0);
    if (status != macaSuccess) return 0;

    int threads_per_sm = blocks_per_sm * block_size;
    if (threads_per_sm > props.maxThreadsPerMultiProcessor) {
      blocks_per_sm  = props.maxThreadsPerMultiProcessor / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }

    if (blocks_per_sm >= min_blocks_per_sm) {
      if ((threads_per_sm > best_threads_per_sm) ||
          ((block_size >= 128) && (threads_per_sm == best_threads_per_sm))) {
        best_block_size     = block_size;
        best_threads_per_sm = threads_per_sm;
      }
    }
  }

  return best_block_size;
}

struct MacaLaunchTuningFunctor {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using scratch_space   = typename execution_space::scratch_memory_space;
  using scratch_view =
      Kokkos::View<int*, scratch_space, Kokkos::MemoryUnmanaged>;

  Kokkos::View<int*, execution_space> totals;
  int scratch_words = 0;

  size_t team_shmem_size(int) const {
    return scratch_view::shmem_size(scratch_words);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    scratch_view scratch(team.team_scratch(0), scratch_words);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, scratch_words),
                         [&](const int i) {
                           scratch(i) =
                               (team.league_rank() + 1) * (i % 11 + 1);
                         });
    team.team_barrier();

    int team_total = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, scratch_words),
        [&](const int i, int& val) { val += scratch(i); }, team_total);

    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { totals(team.league_rank()) = team_total; });
  }
};

TEST(maca, launch_tuning_desired_occupancy_team_scratch) {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  constexpr int league_size   = 13;
  constexpr int team_size     = 8;
  constexpr int scratch_words = 257;

  Kokkos::View<int*, execution_space> totals("totals", league_size);
  MacaLaunchTuningFunctor functor{totals, scratch_words};

  auto policy = Kokkos::Experimental::prefer(
      policy_type(league_size, team_size),
      Kokkos::Experimental::DesiredOccupancy(50));
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(functor.team_shmem_size(team_size)));

  Kokkos::parallel_for("maca_launch_tuning_team_scratch", policy, functor);

  auto totals_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), totals);
  for (int league_rank = 0; league_rank < league_size; ++league_rank) {
    int expected = 0;
    for (int i = 0; i < scratch_words; ++i) {
      expected += (league_rank + 1) * (i % 11 + 1);
    }
    ASSERT_EQ(totals_host(league_rank), expected);
  }
}

TEST(maca, launch_tuning_maximize_occupancy_team_scratch) {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  constexpr int league_size   = 7;
  constexpr int team_size     = 4;
  constexpr int scratch_words = 65;

  Kokkos::View<int*, execution_space> totals("totals", league_size);
  MacaLaunchTuningFunctor functor{totals, scratch_words};

  auto policy = Kokkos::Experimental::prefer(
      policy_type(league_size, team_size),
      Kokkos::Experimental::MaximizeOccupancy{});
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(functor.team_shmem_size(team_size)));

  Kokkos::parallel_for("maca_launch_tuning_max_occ", policy, functor);

  auto totals_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), totals);
  for (int league_rank = 0; league_rank < league_size; ++league_rank) {
    int expected = 0;
    for (int i = 0; i < scratch_words; ++i) {
      expected += (league_rank + 1) * (i % 11 + 1);
    }
    ASSERT_EQ(totals_host(league_rank), expected);
  }
}

TEST(maca, team_reduce_block_count_matches_cuda_shuffle_path) {
  EXPECT_EQ(Kokkos::Impl::maca_team_reduce_block_count(0, 64, true), 1);
  EXPECT_EQ(Kokkos::Impl::maca_team_reduce_block_count(7, 64, true), 7);
  EXPECT_EQ(Kokkos::Impl::maca_team_reduce_block_count(50000, 64, true),
            32768);
}

TEST(maca, team_reduce_block_count_matches_cuda_shared_memory_path) {
  EXPECT_EQ(Kokkos::Impl::maca_team_reduce_block_count(0, 64, false), 1);
  EXPECT_EQ(Kokkos::Impl::maca_team_reduce_block_count(7, 64, false), 7);
  EXPECT_EQ(Kokkos::Impl::maca_team_reduce_block_count(257, 64, false), 64);
}

TEST(maca, range_blocksize_no_shmem_matches_explicit_occupancy_scan) {
  using policy_type   = Kokkos::RangePolicy<Kokkos::Maca>;
  using launch_bounds = typename policy_type::launch_bounds;
  using driver_type =
      Kokkos::Impl::ParallelFor<MacaRangeBlocksizeProbeFunctor, policy_type,
                                Kokkos::Maca>;

  Kokkos::Maca exec;
  auto const* maca_instance = exec.impl_internal_space_instance();

  int const expected =
      explicit_no_shmem_block_size_scan<driver_type, launch_bounds>(
          maca_instance);
  int const actual = Kokkos::Impl::maca_get_opt_block_size_no_shmem<
      driver_type, launch_bounds>(maca_instance);
  ASSERT_GT(expected, 0);
  EXPECT_EQ(actual, expected);
}

struct MacaSlowGlobalLaunchFunctor {
  using execution_space = Kokkos::Maca;
  Kokkos::View<double*, execution_space> values;
  std::array<std::byte, 40000> padding = {};

  KOKKOS_FUNCTION void operator()(int i) const {
    double value = values(i) + 1.0;
    for (int iter = 0; iter < 2048; ++iter) {
      value = Kokkos::sqrt(value);
      value = value + 1.0;
      value = value * value;
    }
    values(i) = value;
  }
};

struct MacaTinyGlobalLaunchFunctor {
  using execution_space = Kokkos::Maca;
  Kokkos::View<int, execution_space> flag;
  std::array<std::byte, 40000> padding = {};

  KOKKOS_FUNCTION void operator()(int) const { flag() = 1; }
};

TEST(maca, global_launch_submission_does_not_fence_previous_work) {
  using execution_space = Kokkos::Maca;

  execution_space exec;
  Kokkos::View<double*, execution_space> values("values", 1 << 18);
  Kokkos::View<int, execution_space> flag("flag");

  auto slow_policy = Kokkos::RangePolicy<execution_space>(exec, 0, values.extent_int(0));
  auto tiny_policy = Kokkos::RangePolicy<execution_space>(exec, 0, 1);

  Kokkos::parallel_for("maca_warmup_global_launch", tiny_policy,
                       MacaTinyGlobalLaunchFunctor{flag});
  exec.fence();

  Kokkos::parallel_for("maca_slow_global_launch", slow_policy,
                       MacaSlowGlobalLaunchFunctor{values});

  Kokkos::Timer timer;
  Kokkos::parallel_for("maca_tiny_global_launch", tiny_policy,
                       MacaTinyGlobalLaunchFunctor{flag});
  const double submit_seconds = timer.seconds();

  exec.fence();

  auto host_flag =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flag);
  ASSERT_EQ(host_flag(), 1);
  ASSERT_LT(submit_seconds, 0.05);
}

}  // namespace Test
