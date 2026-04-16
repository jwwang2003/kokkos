// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestMaca_Category.hpp>

#include <Maca/Kokkos_Maca_Vectorization.hpp>

#include <algorithm>

namespace Test {

template <int VectorLength>
struct TeamCollectiveFunctor {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;

  Kokkos::View<const int**, execution_space> values;
  Kokkos::View<int*, execution_space> reduce_out;
  Kokkos::View<int**, execution_space> scan_out;
  int active_count = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const int league_rank = team.league_rank();

    int team_sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, active_count),
        [&](const int i, int& sum) { sum += values(league_rank, i); }, team_sum);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { reduce_out(league_rank) = team_sum; });

    Kokkos::parallel_scan(
        Kokkos::TeamThreadRange(team, active_count),
        [&](const int i, int& prefix, const bool final) {
          prefix += values(league_rank, i);
          if (final) scan_out(league_rank, i) = prefix;
        });
  }
};

struct TeamMemberCollectiveFunctor {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;

  Kokkos::View<int*, execution_space> reduce_out;
  Kokkos::View<int**, execution_space> scan_out;
  Kokkos::View<int, execution_space> global_accum;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const int league_rank = team.league_rank();
    const int rank        = team.team_rank();
    const int value       = (league_rank + 1) * (rank + 1);

    int max_value = value;
    team.team_reduce(Kokkos::Max<int>(max_value));
    if (rank == 0) reduce_out(league_rank) = max_value;

    const int prefix = team.team_scan(value, global_accum.data());
    scan_out(league_rank, rank) = prefix;
  }
};

template <int VectorLength>
void run_team_collective_case(int league_size, int requested_team_size,
                              int active_count) {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  const int team_size = std::max(requested_team_size, 1);
  Kokkos::View<int**, execution_space> values("values", league_size,
                                              active_count);
  Kokkos::View<int*, execution_space> reduce_out("reduce_out", league_size);
  Kokkos::View<int**, execution_space> scan_out("scan_out", league_size,
                                                active_count);

  Kokkos::parallel_for(
      "maca_team_collective_init",
      Kokkos::RangePolicy<execution_space>(0, league_size * active_count),
      KOKKOS_LAMBDA(const int i) {
        const int league_rank = i / active_count;
        const int idx         = i % active_count;
        values(league_rank, idx) = (league_rank + 1) * (idx + 1);
      });

  TeamCollectiveFunctor<VectorLength> functor{
      values, reduce_out, scan_out, active_count};
  Kokkos::parallel_for("maca_team_collective_case",
                       policy_type(league_size, team_size, VectorLength),
                       functor);

  auto values_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  auto reduce_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reduce_out);
  auto scan_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scan_out);

  for (int league_rank = 0; league_rank < league_size; ++league_rank) {
    int expected_sum    = 0;
    int expected_prefix = 0;
    for (int idx = 0; idx < active_count; ++idx) {
      expected_sum += values_host(league_rank, idx);
      expected_prefix += values_host(league_rank, idx);
      ASSERT_EQ(scan_host(league_rank, idx), expected_prefix);
    }
    ASSERT_EQ(reduce_host(league_rank), expected_sum);
  }
}

void run_team_member_collective_case(int league_size, int team_size) {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  Kokkos::View<int*, execution_space> reduce_out("reduce_out", league_size);
  Kokkos::View<int**, execution_space> scan_out("scan_out", league_size,
                                                team_size);
  Kokkos::View<int, execution_space> global_accum("global_accum");
  Kokkos::deep_copy(global_accum, 0);

  Kokkos::parallel_for("maca_team_member_collective_case",
                       policy_type(league_size, team_size, 1),
                       TeamMemberCollectiveFunctor{reduce_out, scan_out,
                                                   global_accum});

  auto reduce_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reduce_out);
  auto scan_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scan_out);
  int global_accum_host = 0;
  Kokkos::deep_copy(global_accum_host, global_accum);

  int expected_global_total = 0;
  for (int league_rank = 0; league_rank < league_size; ++league_rank) {
    const int team_offset = scan_host(league_rank, 0);
    int expected_team_sum = 0;
    int expected_prefix   = 0;
    for (int rank = 0; rank < team_size; ++rank) {
      const int value = (league_rank + 1) * (rank + 1);
      ASSERT_EQ(scan_host(league_rank, rank), team_offset + expected_prefix);
      expected_prefix += value;
      expected_team_sum += value;
    }
    ASSERT_EQ(reduce_host(league_rank), (league_rank + 1) * team_size);
    expected_global_total += expected_team_sum;
  }

  ASSERT_EQ(global_accum_host, expected_global_total);
}

TEST(maca, team_collectives_vector_length_one) {
  run_team_collective_case<1>(17, 5, 73);
}

TEST(maca, team_collectives_vector_length_shuffle_enabled) {
  const int vector_length_max =
      Kokkos::TeamPolicy<Kokkos::Maca>::vector_length_max();
  if (vector_length_max >= 4) {
    run_team_collective_case<4>(19, 7, 67);
  } else if (vector_length_max >= 2) {
    run_team_collective_case<2>(19, 7, 67);
  } else {
    run_team_collective_case<1>(19, 7, 67);
  }
}

TEST(maca, team_collectives_empty_league) {
  using execution_space = Kokkos::Maca;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;

  Kokkos::View<int**, execution_space> values("values", 0, 1);
  Kokkos::View<int*, execution_space> reduce_out("reduce_out", 0);
  Kokkos::View<int**, execution_space> scan_out("scan_out", 0, 1);

  TeamCollectiveFunctor<1> functor{values, reduce_out, scan_out, 1};
  Kokkos::parallel_for("maca_team_collective_empty_league",
                       policy_type(0, 1, 1), functor);
  SUCCEED();
}

TEST(maca, team_member_collectives_partial_wave) {
  run_team_member_collective_case(13, 17);
}

TEST(maca, shuffle_group_mask_wave64) {
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(64, 0),
            0xffffffffffffffffULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(1, 0), 0x1ULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(1, 63),
            0x8000000000000000ULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(4, 0), 0xFULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(4, 3), 0xFULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(4, 4), 0xF0ULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(8, 31),
            0xFF000000ULL);
  EXPECT_EQ(Kokkos::Impl::maca_shuffle_group_mask(8, 63),
            0xFF00000000000000ULL);
}

}  // namespace Test
