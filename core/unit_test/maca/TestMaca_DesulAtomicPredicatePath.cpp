// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>

// This translation unit opts into the alternate MACA desul fetch-op include
// path so the build proves that the selector path compiles.
#define DESUL_MACA_ATOMICS_USE_PREDICATE_PATH

#include <Kokkos_Core.hpp>
#include <TestMaca_Category.hpp>

#ifndef DESUL_IMPL_MACA_PREDICATE_PATH_ACTIVE
#error "MACA predicate fetch-op path was not selected"
#endif

namespace Test {

template <class T>
void run_desul_atomic_predicate_path_inc_dec_case() {
  using execution_space = Kokkos::Maca;

  Kokkos::View<T, execution_space> inc("inc");
  Kokkos::View<T, execution_space> dec("dec");
  Kokkos::deep_copy(inc, T(0));
  Kokkos::deep_copy(dec, T(64));

  Kokkos::parallel_for(
      "maca_desul_atomic_predicate_path_inc_dec",
      Kokkos::RangePolicy<execution_space>(0, 64), KOKKOS_LAMBDA(const int) {
        (void)Kokkos::atomic_fetch_inc(inc.data());
        (void)Kokkos::atomic_fetch_dec(dec.data());
      });

  T inc_host = 0;
  T dec_host = 0;
  Kokkos::deep_copy(inc_host, inc);
  Kokkos::deep_copy(dec_host, dec);

  ASSERT_EQ(inc_host, T(64));
  ASSERT_EQ(dec_host, T(0));
}

TEST(maca, desul_atomic_predicate_path_inc_dec_unsigned_int) {
  run_desul_atomic_predicate_path_inc_dec_case<unsigned int>();
}

TEST(maca, desul_atomic_predicate_path_inc_dec_unsigned_long_long) {
  run_desul_atomic_predicate_path_inc_dec_case<unsigned long long>();
}

}  // namespace Test
