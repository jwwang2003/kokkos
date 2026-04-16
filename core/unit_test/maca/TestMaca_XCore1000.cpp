// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestMaca_Category.hpp>

#include <Maca/Kokkos_Maca_Instance.hpp>

#include <type_traits>

namespace Test {

TEST(maca, xcore1000_traits_are_explicit) {
#if defined(KOKKOS_ARCH_XCORE1000)
  static_assert(Kokkos::Impl::MacaTraits::WarpSize == 64);
  static_assert(Kokkos::Impl::MacaTraits::WarpIndexShift == 6);
  static_assert(Kokkos::Impl::MacaTraits::WarpIndexMask == 0x003f);
#endif
  SUCCEED();
}

TEST(maca, random_access_views_do_not_use_raw_pointer_handle) {
  using view_t =
      Kokkos::View<const int*, Kokkos::MacaSpace, Kokkos::MemoryRandomAccess>;
  using traits_t  = typename view_t::traits;
  using handle_t  = typename Kokkos::Impl::ViewDataHandle<traits_t>::handle_type;
  using return_t  = typename Kokkos::Impl::ViewDataHandle<traits_t>::return_type;

  static_assert(!std::is_same_v<handle_t, const int*>);
  static_assert(std::is_same_v<return_t, const int>);
  SUCCEED();
}

}  // namespace Test
