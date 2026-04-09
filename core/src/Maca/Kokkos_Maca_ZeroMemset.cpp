// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Maca/Kokkos_Maca_ZeroMemset.hpp>
#include <Maca/Kokkos_Maca_ParallelFor_Range.hpp>

namespace Kokkos {
namespace Impl {

// alternative to hipMemsetAsync, which sets the first `cnt` bytes of `dst` to 0
void zero_with_hip_kernel(const Maca& exec_space, void* dst, size_t cnt) {
  Kokkos::parallel_for(
      "Kokkos::ZeroMemset via parallel_for",
      Kokkos::RangePolicy<Kokkos::Maca, Kokkos::IndexType<size_t>>(exec_space, 0,
                                                                  cnt),
      KOKKOS_LAMBDA(size_t i) { static_cast<char*>(dst)[i] = 0; });
}

}  // namespace Impl
}  // namespace Kokkos
