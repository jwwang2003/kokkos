// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Maca/Kokkos_Maca_DeepCopy.hpp>
#include <Maca/Kokkos_Maca_Error.hpp>  // MACA_SAFE_CALL
#include <Maca/Kokkos_Maca.hpp>
#include <Maca/Kokkos_Maca_Instance.hpp>

namespace Kokkos {
namespace Impl {
namespace {
macaStream_t get_deep_copy_stream() {
  static macaStream_t s = nullptr;
  if (s == nullptr) {
    KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamCreate(&s));
  }
  return s;
}
}  // namespace

void DeepCopyMaca(void* dst, void const* src, size_t n) {
  KOKKOS_IMPL_MACA_SAFE_CALL(macaMemcpyAsync(dst, src, n, macaMemcpyDefault));
}

void DeepCopyAsyncMaca(const Maca& instance, void* dst, void const* src,
                       size_t n) {
  KOKKOS_IMPL_MACA_SAFE_CALL(
      instance.impl_internal_space_instance()->maca_memcpy_async_wrapper(
          dst, src, n, macaMemcpyDefault));
}

void DeepCopyAsyncMaca(void* dst, void const* src, size_t n) {
  macaStream_t s = get_deep_copy_stream();
  KOKKOS_IMPL_MACA_SAFE_CALL(macaMemcpyAsync(dst, src, n, macaMemcpyDefault, s));
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Maca>(
      "Kokkos::Impl::DeepCopyAsyncMaca: Post Deep Copy Fence on Deep-Copy "
      "stream",
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          DeepCopyResourceSynchronization,
      [&]() { KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamSynchronize(s)); });
}
}  // namespace Impl
}  // namespace Kokkos
