// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOS_MACA_ZEROMEMSET_HPP
#define KOKKOS_MACA_ZEROMEMSET_HPP

#include <Kokkos_Macros.hpp>
#include <Maca/Kokkos_Maca.hpp>
#include <Maca/Kokkos_Maca_Instance.hpp>
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos {
namespace Impl {

// macaMemsetAsync sets the first `cnt` bytes of `dst` to the provided value
void zero_with_maca_kernel(const Maca& exec_space, void* dst, size_t cnt);

template <>
struct ZeroMemset<Maca> {
  ZeroMemset(const Maca& exec_space, void* dst, size_t cnt) {
    // We allow user on an AMD APU with unified memory to `malloc` and wrap that
    // in an unmanaged SharedSpace view. In ROCm <= 6.2.1 (and possibly later),
    // macaMemsetAsync on a host-allocated pointer returns an invalid value
    // error, but accessing the data via a GPU kernel works as long as xnack is
    // present and enabled (HSA_XNACK=1)
#if defined(KOKKOS_IMPL_MACA_UNIFIED_MEMORY)
    zero_with_maca_kernel(exec_space, dst, cnt);
#else
    KOKKOS_IMPL_MACA_SAFE_CALL(
        exec_space.impl_internal_space_instance()->maca_memset_async_wrapper(
            dst, 0, cnt));
#endif
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // !defined(KOKKOS_MACA_ZEROMEMSET_HPP)
