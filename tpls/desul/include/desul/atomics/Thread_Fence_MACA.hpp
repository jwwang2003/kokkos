/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_MACA_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_MACA_HPP_

#include <desul/atomics/Adapt_MACA.hpp>

namespace desul {
namespace Impl {

#define DESUL_IMPL_MACA_DEVICE_THREAD_FENCE(MEMORY_ORDER)                             \
  template <class MemoryScope>                                                       \
  inline __device__ void device_atomic_thread_fence(MEMORY_ORDER, MemoryScope) {     \
    __atomic_thread_fence(MACAMemoryOrder<MEMORY_ORDER>::value);                     \
  }

DESUL_IMPL_MACA_DEVICE_THREAD_FENCE(MemoryOrderRelease)
DESUL_IMPL_MACA_DEVICE_THREAD_FENCE(MemoryOrderAcquire)
DESUL_IMPL_MACA_DEVICE_THREAD_FENCE(MemoryOrderAcqRel)
DESUL_IMPL_MACA_DEVICE_THREAD_FENCE(MemoryOrderSeqCst)

#undef DESUL_IMPL_MACA_DEVICE_THREAD_FENCE

}  // namespace Impl
}  // namespace desul

#endif
