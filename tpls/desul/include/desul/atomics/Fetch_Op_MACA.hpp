/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FETCH_OP_MACA_HPP_
#define DESUL_ATOMICS_FETCH_OP_MACA_HPP_

#include <desul/atomics/Thread_Fence_MACA.hpp>

namespace desul {
namespace Impl {

#define DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD(TYPE)                                 \
  template <class MemoryScope>                                                        \
  inline __device__ TYPE device_atomic_fetch_add(                                     \
      TYPE* ptr, TYPE val, MemoryOrderRelaxed, MemoryScope) {                         \
    return atomicAdd(ptr, val);                                                       \
  }                                                                                   \
  template <class MemoryScope>                                                        \
  inline __device__ TYPE device_atomic_fetch_sub(                                     \
      TYPE* ptr, TYPE val, MemoryOrderRelaxed, MemoryScope) {                         \
    return atomicAdd(ptr, -val);                                                      \
  }

DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD(int)
DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD(unsigned int)
DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD(unsigned long long)
DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD(float)
DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD(double)

#undef DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ADD

#define DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER(OP, TYPE)                           \
  template <class MemoryScope>                                                        \
  inline __device__ TYPE device_atomic_##OP(                                          \
      TYPE* ptr, TYPE val, MemoryOrderRelease, MemoryScope scope) {                   \
    TYPE return_val = device_atomic_##OP(ptr, val, MemoryOrderRelaxed(), scope);      \
    device_atomic_thread_fence(MemoryOrderRelease(), scope);                          \
    return return_val;                                                                \
  }                                                                                   \
  template <class MemoryScope>                                                        \
  inline __device__ TYPE device_atomic_##OP(                                          \
      TYPE* ptr, TYPE val, MemoryOrderAcquire, MemoryScope scope) {                   \
    device_atomic_thread_fence(MemoryOrderAcquire(), scope);                          \
    return device_atomic_##OP(ptr, val, MemoryOrderRelaxed(), scope);                 \
  }                                                                                   \
  template <class MemoryScope>                                                        \
  inline __device__ TYPE device_atomic_##OP(                                          \
      TYPE* ptr, TYPE val, MemoryOrderAcqRel, MemoryScope scope) {                    \
    device_atomic_thread_fence(MemoryOrderAcquire(), scope);                          \
    TYPE return_val = device_atomic_##OP(ptr, val, MemoryOrderRelaxed(), scope);      \
    device_atomic_thread_fence(MemoryOrderRelease(), scope);                          \
    return return_val;                                                                \
  }                                                                                   \
  template <class MemoryScope>                                                        \
  inline __device__ TYPE device_atomic_##OP(                                          \
      TYPE* ptr, TYPE val, MemoryOrderSeqCst, MemoryScope scope) {                    \
    device_atomic_thread_fence(MemoryOrderAcquire(), scope);                          \
    TYPE return_val = device_atomic_##OP(ptr, val, MemoryOrderRelaxed(), scope);      \
    device_atomic_thread_fence(MemoryOrderRelease(), scope);                          \
    return return_val;                                                                \
  }

#define DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER_ALL(OP) \
  DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER(OP, int) \
  DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER(OP, unsigned int) \
  DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER(OP, unsigned long long) \
  DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER(OP, float) \
  DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER(OP, double)

DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER_ALL(fetch_add)
DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER_ALL(fetch_sub)

#undef DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER_ALL
#undef DESUL_IMPL_MACA_DEVICE_ATOMIC_FETCH_ORDER

}  // namespace Impl
}  // namespace desul

#endif
