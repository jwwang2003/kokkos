/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Compare_Exchange_MACA.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Mon 20 Apr 2026 12:12:14 AM CST
================================================================*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_MACA_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_MACA_HPP_

#include <desul/atomics/Adapt_MACA.hpp>
#include <desul/atomics/Common.hpp>
#include <desul/atomics/Lock_Array_MACA.hpp>
#include <desul/atomics/Thread_Fence_MACA.hpp>

#include <type_traits>

namespace desul {
namespace Impl {

template <class T>
inline constexpr bool device_atomic_always_lock_free<T, void> =
    (std::is_trivially_copyable<T>::value) &&
    ((sizeof(T) == 1) || (sizeof(T) == 2) || (sizeof(T) == 4) ||
     (sizeof(T) == 8));

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2, T>
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  __atomic_compare_exchange(dest, &compare, &value, false, __ATOMIC_RELAXED,
                            __ATOMIC_RELAXED);
  return compare;
}

template <class T, class MemoryScope>
__device__ std::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2, T>
device_atomic_exchange(T* const dest, T value, MemoryOrderRelaxed, MemoryScope) {
  T old;
  __atomic_exchange(dest, &value, &old, __ATOMIC_RELAXED);
  return old;
}

#include <desul/atomics/maca/MACA_asm_exchange.hpp>

#define DESUL_IMPL_MACA_DEVICE_ATOMIC_EXCHANGE_ORDER(OP)                            \
  template <class T, class MemoryScope>                                             \
  __device__ std::enable_if_t<device_atomic_always_lock_free<T>, T>                 \
  device_atomic_compare_exchange(                                                   \
      T* const dest, T compare, T value, OP, MemoryScope scope) {                   \
    if constexpr (std::is_same_v<OP, MemoryOrderAcquire> ||                         \
                  std::is_same_v<OP, MemoryOrderAcqRel> ||                          \
                  std::is_same_v<OP, MemoryOrderSeqCst>)                            \
      device_atomic_thread_fence(MemoryOrderAcquire(), scope);                      \
    T return_val = device_atomic_compare_exchange(                                  \
        dest, compare, value, MemoryOrderRelaxed(), scope);                         \
    if constexpr (std::is_same_v<OP, MemoryOrderRelease> ||                         \
                  std::is_same_v<OP, MemoryOrderAcqRel> ||                          \
                  std::is_same_v<OP, MemoryOrderSeqCst>)                            \
      device_atomic_thread_fence(MemoryOrderRelease(), scope);                      \
    return return_val;                                                              \
  }                                                                                 \
  template <class T, class MemoryScope>                                             \
  __device__ std::enable_if_t<device_atomic_always_lock_free<T>, T>                 \
  device_atomic_exchange(T* const dest, T value, OP, MemoryScope scope) {           \
    if constexpr (std::is_same_v<OP, MemoryOrderAcquire> ||                         \
                  std::is_same_v<OP, MemoryOrderAcqRel> ||                          \
                  std::is_same_v<OP, MemoryOrderSeqCst>)                            \
      device_atomic_thread_fence(MemoryOrderAcquire(), scope);                      \
    T return_val =                                                                  \
        device_atomic_exchange(dest, value, MemoryOrderRelaxed(), scope);           \
    if constexpr (std::is_same_v<OP, MemoryOrderRelease> ||                         \
                  std::is_same_v<OP, MemoryOrderAcqRel> ||                          \
                  std::is_same_v<OP, MemoryOrderSeqCst>)                            \
      device_atomic_thread_fence(MemoryOrderRelease(), scope);                      \
    return return_val;                                                              \
  }

DESUL_IMPL_MACA_DEVICE_ATOMIC_EXCHANGE_ORDER(MemoryOrderRelease)
DESUL_IMPL_MACA_DEVICE_ATOMIC_EXCHANGE_ORDER(MemoryOrderAcquire)
DESUL_IMPL_MACA_DEVICE_ATOMIC_EXCHANGE_ORDER(MemoryOrderAcqRel)
DESUL_IMPL_MACA_DEVICE_ATOMIC_EXCHANGE_ORDER(MemoryOrderSeqCst)

#undef DESUL_IMPL_MACA_DEVICE_ATOMIC_EXCHANGE_ORDER

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<!device_atomic_always_lock_free<T>, T>
device_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  T return_val{};
  int done                   = 0;
  unsigned long long active  = __activemask();
  unsigned long long retired = 0;
  while (active != retired) {
    if (!done && lock_address_maca((void*)dest, scope)) {
      if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
      device_atomic_thread_fence(MemoryOrderAcquire(), scope);
      return_val = *dest;
      if (return_val == compare) {
        *dest = value;
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
      }
      unlock_address_maca((void*)dest, scope);
      done = 1;
    }
    retired = __ballot_sync(active, done ? 1 : 0);
  }
  return return_val;
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<!device_atomic_always_lock_free<T>, T>
device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope scope) {
  T return_val{};
  int done                   = 0;
  unsigned long long active  = __activemask();
  unsigned long long retired = 0;
  while (active != retired) {
    if (!done && lock_address_maca((void*)dest, scope)) {
      if (std::is_same<MemoryOrder, MemoryOrderSeqCst>::value)
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
      device_atomic_thread_fence(MemoryOrderAcquire(), scope);
      return_val = *dest;
      *dest      = value;
      device_atomic_thread_fence(MemoryOrderRelease(), scope);
      unlock_address_maca((void*)dest, scope);
      done = 1;
    }
    retired = __ballot_sync(active, done ? 1 : 0);
  }
  return return_val;
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<device_atomic_always_lock_free<T>, T> device_atomic_load(
    const T* const dest, MemoryOrder order, MemoryScope) {
  T value;
  __atomic_load(const_cast<T*>(dest), &value, MACAMemoryOrder<MemoryOrder>::value);
  return value;
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<device_atomic_always_lock_free<T>, void>
device_atomic_store(T* const dest, const T val, MemoryOrder order, MemoryScope) {
  T value = val;
  __atomic_store(dest, &value, MACAMemoryOrder<MemoryOrder>::value);
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<!device_atomic_always_lock_free<T>, T> device_atomic_load(
    const T* const dest, MemoryOrder, MemoryScope scope) {
  T return_val{};
  int done                   = 0;
  unsigned long long active  = __activemask();
  unsigned long long retired = 0;
  auto* const mutable_dest   = const_cast<T*>(dest);
  while (active != retired) {
    if (!done && lock_address_maca((void*)mutable_dest, scope)) {
      device_atomic_thread_fence(MemoryOrderAcquire(), scope);
      return_val = *mutable_dest;
      unlock_address_maca((void*)mutable_dest, scope);
      done = 1;
    }
    retired = __ballot_sync(active, done ? 1 : 0);
  }
  return return_val;
}

template <class T, class MemoryOrder, class MemoryScope>
__device__ std::enable_if_t<!device_atomic_always_lock_free<T>, void>
device_atomic_store(T* const dest, const T val, MemoryOrder, MemoryScope scope) {
  int done                   = 0;
  unsigned long long active  = __activemask();
  unsigned long long retired = 0;
  while (active != retired) {
    if (!done && lock_address_maca((void*)dest, scope)) {
      *dest = val;
      device_atomic_thread_fence(MemoryOrderRelease(), scope);
      unlock_address_maca((void*)dest, scope);
      done = 1;
    }
    retired = __ballot_sync(active, done ? 1 : 0);
  }
}

}  // namespace Impl
}  // namespace desul

#endif
