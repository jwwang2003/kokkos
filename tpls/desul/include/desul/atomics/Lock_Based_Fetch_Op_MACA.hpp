/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_BASED_FETCH_OP_MACA_HPP_
#define DESUL_ATOMICS_LOCK_BASED_FETCH_OP_MACA_HPP_

#include <desul/atomics/Common.hpp>
#include <desul/atomics/Compare_Exchange_MACA.hpp>
#include <desul/atomics/Lock_Array_MACA.hpp>
#include <desul/atomics/Operator_Function_Objects.hpp>
#include <desul/atomics/Thread_Fence_MACA.hpp>

#include <type_traits>

namespace desul {
namespace Impl {

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          std::enable_if_t<!device_atomic_always_lock_free<T>, int> = 0>
__device__ T device_atomic_fetch_oper(const Oper& op,
                                      T* const dest,
                                      dont_deduce_this_parameter_t<const T> val,
                                      MemoryOrder /*order*/,
                                      MemoryScope scope) {
  T return_val{};
  int done                   = 0;
  unsigned long long active  = __activemask();
  unsigned long long retired = 0;
  while (active != retired) {
    if (!done) {
      if (lock_address_maca((void*)dest, scope)) {
        device_atomic_thread_fence(MemoryOrderAcquire(), scope);
        if constexpr (!std::is_same_v<Oper, _store_fetch_operator<T, const T>>)
          return_val = *dest;
        *dest = op.apply(return_val, val);
        device_atomic_thread_fence(MemoryOrderRelease(), scope);
        unlock_address_maca((void*)dest, scope);
        done = 1;
      }
    }
    retired = __ballot_sync(active, done ? 1 : 0);
  }
  return return_val;
}

}  // namespace Impl
}  // namespace desul

#endif
