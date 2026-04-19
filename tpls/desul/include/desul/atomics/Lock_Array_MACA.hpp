/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_ARRAY_MACA_HPP_
#define DESUL_ATOMICS_LOCK_ARRAY_MACA_HPP_

#if __has_include(<mcr/mc_runtime_api.h>)
#include <mcr/mc_runtime_api.h>
#include <mcr/mc_runtime_api_template_wrapper.h>
#elif __has_include(<mc/mc_runtime_api.h>)
#include <mc/mc_runtime_api.h>
#include <mc/mc_runtime_api_template_wrapper.h>
#else
#error "Unable to find MACA runtime headers (expected mc/ or mcr/ include layout)"
#endif

#include <cstdint>

#include <desul/atomics/Common.hpp>
#include <desul/atomics/Macros.hpp>

namespace desul {
namespace Impl {

DESUL_IMPL_EXPORT extern int32_t* MACA_SPACE_ATOMIC_LOCKS_DEVICE_h;
DESUL_IMPL_EXPORT extern int32_t* MACA_SPACE_ATOMIC_LOCKS_NODE_h;

template <typename /*AlwaysInt*/ = int>
void init_lock_arrays_maca();

template <typename /*AlwaysInt*/ = int>
void finalize_lock_arrays_maca();

#ifdef DESUL_ATOMICS_ENABLE_MACA_SEPARABLE_COMPILATION
extern
#endif
    __device__ __constant__ int32_t* MACA_SPACE_ATOMIC_LOCKS_DEVICE;

#ifdef DESUL_ATOMICS_ENABLE_MACA_SEPARABLE_COMPILATION
extern
#endif
    __device__ __constant__ int32_t* MACA_SPACE_ATOMIC_LOCKS_NODE;

#define MACA_SPACE_ATOMIC_MASK 0x7FFF

__device__ inline int32_t* get_lock_array_maca(desul::MemoryScopeCore) {
  return MACA_SPACE_ATOMIC_LOCKS_DEVICE;
}

__device__ inline int32_t* get_lock_array_maca(desul::MemoryScopeDevice) {
  return MACA_SPACE_ATOMIC_LOCKS_DEVICE;
}

__device__ inline int32_t* get_lock_array_maca(desul::MemoryScopeNode) {
  return MACA_SPACE_ATOMIC_LOCKS_NODE;
}

__device__ inline int32_t* get_lock_array_maca(desul::MemoryScopeSystem) {
  return MACA_SPACE_ATOMIC_LOCKS_NODE;
}

template <class MemoryScope>
__device__ inline bool lock_address_maca(void* ptr, MemoryScope scope) {
  size_t offset = size_t(ptr);
  offset        = offset >> 2;
  offset        = offset & MACA_SPACE_ATOMIC_MASK;
  return (0 == atomicCAS(&get_lock_array_maca(scope)[offset], 0, 1));
}

template <class MemoryScope>
__device__ inline void unlock_address_maca(void* ptr, MemoryScope scope) {
  size_t offset = size_t(ptr);
  offset        = offset >> 2;
  offset        = offset & MACA_SPACE_ATOMIC_MASK;
  __threadfence();
  get_lock_array_maca(scope)[offset] = 0;
}

#ifdef DESUL_ATOMICS_ENABLE_MACA_SEPARABLE_COMPILATION
inline
#else
inline static
#endif
    void
    copy_maca_lock_arrays_to_device() {
  static bool once = []() {
    (void)mcMemcpyToSymbol(MC_SYMBOL(MACA_SPACE_ATOMIC_LOCKS_DEVICE),
                           &MACA_SPACE_ATOMIC_LOCKS_DEVICE_h,
                           sizeof(int32_t*));
    (void)mcMemcpyToSymbol(MC_SYMBOL(MACA_SPACE_ATOMIC_LOCKS_NODE),
                           &MACA_SPACE_ATOMIC_LOCKS_NODE_h,
                           sizeof(int32_t*));
    return true;
  }();
  (void)once;
}

}  // namespace Impl

#ifdef DESUL_ATOMICS_ENABLE_MACA_SEPARABLE_COMPILATION
inline void ensure_maca_lock_arrays_on_device() {}
#else
static inline void ensure_maca_lock_arrays_on_device() {
  Impl::copy_maca_lock_arrays_to_device();
}
#endif

}  // namespace desul

#endif
