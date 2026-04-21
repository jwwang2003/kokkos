/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Adapt_MACA.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Mon 20 Apr 2026 12:12:14 AM CST
================================================================*/

/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_ADAPT_MACA_HPP_
#define DESUL_ATOMICS_ADAPT_MACA_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

template <class To, class From>
DESUL_FORCEINLINE_FUNCTION To maca_atomic_bit_cast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  union {
    From from;
    To to;
  } bits = {value};
  return bits.to;
}

template <class MemoryOrder>
struct MACAMemoryOrder;

template <>
struct MACAMemoryOrder<MemoryOrderRelaxed> {
  static constexpr int value = __ATOMIC_RELAXED;
};

template <>
struct MACAMemoryOrder<MemoryOrderAcquire> {
  static constexpr int value = __ATOMIC_ACQUIRE;
};

template <>
struct MACAMemoryOrder<MemoryOrderRelease> {
  static constexpr int value = __ATOMIC_RELEASE;
};

template <>
struct MACAMemoryOrder<MemoryOrderAcqRel> {
  static constexpr int value = __ATOMIC_ACQ_REL;
};

template <>
struct MACAMemoryOrder<MemoryOrderSeqCst> {
  static constexpr int value = __ATOMIC_SEQ_CST;
};

template <class MemoryScope>
struct MACAMemoryScope;

template <>
struct MACAMemoryScope<MemoryScopeSystem> {
  static constexpr memory_scope value = memory_scope_system;
};

template <>
struct MACAMemoryScope<MemoryScopeNode> {
  static constexpr memory_scope value = memory_scope_system;
};

template <>
struct MACAMemoryScope<MemoryScopeDevice> {
  static constexpr memory_scope value = memory_scope_device;
};

template <>
struct MACAMemoryScope<MemoryScopeCore> {
  static constexpr memory_scope value = memory_scope_block;
};

template <>
struct MACAMemoryScope<MemoryScopeCaller> {
  static constexpr memory_scope value = memory_scope_single_thread;
};

}  // namespace Impl
}  // namespace desul

#endif
