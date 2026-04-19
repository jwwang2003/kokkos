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

}  // namespace Impl
}  // namespace desul

#endif
