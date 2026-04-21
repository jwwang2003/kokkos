/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_Abort.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_ABORT_HPP
#define KOKKOS_MACA_ABORT_HPP

#include <Kokkos_Macros.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

namespace Kokkos {
namespace Impl {

// The two keywords below are not contradictory. `noinline` is a
// directive to the optimizer.
[[noreturn]] __device__ __attribute__((noinline)) inline void maca_abort(
    char const *msg) {
  const char empty[] = "";
  __assert_fail(msg, empty, 0, empty);
  // This loop is never executed. It's intended to suppress warnings that the
  // function returns, even though it does not. This is necessary because
  // abort() is not marked as [[noreturn]], even though it does not return.
  while (true)
    ;
}

}  // namespace Impl
}  // namespace Kokkos

#endif
