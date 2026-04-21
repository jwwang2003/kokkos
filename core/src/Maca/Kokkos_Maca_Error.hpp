/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_Error.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_ERROR_HPP
#define KOKKOS_MACA_ERROR_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

namespace Kokkos {
namespace Impl {

void maca_internal_error_throw(macaError_t e, const char* name,
                              const char* file = nullptr, const int line = 0);

void maca_internal_error_abort(macaError_t e, const char* name,
                              const char* file = nullptr, const int line = 0);

[[nodiscard]] inline bool maca_internal_error_is_fatal(macaError_t e) {
  switch (e) {
    case macaErrorIllegalAddress:
    case macaErrorAssert:
    case macaErrorLaunchFailure:
    case static_cast<macaError_t>(mcErrorHardwareStackError):
    case static_cast<macaError_t>(mcErrorIllegalInstruction):
    case static_cast<macaError_t>(mcErrorMisalignedAddress):
    case static_cast<macaError_t>(mcErrorInvalidAddressSpace):
    case static_cast<macaError_t>(mcErrorInvalidPc): return true;
    default: return false;
  }
}

inline void maca_internal_safe_call(macaError_t e, const char* name,
                                   const char* file = nullptr,
                                   const int line   = 0) {
  if (e == macaSuccess) return;
  if (maca_internal_error_is_fatal(e))
    maca_internal_error_abort(e, name, file, line);
  else
    maca_internal_error_throw(e, name, file, line);
}

}  // namespace Impl
}  // namespace Kokkos

#define KOKKOS_IMPL_MACA_SAFE_CALL(call) \
  Kokkos::Impl::maca_internal_safe_call(call, #call, __FILE__, __LINE__)

#endif
