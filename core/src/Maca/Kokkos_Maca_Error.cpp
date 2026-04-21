/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_Error.cpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <impl/Kokkos_Error.hpp>

#include <sstream>

namespace Kokkos {
namespace Impl {
void maca_internal_error_throw(macaError_t e, const char *name, const char *file,
                              const int line) {
  std::ostringstream out;
  out << name << " error( " << mcGetErrorName(e)
      << "): " << mcGetErrorString(e);
  if (file) {
    out << " " << file << ":" << line;
  }
  throw_runtime_exception(out.str());
}

void maca_internal_error_abort(macaError_t e, const char *name, const char *file,
                              const int line) {
  std::ostringstream out;
  out << name << " error( " << mcGetErrorName(e)
      << "): " << mcGetErrorString(e);
  if (file) {
    out << " " << file << ":" << line;
  }
  host_abort(out.str().c_str());
}
}  // namespace Impl
}  // namespace Kokkos
