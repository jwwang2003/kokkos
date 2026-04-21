/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_SharedAllocationRecord.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:51:10 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_SHARED_ALLOCATION_RECORD_HPP
#define KOKKOS_MACA_SHARED_ALLOCATION_RECORD_HPP

#include <Maca/Kokkos_Maca_Space.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_SPECIALIZATION(
    Kokkos::MacaSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::MacaHostPinnedSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(Kokkos::MacaManagedSpace);

#endif
