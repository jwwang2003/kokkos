// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Maca/Kokkos_Maca.hpp>
#include <Maca/Kokkos_Maca_DeepCopy.hpp>
#include <Maca/Kokkos_Maca_SharedAllocationRecord.hpp>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::MacaSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::MacaHostPinnedSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::MacaManagedSpace);
