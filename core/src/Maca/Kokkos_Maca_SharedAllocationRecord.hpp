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
