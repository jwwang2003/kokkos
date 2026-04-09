// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SETUP_MACA_HPP_
#define KOKKOS_SETUP_MACA_HPP_

#if defined(KOKKOS_ENABLE_MACA)

#include <Maca/Kokkos_Maca_Runtime.hpp>

#define KOKKOS_LAMBDA [=] __host__ __device__
#define KOKKOS_CLASS_LAMBDA [ =, *this ] __host__ __device__

#define KOKKOS_DEDUCTION_GUIDE __host__ __device__

#define KOKKOS_IMPL_FORCEINLINE_FUNCTION __device__ __host__ __forceinline__
#define KOKKOS_IMPL_INLINE_FUNCTION __device__ __host__ inline
#define KOKKOS_IMPL_FUNCTION __device__ __host__
#define KOKKOS_IMPL_HOST_FUNCTION __host__
#define KOKKOS_IMPL_DEVICE_FUNCTION __device__

#ifdef KOKKOS_ENABLE_MACA_RELOCATABLE_DEVICE_CODE
#define KOKKOS_IMPL_RELOCATABLE_FUNCTION __device__ __host__
#else
#define KOKKOS_IMPL_RELOCATABLE_FUNCTION @"KOKKOS_RELOCATABLE_FUNCTION requires Kokkos_ENABLE_MACA_RELOCATABLE_DEVICE_CODE=ON"
#endif

#ifdef KOKKOS_ENABLE_IMPL_MACA_UNIFIED_MEMORY
#define KOKKOS_IMPL_MACA_UNIFIED_MEMORY
#endif

#endif

#endif
