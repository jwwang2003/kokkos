// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_RUNTIME_HPP
#define KOKKOS_MACA_RUNTIME_HPP

#if __has_include(<mcr/hip_to_maca_adaptor.h>)
#include <mcr/hip_to_maca_adaptor.h>
#elif __has_include(<mc/hip_to_maca_adaptor.h>)
#include <mc/hip_to_maca_adaptor.h>
#elif __has_include(<mc/mc_runtime.h>)
#include <mc/mc_runtime.h>
#include <mc/mc_runtime_api.h>
#elif __has_include(<mcr/mc_runtime.h>)
#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#else
#error "Unable to find MACA runtime headers (expected mc/ or mcr/ include layout)"
#endif

#ifndef hipUUID_t
using hipUUID_t = mcUuid_t;
#endif

#ifndef hipMemPool_t
using hipMemPool_t = mcMemPool_t;
#endif

#ifndef hipMallocAsync
#define hipMallocAsync mcMallocAsync
#endif

#ifndef hipFreeAsync
#define hipFreeAsync mcFreeAsync
#endif

#ifndef hipStreamGetDevice
#define hipStreamGetDevice mcStreamGetDevice
#endif

#endif
