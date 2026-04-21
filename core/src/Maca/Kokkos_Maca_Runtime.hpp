/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_Runtime.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_RUNTIME_HPP
#define KOKKOS_MACA_RUNTIME_HPP

#if __has_include(<mcr/mc_runtime_api.h>)
#include <mcr/mc_runtime_api.h>
#include <mcr/mc_runtime_api_template_wrapper.h>
#elif __has_include(<mc/mc_runtime_api.h>)
#include <mc/mc_runtime_api.h>
#include <mc/mc_runtime_api_template_wrapper.h>
#else
#error "Unable to find MACA runtime headers (expected mc/ or mcr/ include layout)"
#endif

#define MACA_SYMBOL MC_SYMBOL
#define MACA_VERSION MACART_VERSION
#define MACA_VERSION_MAJOR (MACA_VERSION / 1000)
#define MACA_VERSION_MINOR ((MACA_VERSION % 1000) / 10)
#define MACA_VERSION_PATCH (MACA_VERSION % 10)

using macaError_t          = mcError_t;
using macaStream_t         = mcStream_t;
using macaEvent_t          = mcEvent_t;
using macaGraph_t          = mcGraph_t;
using macaGraphExec_t      = mcGraphExec_t;
using macaGraphNode_t      = mcGraphNode_t;
using macaDeviceProp_t     = mcDeviceProp_t;
using macaFuncAttributes   = mcFuncAttributes;
using macaFuncAttribute    = mcFuncAttribute;
using macaFuncCache_t      = mcFuncCache_t;
using macaMemcpyKind       = mcMemcpyKind;
using macaKernelNodeParams = mcKernelNodeParams;
using macaHostNodeParams   = mcHostNodeParams;
using macaUUID_t           = mcUuid_t;
using macaMemPool_t        = mcMemPool_t;

#define macaSuccess mcSuccess

#define macaErrorInvalidValue mcErrorInvalidValue
#define macaErrorOutOfMemory mcErrorMemoryAllocation
#define macaErrorInitializationError mcErrorInitializationError
#define macaErrorDeinitialized mcErrorDeinitialized
#define macaErrorInvalidConfiguration mcErrorInvalidConfiguration
#define macaErrorInvalidSymbol mcErrorInvalidSymbol
#define macaErrorInvalidDevicePointer mcErrorInvalidDevicePointer
#define macaErrorInvalidMemcpyDirection mcErrorInvalidMemcpyDirection
#define macaErrorInsufficientDriver mcErrorInsufficientDriver
#define macaErrorMissingConfiguration mcErrorMissingConfiguration
#define macaErrorPriorLaunchFailure mcErrorPriorLaunchFailure
#define macaErrorInvalidDeviceFunction mcErrorInvalidDeviceFunction
#define macaErrorNoDevice mcErrorNoDevice
#define macaErrorInvalidDevice mcErrorInvalidDevice
#define macaErrorInvalidContext mcErrorDeviceUninitialized
#define macaErrorNoBinaryForGpu mcErrorNoKernelImageForDevice
#define macaErrorInvalidSource mcErrorInvalidSource
#define macaErrorIllegalState mcErrorIllegalState
#define macaErrorNotFound mcErrorSymbolNotFound
#define macaErrorIllegalAddress mcErrorIllegalAddress
#define macaErrorLaunchOutOfResources mcErrorLaunchOutOfResources
#define macaErrorLaunchTimeOut mcErrorLaunchTimeout
#define macaErrorAssert mcErrorAssert
#define macaErrorLaunchFailure mcErrorLaunchFailure
#define macaErrorNotSupported mcErrorNotSupported
#define macaErrorStreamCaptureUnsupported mcErrorStreamCaptureUnsupported
#define macaErrorCapturedEvent mcErrorCapturedEvent
#define macaErrorGraphExecUpdateFailure mcErrorGraphExecUpdateFailure
#define macaErrorUnknown mcErrorUnknown

#define macaEventDisableTiming mcEventDisableTiming

#define macaFuncAttributePreferredSharedMemoryCarveout \
  mcFuncAttributePreferredSharedMemoryCarveout

#define macaFuncCachePreferShared mcFuncCachePreferShared

#define macaHostMallocDefault mcMallocHostDefault

#define macaMemcpyDefault mcMemcpyDefault
#define macaMemcpyHostToDevice mcMemcpyHostToDevice

#define macaDeviceAttributeManagedMemory mcDeviceAttributeManagedMemory
#define macaDeviceAttributePageableMemoryAccess \
  mcDeviceAttributePageableMemoryAccess

#define macaMemAdviseSetCoarseGrain mcMemAdviseSetCoarseGrain
#define macaMemAdviseUnsetCoarseGrain mcMemAdviseUnsetCoarseGrain

#define macaStreamCaptureModeGlobal mcStreamCaptureModeGlobal

#define macaDeviceGetAttribute mcDeviceGetAttribute
#define macaDeviceSynchronize mcDeviceSynchronize

#define macaEventCreateWithFlags mcEventCreateWithFlags
#define macaEventDestroy mcEventDestroy
#define macaEventRecord mcEventRecord
#define macaEventSynchronize mcEventSynchronize

#define macaFree mcFree
#define macaFreeAsync mcFreeAsync

#define macaFuncGetAttributes mcFuncGetAttributes
#define macaFuncSetAttribute mcFuncSetAttribute
#define macaFuncSetCacheConfig mcFuncSetCacheConfig

#define macaGetDeviceProperties mcGetDeviceProperties
#define macaGetLastError mcGetLastError

#define macaGraphAddChildGraphNode mcGraphAddChildGraphNode
#define macaGraphAddDependencies mcGraphAddDependencies
#define macaGraphAddEmptyNode mcGraphAddEmptyNode
#define macaGraphAddHostNode mcGraphAddHostNode
#define macaGraphAddKernelNode mcGraphAddKernelNode
#define macaGraphCreate mcGraphCreate
#define macaGraphDestroy mcGraphDestroy
#define macaGraphExecDestroy mcGraphExecDestroy
#define macaGraphInstantiate mcGraphInstantiate
#define macaGraphLaunch mcGraphLaunch

#define macaHostFree mcFreeHost
#define macaHostMalloc mcMallocHost

#define macaInit mcInit
#define macaMalloc mcMalloc
#define macaMallocAsync mcMallocAsync
#define macaMallocManaged mcMallocManaged

#define macaMemAdvise mcMemAdvise

#define macaGetDevice mcGetDevice
#define macaGetDeviceCount mcGetDeviceCount
#define macaMemcpyAsync mcMemcpyAsync
#define macaMemcpyToSymbolAsync mcMemcpyToSymbolAsync

#define macaMemset mcMemset
#define macaMemsetAsync mcMemsetAsync

#define macaOccupancyMaxActiveBlocksPerMultiprocessor \
  mcOccupancyMaxActiveBlocksPerMultiprocessor
#define macaOccupancyMaxPotentialBlockSize mcOccupancyMaxPotentialBlockSize

#define macaSetDevice mcSetDevice

#define macaStreamBeginCapture mcStreamBeginCapture
#define macaStreamCreate mcStreamCreate
#define macaStreamDestroy mcStreamDestroy
#define macaStreamEndCapture mcStreamEndCapture
#define macaStreamGetDevice mcStreamGetDevice
#define macaStreamSynchronize mcStreamSynchronize

#endif
