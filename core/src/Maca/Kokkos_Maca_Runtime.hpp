// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_RUNTIME_HPP
#define KOKKOS_MACA_RUNTIME_HPP

#include <mc/mc_runtime.h>
#include <mc/mc_runtime_api.h>

// MXMACA is close enough to the Maca runtime surface that the backend can
// reuse the Maca implementation structure while keeping a distinct public
// execution space in Kokkos.
using hipError_t      = mcError_t;
using hipStream_t     = mcStream_t;
using hipEvent_t      = mcEvent_t;
using hipDeviceProp_t = mcDeviceProp_t;
using hipGraph_t      = mcGraph_t;
using hipGraphExec_t  = mcGraphExec_t;
using hipGraphNode_t  = mcGraphNode_t;
using hipLaunchParams = mcLaunchParams;
using hipUUID_t       = mcUuid_t;
using hipMemPool_t    = mcMemPool_t;

#define hipSuccess mcSuccess
#define hipErrorInvalidValue mcErrorInvalidValue
#define hipErrorOutOfMemory mcErrorOutOfMemory
#define hipErrorInitializationError mcErrorInitializationError
#define hipErrorDeinitialized mcErrorDeinitialized
#define hipErrorInvalidConfiguration mcErrorInvalidConfiguration
#define hipErrorInvalidSymbol mcErrorInvalidSymbol
#define hipErrorInvalidDevicePointer mcErrorInvalidDevicePointer
#define hipErrorInvalidMemcpyDirection mcErrorInvalidMemcpyDirection
#define hipErrorInsufficientDriver mcErrorInsufficientDriver
#define hipErrorMissingConfiguration mcErrorMissingConfiguration
#define hipErrorPriorLaunchFailure mcErrorPriorLaunchFailure
#define hipErrorInvalidDeviceFunction mcErrorInvalidDeviceFunction
#define hipErrorNoDevice mcErrorNoDevice
#define hipErrorInvalidDevice mcErrorInvalidDevice
#define hipErrorInvalidContext mcErrorInvalidContext
#define hipErrorNoBinaryForGpu mcErrorNoBinaryForGpu
#define hipErrorInvalidSource mcErrorInvalidSource
#define hipErrorIllegalState mcErrorIllegalState
#define hipErrorNotFound mcErrorNotFound
#define hipErrorIllegalAddress mcErrorIllegalAddress
#define hipErrorLaunchOutOfResources mcErrorLaunchOutOfResources
#define hipErrorLaunchTimeOut mcErrorLaunchTimeOut
#define hipErrorAssert mcErrorAssert
#define hipErrorLaunchFailure mcErrorLaunchFailure
#define hipErrorNotSupported mcErrorNotSupported
#define hipErrorStreamCaptureUnsupported mcErrorStreamCaptureUnsupported
#define hipErrorCapturedEvent mcErrorCapturedEvent
#define hipErrorGraphExecUpdateFailure mcErrorGraphExecUpdateFailure
#define hipErrorUnknown mcErrorUnknown

#define hipMemcpyHostToDevice mcMemcpyHostToDevice
#define hipMemcpyDeviceToHost mcMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice mcMemcpyDeviceToDevice
#define hipMemcpyDefault mcMemcpyDefault

#define hipMemAttachGlobal mcMemAttachGlobal
#define hipMemAdviseSetPreferredLocation mcMemAdviseSetPreferredLocation
#define hipMemAdviseSetCoarseGrain mcMemAdviseSetCoarseGrain
#define hipHostMallocNonCoherent 0
#define hipHostMallocMapped mcHostAllocMapped
#define hipHostMallocNumaUser mcHostAllocDefault
#define hipDeviceMallocFinegrained 0

#define hipEventDisableTiming mcEventDisableTiming
#define hipEventBlockingSync mcEventBlockingSync
#define hipEventWaitDefault 0
#define hipStreamNonBlocking mcStreamNonBlocking
#define hipStreamDefault mcStreamDefault
#define hipStreamPerThread mcStreamPerThread
#define hipStreamLegacy mcStreamLegacy
#define hipStreamCaptureModeGlobal mcStreamCaptureModeGlobal

#define hipDeviceAttributeManagedMemory mcDevAttrManagedMemory
#define hipDeviceAttributeDirectManagedMemAccessFromHost mcDevAttrDirectManagedMemAccessFromHost
#define hipDeviceAttributeCooperativeLaunch mcDevAttrCooperativeLaunch
#define hipDeviceAttributeCooperativeMultiDeviceLaunch mcDevAttrCooperativeMultiDeviceLaunch

#define hipGetErrorString mcGetErrorString
#define hipGetDevice mcGetDevice
#define hipSetDevice mcSetDevice
#define hipGetDeviceCount mcGetDeviceCount
#define hipGetDeviceProperties mcGetDeviceProperties
#define hipDeviceGetAttribute mcDeviceGetAttribute
#define hipDeviceSynchronize mcDeviceSynchronize
#define hipDeviceReset mcDeviceReset
#define hipDeviceCanAccessPeer mcDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess mcDeviceEnableAccessPeer
#define hipDeviceDisablePeerAccess mcDeviceDisableAccessPeer
#define hipMalloc mcMalloc
#define hipFree mcFree
#define hipMallocAsync mcMallocAsync
#define hipFreeAsync mcFreeAsync
#define hipMallocManaged mcMallocManaged
#define hipHostMalloc mcMallocHost
#define hipHostFree mcFreeHost
#define hipMemcpy mcMemcpy
#define hipMemcpyAsync mcMemcpyAsync
#define hipMemcpyPeerAsync mcMemcpyPeerAsync
#define hipMemcpyHtoD mcMemcpyHtoD
#define hipMemcpyDtoH mcMemcpyDtoH
#define hipMemcpyHtoDAsync mcMemcpyHtoDAsync
#define hipMemcpyDtoHAsync mcMemcpyDtoHAsync
#define hipMemset mcMemset
#define hipMemsetAsync mcMemsetAsync
#define hipMemAdvise mcMemAdvise
#define hipGetLastError mcGetLastError
#define hipPeekAtLastError mcPeekAtLastError
#define hipStreamCreate mcStreamCreate
#define hipStreamCreateWithFlags mcStreamCreateWithFlags
#define hipStreamCreateWithPriority mcStreamCreateWithPriority
#define hipStreamDestroy mcStreamDestroy
#define hipStreamSynchronize mcStreamSynchronize
#define hipStreamWaitEvent mcStreamWaitEvent
#define hipStreamQuery mcStreamQuery
#define hipStreamGetDevice mcStreamGetDevice
#define hipStreamBeginCapture mcStreamBeginCapture
#define hipStreamEndCapture mcStreamEndCapture
#define hipEventCreate mcEventCreate
#define hipEventCreateWithFlags mcEventCreateWithFlags
#define hipEventDestroy mcEventDestroy
#define hipEventRecord mcEventRecord
#define hipEventSynchronize mcEventSynchronize
#define hipGraphCreate mcGraphCreate
#define hipGraphDestroy mcGraphDestroy
#define hipGraphExecDestroy mcGraphExecDestroy
#define hipGraphInstantiate mcGraphInstantiate
#define hipGraphLaunch mcGraphLaunch
#define hipGraphExecUpdate mcGraphExecUpdate
#define hipGraphAddDependencies mcGraphAddDependencies
#define hipGraphAddEmptyNode mcGraphAddEmptyNode
#define hipGraphAddKernelNode mcGraphAddKernelNode
#define hipGraphAddMemcpyNode1D mcGraphAddMemcpyNode1D
#define hipExtLaunchMultiKernelMultiDevice mcExtLaunchMultiKernelMultiDevice
#define hipLaunchCooperativeKernelMultiDevice mcLaunchCooperativeKernelMultiDevice
#define hipFuncGetAttributes mcFuncGetAttributes

#endif
