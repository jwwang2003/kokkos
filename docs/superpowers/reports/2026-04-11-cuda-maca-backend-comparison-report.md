# CUDA vs MACA Backend Comparison Report

Date: 2026-04-11
Repo: `/home/wjw/workspace/kokkos`
Compared folders:
- `core/src/Cuda`
- `core/src/Maca`

## Executive Summary

The `Maca` backend was implemented by reproducing the same major backend responsibilities as `Cuda`, but routing them through a HIP-shaped runtime model that is adapted to MACA. Structurally, `Maca` looks like a sibling of `Cuda`: it has an execution-space class, memory spaces, instance management, kernel launch code, graph support, reductions, teams, half support, vectorization, unique-token support, and zero-memset support.

However, it is not a pure CUDA-to-MACA API translation. The implementation strategy is:

1. preserve the overall Kokkos GPU backend architecture used by `Cuda`
2. replace the low-level runtime/compiler interface with MACA headers and HIP-style entry points
3. split some CUDA monolithic files into more specialized MACA files
4. add MACA-specific bring-up layers where CUDA has no equivalent, especially runtime adaptation, deep copy plumbing, shared constant-memory coordination, and XNACK/HMM checks

The current tree still shows strong evidence of HIP ancestry inside the MACA backend. This is visible in file names, helper type names, comments, and runtime calls such as `hipSetDevice`, `hipMalloc`, `hipMemcpyAsync`, `hipGraph*`, and `hipStream*`, all included through the MACA runtime adapter.

## Comparison at a Glance

- `core/src/Cuda` contains 26 files.
- `core/src/Maca` contains 41 files.
- MACA has broader file granularity than CUDA.
- MACA adds explicit runtime-adapter files that CUDA does not need.
- MACA splits several CUDA files into separate `ParallelFor`, `ParallelReduce`, and `ParallelScan` files.

## High-Level Folder Mapping

Direct conceptual matches:

| CUDA | MACA | Role |
| --- | --- | --- |
| `Kokkos_Cuda.hpp` | `Kokkos_Maca.hpp` | execution-space public class |
| `Kokkos_CudaSpace.hpp/.cpp` | `Kokkos_Maca_Space.hpp/.cpp` | device, pinned, managed memory spaces |
| `Kokkos_Cuda_Instance.hpp/.cpp` | `Kokkos_Maca_Instance.hpp/.cpp` | device/stream instance state and wrappers |
| `Kokkos_Cuda_KernelLaunch.hpp` | `Kokkos_Maca_KernelLaunch.hpp` | launch mechanism selection and kernel invocation |
| `Kokkos_Cuda_BlockSize_Deduction.hpp` | `Kokkos_Maca_BlockSize_Deduction.hpp` | occupancy and launch-bound deduction |
| `Kokkos_Cuda_MDRangePolicy.hpp` | `Kokkos_Maca_MDRangePolicy.hpp` | MDRange backend policy details |
| `Kokkos_Cuda_ReduceScan.hpp` | `Kokkos_Maca_ReduceScan.hpp` | reduction/scan infrastructure |
| `Kokkos_Cuda_Team.hpp` | `Kokkos_Maca_Team.hpp` | team-level execution implementation |
| `Kokkos_Cuda_Vectorization.hpp` | `Kokkos_Maca_Vectorization.hpp` | warp/wave shuffle operations |
| `Kokkos_Cuda_UniqueToken.hpp` | `Kokkos_Maca_UniqueToken.hpp` | unique-token support |
| `Kokkos_Cuda_Graph_Impl.hpp` | `Kokkos_Maca_Graph_Impl.hpp` | graph support |
| `Kokkos_Cuda_GraphNodeKernel.hpp` | `Kokkos_Maca_GraphNodeKernel.hpp` | kernel graph nodes |
| `Kokkos_Cuda_WorkGraphPolicy.hpp` | `Kokkos_Maca_WorkGraphPolicy.hpp` | work graph policy |
| `Kokkos_Cuda_ZeroMemset.hpp` | `Kokkos_Maca_ZeroMemset.hpp/.cpp` | zero-fill helper |
| `Kokkos_Cuda_Half_*` | `Kokkos_Maca_Half_*` | half/bhalf type support |

MACA-only support files:

| MACA file | Why it exists |
| --- | --- |
| `Kokkos_Maca_Runtime.hpp` | adapts available MACA headers into HIP-like symbols and types |
| `Kokkos_Maca_DeepCopy.hpp/.cpp` | separates deep-copy logic that CUDA keeps in `Kokkos_CudaSpace.cpp` |
| `Kokkos_Maca_SharedAllocationRecord.hpp/.cpp` | shared allocation bookkeeping brought out explicitly |
| `Kokkos_Maca_IsXnack.hpp/.cpp` | environment and boot-config checks for managed/HMM behavior |
| `Kokkos_Maca_Abort.hpp` | MACA-specific abort support |
| `Kokkos_Maca_ParallelFor_*` | split-out parallel_for implementations |
| `Kokkos_Maca_ParallelReduce_*` | split-out parallel_reduce implementations |
| `Kokkos_Maca_ParallelScan_Range.hpp` | split-out parallel_scan implementation |
| `Kokkos_Maca_TeamPolicyInternal.hpp` | extra team-policy internals split from monolithic CUDA code |
| `Kokkos_Maca_Error.cpp` | explicit implementation file for MACA error helpers |

CUDA-only support file with no direct MACA peer:

| CUDA file | Notes |
| --- | --- |
| `Kokkos_Cuda_View.hpp` | CUDA-specific LDG/random-access view optimization is not mirrored as a dedicated MACA file |

## Core Conclusion: How MACA Was Implemented

The MACA backend was implemented in three layers.

### 1. Reuse the Kokkos GPU backend shape

The outer architecture closely follows CUDA:

- execution space class
- memory spaces
- device-instance object with stream/device wrappers
- kernel launch abstraction
- team/reduction/scan machinery
- graph support
- half/vectorization helpers

This is visible from the direct file mapping and from the public `Maca` class in `core/src/Maca/Kokkos_Maca.hpp`, which mirrors the public responsibilities of `Kokkos::Cuda` in `core/src/Cuda/Kokkos_Cuda.hpp`.

Evidence:

- `Kokkos::Cuda` exposes `memory_space`, `device_type`, `scratch_memory_space`, stream/device accessors, and `impl_initialize/impl_finalize`.
- `Kokkos::Maca` exposes the same categories, with MACA naming but the same backend contract.

### 2. Replace CUDA runtime coupling with a MACA runtime adapter

This is the single most important design choice.

`core/src/Maca/Kokkos_Maca_Runtime.hpp` checks for:

- `mcr/hip_to_maca_adaptor.h`
- `mc/hip_to_maca_adaptor.h`
- fallback MACA runtime headers under `mc/` or `mcr/`

It then aliases HIP symbols onto MACA symbols where needed:

- `hipUUID_t -> mcUuid_t`
- `hipMemPool_t -> mcMemPool_t`
- `hipMallocAsync -> mcMallocAsync`
- `hipFreeAsync -> mcFreeAsync`
- `hipStreamGetDevice -> mcStreamGetDevice`

This means the backend implementation can stay largely HIP-shaped internally while still targeting MACA. In practice, MACA is implemented as a Kokkos backend that speaks to a MACA runtime through HIP-compatible entry points.

### 3. Adapt and split the backend internals

Once the runtime layer is in place, the implementation copies the same internal concepts used by CUDA but restructures them for the MACA bring-up:

- deep copy is separated into dedicated files
- `parallel_for`, `parallel_reduce`, and `parallel_scan` are split across multiple files
- shared-resource synchronization for constant-memory launches is more explicit
- XNACK/HMM handling is surfaced in both initialization and managed-memory allocation

This is why the MACA folder is larger than the CUDA folder even though many concepts are parallel.

## Detailed Technical Comparison

### 1. Public execution-space class

`core/src/Cuda/Kokkos_Cuda.hpp` and `core/src/Maca/Kokkos_Maca.hpp` define equivalent backend-facing execution spaces.

Common pattern:

- execution space alias
- preferred memory space
- `Device<execution_space, memory_space>`
- `LayoutLeft`
- scratch-memory space
- `impl_initialize`, `impl_finalize`, `fence`, `concurrency`
- `DeviceTypeTraits` for tools

MACA-specific differences:

- `Maca` includes `Kokkos_Maca_Runtime.hpp` directly because the runtime adaptation is fundamental to the class.
- The MACA class uses `hipStream_t` rather than a native `mc*` stream type directly.
- `Kokkos::Maca` is registered as `DeviceType::Maca`.

Interpretation:

The public Kokkos contract was preserved almost one-for-one. The backend behaves like a normal first-class Kokkos execution space, not like an external plugin.

### 2. Memory spaces

CUDA uses:

- `CudaSpace`
- `CudaUVMSpace`
- `CudaHostPinnedSpace`

MACA uses:

- `MacaSpace`
- `MacaHostPinnedSpace`
- `MacaManagedSpace`

Implementation observations:

- `MacaSpace` mirrors `CudaSpace` allocation and deallocation flow, but uses `hipMalloc`, `hipMallocAsync`, `hipFree`, and `hipFreeAsync`.
- `MacaHostPinnedSpace` uses `hipHostMalloc` and `hipHostFree`.
- `MacaManagedSpace` uses `hipMallocManaged` plus `hipMemAdvise`.

Important divergence:

- CUDA keeps deep-copy logic in `Kokkos_CudaSpace.cpp`.
- MACA splits deep-copy handling into `Kokkos_Maca_DeepCopy.hpp/.cpp`.

Another important divergence:

- `MacaManagedSpace` explicitly checks page migration support and warns about missing XNACK/HMM conditions.
- CUDA’s UVM handling is present, but the MACA path is more defensive and more visibly adapted during bring-up.

Notable rough edges in the current MACA tree:

- `MacaHostPinnedSpace::name()` returns `"HIPHostPinned"`.
- `MacaManagedSpace::name()` returns `"HIPManaged"`.
- warning strings still mention `HIPManaged`.

That is strong evidence that MACA memory-space code was derived from a HIP-oriented implementation and then partially renamed.

### 3. Device instance and runtime wrappers

`CudaInternal` and `MacaInternal` play the same architectural role:

- store current device id
- hold stream state
- own scratch buffers and team scratch pools
- provide wrapper functions around the underlying runtime
- serve as the backend’s concrete execution-space instance

MACA differences:

- `MacaInternal` uses `m_hipDev`, `hipStream_t`, `hipGraph_t`, `hipEvent_t`, and `hipDeviceProp_t`.
- wrapper methods are named `maca_*_wrapper` in some places but still call HIP-shaped functions.
- constant-memory resource reuse is guarded with a dedicated `SharedResourceLock` type.

The `SharedResourceLock` addition is important. CUDA uses per-device maps of staging buffers, events, and mutexes; MACA wraps this more explicitly in a reusable lock object that tracks whether a synchronization is still needed before another stream reuses the shared constant-memory resource.

Interpretation:

This part of the implementation preserved the CUDA backend’s instance-oriented design, but the author made stream/resource safety more explicit for MACA constant-memory launches.

### 4. Kernel launch strategy

This is where the lineage is clearest.

`core/src/Maca/Kokkos_Maca_KernelLaunch.hpp` contains explicit comments:

- "The hip_parallel_launch_*_memory code is identical to the cuda code"
- "The following code is identical to the cuda code"

The MACA launch layer keeps the same high-level mechanisms as CUDA:

- local-memory launch
- constant-memory launch
- global-memory launch
- launch-bound specializations
- launch-mechanism deduction based on functor size and work-item properties

But the implementation is translated into a HIP/MACA execution model:

- kernels are named `hip_parallel_launch_*`
- function attributes are queried via `hipFuncGetAttributes`
- launches are dispatched through `hip*` wrappers
- constant-memory staging uses `hipMemcpyToSymbolAsync`

This means the launch algorithm itself is CUDA-style, but the concrete runtime path is HIP-shaped and MACA-adapted.

### 5. Parallel dispatch files

CUDA groups more backend logic into:

- `Kokkos_Cuda_Parallel_Range.hpp`
- `Kokkos_Cuda_Parallel_MDRange.hpp`
- `Kokkos_Cuda_Parallel_Team.hpp`

MACA splits this into:

- `Kokkos_Maca_ParallelFor_Range.hpp`
- `Kokkos_Maca_ParallelFor_MDRange.hpp`
- `Kokkos_Maca_ParallelFor_Team.hpp`
- `Kokkos_Maca_ParallelReduce_Range.hpp`
- `Kokkos_Maca_ParallelReduce_MDRange.hpp`
- `Kokkos_Maca_ParallelReduce_Team.hpp`
- `Kokkos_Maca_ParallelScan_Range.hpp`

Interpretation:

MACA keeps the same execution concepts but decomposes them by operation type. This likely made the bring-up easier because each execution pattern could be adjusted independently.

### 6. Team, shuffle, and reduction internals

`Kokkos_Maca_Team.hpp`, `Kokkos_Maca_Vectorization.hpp`, and `Kokkos_Maca_ReduceScan.hpp` strongly resemble CUDA/HIP GPU backend internals:

- team reductions and joins still use HIP-oriented helper names such as `HIPJoinFunctor`
- shuffle operations use synchronized shuffle intrinsics
- reduction comments and helper names retain `HIP` references

The vectorization layer is especially telling:

- CUDA uses CUDA warp shuffles
- MACA uses `__shfl_sync`, `__shfl_up_sync`, and `__shfl_down_sync`
- a later branch commit explicitly switched MACA to sync shuffle intrinsics

Interpretation:

The backend retains GPU execution semantics close to CUDA/HIP warp-level programming, then adapts them just enough for the MACA toolchain/runtime to compile and execute correctly.

### 7. Graph support

Both backends implement graph support with dedicated graph and graph-node files. The MACA graph path again uses HIP-shaped runtime calls:

- `hipGraphCreate`
- `hipGraphAddKernelNode`
- `hipGraphInstantiate`
- `hipGraphLaunch`

This fits the same pattern as the rest of the backend: reuse Kokkos’ existing GPU backend architecture, but route the concrete operations through HIP-compatible MACA entry points.

### 8. Initialization and architecture handling

`Kokkos_Maca.cpp` performs backend initialization that is analogous to CUDA initialization, but with MACA-specific checks:

- choose visible device
- call `hipGetDeviceProperties`
- set the selected device
- compute backend concurrency limits
- initialize desul lock arrays
- create the default execution-space instance and default stream

MACA-specific additions:

- compares runtime `mxArchName` against the configured Kokkos MACA architecture
- carries warnings for architecture mismatch
- includes XNACK/HMM diagnostics

This is more bring-up-oriented than CUDA because the author is validating a new backend on top of an adaptor layer rather than on a mature native runtime path.

## What the Folder Comparison Says About the Implementation Approach

The folder comparison supports the following explanation.

### MACA was not written as a thin CUDA rename

Reasons:

- the runtime is not CUDA-native; it is MACA exposed through HIP-style types and calls
- several files remain HIP-oriented internally
- CUDA’s monolithic parallel-dispatch files were reorganized for MACA
- MACA adds bring-up helpers not present in CUDA

### MACA was also not written from scratch

Reasons:

- file roles align very closely with the CUDA backend
- several sections explicitly say they are identical to CUDA
- many helper names and comments still reference HIP
- the kernel-launch, team, reduction, and managed-memory code clearly follow an existing GPU backend pattern

### The best description is:

MACA was implemented as a Kokkos GPU backend by taking the established CUDA-style backend architecture, reusing a large amount of GPU backend logic through HIP-shaped code paths, and introducing a MACA runtime adapter plus a set of MACA-specific bring-up fixes around memory, launch, and environment handling.

## Strengths of the Current MACA Implementation

- It is integrated as a first-class Kokkos backend, not as an external patch layer.
- The public execution-space and memory-space model matches established Kokkos backend conventions.
- The runtime adapter isolates platform-specific header and symbol differences in one place.
- The folder structure is comprehensive enough to support graphs, teams, reductions, scans, and managed memory.
- The split dispatch files likely make future debugging easier than a single monolithic file.

## Limitations and Technical Debt Visible from the Comparison

- Many identifiers, warnings, and helper names still say `HIP` rather than `Maca`.
- Some memory-space names still report `HIPHostPinned` and `HIPManaged`.
- Several comments explicitly say the code is copied or identical, which suggests cleanup is incomplete.
- The backend still appears semantically closer to a HIP-derived implementation than to a fully MACA-native backend.
- CUDA-specific features such as `Kokkos_Cuda_View.hpp` do not appear to have a dedicated MACA analogue.

## Bottom-Line Explanation

If this backend had to be described in one sentence:

`Maca` was implemented by cloning the overall structure of Kokkos’ CUDA GPU backend, replacing the low-level runtime with a MACA-to-HIP compatibility path, and then incrementally fixing the mismatches in memory handling, kernel launch, reductions, atomics, and test coverage until the backend behaved like a proper Kokkos execution space.
