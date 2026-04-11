// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_KERNEL_LAUNCH_HPP
#define KOKKOS_MACA_KERNEL_LAUNCH_HPP

#include <Kokkos_Macros.hpp>

#if defined(__MACACC__)

#include <Maca/Kokkos_Maca_Error.hpp>
#include <Maca/Kokkos_Maca_GraphNodeKernel.hpp>
#include <Maca/Kokkos_Maca_Instance.hpp>
#include <Maca/Kokkos_Maca_Space.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>

// Must use global variable on the device with Maca-Clang
#ifdef __MACACC__
#ifdef KOKKOS_ENABLE_MACA_RELOCATABLE_DEVICE_CODE
__device__ __constant__ extern unsigned long
    kokkos_impl_hip_constant_memory_buffer[];
#else
__device__ __constant__ unsigned long kokkos_impl_hip_constant_memory_buffer
    [Kokkos::Impl::MacaTraits::ConstantMemoryUsage / sizeof(unsigned long)];
#endif
#endif

namespace Kokkos {
template <typename T>
inline __device__ T *kokkos_impl_hip_shared_memory() {
  extern __shared__ Kokkos::MacaSpace::size_type sh[];
  return (T *)sh;
}
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

#define KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(maxTperB, minBperSM) \
  __launch_bounds__(maxTperB)

// The hip_parallel_launch_*_memory code is identical to the cuda code
template <typename DriverType>
__global__ static void hip_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_hip_constant_memory_buffer));

  driver();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(
    maxTperB, minBperSM) static void hip_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_hip_constant_memory_buffer));

  driver();
}

template <class DriverType>
__global__ static void hip_parallel_launch_local_memory(
    const DriverType driver) {
  driver();
}

template <class DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(
    maxTperB, minBperSM) static void hip_parallel_launch_local_memory(
    const DriverType driver) {
  driver();
}

template <typename DriverType>
__global__ static void hip_parallel_launch_global_memory(
    const DriverType *driver) {
  driver->operator()();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(
    maxTperB, minBperSM) static void hip_parallel_launch_global_memory(
    const DriverType *driver) {
  driver->operator()();
}

enum class MacaLaunchMechanism : unsigned {
  Default        = 0,
  ConstantMemory = 1,
  GlobalMemory   = 2,
  LocalMemory    = 4
};

#undef KOKKOS_IMPL_MACA_LAUNCH_BOUNDS

constexpr inline MacaLaunchMechanism operator|(MacaLaunchMechanism p1,
                                              MacaLaunchMechanism p2) {
  return static_cast<MacaLaunchMechanism>(static_cast<unsigned>(p1) |
                                         static_cast<unsigned>(p2));
}
constexpr inline MacaLaunchMechanism operator&(MacaLaunchMechanism p1,
                                              MacaLaunchMechanism p2) {
  return static_cast<MacaLaunchMechanism>(static_cast<unsigned>(p1) &
                                         static_cast<unsigned>(p2));
}

// Use local memory up to ConstantMemoryUseThreshold
// Use global memory above ConstantMemoryUsage
// In between use ConstantMemory
// The following code is identical to the cuda code
template <typename DriverType>
struct DeduceHIPLaunchMechanism {
  static constexpr Kokkos::Experimental::WorkItemProperty::HintLightWeight_t
      light_weight = Kokkos::Experimental::WorkItemProperty::HintLightWeight;
  static constexpr Kokkos::Experimental::WorkItemProperty::HintHeavyWeight_t
      heavy_weight = Kokkos::Experimental::WorkItemProperty::HintHeavyWeight;
  static constexpr Kokkos::Experimental::WorkItemProperty::
      ImplForceGlobalLaunch_t force_global_launch =
          Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch;
  static constexpr typename DriverType::Policy::work_item_property property =
      typename DriverType::Policy::work_item_property();

  static constexpr MacaLaunchMechanism valid_launch_mechanism =
      // BuildValidMask
      (sizeof(DriverType) < MacaTraits::KernelArgumentLimit
           ? MacaLaunchMechanism::LocalMemory
           : MacaLaunchMechanism::Default) |
      (sizeof(DriverType) < MacaTraits::ConstantMemoryUsage
           ? MacaLaunchMechanism::ConstantMemory
           : MacaLaunchMechanism::Default) |
      MacaLaunchMechanism::GlobalMemory;

  static constexpr MacaLaunchMechanism requested_launch_mechanism =
      (((property & light_weight) == light_weight)
           ? MacaLaunchMechanism::LocalMemory
           : MacaLaunchMechanism::ConstantMemory) |
      MacaLaunchMechanism::GlobalMemory;

  static constexpr MacaLaunchMechanism default_launch_mechanism =
      // BuildValidMask
      (sizeof(DriverType) < MacaTraits::ConstantMemoryUseThreshold)
          ? MacaLaunchMechanism::LocalMemory
          : ((sizeof(DriverType) < MacaTraits::ConstantMemoryUsage)
                 ? MacaLaunchMechanism::ConstantMemory
                 : MacaLaunchMechanism::GlobalMemory);

  //              None                LightWeight    HeavyWeight
  // F<UseT       LCG  LCG L  L       LCG  LG L  L   LCG  CG L  C
  // UseT<F<KAL   LCG  LCG C  C       LCG  LG C  L   LCG  CG C  C
  // Kal<F<CMU     CG  LCG C  C        CG  LG C  G    CG  CG C  C
  // CMU<F          G  LCG G  G         G  LG G  G     G  CG G  G
  static constexpr MacaLaunchMechanism launch_mechanism =
      ((property & force_global_launch) == force_global_launch)
          ? MacaLaunchMechanism::GlobalMemory
      : ((property & light_weight) == light_weight)
          ? (sizeof(DriverType) < MacaTraits::KernelArgumentLimit
                 ? MacaLaunchMechanism::LocalMemory
                 : MacaLaunchMechanism::GlobalMemory)
          : (((property & heavy_weight) == heavy_weight)
                 ? (sizeof(DriverType) < MacaTraits::ConstantMemoryUsage
                        ? MacaLaunchMechanism::ConstantMemory
                        : MacaLaunchMechanism::GlobalMemory)
                 : (default_launch_mechanism));
};

template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelFuncData {
  static unsigned int get_scratch_size(
      hipFuncAttributes const &hip_func_attributes) {
    return hip_func_attributes.localSizeBytes;
  }

  // These functions need to be templated on DriverType and LaunchBounds
  // so that the static bool is unique for each type combo
  // KernelFuncPtr does not necessarily contain that type information.
  static hipFuncAttributes get_hip_func_attributes(const int maca_device,
                                                   void const *kernel_func) {
    // Only call hipFuncGetAttributes once for each unique kernel
    // and device by leveraging static variable initialization rules
    static std::map<int, hipFuncAttributes> func_attr;
    if (func_attr.find(maca_device) == func_attr.end()) {
      hipFuncAttributes attr;
      KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(maca_device));
      KOKKOS_IMPL_MACA_SAFE_CALL(hipFuncGetAttributes(&attr, kernel_func));
      func_attr.emplace(maca_device, attr);
    }
    return func_attr[maca_device];
  }
};

//---------------------------------------------------------------//
// Helper function                                               //
//---------------------------------------------------------------//
inline bool is_empty_launch(dim3 const &grid, dim3 const &block) {
  return (grid.x == 0) || ((block.x * block.y * block.z) == 0);
}

//---------------------------------------------------------------//
// HIPParallelLaunchKernelFunc structure and its specializations //
//---------------------------------------------------------------//
template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelFunc;

// MacaLaunchMechanism::LocalMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    MacaLaunchMechanism::LocalMemory> {
  using funcdata_t = HIPParallelLaunchKernelFuncData<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      MacaLaunchMechanism::LocalMemory>;
  static auto get_kernel_func() {
    return hip_parallel_launch_local_memory<DriverType, MaxThreadsPerBlock,
                                            MinBlocksPerSM>;
  }

  static constexpr auto default_launchbounds() { return false; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_hip_func_attributes(maca_device));
  }

  static hipFuncAttributes get_hip_func_attributes(const int maca_device) {
    return funcdata_t::get_hip_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   MacaLaunchMechanism::LocalMemory> {
  using funcdata_t =
      HIPParallelLaunchKernelFuncData<DriverType, Kokkos::LaunchBounds<0, 0>,
                                      MacaLaunchMechanism::LocalMemory>;
  static auto get_kernel_func() {
    return hip_parallel_launch_local_memory<DriverType>;
  }

  static constexpr auto default_launchbounds() { return true; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_hip_func_attributes(maca_device));
  }

  static hipFuncAttributes get_hip_func_attributes(const int maca_device) {
    return funcdata_t::get_hip_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

// MacaLaunchMechanism::GlobalMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    MacaLaunchMechanism::GlobalMemory> {
  using funcdata_t = HIPParallelLaunchKernelFuncData<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      MacaLaunchMechanism::GlobalMemory>;
  static auto get_kernel_func() {
    return hip_parallel_launch_global_memory<DriverType, MaxThreadsPerBlock,
                                             MinBlocksPerSM>;
  }

  static constexpr auto default_launchbounds() { return false; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_hip_func_attributes(maca_device));
  }

  static hipFuncAttributes get_hip_func_attributes(const int maca_device) {
    return funcdata_t::get_hip_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   MacaLaunchMechanism::GlobalMemory> {
  using funcdata_t =
      HIPParallelLaunchKernelFuncData<DriverType, Kokkos::LaunchBounds<0, 0>,
                                      MacaLaunchMechanism::GlobalMemory>;
  static auto get_kernel_func() {
    return hip_parallel_launch_global_memory<DriverType>;
  }

  static constexpr auto default_launchbounds() { return true; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_hip_func_attributes(maca_device));
  }

  static hipFuncAttributes get_hip_func_attributes(const int maca_device) {
    return funcdata_t::get_hip_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

// MacaLaunchMechanism::ConstantMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    MacaLaunchMechanism::ConstantMemory> {
  using funcdata_t = HIPParallelLaunchKernelFuncData<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      MacaLaunchMechanism::ConstantMemory>;
  static auto get_kernel_func() {
    return hip_parallel_launch_constant_memory<DriverType, MaxThreadsPerBlock,
                                               MinBlocksPerSM>;
  }

  static constexpr auto default_launchbounds() { return false; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_hip_func_attributes(maca_device));
  }

  static hipFuncAttributes get_hip_func_attributes(const int maca_device) {
    return funcdata_t::get_hip_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   MacaLaunchMechanism::ConstantMemory> {
  using funcdata_t =
      HIPParallelLaunchKernelFuncData<DriverType, Kokkos::LaunchBounds<0, 0>,
                                      MacaLaunchMechanism::ConstantMemory>;
  static auto get_kernel_func() {
    return hip_parallel_launch_constant_memory<DriverType>;
  }
  static constexpr auto default_launchbounds() { return true; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_hip_func_attributes(maca_device));
  }

  static hipFuncAttributes get_hip_func_attributes(const int maca_device) {
    return funcdata_t::get_hip_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

//------------------------------------------------------------------//
// HIPParallelLaunchKernelInvoker structure and its specializations //
//------------------------------------------------------------------//
template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelInvoker;

// MacaLaunchMechanism::LocalMemory specialization
template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      MacaLaunchMechanism::LocalMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  MacaLaunchMechanism::LocalMemory> {
  using base_t = HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                             MacaLaunchMechanism::LocalMemory>;

  static void invoke_kernel(DriverType const &driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            MacaInternal const *hip_instance) {
    // Set hip device before launching kernel
    hip_instance->set_maca_device();
    (base_t::get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>(
        driver);
  }

  static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block, int shmem,
      MacaInternal const *hip_instance) {
    auto const &graph = get_hip_graph_from_kernel(driver);
    KOKKOS_EXPECTS(graph);
    auto &graph_node = get_hip_graph_node_from_kernel(driver);
    // Expect node not yet initialized
    KOKKOS_EXPECTS(!graph_node);

    if (!is_empty_launch(grid, block)) {
      void const *args[] = {&driver};

      hipKernelNodeParams params = {};

      params.blockDim       = block;
      params.gridDim        = grid;
      params.sharedMemBytes = shmem;
      // Casting a function pointer to a data pointer...
      params.func         = reinterpret_cast<void *>(base_t::get_kernel_func());
      params.kernelParams = const_cast<void **>(args);
      params.extra        = nullptr;

      KOKKOS_IMPL_MACA_SAFE_CALL(hip_instance->maca_graph_add_kernel_node_wrapper(
          &graph_node, graph, /* dependencies = */ nullptr,
          /* numDependencies = */ 0, &params));
    } else {
      // We still need an empty node for the dependency structure
      KOKKOS_IMPL_MACA_SAFE_CALL(hip_instance->maca_graph_add_empty_node_wrapper(
          &graph_node, graph,
          /* dependencies = */ nullptr,
          /* numDependencies = */ 0));
    }
    KOKKOS_ENSURES(graph_node);
  }
};

// MacaLaunchMechanism::GlobalMemory specialization
template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      MacaLaunchMechanism::GlobalMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  MacaLaunchMechanism::GlobalMemory> {
  using base_t = HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                             MacaLaunchMechanism::GlobalMemory>;

  static void invoke_kernel(DriverType const &driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            MacaInternal const *hip_instance) {
    // Wait until the previous kernel that uses m_scratchFuntor is done
    std::lock_guard<std::mutex> lock(MacaInternal::scratchFunctorMutex);
    DriverType *driver_ptr = reinterpret_cast<DriverType *>(
        hip_instance->stage_functor_for_execution(
            reinterpret_cast<void const *>(&driver), sizeof(DriverType)));

    // Set hip device before launching kernel
    hip_instance->set_maca_device();
    (base_t::get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>(
        driver_ptr);
  }

  static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block, int shmem,
      MacaInternal const *hip_instance) {
    auto const &graph = get_hip_graph_from_kernel(driver);
    KOKKOS_EXPECTS(graph);
    auto &graph_node = get_hip_graph_node_from_kernel(driver);
    // Expect node not yet initialized
    KOKKOS_EXPECTS(!graph_node);

    if (!Impl::is_empty_launch(grid, block)) {
      auto *driver_ptr = Impl::allocate_driver_storage_for_kernel(
          Maca(hip_instance->m_stream, ManageStream::no), driver);

      // Unlike in the non-graph case, we can get away with doing an async copy
      // here because the `DriverType` instance is held in the GraphNodeImpl
      // which is guaranteed to be alive until the graph instance itself is
      // destroyed, where there should be a fence ensuring that the allocation
      // associated with this kernel on the device side isn't deleted.
      KOKKOS_IMPL_MACA_SAFE_CALL(hip_instance->maca_memcpy_async_wrapper(
          driver_ptr, &driver, sizeof(DriverType), hipMemcpyDefault));

      // FIXME_HIP Modifying the assignment to args causes a segfault in
      // hip_graph.force_global_launch
      // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
      void *args[] = {&driver_ptr};

      hipKernelNodeParams params = {};

      params.blockDim       = block;
      params.gridDim        = grid;
      params.sharedMemBytes = shmem;
      // Casting a function pointer to a data pointer...
      params.func         = reinterpret_cast<void *>(base_t::get_kernel_func());
      params.kernelParams = args;
      params.extra        = nullptr;

      KOKKOS_IMPL_MACA_SAFE_CALL(hip_instance->maca_graph_add_kernel_node_wrapper(
          &graph_node, graph, /* dependencies = */ nullptr,
          /* numDependencies = */ 0, &params));
    } else {
      // We still need an empty node for the dependency structure
      KOKKOS_IMPL_MACA_SAFE_CALL(hip_instance->maca_graph_add_empty_node_wrapper(
          &graph_node, graph,
          /* dependencies = */ nullptr,
          /* numDependencies = */ 0));
    }
    KOKKOS_ENSURES(bool(graph_node))
  }
};

// MacaLaunchMechanism::ConstantMemory specializations
template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      MacaLaunchMechanism::ConstantMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  MacaLaunchMechanism::ConstantMemory> {
  using base_t =
      HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  MacaLaunchMechanism::ConstantMemory>;
  static_assert(sizeof(DriverType) < MacaTraits::ConstantMemoryUsage,
                "Kokkos Error: Requested HIPLaunchConstantMemory with a "
                "Functor larger than 32kB.");

  static void invoke_kernel(DriverType const &driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            MacaInternal const *hip_instance) {
    const auto maca_device = hip_instance->m_hipDev;

    auto lock = MacaInternal::constantMemReusable[maca_device].acquire();

    // Copy functor (synchronously) to staging buffer in pinned host memory
    unsigned long *staging = hip_instance->constantMemHostStaging[maca_device];
    std::memcpy(static_cast<void *>(staging),
                static_cast<const void *>(&driver), sizeof(DriverType));

    // Copy functor asynchronously from there to constant memory on the device
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmaca-compat"
#endif
    KOKKOS_IMPL_MACA_SAFE_CALL(hip_instance->maca_memcpy_to_symbol_async_wrapper(
        HIP_SYMBOL(kokkos_impl_hip_constant_memory_buffer), staging,
        sizeof(DriverType), 0, hipMemcpyHostToDevice));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    // Set hip device before launching kernel
    hip_instance->set_maca_device();

    // Invoke the driver function on the device
    (base_t::
         get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>();

    MacaInternal::constantMemReusable[maca_device].release(
        std::move(lock), hip_instance->m_stream);
  }

  static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block, int shmem,
      MacaInternal const *hip_instance) {
    // Just use global memory; coordinating through events to share constant
    // memory with the non-graph interface is not really reasonable since
    // events don't work with Graphs directly, and this would anyway require
    // a much more complicated structure that finds previous nodes in the
    // dependency structure of the graph and creates an implicit dependence
    // based on the need for constant memory (which we would then have to
    // somehow go and prove was not creating a dependency cycle, and I don't
    // even know if there's an efficient way to do that, let alone in the
    // structure we currenty have).
    using global_launch_impl_t =
        HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                       MacaLaunchMechanism::GlobalMemory>;
    global_launch_impl_t::create_parallel_launch_graph_node(
        driver, grid, block, shmem, hip_instance);
  }
};

//-----------------------------//
// HIPParallelLaunch structure //
//-----------------------------//
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceHIPLaunchMechanism<DriverType>::launch_mechanism>
struct HIPParallelLaunch;

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM, MacaLaunchMechanism LaunchMechanism>
struct HIPParallelLaunch<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    LaunchMechanism>
    : HIPParallelLaunchKernelInvoker<
          DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
          LaunchMechanism> {
  using base_t = HIPParallelLaunchKernelInvoker<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      LaunchMechanism>;

  HIPParallelLaunch(const DriverType &driver, const dim3 &grid,
                    const dim3 &block, const unsigned int shmem,
                    const MacaInternal *hip_instance,
                    const bool /*prefer_shmem*/) {
    if (!is_empty_launch(grid, block)) {
      if (hip_instance->m_deviceProp.sharedMemPerBlock < shmem) {
        Kokkos::Impl::throw_runtime_exception(
            "HIPParallelLaunch FAILED: shared memory request is too large");
      }

      // Invoke the driver function on the device
      base_t::invoke_kernel(driver, grid, block, shmem, hip_instance);

#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
      KOKKOS_IMPL_MACA_SAFE_CALL(hipGetLastError());
      hip_instance->fence(
          "Kokkos::Impl::HIParallelLaunch: Debug Only Check for "
          "Execution Error");
#endif
    }
  }
};

// convenience method to launch the correct kernel given the launch bounds et
// al.
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceHIPLaunchMechanism<DriverType>::launch_mechanism,
          bool DoGraph = DriverType::Policy::is_graph_kernel::value>
void hip_parallel_launch(const DriverType &driver, const dim3 &grid,
                         const dim3 &block, const int shmem,
                         const MacaInternal *hip_instance,
                         const bool prefer_shmem) {
  if (!is_empty_launch(grid, block)) {
    desul::Impl::ensure_lock_arrays_on_device();
  }

  if constexpr (DoGraph) {
    // Graph launch
    using base_t = HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                                  LaunchMechanism>;
    base_t::create_parallel_launch_graph_node(driver, grid, block, shmem,
                                              hip_instance);
  } else {
    // Regular kernel launch
#ifndef KOKKOS_ENABLE_MACA_MULTIPLE_KERNEL_INSTANTIATIONS
    HIPParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>(
        driver, grid, block, shmem, hip_instance, prefer_shmem);
#else
    if constexpr (!HIPParallelLaunch<DriverType, LaunchBounds,
                                     LaunchMechanism>::default_launchbounds()) {
      // for user defined, we *always* honor the request
      HIPParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>(
          driver, grid, block, shmem, hip_instance, prefer_shmem);
    } else {
      // we can do what we like
      const unsigned flat_block_size = block.x * block.y * block.z;
      if (flat_block_size <= MacaTraits::ConservativeThreadsPerBlock) {
        // we have to use the large blocksize
        HIPParallelLaunch<
            DriverType,
            Kokkos::LaunchBounds<MacaTraits::ConservativeThreadsPerBlock, 1>,
            LaunchMechanism>(driver, grid, block, shmem, hip_instance,
                             prefer_shmem);
      } else {
        HIPParallelLaunch<
            DriverType, Kokkos::LaunchBounds<MacaTraits::MaxThreadsPerBlock, 1>,
            LaunchMechanism>(driver, grid, block, shmem, hip_instance,
                             prefer_shmem);
      }
    }
#endif
  }
}
}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
