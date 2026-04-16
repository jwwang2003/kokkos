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

#include <algorithm>

// Must use global variable on the device with Maca-Clang
#ifdef __MACACC__
#ifdef KOKKOS_ENABLE_MACA_RELOCATABLE_DEVICE_CODE
__device__ __constant__ extern unsigned long
    kokkos_impl_maca_constant_memory_buffer[];
#else
__device__ __constant__ unsigned long kokkos_impl_maca_constant_memory_buffer
    [Kokkos::Impl::MacaTraits::ConstantMemoryUsage / sizeof(unsigned long)];
#endif
#endif

namespace Kokkos {
template <typename T>
inline __device__ T *kokkos_impl_maca_shared_memory() {
  extern __shared__ Kokkos::MacaSpace::size_type sh[];
  return (T *)sh;
}
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

#define KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(maxTperB, minBperSM) \
  __launch_bounds__(maxTperB)

// The maca_parallel_launch_*_memory code is identical to the cuda code
template <typename DriverType>
__global__ static void maca_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_maca_constant_memory_buffer));

  driver();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(
    maxTperB, minBperSM) static void maca_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_maca_constant_memory_buffer));

  driver();
}

template <class DriverType>
__global__ static void maca_parallel_launch_local_memory(
    const DriverType driver) {
  driver();
}

template <class DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(
    maxTperB, minBperSM) static void maca_parallel_launch_local_memory(
    const DriverType driver) {
  driver();
}

template <typename DriverType>
__global__ static void maca_parallel_launch_global_memory(
    const DriverType *driver) {
  driver->operator()();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ KOKKOS_IMPL_MACA_LAUNCH_BOUNDS(
    maxTperB, minBperSM) static void maca_parallel_launch_global_memory(
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
struct DeduceMacaLaunchMechanism {
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
struct MacaParallelLaunchKernelFuncData {
  static unsigned int get_scratch_size(
      macaFuncAttributes const &maca_func_attributes) {
    return maca_func_attributes.localSizeBytes;
  }

  // These functions need to be templated on DriverType and LaunchBounds
  // so that the static bool is unique for each type combo
  // KernelFuncPtr does not necessarily contain that type information.
  static macaFuncAttributes get_maca_func_attributes(const int maca_device,
                                                    void const *kernel_func) {
    // Only call macaFuncGetAttributes once for each unique kernel
    // and device by leveraging static variable initialization rules
    static std::map<int, macaFuncAttributes> func_attr;
    if (func_attr.find(maca_device) == func_attr.end()) {
      macaFuncAttributes attr;
      KOKKOS_IMPL_MACA_SAFE_CALL(macaSetDevice(maca_device));
      KOKKOS_IMPL_MACA_SAFE_CALL(macaFuncGetAttributes(&attr, kernel_func));
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

inline void check_shmem_request(MacaInternal const *maca_instance, int shmem) {
  if (maca_instance->m_deviceProp.sharedMemPerBlock < shmem) {
    Kokkos::Impl::throw_runtime_exception(
        "MacaParallelLaunch FAILED: shared memory request is too large");
  }
}

template <class DriverType, class LaunchBounds, class KernelFuncPtr>
const macaFuncAttributes &get_maca_kernel_func_attributes(
    const MacaInternal *maca_instance, const KernelFuncPtr &func) {
  const auto maca_device = maca_instance->m_macaDev;
  static std::map<int, macaFuncAttributes> func_attr;
  if (func_attr.find(maca_device) == func_attr.end()) {
    macaFuncAttributes attr;
    KOKKOS_IMPL_MACA_SAFE_CALL(
        (maca_instance->maca_func_get_attributes_wrapper(&attr, func)));
    func_attr.emplace(maca_device, attr);
  }
  return func_attr[maca_device];
}

template <class DriverType, class LaunchBounds, class KernelFuncPtr>
inline void configure_shmem_preference(const MacaInternal *maca_instance,
                                       const KernelFuncPtr &func,
                                       const size_t block_size, int &shmem,
                                       const int occupancy,
                                       const bool prefer_shmem) {
  auto const &func_attr =
      get_maca_kernel_func_attributes<DriverType, LaunchBounds>(maca_instance,
                                                               func);
  auto const &device_props = maca_instance->m_deviceProp;

  if (prefer_shmem) {
    static std::map<int, int> cache_config_preference_cached;
    auto const maca_device = maca_instance->m_macaDev;
    if (cache_config_preference_cached[maca_device] !=
        int(macaFuncCachePreferShared)) {
      if (maca_instance->maca_func_set_cache_config_wrapper(
              func, macaFuncCachePreferShared) == macaSuccess) {
        cache_config_preference_cached[maca_device] =
            int(macaFuncCachePreferShared);
      }
    }
  }

  const int clamped_occupancy = std::clamp(occupancy, 1, 100);
  if ((clamped_occupancy == 100) && !prefer_shmem) return;

  const size_t warp_size = std::max(
      1, int(device_props.warpSize > 0 ? device_props.warpSize : MacaTraits::WarpSize));
  const size_t max_threads_per_sm =
      std::max(1, int(device_props.maxThreadsPerMultiProcessor));
  const size_t max_shmem_per_sm =
      std::max<size_t>(device_props.maxSharedMemoryPerMultiProcessor,
                       device_props.sharedMemPerBlock);

  const size_t num_threads_desired =
      std::max(warp_size,
               ((max_threads_per_sm * clamped_occupancy / 100 + warp_size - 1) /
                warp_size) *
                   warp_size);
  const size_t num_blocks_desired =
      std::max<size_t>(1, (num_threads_desired + block_size - 1) / block_size);

  size_t shmem_per_block = static_cast<size_t>(shmem) + func_attr.sharedSizeBytes;
  constexpr size_t min_shmem_size_per_sm = 8192;
  if (((clamped_occupancy < 100) || prefer_shmem) &&
      (shmem_per_block * num_blocks_desired < min_shmem_size_per_sm)) {
    shmem_per_block = (min_shmem_size_per_sm + num_blocks_desired - 1) /
                      num_blocks_desired;
    if (shmem_per_block > func_attr.sharedSizeBytes) {
      shmem = int(shmem_per_block - func_attr.sharedSizeBytes);
    }
  }

  size_t carveout = 100;
  if (max_shmem_per_sm > 0) {
    carveout = (100 * std::min(max_shmem_per_sm,
                               num_blocks_desired *
                                   std::max<size_t>(shmem_per_block, 1))) /
               max_shmem_per_sm;
    carveout = std::clamp<size_t>(carveout, prefer_shmem ? 100 : 1, 100);
  }

  static std::map<int, int> carveout_cached;
  auto const maca_device = maca_instance->m_macaDev;
  if (carveout_cached[maca_device] != int(carveout)) {
    if (maca_instance->maca_func_set_attribute_wrapper(
            func, macaFuncAttributePreferredSharedMemoryCarveout,
            int(carveout)) == macaSuccess) {
      carveout_cached[maca_device] = int(carveout);
    }
  }
}

//---------------------------------------------------------------//
// MacaParallelLaunchKernelFunc structure and its specializations //
//---------------------------------------------------------------//
template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism>
struct MacaParallelLaunchKernelFunc;

// MacaLaunchMechanism::LocalMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct MacaParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    MacaLaunchMechanism::LocalMemory> {
  using funcdata_t = MacaParallelLaunchKernelFuncData<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      MacaLaunchMechanism::LocalMemory>;
  static auto get_kernel_func() {
    return maca_parallel_launch_local_memory<DriverType, MaxThreadsPerBlock,
                                            MinBlocksPerSM>;
  }

  static constexpr auto default_launchbounds() { return false; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_maca_func_attributes(maca_device));
  }

  static macaFuncAttributes get_maca_func_attributes(const int maca_device) {
    return funcdata_t::get_maca_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

template <typename DriverType>
struct MacaParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                    MacaLaunchMechanism::LocalMemory> {
  using funcdata_t =
      MacaParallelLaunchKernelFuncData<DriverType, Kokkos::LaunchBounds<0, 0>,
                                       MacaLaunchMechanism::LocalMemory>;
  static auto get_kernel_func() {
    return maca_parallel_launch_local_memory<DriverType>;
  }

  static constexpr auto default_launchbounds() { return true; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_maca_func_attributes(maca_device));
  }

  static macaFuncAttributes get_maca_func_attributes(const int maca_device) {
    return funcdata_t::get_maca_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

// MacaLaunchMechanism::GlobalMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct MacaParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    MacaLaunchMechanism::GlobalMemory> {
  using funcdata_t = MacaParallelLaunchKernelFuncData<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      MacaLaunchMechanism::GlobalMemory>;
  static auto get_kernel_func() {
    return maca_parallel_launch_global_memory<DriverType, MaxThreadsPerBlock,
                                             MinBlocksPerSM>;
  }

  static constexpr auto default_launchbounds() { return false; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_maca_func_attributes(maca_device));
  }

  static macaFuncAttributes get_maca_func_attributes(const int maca_device) {
    return funcdata_t::get_maca_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

template <typename DriverType>
struct MacaParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                    MacaLaunchMechanism::GlobalMemory> {
  using funcdata_t =
      MacaParallelLaunchKernelFuncData<DriverType, Kokkos::LaunchBounds<0, 0>,
                                       MacaLaunchMechanism::GlobalMemory>;
  static auto get_kernel_func() {
    return maca_parallel_launch_global_memory<DriverType>;
  }

  static constexpr auto default_launchbounds() { return true; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_maca_func_attributes(maca_device));
  }

  static macaFuncAttributes get_maca_func_attributes(const int maca_device) {
    return funcdata_t::get_maca_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

// MacaLaunchMechanism::ConstantMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct MacaParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    MacaLaunchMechanism::ConstantMemory> {
  using funcdata_t = MacaParallelLaunchKernelFuncData<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      MacaLaunchMechanism::ConstantMemory>;
  static auto get_kernel_func() {
    return maca_parallel_launch_constant_memory<DriverType, MaxThreadsPerBlock,
                                               MinBlocksPerSM>;
  }

  static constexpr auto default_launchbounds() { return false; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_maca_func_attributes(maca_device));
  }

  static macaFuncAttributes get_maca_func_attributes(const int maca_device) {
    return funcdata_t::get_maca_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

template <typename DriverType>
struct MacaParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                    MacaLaunchMechanism::ConstantMemory> {
  using funcdata_t =
      MacaParallelLaunchKernelFuncData<DriverType, Kokkos::LaunchBounds<0, 0>,
                                       MacaLaunchMechanism::ConstantMemory>;
  static auto get_kernel_func() {
    return maca_parallel_launch_constant_memory<DriverType>;
  }
  static constexpr auto default_launchbounds() { return true; }

  static auto get_scratch_size(const int maca_device) {
    return funcdata_t::get_scratch_size(get_maca_func_attributes(maca_device));
  }

  static macaFuncAttributes get_maca_func_attributes(const int maca_device) {
    return funcdata_t::get_maca_func_attributes(
        maca_device, reinterpret_cast<void const *>(get_kernel_func()));
  }
};

//------------------------------------------------------------------//
// MacaParallelLaunchKernelInvoker structure and its specializations //
//------------------------------------------------------------------//
template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism>
struct MacaParallelLaunchKernelInvoker;

// MacaLaunchMechanism::LocalMemory specialization
template <typename DriverType, typename LaunchBounds>
struct MacaParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                       MacaLaunchMechanism::LocalMemory>
    : MacaParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                   MacaLaunchMechanism::LocalMemory> {
  using base_t = MacaParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                              MacaLaunchMechanism::LocalMemory>;

  static void invoke_kernel(DriverType const &driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            MacaInternal const *maca_instance) {
    // Set the Maca device before launching the kernel
    maca_instance->set_maca_device();
    (base_t::get_kernel_func())<<<grid, block, shmem, maca_instance->m_stream>>>(
        driver);
  }

  static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block, int shmem,
      MacaInternal const *maca_instance) {
    auto const &graph = get_maca_graph_from_kernel(driver);
    KOKKOS_EXPECTS(graph);
    auto &graph_node = get_maca_graph_node_from_kernel(driver);
    // Expect node not yet initialized
    KOKKOS_EXPECTS(!graph_node);

    if (!is_empty_launch(grid, block)) {
      void const *args[] = {&driver};

      macaKernelNodeParams params = {};

      params.blockDim       = block;
      params.gridDim        = grid;
      params.sharedMemBytes = shmem;
      // Casting a function pointer to a data pointer...
      params.func         = reinterpret_cast<void *>(base_t::get_kernel_func());
      params.kernelParams = const_cast<void **>(args);
      params.extra        = nullptr;

      KOKKOS_IMPL_MACA_SAFE_CALL(maca_instance->maca_graph_add_kernel_node_wrapper(
          &graph_node, graph, /* dependencies = */ nullptr,
          /* numDependencies = */ 0, &params));
    } else {
      // We still need an empty node for the dependency structure
      KOKKOS_IMPL_MACA_SAFE_CALL(maca_instance->maca_graph_add_empty_node_wrapper(
          &graph_node, graph,
          /* dependencies = */ nullptr,
          /* numDependencies = */ 0));
    }
    KOKKOS_ENSURES(graph_node);
  }
};

// MacaLaunchMechanism::GlobalMemory specialization
template <typename DriverType, typename LaunchBounds>
struct MacaParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                       MacaLaunchMechanism::GlobalMemory>
    : MacaParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                   MacaLaunchMechanism::GlobalMemory> {
  using base_t = MacaParallelLaunchKernelFunc<
      DriverType, LaunchBounds, MacaLaunchMechanism::GlobalMemory>;

  static void invoke_kernel(DriverType const &driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            MacaInternal const *maca_instance) {
    // Wait until the previous kernel that uses m_scratchFuntor is done
    std::lock_guard<std::mutex> lock(MacaInternal::scratchFunctorMutex);
    DriverType *driver_ptr = reinterpret_cast<DriverType *>(
        maca_instance->stage_functor_for_execution(
            reinterpret_cast<void const *>(&driver), sizeof(DriverType)));

    // Set the Maca device before launching the kernel
    maca_instance->set_maca_device();
    (base_t::get_kernel_func())<<<grid, block, shmem, maca_instance->m_stream>>>(
        driver_ptr);
    maca_instance->mark_functor_for_execution(driver_ptr);
  }

  static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block, int shmem,
      MacaInternal const *maca_instance) {
    auto const &graph = get_maca_graph_from_kernel(driver);
    KOKKOS_EXPECTS(graph);
    auto &graph_node = get_maca_graph_node_from_kernel(driver);
    // Expect node not yet initialized
    KOKKOS_EXPECTS(!graph_node);

    if (!Impl::is_empty_launch(grid, block)) {
      auto *driver_ptr = Impl::allocate_driver_storage_for_kernel(
          Maca(maca_instance->m_stream, ManageStream::no), driver);

      // Unlike in the non-graph case, we can get away with doing an async copy
      // here because the `DriverType` instance is held in the GraphNodeImpl
      // which is guaranteed to be alive until the graph instance itself is
      // destroyed, where there should be a fence ensuring that the allocation
      // associated with this kernel on the device side isn't deleted.
      KOKKOS_IMPL_MACA_SAFE_CALL(maca_instance->maca_memcpy_async_wrapper(
          driver_ptr, &driver, sizeof(DriverType), macaMemcpyDefault));

      // FIXME_MACA Modifying the assignment to args causes a segfault in
      // maca_graph.force_global_launch
      // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
      void *args[] = {&driver_ptr};

      macaKernelNodeParams params = {};

      params.blockDim       = block;
      params.gridDim        = grid;
      params.sharedMemBytes = shmem;
      // Casting a function pointer to a data pointer...
      params.func         = reinterpret_cast<void *>(base_t::get_kernel_func());
      params.kernelParams = args;
      params.extra        = nullptr;

      KOKKOS_IMPL_MACA_SAFE_CALL(maca_instance->maca_graph_add_kernel_node_wrapper(
          &graph_node, graph, /* dependencies = */ nullptr,
          /* numDependencies = */ 0, &params));
    } else {
      // We still need an empty node for the dependency structure
      KOKKOS_IMPL_MACA_SAFE_CALL(maca_instance->maca_graph_add_empty_node_wrapper(
          &graph_node, graph,
          /* dependencies = */ nullptr,
          /* numDependencies = */ 0));
    }
    KOKKOS_ENSURES(bool(graph_node))
  }
};

// MacaLaunchMechanism::ConstantMemory specializations
template <typename DriverType, typename LaunchBounds>
struct MacaParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                       MacaLaunchMechanism::ConstantMemory>
    : MacaParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                   MacaLaunchMechanism::ConstantMemory> {
  using base_t = MacaParallelLaunchKernelFunc<
      DriverType, LaunchBounds, MacaLaunchMechanism::ConstantMemory>;
  static_assert(sizeof(DriverType) < MacaTraits::ConstantMemoryUsage,
                "Kokkos Error: Requested MacaLaunchConstantMemory with a "
                "Functor larger than 32kB.");

  static void invoke_kernel(DriverType const &driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            MacaInternal const *maca_instance) {
    const auto maca_device = maca_instance->m_macaDev;

    auto lock = MacaInternal::constantMemReusable[maca_device].acquire();

    // Copy functor (synchronously) to staging buffer in pinned host memory
    unsigned long *staging = maca_instance->constantMemHostStaging[maca_device];
    std::memcpy(static_cast<void *>(staging),
                static_cast<const void *>(&driver), sizeof(DriverType));

    // Copy functor asynchronously from there to constant memory on the device
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmaca-compat"
#endif
    KOKKOS_IMPL_MACA_SAFE_CALL(maca_instance->maca_memcpy_to_symbol_async_wrapper(
        MACA_SYMBOL(kokkos_impl_maca_constant_memory_buffer), staging,
        sizeof(DriverType), 0, macaMemcpyHostToDevice));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    // Set the Maca device before launching the kernel
    maca_instance->set_maca_device();

    // Invoke the driver function on the device
    (base_t::
         get_kernel_func())<<<grid, block, shmem, maca_instance->m_stream>>>();

    MacaInternal::constantMemReusable[maca_device].release(
        std::move(lock), maca_instance->m_stream);
  }

  static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block, int shmem,
      MacaInternal const *maca_instance) {
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
        MacaParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                        MacaLaunchMechanism::GlobalMemory>;
    global_launch_impl_t::create_parallel_launch_graph_node(
        driver, grid, block, shmem, maca_instance);
  }
};

//-----------------------------//
// MacaParallelLaunch structure //
//-----------------------------//
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
struct MacaParallelLaunch;

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM, MacaLaunchMechanism LaunchMechanism>
struct MacaParallelLaunch<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    LaunchMechanism>
    : MacaParallelLaunchKernelInvoker<
          DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
          LaunchMechanism> {
  using base_t = MacaParallelLaunchKernelInvoker<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      LaunchMechanism>;

  MacaParallelLaunch(const DriverType &driver, const dim3 &grid,
                     const dim3 &block, const unsigned int shmem,
                     const MacaInternal *maca_instance,
                     const bool prefer_shmem) {
    if (!is_empty_launch(grid, block)) {
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);

      int launch_shmem = int(shmem);
      check_shmem_request(maca_instance, launch_shmem);

      if constexpr (DriverType::Policy::
                        experimental_contains_desired_occupancy) {
        int desired_occupancy =
            driver.get_policy().impl_get_desired_occupancy().value();
        size_t block_size = static_cast<size_t>(block.x) * block.y * block.z;
        configure_shmem_preference<
            DriverType,
            Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>>(
            maca_instance, base_t::get_kernel_func(), block_size, launch_shmem,
            desired_occupancy, prefer_shmem);
      } else if (prefer_shmem) {
        size_t block_size = static_cast<size_t>(block.x) * block.y * block.z;
        configure_shmem_preference<
            DriverType,
            Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>>(
            maca_instance, base_t::get_kernel_func(), block_size, launch_shmem,
            100, true);
      }

      // Invoke the driver function on the device
      base_t::invoke_kernel(driver, grid, block, launch_shmem, maca_instance);

#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
      KOKKOS_IMPL_MACA_SAFE_CALL(macaGetLastError());
      maca_instance->fence(
          "Kokkos::Impl::MacaParallelLaunch: Debug Only Check for "
          "Execution Error");
#endif
    }
  }
};

// convenience method to launch the correct kernel given the launch bounds et
// al.
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism,
          bool DoGraph = DriverType::Policy::is_graph_kernel::value>
void maca_parallel_launch(const DriverType &driver, const dim3 &grid,
                          const dim3 &block, const int shmem,
                          const MacaInternal *maca_instance,
                          const bool prefer_shmem) {
  if (!is_empty_launch(grid, block)) {
    desul::Impl::ensure_lock_arrays_on_device();
  }

  if constexpr (DoGraph) {
    // Graph launch
    using base_t = MacaParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                                   LaunchMechanism>;
    base_t::create_parallel_launch_graph_node(driver, grid, block, shmem,
                                              maca_instance);
  } else {
    // Regular kernel launch
#ifndef KOKKOS_ENABLE_MACA_MULTIPLE_KERNEL_INSTANTIATIONS
    MacaParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>(
        driver, grid, block, shmem, maca_instance, prefer_shmem);
#else
    if constexpr (!MacaParallelLaunch<DriverType, LaunchBounds,
                                      LaunchMechanism>::default_launchbounds()) {
      // for user defined, we *always* honor the request
      MacaParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>(
          driver, grid, block, shmem, maca_instance, prefer_shmem);
    } else {
      // we can do what we like
      const unsigned flat_block_size = block.x * block.y * block.z;
      if (flat_block_size <= MacaTraits::ConservativeThreadsPerBlock) {
        // we have to use the large blocksize
        MacaParallelLaunch<
            DriverType,
            Kokkos::LaunchBounds<MacaTraits::ConservativeThreadsPerBlock, 1>,
            LaunchMechanism>(driver, grid, block, shmem, maca_instance,
                             prefer_shmem);
      } else {
        MacaParallelLaunch<
            DriverType, Kokkos::LaunchBounds<MacaTraits::MaxThreadsPerBlock, 1>,
            LaunchMechanism>(driver, grid, block, shmem, maca_instance,
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
