// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_MACA_INSTANCE_HPP
#define KOKKOS_MACA_INSTANCE_HPP

#include <Maca/Kokkos_Maca_Space.hpp>
#include <Maca/Kokkos_Maca_Error.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <set>

namespace Kokkos {
namespace Impl {

struct MacaTraits {
#if defined(KOKKOS_ARCH_AMD_GFX906) || defined(KOKKOS_ARCH_AMD_GFX908) ||     \
    defined(KOKKOS_ARCH_AMD_GFX90A) || defined(KOKKOS_ARCH_AMD_GFX940) ||     \
    defined(KOKKOS_ARCH_AMD_GFX942) || defined(KOKKOS_ARCH_AMD_GFX942_APU) || \
    defined(KOKKOS_ARCH_AMD_GFX950)
  static constexpr int WarpSize       = 64;
  static constexpr int WarpIndexMask  = 0x003f; /* hexadecimal for 63 */
  static constexpr int WarpIndexShift = 6;      /* WarpSize == 1 << WarpShift*/
#elif defined(KOKKOS_ARCH_AMD_GFX1030) || defined(KOKKOS_ARCH_AMD_GFX1100) || \
    defined(KOKKOS_ARCH_AMD_GFX1103) || defined(KOKKOS_ARCH_AMD_GFX1201)
  static constexpr int WarpSize       = 32;
  static constexpr int WarpIndexMask  = 0x001f; /* hexadecimal for 31 */
  static constexpr int WarpIndexShift = 5;      /* WarpSize == 1 << WarpShift*/
#elif defined(KOKKOS_ARCH_XCORE1000)
  static constexpr int WarpSize       = 64;
  static constexpr int WarpIndexMask  = 0x003f; /* hexadecimal for 63 */
  static constexpr int WarpIndexShift = 6;      /* WarpSize == 1 << WarpShift*/
#else
  // MXMACA targets like xcore1000 do not currently map onto Kokkos' AMD GFX
  // arch list. Use the conservative wavefront size used by most wave64 GPUs.
  static constexpr int WarpSize       = 64;
  static constexpr int WarpIndexMask  = 0x003f;
  static constexpr int WarpIndexShift = 6;
#endif
  static constexpr int ConservativeThreadsPerBlock =
      256;  // conservative fallback blocksize in case of spills
  static constexpr int MaxThreadsPerBlock =
      1024;  // the maximum we can fit in a block
  static constexpr int ConstantMemoryUsage        = 0x008000; /* 32k bytes */
  static constexpr int KernelArgumentLimit        = 0x001000; /*  4k bytes */
  static constexpr int ConstantMemoryUseThreshold = 0x000200; /* 512 bytes */
};

//----------------------------------------------------------------------------

Maca::size_type *maca_internal_scratch_space(const Maca &instance,
                                           const std::size_t size);
Maca::size_type *maca_internal_scratch_flags(const Maca &instance,
                                           const std::size_t size);

//----------------------------------------------------------------------------

// Helper to protect a shared resource from being used by multiple streams
// simultaneously.
// If used properly, only one stream at a time will be able to use the shared
// resource.
// This helper should be used in a thread-safe way.
//
// Typical usage:
// @code
// auto lock_for_acquisition = shared_resource.acquire();
//
// ... do stuff on the shared resource ...
//
// shared_resource.release(std::move(lock_for_acquisition), stream);
//
// ...
//
// auto lock_for_involvement = shared_resource.lock();
// stream synchronization happens here;
// shared_resource.check_if_involved_and_unlock(std::move(lock_for_involvement),
//                                              stream);
// stream destruction happens here.
// @endcode
struct SharedResourceLock {
  bool m_need_sync = false;
  std::mutex m_mutex{};
  macaEvent_t m_event   = nullptr;
  macaStream_t m_stream = nullptr;

  // Acquire the right to interact in a thread-safe way.
  [[nodiscard]] auto lock() { return std::unique_lock<std::mutex>{m_mutex}; }

  // The event is created for the current device. The instance is locked first.
  void initialize() {
    auto lock = this->lock();
    if (!m_event) {
      KOKKOS_IMPL_MACA_SAFE_CALL(
          macaEventCreateWithFlags(&m_event, macaEventDisableTiming));
    }
  }

  // Destroying an event can be done even if it is not bound to the current
  // device.
  void finalize() {
    auto lock = this->lock();
    if (m_event) {
      KOKKOS_IMPL_MACA_SAFE_CALL(macaEventDestroy(m_event));
    }
  }

  SharedResourceLock() = default;

  SharedResourceLock(SharedResourceLock const &other)            = delete;
  SharedResourceLock(SharedResourceLock &&other)                 = delete;
  SharedResourceLock &operator=(SharedResourceLock const &other) = delete;
  SharedResourceLock &operator=(SharedResourceLock &&other)      = delete;
  ~SharedResourceLock()                                          = default;

  // Acquire the right to use the shared resource. The instance is locked first.
  [[nodiscard]] auto acquire() {
    auto lock = this->lock();
    if (m_need_sync) KOKKOS_IMPL_MACA_SAFE_CALL(macaEventSynchronize(m_event));
    return lock;
  }

  // Record an event in a stream to signal when it's done with the shared
  // resource.
  void release(std::unique_lock<std::mutex> lock, macaStream_t stream) {
    KOKKOS_ENSURES(lock.owns_lock());
    KOKKOS_ENSURES((lock.mutex() == std::addressof(m_mutex)));
    m_stream = stream;
    KOKKOS_IMPL_MACA_SAFE_CALL(macaEventRecord(m_event, m_stream));
    m_need_sync = true;
    lock.unlock();
  }

  // Check if the stream is the one that was used for the event recording.
  // We assume that this function is called once the stream has been
  // synchronized, and we mark that the shared resource lock does not need to
  // synchronize the next time it is acquired.
  // Doing so allows the current stream to be properly destroyed, while ensuring
  // that the next constant memory launch will work fine.
  // See https://github.com/kokkos/kokkos/issues/8006 for more details.
  void check_if_involved_and_unlock(std::unique_lock<std::mutex> lock,
                                    macaStream_t stream) {
    KOKKOS_ENSURES(lock.owns_lock());
    KOKKOS_ENSURES((lock.mutex() == std::addressof(m_mutex)));
    if (m_stream == stream) m_need_sync = false;
    lock.unlock();
  }
};

class MacaInternal {
 public:
  using size_type = ::Kokkos::Maca::size_type;
  static constexpr unsigned scratch_functor_slot_count = 4;

  struct ScratchFunctorSlot {
    std::size_t size     = 0;
    size_type *device    = nullptr;
    size_type *host      = nullptr;
    macaEvent_t reusable = nullptr;
    bool pending         = false;
  };

  int m_macaDev = -1;
  static int m_maxThreadsPerSM;

  static HostSharedPtr<MacaInternal> default_instance;

  static macaDeviceProp_t m_deviceProp;

  static int concurrency();

  // Scratch Spaces for Reductions
  std::size_t m_scratchSpaceCount = 0;
  std::size_t m_scratchFlagsCount = 0;

  size_type *m_scratchSpace               = nullptr;
  size_type *m_scratchFlags               = nullptr;
  mutable std::array<ScratchFunctorSlot, scratch_functor_slot_count>
      m_scratchFunctorSlots = {};
  mutable unsigned m_nextScratchFunctorSlot = 0;
  static std::mutex scratchFunctorMutex;

  macaStream_t m_stream = nullptr;
  bool m_allow_post_finalize_destruction = false;
  uint32_t m_instance_id =
      Kokkos::Tools::Experimental::Impl::idForInstance<Maca>(
          reinterpret_cast<uintptr_t>(this));

  // Team Scratch Level 1 Space
  int m_n_team_scratch                            = 10;
  mutable int64_t m_team_scratch_current_size[10] = {};
  mutable void *m_team_scratch_ptr[10]            = {};
  mutable std::atomic_int m_team_scratch_pool[10] = {};
  int32_t *m_scratch_locks                        = nullptr;
  size_t m_num_scratch_locks                      = 0;

  static std::set<int> maca_devices;
  static std::map<int, unsigned long *> constantMemHostStaging;
  static std::map<int, SharedResourceLock> constantMemReusable;

  int verify_is_initialized(const char *const label) const;

  MacaInternal(macaStream_t stream);
  ~MacaInternal();
  MacaInternal(const MacaInternal &)            = delete;
  MacaInternal &operator=(const MacaInternal &) = delete;

  void print_configuration(std::ostream &) const;

  void fence() const;
  void fence(const std::string &) const;

  // Using Maca API function/objects will be w.r.t. device 0 unless
  // macaSetDevice(device_id) is called with the correct device_id.
  // The correct device_id is stored in the variable
  // MacaInternal::m_macaDev set in Maca::impl_initialize(). In the case
  // where multiple Maca instances are used, or threads are launched
  // using non-default Maca execution space after initialization, all Maca
  // API calls must follow a call to macaSetDevice(device_id) when an
  // execution space or MacaInternal object is provided to ensure all
  // computation is done on the correct device.

  // FIXME: Not all Maca API calls require us to set device. Potential
  // performance gain by selectively setting device.

  // Set the device in to the device stored by this instance for Maca API calls.
  void set_maca_device() const {
    verify_is_initialized("set_maca_device");
    KOKKOS_IMPL_MACA_SAFE_CALL(macaSetDevice(m_macaDev));
  }

  macaError_t maca_free_wrapper(void *ptr) const {
    set_maca_device();
    return macaFree(ptr);
  }

  template <typename T>
  macaError_t maca_func_get_attributes_wrapper(macaFuncAttributes* attr,
                                              T* entry) const {
    set_maca_device();
    return macaFuncGetAttributes(attr, reinterpret_cast<void const*>(entry));
  }

  template <typename T>
  macaError_t maca_func_set_attribute_wrapper(T* entry, macaFuncAttribute attr,
                                             int value) const {
    set_maca_device();
    return macaFuncSetAttribute(reinterpret_cast<void const*>(entry), attr,
                               value);
  }

  template <typename T>
  macaError_t maca_func_set_cache_config_wrapper(T* entry,
                                                macaFuncCache_t value) const {
    set_maca_device();
    return macaFuncSetCacheConfig(reinterpret_cast<void const*>(entry), value);
  }

  macaError_t maca_graph_add_dependencies_wrapper(macaGraph_t graph,
                                                 const macaGraphNode_t *from,
                                                 const macaGraphNode_t *to,
                                                 size_t numDependencies) const {
    set_maca_device();
    return macaGraphAddDependencies(graph, from, to, numDependencies);
  }

  macaError_t maca_graph_add_empty_node_wrapper(
      macaGraphNode_t *pGraphNode, macaGraph_t graph,
      const macaGraphNode_t *pDependencies, size_t numDependencies) const {
    set_maca_device();
    return macaGraphAddEmptyNode(pGraphNode, graph, pDependencies,
                                numDependencies);
  }

  macaError_t maca_graph_add_kernel_node_wrapper(
      macaGraphNode_t *pGraphNode, macaGraph_t graph,
      const macaGraphNode_t *pDependencies, size_t numDependencies,
      const macaKernelNodeParams *pNodeParams) const {
    set_maca_device();
    return macaGraphAddKernelNode(pGraphNode, graph, pDependencies,
                                 numDependencies, pNodeParams);
  }

  macaError_t maca_graph_create_wrapper(macaGraph_t *pGraph,
                                       unsigned int flags) const {
    set_maca_device();
    return macaGraphCreate(pGraph, flags);
  }

  macaError_t maca_graph_destroy_wrapper(macaGraph_t graph) const {
    set_maca_device();
    return macaGraphDestroy(graph);
  }

  macaError_t maca_graph_exec_destroy_wrapper(macaGraphExec_t graphExec) const {
    set_maca_device();
    return macaGraphExecDestroy(graphExec);
  }

  macaError_t maca_graph_instantiate_wrapper(macaGraphExec_t *pGraphExec,
                                            macaGraph_t graph,
                                            macaGraphNode_t *pErrorNode,
                                            char *pLogBuffer,
                                            size_t bufferSize) const {
    set_maca_device();
    return macaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer,
                               bufferSize);
  }

  macaError_t maca_graph_launch_wrapper(macaGraphExec_t graphExec) const {
    set_maca_device();
    return macaGraphLaunch(graphExec, m_stream);
  }

  macaError_t maca_host_malloc_wrapper(
      void **ptr, size_t size,
      unsigned int flags = macaHostMallocDefault) const {
    set_maca_device();
    return macaHostMalloc(ptr, size, flags);
  }

  macaError_t maca_memcpy_async_wrapper(void *dst, const void *src,
                                      size_t sizeBytes,
                                      macaMemcpyKind kind) const {
    set_maca_device();
    return macaMemcpyAsync(dst, src, sizeBytes, kind, m_stream);
  }

  macaError_t maca_memcpy_to_symbol_async_wrapper(const void *symbol,
                                                const void *src,
                                                size_t sizeBytes, size_t offset,
                                                macaMemcpyKind kind) const {
    set_maca_device();
    return macaMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind,
                                  m_stream);
  }

  macaError_t maca_memset_wrapper(void *dst, int value,
                                 size_t sizeBytes) const {
    set_maca_device();
    return macaMemset(dst, value, sizeBytes);
  }

  macaError_t maca_memset_async_wrapper(void *dst, int value,
                                       size_t sizeBytes) const {
    set_maca_device();
    return macaMemsetAsync(dst, value, sizeBytes, m_stream);
  }

  macaError_t maca_stream_create_wrapper(macaStream_t *pStream) const {
    set_maca_device();
    return macaStreamCreate(pStream);
  }

  // Resizing of reduction related scratch spaces
  size_type *scratch_space(std::size_t const size);
  size_type *scratch_flags(std::size_t const size);
  size_type *stage_functor_for_execution(void const *driver,
                                         std::size_t const size) const;
  void mark_functor_for_execution(void const *driver_ptr) const;
  uint32_t impl_get_instance_id() const noexcept;
  int acquire_team_scratch_space();
  // Resizing of team level 1 scratch
  void *resize_team_scratch_space(int scratch_pool_id, std::int64_t bytes,
                                  bool force_shrink = false);
  void release_team_scratch_space(int scratch_pool_id);
};
}  // namespace Impl

namespace Experimental::Impl {
// For each space in partition, create a new stream on the same device as
// base_instance, ignoring weights
template <class T>
std::vector<Maca> impl_partition_space(const Maca &base_instance,
                                      const std::vector<T> &weights) {
  std::vector<Maca> instances;
  instances.reserve(weights.size());
  std::generate_n(
      std::back_inserter(instances), weights.size(), [&base_instance]() {
        macaStream_t stream;
        KOKKOS_IMPL_MACA_SAFE_CALL(base_instance.impl_internal_space_instance()
                                      ->maca_stream_create_wrapper(&stream));
        return Maca(stream, Kokkos::Impl::ManageStream::yes);
      });

  return instances;
}
}  // namespace Experimental::Impl
}  // namespace Kokkos

#endif
