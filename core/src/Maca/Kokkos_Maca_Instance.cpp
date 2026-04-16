// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <Maca/Kokkos_Maca_Instance.hpp>
#include <Maca/Kokkos_Maca.hpp>
#include <Maca/Kokkos_Maca_Space.hpp>
#include <Maca/Kokkos_Maca_IsXnack.hpp>
#include <impl/Kokkos_CheckedIntegerOps.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <iostream>
#include <string>
#include <vector>

#ifdef KOKKOS_ENABLE_MACA_RELOCATABLE_DEVICE_CODE
__device__ __constant__ unsigned long kokkos_impl_maca_constant_memory_buffer
    [Kokkos::Impl::MacaTraits::ConstantMemoryUsage / sizeof(unsigned long)];
#endif

namespace Kokkos {
namespace Impl {
Kokkos::View<uint32_t *, MacaSpace> maca_global_unique_token_locks(
    bool deallocate) {
  static Kokkos::View<uint32_t *, MacaSpace> locks =
      Kokkos::View<uint32_t *, MacaSpace>();
  if (!deallocate && locks.extent(0) == 0)
    locks = Kokkos::View<uint32_t *, MacaSpace>(
        "Kokkos::UniqueToken<Maca>::m_locks", MacaInternal::concurrency());
  if (deallocate) locks = Kokkos::View<uint32_t *, MacaSpace>();
  return locks;
}
}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

namespace {

using ScratchGrain = Kokkos::Maca::size_type[Impl::MacaTraits::WarpSize];
constexpr auto sizeScratchGrain = sizeof(ScratchGrain);

std::size_t scratch_count(const std::size_t size) {
  return (size + sizeScratchGrain - 1) / sizeScratchGrain;
}

}  // namespace

//----------------------------------------------------------------------------

int MacaInternal::concurrency() {
  static int const concurrency =
      m_maxThreadsPerSM * m_deviceProp.multiProcessorCount;

  return concurrency;
}

void MacaInternal::print_configuration(std::ostream &s) const {
  s << "macro  KOKKOS_ENABLE_MACA : defined" << '\n';
#if defined(MACA_VERSION)
  s << "macro  MACA_VERSION : " << MACA_VERSION << " = version "
    << MACA_VERSION_MAJOR << '.' << MACA_VERSION_MINOR << '.' << MACA_VERSION_PATCH
    << '\n';
#endif

  s << "macro KOKKOS_ENABLE_ROCTHRUST : "
#if defined(KOKKOS_ENABLE_ROCTHRUST)
    << "defined\n";
#else
    << "undefined\n";
#endif

  s << "macro KOKKOS_ENABLE_IMPL_MACA_MALLOC_ASYNC: ";
#ifdef KOKKOS_ENABLE_IMPL_MACA_MALLOC_ASYNC
  s << "yes\n";
#else
  s << "no\n";
#endif

  for (int i : get_visible_devices()) {
    macaDeviceProp_t maca_prop;
    KOKKOS_IMPL_MACA_SAFE_CALL(macaGetDeviceProperties(&maca_prop, i));
    auto const support = query_maca_managed_memory_support(i);
    std::string gpu_type = maca_prop.integrated == 1 ? "APU" : "dGPU";

    s << "Kokkos::Maca[ " << i << " ] "
      << "mxArch " << maca_prop.mxArchName;
    if (m_macaDev == i)
      s << " : Selected";
    else
      s << " : Not Selected";
    s << '\n'
      << "  Total Global Memory: "
      << ::Kokkos::Impl::human_memory_size(maca_prop.totalGlobalMem) << '\n'
      << "  Shared Memory per Block: "
      << ::Kokkos::Impl::human_memory_size(maca_prop.sharedMemPerBlock) << '\n'
      << "  APU or dGPU: " << gpu_type << '\n'
      << "  Is Large Bar: " << maca_prop.isLargeBar << '\n'
      << "  Supports Managed Memory: "
      << support.has_managed_memory_attribute << '\n'
      << "  Pageable Memory Access: "
      << support.has_pageable_memory_access << '\n'
      << "  Architecture capable of accessing system allocated memory: "
      << support.gpu_arch_can_access_system_memory << '\n'
      << "  HMM mirror enabled in kernel config: "
      << support.hmm_mirror_enabled_in_kernel_config << '\n'
      << "  XNACK enabled in environment: "
      << support.xnack_enabled_in_environment << '\n'
      << "  System allows accessing system allocated memory on GPU: "
      << support.fully_supported()
      << '\n'
      << "  Wavefront Size: " << maca_prop.warpSize << '\n';
  }
}

//----------------------------------------------------------------------------

int MacaInternal::verify_is_initialized(const char *const label) const {
  if (m_macaDev < 0) {
    Kokkos::abort((std::string("Kokkos::Maca::") + label +
                   " : ERROR device not initialized\n")
                      .c_str());
  }
  return 0 <= m_macaDev;
}

uint32_t MacaInternal::impl_get_instance_id() const noexcept {
  return m_instance_id;
}

void MacaInternal::fence() const {
  fence("Kokkos::MacaInternal::fence: Unnamed Internal Fence");
}
void MacaInternal::fence(const std::string &name) const {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Kokkos::Maca>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{
          impl_get_instance_id()},
      [&]() { KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamSynchronize(m_stream)); });
}

MacaInternal::MacaInternal(macaStream_t stream) : m_stream(stream) {
  KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamGetDevice(m_stream, &m_macaDev));
  KOKKOS_IMPL_MACA_SAFE_CALL(macaSetDevice(m_macaDev));
  maca_devices.insert(m_macaDev);

  // Allocate a staging buffer for constant mem in pinned host memory.
  if (!constantMemHostStaging[m_macaDev]) {
    void *constant_mem_void_ptr = nullptr;
    KOKKOS_IMPL_MACA_SAFE_CALL(maca_host_malloc_wrapper(
        &constant_mem_void_ptr, Impl::MacaTraits::ConstantMemoryUsage));
    constantMemHostStaging[m_macaDev] =
        static_cast<unsigned long *>(constant_mem_void_ptr);
  }

  // Initialize the shared resource locking to avoid overwriting the driver of
  // the previous kernel launch.
  constantMemReusable[m_macaDev].initialize();

  //----------------------------------
  // Multiblock reduction uses scratch flags for counters
  // and scratch space for partial reduction values.
  // Allocate some initial space.  This will grow as needed.
  {
    // Maximum number of warps,
    // at most one warp per thread in a warp for reduction.
    unsigned int maxWarpCount =
        m_deviceProp.maxThreadsPerBlock / Impl::MacaTraits::WarpSize;
    if (Impl::MacaTraits::WarpSize < maxWarpCount) {
      maxWarpCount = Impl::MacaTraits::WarpSize;
    }

    const unsigned reduce_block_count =
        maxWarpCount * Impl::MacaTraits::WarpSize;

    (void)scratch_flags(static_cast<size_t>(reduce_block_count * 2) *
                        sizeof(size_type));
    (void)scratch_space(static_cast<size_t>(reduce_block_count * 16) *
                        sizeof(size_type));
  }

  m_num_scratch_locks = concurrency();
  KOKKOS_IMPL_MACA_SAFE_CALL(
      macaMalloc(&m_scratch_locks, sizeof(int32_t) * m_num_scratch_locks));
  KOKKOS_IMPL_MACA_SAFE_CALL(
      macaMemset(m_scratch_locks, 0, sizeof(int32_t) * m_num_scratch_locks));
}

//----------------------------------------------------------------------------

Kokkos::Maca::size_type *MacaInternal::scratch_space(const std::size_t size) {
  if (verify_is_initialized("scratch_space") &&
      m_scratchSpaceCount < scratch_count(size)) {
    auto mem_space = Kokkos::MacaSpace::impl_create(m_macaDev, m_stream);

    if (m_scratchSpace) {
      mem_space.deallocate(m_scratchSpace,
                           m_scratchSpaceCount * sizeScratchGrain);
    }

    m_scratchSpaceCount = scratch_count(size);

    std::size_t alloc_size =
        multiply_overflow_abort(m_scratchSpaceCount, sizeScratchGrain);
    m_scratchSpace = static_cast<size_type *>(
        mem_space.allocate("Kokkos::InternalScratchSpace", alloc_size));
  }

  return m_scratchSpace;
}

Kokkos::Maca::size_type *MacaInternal::scratch_flags(const std::size_t size) {
  if (verify_is_initialized("scratch_flags") &&
      m_scratchFlagsCount < scratch_count(size)) {
    auto mem_space = Kokkos::MacaSpace::impl_create(m_macaDev, m_stream);

    if (m_scratchFlags) {
      mem_space.deallocate(m_scratchFlags,
                           m_scratchFlagsCount * sizeScratchGrain);
    }

    m_scratchFlagsCount = scratch_count(size);

    std::size_t alloc_size =
        multiply_overflow_abort(m_scratchFlagsCount, sizeScratchGrain);
    m_scratchFlags = static_cast<size_type *>(
        mem_space.allocate("Kokkos::InternalScratchFlags", alloc_size));

    // We only zero-initialize the allocation when we actually allocate.
    // It's the responsibility of the features using scratch_flags,
    // namely parallel_reduce and parallel_scan, to reset the used values to 0.
    KOKKOS_IMPL_MACA_SAFE_CALL(
        maca_memset_wrapper(m_scratchFlags, 0, alloc_size));
  }

  return m_scratchFlags;
}

Kokkos::Maca::size_type *MacaInternal::stage_functor_for_execution(
    void const *driver, std::size_t const size) const {
  if (verify_is_initialized("scratch_functor") && m_scratchFunctorSize < size) {
    auto device_mem_space =
        Kokkos::MacaSpace::impl_create(m_macaDev, m_stream);
    auto host_mem_space =
        Kokkos::MacaHostPinnedSpace::impl_create(m_macaDev, m_stream);

    if (m_scratchFunctor) {
      device_mem_space.deallocate(m_scratchFunctor, m_scratchFunctorSize);
      host_mem_space.deallocate(m_scratchFunctorHost, m_scratchFunctorSize);
    }

    m_scratchFunctorSize = size;

    m_scratchFunctor     = static_cast<size_type *>(device_mem_space.allocate(
        "Kokkos::InternalScratchFunctor", m_scratchFunctorSize));
    m_scratchFunctorHost = static_cast<size_type *>(host_mem_space.allocate(
        "Kokkos::InternalScratchFunctorHost", m_scratchFunctorSize));
  }

  // When using HSA_XNACK=1, it is necessary to copy the driver to the host to
  // ensure that the driver is not destroyed before the computation is done.
  // Without this fix, all the atomic tests fail. It is not obvious that this
  // problem is limited to HSA_XNACK=1 even if all the tests pass when
  // HSA_XNACK=0. That's why we always copy the driver.
  KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamSynchronize(m_stream));
  std::memcpy(m_scratchFunctorHost, driver, size);
  KOKKOS_IMPL_MACA_SAFE_CALL(maca_memcpy_async_wrapper(
      m_scratchFunctor, m_scratchFunctorHost, size, macaMemcpyDefault));

  return m_scratchFunctor;
}

int MacaInternal::acquire_team_scratch_space() {
  int current_team_scratch = 0;
  int zero                 = 0;
  while (!m_team_scratch_pool[current_team_scratch].compare_exchange_weak(
      zero, 1, std::memory_order_release, std::memory_order_relaxed)) {
    current_team_scratch = (current_team_scratch + 1) % m_n_team_scratch;
  }

  return current_team_scratch;
}

void *MacaInternal::resize_team_scratch_space(int scratch_pool_id,
                                             std::int64_t bytes,
                                             bool force_shrink) {
  // Multiple ParallelFor/Reduce Teams can call this function at the same time
  // and invalidate the m_team_scratch_ptr. We use a pool to avoid any race
  // condition.
  auto mem_space = Kokkos::MacaSpace::impl_create(m_macaDev, m_stream);
  if (m_team_scratch_current_size[scratch_pool_id] == 0) {
    m_team_scratch_current_size[scratch_pool_id] = bytes;
    m_team_scratch_ptr[scratch_pool_id] =
        mem_space.allocate("Kokkos::MacaSpace::TeamScratchMemory",
                           m_team_scratch_current_size[scratch_pool_id]);
  }
  if ((bytes > m_team_scratch_current_size[scratch_pool_id]) ||
      ((bytes < m_team_scratch_current_size[scratch_pool_id]) &&
       (force_shrink))) {
    mem_space.deallocate("Kokkos::MacaSpace::TeamScratchMemory",
                         m_team_scratch_ptr[scratch_pool_id],
                         m_team_scratch_current_size[scratch_pool_id]);
    m_team_scratch_current_size[scratch_pool_id] = bytes;
    m_team_scratch_ptr[scratch_pool_id] =
        mem_space.allocate("Kokkos::MacaSpace::TeamScratchMemory", bytes);
  }
  return m_team_scratch_ptr[scratch_pool_id];
}

void MacaInternal::release_team_scratch_space(int scratch_pool_id) {
  m_team_scratch_pool[scratch_pool_id] = 0;
}

//----------------------------------------------------------------------------

MacaInternal::~MacaInternal() {
  // First, lock the shared resource locking helper.
  // Then, fence the stream and check if it was involved in the last constant
  // memory launch.
  // Locking is required to avoid a race condition, i.e. it prevents another
  // thread from launching another kernel in-between the fence
  // and the 'check_if_involved_and_unlock'.
  auto lock = MacaInternal::constantMemReusable[m_macaDev].lock();
  this->fence("Kokkos::MacaInternal::finalize: fence on destruction");
  MacaInternal::constantMemReusable[m_macaDev].check_if_involved_and_unlock(
      std::move(lock), m_stream);

  auto device_mem_space = Kokkos::MacaSpace::impl_create(m_macaDev, m_stream);
  if (nullptr != m_scratchSpace || nullptr != m_scratchFlags) {
    device_mem_space.deallocate(m_scratchFlags,
                                m_scratchSpaceCount * sizeScratchGrain);
    device_mem_space.deallocate(m_scratchSpace,
                                m_scratchFlagsCount * sizeScratchGrain);

    if (m_scratchFunctorSize > 0) {
      device_mem_space.deallocate(m_scratchFunctor, m_scratchFunctorSize);
      auto host_mem_space =
          Kokkos::MacaHostPinnedSpace::impl_create(m_macaDev, m_stream);
      host_mem_space.deallocate(m_scratchFunctorHost, m_scratchFunctorSize);
    }
  }

  for (int i = 0; i < m_n_team_scratch; ++i) {
    if (m_team_scratch_current_size[i] > 0)
      device_mem_space.deallocate(m_team_scratch_ptr[i],
                                  m_team_scratch_current_size[i]);
  }

  KOKKOS_IMPL_MACA_SAFE_CALL(maca_free_wrapper(m_scratch_locks));
}

int MacaInternal::m_maxThreadsPerSM = 0;

macaDeviceProp_t MacaInternal::m_deviceProp;

std::mutex MacaInternal::scratchFunctorMutex;

HostSharedPtr<MacaInternal> MacaInternal::default_instance;

std::set<int> MacaInternal::maca_devices                             = {};
std::map<int, unsigned long *> MacaInternal::constantMemHostStaging = {};
std::map<int, SharedResourceLock> MacaInternal::constantMemReusable = {};

//----------------------------------------------------------------------------

Kokkos::Maca::size_type *maca_internal_scratch_space(const Maca &instance,
                                                   const std::size_t size) {
  return instance.impl_internal_space_instance()->scratch_space(size);
}

Kokkos::Maca::size_type *maca_internal_scratch_flags(const Maca &instance,
                                                   const std::size_t size) {
  return instance.impl_internal_space_instance()->scratch_flags(size);
}

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
