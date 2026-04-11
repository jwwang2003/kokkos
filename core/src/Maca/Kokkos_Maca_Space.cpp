// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <Maca/Kokkos_Maca_Space.hpp>
#include <Maca/Kokkos_Maca_IsXnack.hpp>

#include <Maca/Kokkos_Maca_DeepCopy.hpp>

#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace {

static std::atomic<bool> is_first_hip_managed_allocation(true);

}  // namespace

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

MacaManagedMemorySupport query_maca_managed_memory_support(int device_id) {
  MacaManagedMemorySupport support{};
  int has_managed_memory   = 0;
  int has_pageable_memory  = 0;

  KOKKOS_IMPL_MACA_SAFE_CALL(hipDeviceGetAttribute(
      &has_managed_memory, hipDeviceAttributeManagedMemory, device_id));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipDeviceGetAttribute(
      &has_pageable_memory, hipDeviceAttributePageableMemoryAccess, device_id));

  support.has_managed_memory_attribute =
      static_cast<bool>(has_managed_memory);
  support.has_pageable_memory_access = static_cast<bool>(has_pageable_memory);
  support.gpu_arch_can_access_system_memory =
      gpu_arch_can_access_system_allocations() ||
      support.has_pageable_memory_access;
  support.hmm_mirror_enabled_in_kernel_config =
      xnack_boot_config_has_hmm_mirror();
  support.xnack_enabled_in_environment = xnack_environment_enabled();

  return support;
}

}  // namespace Impl

MacaSpace::MacaSpace()
    : m_device(Maca().maca_device()), m_stream(Maca().maca_stream()) {}
MacaSpace::MacaSpace(int device_id, hipStream_t stream)
    : m_device(device_id), m_stream(stream) {}

MacaHostPinnedSpace::MacaHostPinnedSpace()
    : m_device(Maca().maca_device()), m_stream(Maca().maca_stream()) {}
MacaHostPinnedSpace::MacaHostPinnedSpace(int device_id, hipStream_t stream)
    : m_device(device_id), m_stream(stream) {}

MacaManagedSpace::MacaManagedSpace()
    : m_device(Maca().maca_device()), m_stream(Maca().maca_stream()) {}
MacaManagedSpace::MacaManagedSpace(int device_id, hipStream_t stream)
    : m_device(device_id), m_stream(stream) {}

void* MacaSpace::allocate(const Maca& exec_space,
                         const size_t arg_alloc_size) const {
  return allocate(exec_space, "[unlabeled]", arg_alloc_size);
}

void* MacaSpace::allocate(const Maca& exec_space, const char* arg_label,
                         const size_t arg_alloc_size,
                         const size_t arg_logical_size) const {
  return impl_allocate(exec_space.maca_device(), exec_space.maca_stream(),
                       arg_label, arg_alloc_size, arg_logical_size, true);
}

void* MacaSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void* MacaSpace::allocate(const char* arg_label, const size_t arg_alloc_size,
                         const size_t arg_logical_size) const {
  return impl_allocate(m_device, m_stream, arg_label, arg_alloc_size,
                       arg_logical_size, false);
}

void* MacaSpace::impl_allocate(const int device_id,
                              [[maybe_unused]] const hipStream_t stream,
                              const char* arg_label,
                              const size_t arg_alloc_size,
                              const size_t arg_logical_size,
                              [[maybe_unused]] bool stream_sync_only) const {
  void* ptr = nullptr;
  // Instead of trying to allocate zero memory, return early.
  if (arg_alloc_size == 0) return ptr;

  KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(device_id));

#ifdef KOKKOS_ENABLE_IMPL_MACA_MALLOC_ASYNC
  auto const error_code = hipMallocAsync(&ptr, arg_alloc_size, stream);
  if (stream_sync_only) {
    KOKKOS_IMPL_MACA_SAFE_CALL(hipStreamSynchronize(stream));
  } else {
    KOKKOS_IMPL_MACA_SAFE_CALL(hipDeviceSynchronize());
  }
#else
  auto const error_code = hipMalloc(&ptr, arg_alloc_size);
#endif

  if (error_code != hipSuccess) {
    // This is the only way to clear the last error, which we should do here
    // since we're turning it into an exception here
    (void)hipGetLastError();
    Kokkos::Impl::throw_bad_alloc(name(), arg_alloc_size, arg_label);
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const auto arg_handle = Kokkos::Tools::make_space_handle(name());
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void* MacaHostPinnedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void* MacaHostPinnedSpace::allocate(const char* arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void* MacaHostPinnedSpace::impl_allocate(
    const char* arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void* ptr = nullptr;

  KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(m_device));
  auto const error_code =
      hipHostMalloc(&ptr, arg_alloc_size, hipHostMallocNonCoherent);
  if (error_code != hipSuccess) {
    // This is the only way to clear the last error, which we should do here
    // since we're turning it into an exception here
    (void)hipGetLastError();
    Kokkos::Impl::throw_bad_alloc(name(), arg_alloc_size, arg_label);
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void* MacaManagedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void* MacaManagedSpace::allocate(const char* arg_label,
                                const size_t arg_alloc_size,
                                const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void* MacaManagedSpace::impl_allocate(
    const char* arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void* ptr = nullptr;

  if (arg_alloc_size > 0) {
    KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(m_device));
    if (is_first_hip_managed_allocation.exchange(false) &&
        Kokkos::show_warnings()) {
      auto const support =
          Kokkos::Impl::query_maca_managed_memory_support(m_device);
      if (!support.fully_supported()) {
        std::cerr
            << "Kokkos::Maca::allocation WARNING: MacaManagedSpace is not "
               "fully supported on this system.\n"
            << "                                 "
               "hipDeviceAttributeManagedMemory: "
            << support.has_managed_memory_attribute << '\n'
            << "                                 "
               "hipDeviceAttributePageableMemoryAccess: "
            << support.has_pageable_memory_access << '\n'
            << "                                 "
               "gpu_arch_can_access_system_memory: "
            << support.gpu_arch_can_access_system_memory << '\n'
            << "                                 "
               "kernel_hmm_mirror_enabled: "
            << support.hmm_mirror_enabled_in_kernel_config << '\n'
            << "                                 "
               "xnack_enabled_in_environment: "
            << support.xnack_enabled_in_environment << '\n';
      }
    }
    auto const error_code = hipMallocManaged(&ptr, arg_alloc_size);
    if (error_code != hipSuccess) {
      // This is the only way to clear the last error, which we should do here
      // since we're turning it into an exception here
      (void)hipGetLastError();
      Kokkos::Impl::throw_bad_alloc(name(), arg_alloc_size, arg_label);
    }
    KOKKOS_IMPL_MACA_SAFE_CALL(hipMemAdvise(
        ptr, arg_alloc_size, hipMemAdviseSetCoarseGrain, m_device));
  }

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}
bool MacaManagedSpace::impl_hip_driver_check_page_migration() const {
  return Kokkos::Impl::query_maca_managed_memory_support(m_device)
      .page_migration_supported();
}

void MacaSpace::deallocate(void* const arg_alloc_ptr,
                          const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void MacaSpace::deallocate(const char* arg_label, void* const arg_alloc_ptr,
                          const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void MacaSpace::impl_deallocate(
    const char* arg_label, void* const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
#ifdef KOKKOS_ENABLE_IMPL_MACA_MALLOC_ASYNC
  KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(m_device));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipFreeAsync(arg_alloc_ptr, m_stream));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipDeviceSynchronize());
#else
  KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(m_device));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipFree(arg_alloc_ptr));
#endif
}

void MacaHostPinnedSpace::deallocate(void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void MacaHostPinnedSpace::deallocate(const char* arg_label,
                                    void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size,
                                    const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void MacaHostPinnedSpace::impl_deallocate(
    const char* arg_label, void* const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(m_device));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipHostFree(arg_alloc_ptr));
}

void MacaManagedSpace::deallocate(void* const arg_alloc_ptr,
                                 const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void MacaManagedSpace::deallocate(const char* arg_label,
                                 void* const arg_alloc_ptr,
                                 const size_t arg_alloc_size,
                                 const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void MacaManagedSpace::impl_deallocate(
    const char* arg_label, void* const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  // We have to unset the CoarseGrain property manually as hipFree does not take
  // care of it. Otherwise, the allocation would continue to linger in the
  // kernel mem page table.
  KOKKOS_IMPL_MACA_SAFE_CALL(hipMemAdvise(
      arg_alloc_ptr, arg_alloc_size, hipMemAdviseUnsetCoarseGrain, m_device));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipSetDevice(m_device));
  KOKKOS_IMPL_MACA_SAFE_CALL(hipFree(arg_alloc_ptr));
}

}  // namespace Kokkos
