// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACASPACE_HPP
#define KOKKOS_MACASPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#include <iosfwd>
#include <typeinfo>
#include <string>
#include <cstddef>
#include <iosfwd>

#include <Kokkos_HostSpace.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Maca/Kokkos_Maca_Error.hpp>  // HIP_SAFE_CALL

#include <impl/Kokkos_Profiling_Interface.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <typename T>
struct is_maca_type_space : public std::false_type {};

}  // namespace Impl

/** \brief  Maca on-device memory management */

class MacaSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space    = MacaSpace;
  using execution_space = Maca;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using size_type = unsigned int;

  /*--------------------------------*/

  MacaSpace();

 private:
  MacaSpace(int device_id, hipStream_t stream);

 public:
  static MacaSpace impl_create(int device_id, hipStream_t stream) {
    return MacaSpace(device_id, stream);
  }

  /**\brief  Allocate untracked memory in the hip space */
#ifdef KOKKOS_IMPL_MACA_UNIFIED_MEMORY
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const size_t arg_alloc_size) const {
    return allocate(arg_alloc_size);
  }
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_label, arg_alloc_size, arg_logical_size);
  }
#endif

  void* allocate(const Maca& exec_space, const size_t arg_alloc_size) const;
  void* allocate(const Maca& exec_space, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the hip space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  void* impl_allocate(const int device_id, const hipStream_t stream,
                      const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size,
                      bool stream_sync_only) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return "Maca"; }

 private:
  int m_device;          // Maca device
  hipStream_t m_stream;  // Maca stream
};

template <>
struct Impl::is_maca_type_space<MacaSpace> : public std::true_type {};

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
/** \brief  Host memory that is accessible to Maca execution space
 *          through Maca's host-pinned memory allocation.
 */
class MacaHostPinnedSpace {
 public:
  //! Tag this class as a kokkos memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  using execution_space = HostSpace::execution_space;
  using memory_space    = MacaHostPinnedSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = unsigned int;

  /*--------------------------------*/

  MacaHostPinnedSpace();

 private:
  MacaHostPinnedSpace(int device_id, hipStream_t stream);

 public:
  static MacaHostPinnedSpace impl_create(int device_id, hipStream_t stream) {
    return MacaHostPinnedSpace(device_id, stream);
  }

  /**\brief  Allocate untracked memory in the space */
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const size_t arg_alloc_size) const {
    return allocate(arg_alloc_size);
  }
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_label, arg_alloc_size, arg_logical_size);
  }
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return "HIPHostPinned"; }

 private:
  int m_device;          // Maca device
  hipStream_t m_stream;  // Maca stream

  /*--------------------------------*/
};

template <>
struct Impl::is_maca_type_space<MacaHostPinnedSpace> : public std::true_type {};

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
/** \brief  Memory that is accessible to Maca execution space
 *          and host through Maca's memory page migration.
 */
class MacaManagedSpace {
 public:
  //! Tag this class as a kokkos memory space
  /** \brief  Memory is unified to both device and host via page migration
   *  and therefore able to be used by HostSpace::execution_space and
   *  DeviceSpace::execution_space.
   */
  //! tag this class as a kokkos memory space
  using memory_space    = MacaManagedSpace;
  using execution_space = Maca;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = unsigned int;

  /*--------------------------------*/

  MacaManagedSpace();

 private:
  MacaManagedSpace(int device_id, hipStream_t stream);

 public:
  static MacaManagedSpace impl_create(int device_id, hipStream_t stream) {
    return MacaManagedSpace(device_id, stream);
  }

  /**\brief  Allocate untracked memory in the space */
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const size_t arg_alloc_size) const {
    return allocate(arg_alloc_size);
  }
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace&, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_label, arg_alloc_size, arg_logical_size);
  }
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

  //  internal only method to determine whether page migration is supported
  bool impl_hip_driver_check_page_migration() const;

 private:
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return "HIPManaged"; }

 private:
  int m_device;          // Maca device
  hipStream_t m_stream;  // Maca stream
  /*--------------------------------*/
};

template <>
struct Impl::is_maca_type_space<MacaManagedSpace> : public std::true_type {};

}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert(Kokkos::Impl::MemorySpaceAccess<MacaSpace, MacaSpace>::assignable);

//----------------------------------------

template <>
struct MemorySpaceAccess<HostSpace, MacaSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<HostSpace, MacaHostPinnedSpace> {
  // HostSpace::execution_space == MacaHostPinnedSpace::execution_space
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<HostSpace, MacaManagedSpace> {
  // HostSpace::execution_space != MacaManagedSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

//----------------------------------------

template <>
struct MemorySpaceAccess<MacaSpace, HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<MacaSpace, MacaHostPinnedSpace> {
  // MacaSpace::execution_space != MacaHostPinnedSpace::execution_space
  enum : bool { assignable = false };
  enum : bool { accessible = true };  // MacaSpace::execution_space
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<MacaSpace, MacaManagedSpace> {
  // MacaSpace::execution_space == MacaManagedSpace::execution_space
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

//----------------------------------------
// MacaHostPinnedSpace::execution_space == HostSpace::execution_space
// MacaHostPinnedSpace accessible to both Maca and Host

template <>
struct MemorySpaceAccess<MacaHostPinnedSpace, HostSpace> {
  enum : bool { assignable = false };  // Cannot access from Maca
  enum : bool { accessible = true };   // MacaHostPinnedSpace::execution_space
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<MacaHostPinnedSpace, MacaSpace> {
  enum : bool { assignable = false };  // Cannot access from Host
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<MacaHostPinnedSpace, MacaManagedSpace> {
  enum : bool { assignable = false };  // different exec_space
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

//----------------------------------------
// MacaManagedSpace::execution_space != HostSpace::execution_space
// MacaManagedSpace accessible to both Maca and Host

template <>
struct MemorySpaceAccess<MacaManagedSpace, HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };  // MacaHostPinnedSpace::execution_space
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<MacaManagedSpace, MacaSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<MacaManagedSpace, MacaHostPinnedSpace> {
  enum : bool { assignable = false };  // different exec_space
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* #define KOKKOS_MACASPACE_HPP */
