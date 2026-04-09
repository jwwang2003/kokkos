// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_HPP
#define KOKKOS_MACA_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_Layout.hpp>
#include <Maca/Kokkos_Maca_Space.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

namespace Kokkos {
namespace Impl {
class MacaInternal;
enum class ManageStream : bool { no, yes };
}  // namespace Impl
/// \class Maca
/// \brief Kokkos device for multicore processors in the host memory space.
class Maca {
 public:
  //------------------------------------
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as a kokkos execution space
  using execution_space = Maca;
  using memory_space    = MacaSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutLeft;
  using size_type    = MacaSpace::size_type;

  using scratch_memory_space = ScratchMemorySpace<Maca>;

  KOKKOS_DEFAULTED_FUNCTION Maca(const Maca&) = default;
  KOKKOS_FUNCTION Maca(Maca&& other) noexcept
      : Maca(static_cast<const Maca&>(other)) {}
  KOKKOS_DEFAULTED_FUNCTION Maca& operator=(const Maca&) = default;
  KOKKOS_FUNCTION Maca& operator=(Maca&& other) noexcept {
    return *this = static_cast<const Maca&>(other);
  }
  ~Maca();
  Maca();

  explicit Maca(hipStream_t stream) : Maca(stream, Impl::ManageStream::no) {}

  Maca(hipStream_t stream, Impl::ManageStream manage_stream);

  //@}
  //------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  /** \brief Wait until all dispatched functors complete.
   *
   * The parallel_for or parallel_reduce dispatch of a functor may return
   * asynchronously, before the functor completes. This method does not return
   * until all dispatched functors on this device have completed.
   */
  static void impl_static_fence(const std::string& name);

  void fence(const std::string& name =
                 "Kokkos::Maca::fence(): Unnamed Instance Fence") const;

  hipStream_t maca_stream() const;

  /// \brief Print configuration information to the given output stream.
  void print_configuration(std::ostream& os, bool verbose = false) const;

  /// \brief Free any resources being consumed by the device.
  static void impl_finalize();

  /** \brief  Initialize the device.
   *
   */
  int maca_device() const;
  static hipDeviceProp_t const& maca_device_prop();

  static void impl_initialize(InitializationSettings const&);

  int concurrency() const;

  static const char* name();

  inline Impl::MacaInternal* impl_internal_space_instance() const {
    return m_space_instance.get();
  }

  uint32_t impl_instance_id() const noexcept;

 private:
  friend bool operator==(Maca const& lhs, Maca const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(Maca const& lhs, Maca const& rhs) {
    return !(lhs == rhs);
  }
  Kokkos::Impl::HostSharedPtr<Impl::MacaInternal> m_space_instance;
};

namespace Impl {
template <>
struct MemorySpaceAccess<MacaSpace, Maca::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};
}  // namespace Impl

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<Maca> {
  static constexpr DeviceType id = DeviceType::Maca;
  static int device_id(const Maca& exec) { return exec.maca_device(); }
};
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

#endif
