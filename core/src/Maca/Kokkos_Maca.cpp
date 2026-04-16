// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <Maca/Kokkos_Maca.hpp>
#include <Maca/Kokkos_Maca_Instance.hpp>
#include <Maca/Kokkos_Maca_IsXnack.hpp>

#include <impl/Kokkos_CheckUsage.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>

#include <Maca/Kokkos_Maca_Runtime.hpp>

#include <iostream>

namespace {

struct {
  void operator()(Kokkos::Impl::MacaInternal* ptr) const {
    macaStream_t stream = ptr->m_stream;
    delete ptr;
    KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamDestroy(stream));
  }
} customDeleterManagesStream;

}  // namespace

namespace Kokkos {

int Maca::concurrency() const { return Impl::MacaInternal::concurrency(); }

void Maca::impl_initialize(InitializationSettings const& settings) {
  const std::vector<int>& visible_devices = Impl::get_visible_devices();
  const int maca_device_id =
      Impl::get_gpu(settings).value_or(visible_devices[0]);

  KOKKOS_IMPL_MACA_SAFE_CALL(
      macaGetDeviceProperties(&Impl::MacaInternal::m_deviceProp, maca_device_id));
  KOKKOS_IMPL_MACA_SAFE_CALL(macaSetDevice(maca_device_id));

  // Check that we are running on the expected architecture. We print a warning
  // instead of erroring out because the runtime architecture string may not
  // exactly match the target selected at compile time.
  if (Kokkos::show_warnings()) {
#ifdef KOKKOS_ARCH_MACA_GPU
    if (std::string_view arch_name =
            Impl::MacaInternal::m_deviceProp.mxArchName;
        arch_name.find(KOKKOS_ARCH_MACA_GPU) != 0) {
      std::cerr
          << "Kokkos::Maca::initialize WARNING: running kernels compiled for "
          << KOKKOS_ARCH_MACA_GPU << " on " << arch_name << " device.\n";
    }
#elif defined(KOKKOS_ARCH_AMD_GPU)
    if (std::string_view arch_name =
            Impl::MacaInternal::m_deviceProp.mxArchName;
        arch_name.find(KOKKOS_ARCH_AMD_GPU) != 0) {
      std::cerr
          << "Kokkos::Maca::initialize WARNING: running kernels compiled for "
          << KOKKOS_ARCH_AMD_GPU << " on " << arch_name << " device.\n";
    }
#else
    std::cerr << "Kokkos::Maca::initialize WARNING: no Kokkos MACA target "
                 "architecture macro is set; runtime device is "
              << Impl::MacaInternal::m_deviceProp.mxArchName << ".\n";
#endif
  }

  // Print a warning if the user did not select the right GFX942 architecture
#ifdef KOKKOS_ARCH_AMD_GFX942
  if ((Kokkos::show_warnings()) &&
      (Impl::MacaInternal::m_deviceProp.integrated == 1)) {
    std::cerr << "Kokkos::Maca::initialize WARNING: running kernels for MI300X "
                 "(discrete GPU) on a MI300A (APU).\n";
  }
#endif
#ifdef KOKKOS_ARCH_AMD_GFX942_APU
  if (!Kokkos::Impl::xnack_environment_enabled()) {
    std::cerr << R"warning(
Kokkos::Maca::initialize WARNING: Could not determine that xnack is enabled.
                                 Kokkos requires xnack to be enabled for
                                 ARCH_AMD_GFX942_APU (MI300A) to access host
                                 allocations from the device. Set HSA_XNACK=1
                                 in your environment. For further information
                                 on HMM support call `Kokkos::print_configuration`,
                                 or run with KOKKOS_PRINT_CONFIGURATION=1 in your
                                 environment.
)warning";
  }

  if ((Kokkos::show_warnings()) &&
      (Impl::MacaInternal::m_deviceProp.integrated == 0)) {
    std::cerr << "Kokkos::Maca::initialize WARNING: running kernels for MI300A "
                 "(APU) on a MI300X (discrete GPU).\n";
  }
#endif

  // On AMD GPUs this wavefront-per-CU limit is typically 32 for gfx9-class
  // parts even when the architectural maximum is higher.
  const int maxWavesPerCU =
      Impl::MacaInternal::m_deviceProp.major <= 9 ? 32 : 64;
  Impl::MacaInternal::m_maxThreadsPerSM =
      maxWavesPerCU * Impl::MacaTraits::WarpSize;

  // Init the array for used for arbitrarily sized atomics
  desul::Impl::init_lock_arrays();  // FIXME

  // Create the default instance.
  macaStream_t stream;
  KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamCreate(&stream));
  Impl::MacaInternal::default_instance = Impl::HostSharedPtr(
      new Impl::MacaInternal(stream), customDeleterManagesStream);
}

void Maca::impl_finalize() {
  (void)Impl::maca_global_unique_token_locks(true);

  desul::Impl::finalize_lock_arrays();  // FIXME

  // TODO C++20 Use std::views::values.
  for (const auto [_, ptr] : Impl::MacaInternal::constantMemHostStaging) {
    KOKKOS_IMPL_MACA_SAFE_CALL(macaHostFree(ptr));
  }

  // TODO C++20 Use std::views::values.
  for (auto& [_, lock] : Impl::MacaInternal::constantMemReusable) {
    lock.finalize();
  }

  // Destroy the default instance.
  Impl::MacaInternal::default_instance = nullptr;
}

Maca::~Maca() { Impl::check_execution_space_destructor_precondition(name()); }

Maca::Maca()
    : m_space_instance(
          (Impl::check_execution_space_constructor_precondition(name()),
           Impl::MacaInternal::default_instance)) {}

Maca::Maca(macaStream_t const stream, Impl::ManageStream manage_stream)
    : m_space_instance(
          (Impl::check_execution_space_constructor_precondition(name()),
           static_cast<bool>(manage_stream)
               ? Impl::HostSharedPtr(new Impl::MacaInternal(stream),
                                     customDeleterManagesStream)
               : Impl::HostSharedPtr(new Impl::MacaInternal(stream)))) {}

void Maca::print_configuration(std::ostream& os, bool /*verbose*/) const {
  os << "Device Execution Space:\n";
  os << "  KOKKOS_ENABLE_MACA: yes\n";

  os << "Maca Options:\n";
  os << "  KOKKOS_ENABLE_MACA_RELOCATABLE_DEVICE_CODE: ";
#ifdef KOKKOS_ENABLE_MACA_RELOCATABLE_DEVICE_CODE
  os << "yes\n";
#else
  os << "no\n";
#endif

  os << "\nRuntime Configuration:\n";
  os << "  XNACK environment variable set: ";
  os << (Kokkos::Impl::xnack_environment_enabled() ? "yes\n" : "no\n");
  os << "  Kernel reports HMM module via `CONFIG_HMM_MIRROR=y` in "
        "`/boot/config`: ";
  os << (Kokkos::Impl::xnack_boot_config_has_hmm_mirror() ? "yes\n" : "no\n");

  m_space_instance->print_configuration(os);
}

uint32_t Maca::impl_instance_id() const noexcept {
  return m_space_instance->impl_get_instance_id();
}
void Maca::impl_static_fence(const std::string& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Maca>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [&]() {
        for (const auto maca_device : Impl::MacaInternal::maca_devices) {
          KOKKOS_IMPL_MACA_SAFE_CALL(macaSetDevice(maca_device));
          KOKKOS_IMPL_MACA_SAFE_CALL(macaDeviceSynchronize());
        }
      });
}

void Maca::fence(const std::string& name) const {
  m_space_instance->fence(name);
}

macaStream_t Maca::maca_stream() const { return m_space_instance->m_stream; }

int Maca::maca_device() const {
  return impl_internal_space_instance()->m_macaDev;
}

macaDeviceProp_t const& Maca::maca_device_prop() {
  return Impl::MacaInternal::default_instance->m_deviceProp;
}

const char* Maca::name() { return "Maca"; }

namespace Impl {

int g_maca_space_factory_initialized = initialize_space_factory<Maca>("150_MACA");

}  // namespace Impl

}  // namespace Kokkos
