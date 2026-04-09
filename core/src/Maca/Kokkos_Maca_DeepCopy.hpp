// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_DEEP_COPY_HPP
#define KOKKOS_MACA_DEEP_COPY_HPP

#include <Maca/Kokkos_Maca_Space.hpp>
#include <Maca/Kokkos_Maca_Error.hpp>  // HIP_SAFE_CALL

#include <Maca/Kokkos_Maca_Runtime.hpp>

namespace Kokkos {
namespace Impl {

void DeepCopyHIP(void* dst, const void* src, size_t n);
void DeepCopyAsyncHIP(const Maca& instance, void* dst, const void* src,
                      size_t n);
void DeepCopyAsyncHIP(void* dst, const void* src, size_t n);

template <class MemSpace>
struct DeepCopy<MemSpace, HostSpace, Maca,
                std::enable_if_t<is_maca_type_space<MemSpace>::value>> {
  DeepCopy(void* dst, const void* src, size_t n) { DeepCopyHIP(dst, src, n); }
  DeepCopy(const Maca& instance, void* dst, const void* src, size_t n) {
    DeepCopyAsyncHIP(instance, dst, src, n);
  }
};

template <class MemSpace>
struct DeepCopy<HostSpace, MemSpace, Maca,
                std::enable_if_t<is_maca_type_space<MemSpace>::value>> {
  DeepCopy(void* dst, const void* src, size_t n) { DeepCopyHIP(dst, src, n); }
  DeepCopy(const Maca& instance, void* dst, const void* src, size_t n) {
    DeepCopyAsyncHIP(instance, dst, src, n);
  }
};

template <class MemSpace1, class MemSpace2>
struct DeepCopy<MemSpace1, MemSpace2, Maca,
                std::enable_if_t<is_maca_type_space<MemSpace1>::value &&
                                 is_maca_type_space<MemSpace2>::value>> {
  DeepCopy(void* dst, const void* src, size_t n) { DeepCopyHIP(dst, src, n); }
  DeepCopy(const Maca& instance, void* dst, const void* src, size_t n) {
    DeepCopyAsyncHIP(instance, dst, src, n);
  }
};

template <class MemSpace1, class MemSpace2, class ExecutionSpace>
struct DeepCopy<MemSpace1, MemSpace2, ExecutionSpace,
                std::enable_if_t<is_maca_type_space<MemSpace1>::value &&
                                 is_maca_type_space<MemSpace2>::value &&
                                 !std::is_same_v<ExecutionSpace, Maca>>> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyHIP(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence(fence_string());
    DeepCopyAsyncHIP(dst, src, n);
  }

 private:
  static const std::string& fence_string() {
    static const std::string string =
        std::string("Kokkos::Impl::DeepCopy<") + MemSpace1::name() + "Space, " +
        MemSpace2::name() +
        "Space, ExecutionSpace>::DeepCopy: fence before copy";
    return string;
  }
};

template <class MemSpace, class ExecutionSpace>
struct DeepCopy<MemSpace, HostSpace, ExecutionSpace,
                std::enable_if_t<is_maca_type_space<MemSpace>::value &&
                                 !std::is_same_v<ExecutionSpace, Maca>>> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyHIP(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence(fence_string());
    DeepCopyAsyncHIP(dst, src, n);
  }

 private:
  static const std::string& fence_string() {
    static const std::string string =
        std::string("Kokkos::Impl::DeepCopy<") + MemSpace::name() +
        "Space, HostSpace, ExecutionSpace>::DeepCopy: fence before copy";
    return string;
  }
};

template <class MemSpace, class ExecutionSpace>
struct DeepCopy<HostSpace, MemSpace, ExecutionSpace,
                std::enable_if_t<is_maca_type_space<MemSpace>::value &&
                                 !std::is_same_v<ExecutionSpace, Maca>>> {
  inline DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyHIP(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                  size_t n) {
    exec.fence(fence_string());
    DeepCopyAsyncHIP(dst, src, n);
  }

 private:
  static const std::string& fence_string() {
    static const std::string string =
        std::string("Kokkos::Impl::DeepCopy<HostSpace, ") + MemSpace::name() +
        "Space, ExecutionSpace>::DeepCopy: fence before copy";
    return string;
  }
};
}  // namespace Impl
}  // namespace Kokkos

#endif
