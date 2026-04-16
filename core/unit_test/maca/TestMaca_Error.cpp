// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestMaca_Category.hpp>

#include <Maca/Kokkos_Maca_Error.hpp>

namespace Test {

TEST(maca, safe_call_only_aborts_for_fatal_runtime_errors) {
  EXPECT_FALSE(Kokkos::Impl::maca_internal_error_is_fatal(
      macaErrorInitializationError));
  EXPECT_FALSE(
      Kokkos::Impl::maca_internal_error_is_fatal(macaErrorInvalidValue));
  EXPECT_FALSE(Kokkos::Impl::maca_internal_error_is_fatal(
      macaErrorLaunchOutOfResources));

  EXPECT_TRUE(
      Kokkos::Impl::maca_internal_error_is_fatal(macaErrorIllegalAddress));
  EXPECT_TRUE(Kokkos::Impl::maca_internal_error_is_fatal(macaErrorAssert));
  EXPECT_TRUE(
      Kokkos::Impl::maca_internal_error_is_fatal(macaErrorLaunchFailure));
  EXPECT_TRUE(Kokkos::Impl::maca_internal_error_is_fatal(
      static_cast<macaError_t>(mcErrorHardwareStackError)));
  EXPECT_TRUE(Kokkos::Impl::maca_internal_error_is_fatal(
      static_cast<macaError_t>(mcErrorIllegalInstruction)));
  EXPECT_TRUE(Kokkos::Impl::maca_internal_error_is_fatal(
      static_cast<macaError_t>(mcErrorMisalignedAddress)));
  EXPECT_TRUE(Kokkos::Impl::maca_internal_error_is_fatal(
      static_cast<macaError_t>(mcErrorInvalidAddressSpace)));
  EXPECT_TRUE(Kokkos::Impl::maca_internal_error_is_fatal(
      static_cast<macaError_t>(mcErrorInvalidPc)));
}

}  // namespace Test
