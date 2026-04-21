// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_DECLARE_MACA_HPP
#define KOKKOS_DECLARE_MACA_HPP

#if defined(KOKKOS_ENABLE_MACA)
#include <Maca/Kokkos_Maca.hpp>
#include <Maca/Kokkos_Maca_Space.hpp>
#include <Maca/Kokkos_Maca_DeepCopy.hpp>
#include <Maca/Kokkos_Maca_Half_Impl_Type.hpp>
#include <Maca/Kokkos_Maca_Half_Conversion.hpp>
#include <Maca/Kokkos_Maca_Half_MathematicalFunctions.hpp>
#include <Maca/Kokkos_Maca_Instance.hpp>
#include <Maca/Kokkos_Maca_MDRangePolicy.hpp>
#include <Maca/Kokkos_Maca_ParallelFor_Range.hpp>
#include <Maca/Kokkos_Maca_ParallelFor_MDRange.hpp>
#include <Maca/Kokkos_Maca_ParallelFor_Team.hpp>
#include <Maca/Kokkos_Maca_ParallelReduce_Range.hpp>
#include <Maca/Kokkos_Maca_ParallelReduce_MDRange.hpp>
#include <Maca/Kokkos_Maca_ParallelReduce_Team.hpp>
#include <Maca/Kokkos_Maca_ParallelScan_Range.hpp>
#include <Maca/Kokkos_Maca_SharedAllocationRecord.hpp>
#include <Maca/Kokkos_Maca_UniqueToken.hpp>
#include <Maca/Kokkos_Maca_View.hpp>
#include <Maca/Kokkos_Maca_ZeroMemset.hpp>

namespace Kokkos {
namespace Experimental {
using MacaSpace           = ::Kokkos::MacaSpace;
using MacaHostPinnedSpace = ::Kokkos::MacaHostPinnedSpace;
using MacaManagedSpace    = ::Kokkos::MacaManagedSpace;
using Maca                = ::Kokkos::Maca;
}  // namespace Experimental
}  // namespace Kokkos
#endif

#endif
