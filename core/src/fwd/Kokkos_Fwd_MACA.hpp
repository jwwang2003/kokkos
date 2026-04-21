// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_FWD_HPP_
#define KOKKOS_MACA_FWD_HPP_

#if defined(KOKKOS_ENABLE_MACA)
namespace Kokkos {
class MacaSpace;            ///< Memory space on Maca GPU
class MacaHostPinnedSpace;  ///< Memory space on Host accessible to Maca GPU
class MacaManagedSpace;     ///< Memory migratable between Host and Maca GPU
class Maca;                 ///< Execution space for Maca GPU
}  // namespace Kokkos
#endif
#endif
