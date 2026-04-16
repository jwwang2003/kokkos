// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_GRAPHNODE_IMPL_HPP
#define KOKKOS_MACA_GRAPHNODE_IMPL_HPP

#include <Kokkos_Graph_fwd.hpp>

#include <impl/Kokkos_GraphImpl.hpp>

#include <Maca/Kokkos_Maca.hpp>

namespace Kokkos {
namespace Impl {
template <>
struct GraphNodeBackendSpecificDetails<Kokkos::Maca> {
  macaGraphNode_t node = nullptr;

  explicit GraphNodeBackendSpecificDetails() = default;

  explicit GraphNodeBackendSpecificDetails(
      _graph_node_is_root_ctor_tag) noexcept {}
};

template <typename Kernel, typename PredecessorRef>
struct GraphNodeBackendDetailsBeforeTypeErasure<Kokkos::Maca, Kernel,
                                                PredecessorRef> {
 protected:
  GraphNodeBackendDetailsBeforeTypeErasure(
      Kokkos::Maca const &, Kernel &, PredecessorRef const &,
      GraphNodeBackendSpecificDetails<Kokkos::Maca> &) noexcept {}

  GraphNodeBackendDetailsBeforeTypeErasure(
      Kokkos::Maca const &, _graph_node_is_root_ctor_tag,
      GraphNodeBackendSpecificDetails<Kokkos::Maca> &) noexcept {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif
