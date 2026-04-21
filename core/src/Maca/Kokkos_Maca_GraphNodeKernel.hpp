/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_GraphNodeKernel.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_GRAPHNODEKERNEL_HPP
#define KOKKOS_MACA_GRAPHNODEKERNEL_HPP

#include <Kokkos_Graph_fwd.hpp>

#include <impl/Kokkos_GraphImpl.hpp>

#include <Kokkos_Parallel.hpp>
#include <Kokkos_Parallel_Reduce.hpp>

#include <Maca/Kokkos_Maca_GraphNode_Impl.hpp>

namespace Kokkos {
namespace Impl {

template <typename Functor>
struct GraphNodeThenHostImpl<Kokkos::Maca, Functor> {
  Functor m_functor;
  macaGraphNode_t m_node = nullptr;

  explicit GraphNodeThenHostImpl(Functor functor)
      : m_functor(std::move(functor)) {}

  static void callback(void* data) {
    reinterpret_cast<Functor*>(data)->operator()();
  }

  void add_to_graph(macaGraph_t graph) {
    macaHostNodeParams params = {};
    params.fn                = callback;
    params.userData          = &m_functor;

    KOKKOS_IMPL_MACA_SAFE_CALL(
        macaGraphAddHostNode(&m_node, graph, nullptr, 0, &params));
  }
};

template <typename Functor>
struct GraphNodeCaptureImpl<Kokkos::Maca, Functor> {
  Functor m_functor;
  macaGraphNode_t m_node = nullptr;

  void capture(const Kokkos::Maca& exec, macaGraph_t graph) {
    KOKKOS_IMPL_MACA_SAFE_CALL(
        macaStreamBeginCapture(exec.maca_stream(), macaStreamCaptureModeGlobal));

    m_functor(exec);

    macaGraph_t captured_subgraph(nullptr);

    KOKKOS_IMPL_MACA_SAFE_CALL(
        macaStreamEndCapture(exec.maca_stream(), &captured_subgraph));

    KOKKOS_IMPL_MACA_SAFE_CALL(macaGraphAddChildGraphNode(&m_node, graph, nullptr,
                                                        0, captured_subgraph));
  }
};

template <typename PolicyType, typename Functor, typename PatternTag,
          typename... Args>
class GraphNodeKernelImpl<Kokkos::Maca, PolicyType, Functor, PatternTag, Args...>
    : public PatternImplSpecializationFromTag<PatternTag, Functor, PolicyType,
                                              Args..., Kokkos::Maca>::type {
 public:
  using Policy       = PolicyType;
  using graph_kernel = GraphNodeKernelImpl;
  using base_t =
      typename PatternImplSpecializationFromTag<PatternTag, Functor, Policy,
                                                Args..., Kokkos::Maca>::type;

  // TODO use the name and executionspace
  template <typename PolicyDeduced, typename... ArgsDeduced>
  GraphNodeKernelImpl(std::string label_, Maca const&, Functor arg_functor,
                      PolicyDeduced&& arg_policy, ArgsDeduced&&... args)
      : base_t(std::move(arg_functor), (PolicyDeduced&&)arg_policy,
               (ArgsDeduced&&)args...),
        label(std::move(label_)) {}

  template <typename PolicyDeduced>
  GraphNodeKernelImpl(Kokkos::Maca const& exec_space, Functor arg_functor,
                      PolicyDeduced&& arg_policy)
      : GraphNodeKernelImpl("[unlabeled]", exec_space, std::move(arg_functor),
                            (PolicyDeduced&&)arg_policy) {}

  void set_maca_graph_ptr(macaGraph_t* arg_graph_ptr) {
    m_graph_ptr = arg_graph_ptr;
  }

  void set_maca_graph_node_ptr(macaGraphNode_t* arg_node_ptr) {
    m_graph_node_ptr = arg_node_ptr;
  }

  macaGraphNode_t* get_maca_graph_node_ptr() const { return m_graph_node_ptr; }

  macaGraph_t const* get_maca_graph_ptr() const { return m_graph_ptr; }

  base_t* allocate_driver_memory_buffer(const Maca& exec) const {
    KOKKOS_EXPECTS(m_driver_storage == nullptr);
    std::string alloc_label =
        label + " - GraphNodeKernel global memory functor storage";
    auto alloc_space = MacaSpace::impl_create(exec.maca_device(),
                                              exec.maca_stream());
    m_driver_storage = std::shared_ptr<base_t>(
        static_cast<base_t*>(
            MacaSpace().allocate(exec, alloc_label.c_str(), sizeof(base_t))),
        // MACA allocation and deallocation must use the same device/stream
        // identity because the runtime binds subsequent API calls to the
        // current thread's selected device.
        [alloc_label, alloc_space](base_t* ptr) {
          alloc_space.deallocate(alloc_label.c_str(), ptr, sizeof(base_t));
        });
    KOKKOS_ENSURES(m_driver_storage != nullptr);
    return m_driver_storage.get();
  }

  auto get_driver_storage() const { return m_driver_storage; }

 private:
  macaGraph_t const* m_graph_ptr                    = nullptr;
  macaGraphNode_t* m_graph_node_ptr                 = nullptr;
  mutable std::shared_ptr<base_t> m_driver_storage = nullptr;
  std::string label;
};

struct MacaGraphNodeAggregate {};

template <typename KernelType,
          typename Tag =
              typename PatternTagFromImplSpecialization<KernelType>::type>
struct get_graph_node_kernel_type
    : std::type_identity<
          GraphNodeKernelImpl<Kokkos::Maca, typename KernelType::Policy,
                              typename KernelType::functor_type, Tag>> {};

template <typename KernelType>
struct get_graph_node_kernel_type<KernelType, Kokkos::ParallelReduceTag>
    : std::type_identity<GraphNodeKernelImpl<
          Kokkos::Maca, typename KernelType::Policy,
          CombinedFunctorReducer<typename KernelType::functor_type,
                                 typename KernelType::reducer_type>,
          Kokkos::ParallelReduceTag>> {};

template <typename KernelType>
auto* allocate_driver_storage_for_kernel(const Maca& exec,
                                         KernelType const& kernel) {
  using graph_node_kernel_t =
      typename get_graph_node_kernel_type<KernelType>::type;
  auto const& kernel_as_graph_kernel =
      static_cast<graph_node_kernel_t const&>(kernel);

  return kernel_as_graph_kernel.allocate_driver_memory_buffer(exec);
}

template <typename KernelType>
auto const& get_maca_graph_from_kernel(KernelType const& kernel) {
  using graph_node_kernel_t =
      typename get_graph_node_kernel_type<KernelType>::type;
  auto const& kernel_as_graph_kernel =
      static_cast<graph_node_kernel_t const&>(kernel);
  macaGraph_t const* graph_ptr = kernel_as_graph_kernel.get_maca_graph_ptr();
  KOKKOS_EXPECTS(graph_ptr != nullptr);

  return *graph_ptr;
}

template <typename KernelType>
auto& get_maca_graph_node_from_kernel(KernelType const& kernel) {
  using graph_node_kernel_t =
      typename get_graph_node_kernel_type<KernelType>::type;
  auto const& kernel_as_graph_kernel =
      static_cast<graph_node_kernel_t const&>(kernel);
  auto* graph_node_ptr = kernel_as_graph_kernel.get_maca_graph_node_ptr();
  KOKKOS_EXPECTS(graph_node_ptr != nullptr);

  return *graph_node_ptr;
}
}  // namespace Impl
}  // namespace Kokkos

#endif
