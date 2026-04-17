// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

#include <TestMaca_Category.hpp>
#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <Kokkos_Graph.hpp>

#include <gtest/gtest.h>

namespace {

template <typename ViewType>
struct Increment {
  ViewType data;

  KOKKOS_FUNCTION
  void operator()(const int) const { ++data(); }
};

TEST(TEST_CATEGORY, graph_promises_on_native_objects) {
  Kokkos::Experimental::Graph<Kokkos::Maca> graph{};

  macaGraph_t maca_graph = graph.native_graph();

  ASSERT_NE(maca_graph, nullptr);
  ASSERT_EQ(graph.native_graph_exec(), nullptr);

  graph.instantiate();

  macaGraphExec_t maca_graph_exec = graph.native_graph_exec();

  ASSERT_EQ(graph.native_graph(), maca_graph);
  ASSERT_NE(maca_graph_exec, nullptr);

  graph.submit();

  ASSERT_EQ(graph.native_graph(), maca_graph);
  ASSERT_EQ(graph.native_graph_exec(), maca_graph_exec);
}

TEST(TEST_CATEGORY, graph_instantiate_and_debug_dot_print) {
  using view_t = Kokkos::View<int, Kokkos::Maca>;

  const Kokkos::Maca exec{};

  view_t data(Kokkos::view_alloc(exec, "witness"));

  Kokkos::Experimental::Graph graph{
      Kokkos::Experimental::get_device_handle(exec)};

  graph.root_node().then_parallel_for(1, Increment<view_t>{data});

  graph.instantiate();

  size_t num_nodes;

  KOKKOS_IMPL_MACA_SAFE_CALL(
      mcGraphGetNodes(graph.native_graph(), nullptr, &num_nodes));

  ASSERT_EQ(num_nodes, 2u);

  const auto dot = std::filesystem::temp_directory_path() / "maca_graph.dot";

  KOKKOS_IMPL_MACA_SAFE_CALL(mcGraphDebugDotPrint(
      graph.native_graph(), dot.string().c_str(), mcGraphDebugDotFlagsVerbose));

  ASSERT_TRUE(std::filesystem::exists(dot));
  ASSERT_GT(std::filesystem::file_size(dot), 0u);

  const std::string expected("[A-Za-z0-9_]+Increment[A-Za-z0-9_]+RangePolicy");

  std::stringstream buffer;
  buffer << std::ifstream(dot).rdbuf();

  ASSERT_TRUE(std::regex_search(buffer.str(), std::regex(expected)))
      << "Could not find expected signature regex " << std::quoted(expected)
      << " in " << dot;
}

TEST(TEST_CATEGORY, graph_construct_from_native) {
  using view_t = Kokkos::View<int, Kokkos::MacaManagedSpace>;

  macaGraph_t native_graph = nullptr;
  KOKKOS_IMPL_MACA_SAFE_CALL(mcGraphCreate(&native_graph, 0));

  const Kokkos::Maca exec{};

  Kokkos::Experimental::Graph graph_from_native(
      Kokkos::Experimental::get_device_handle(exec), native_graph);

  ASSERT_EQ(native_graph, graph_from_native.native_graph());

  const view_t data(Kokkos::view_alloc(exec, "witness"));

  graph_from_native.root_node().then_parallel_for(1, Increment<view_t>{data});

  graph_from_native.submit(exec);

  exec.fence();

  ASSERT_EQ(data(), 1);
}

TEST(TEST_CATEGORY, graph_destruction_on_explicit_stream) {
  using view_t = Kokkos::View<int, Kokkos::MacaManagedSpace>;

  macaStream_t stream = nullptr;
  KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamCreate(&stream));

  {
    const Kokkos::Maca exec(stream);
    const view_t data(Kokkos::view_alloc(exec, "witness"));

    {
      auto graph = Kokkos::Experimental::create_graph(
          Kokkos::Experimental::get_device_handle(exec), [&](const auto& root) {
            root.then_parallel_for(1, Increment<view_t>{data});
          });

      graph.instantiate();
      graph.submit(exec);
      exec.fence();
      ASSERT_EQ(data(), 1);
    }

    exec.fence();
    ASSERT_EQ(data(), 1);
  }

  KOKKOS_IMPL_MACA_SAFE_CALL(macaStreamDestroy(stream));
}

}  // namespace
