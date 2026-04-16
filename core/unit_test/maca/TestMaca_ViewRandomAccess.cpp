// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestMaca_Category.hpp>

#include <type_traits>

namespace Test {

template <class MemSpace>
struct TestViewMacaRandomAccess {
  enum : int { N = 1024 };

  using base_view_type = Kokkos::View<double*, MemSpace>;
  using random_view_type =
      Kokkos::View<const double*, MemSpace, Kokkos::MemoryRandomAccess>;

  base_view_type base;
  random_view_type random_access;

  struct TagInit {};
  struct TagCheck {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagInit&, const int i) const { base(i) = i + 1.0; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagCheck&, const int i, long& errors) const {
    if (random_access(i) != i + 1.0) ++errors;
  }

  TestViewMacaRandomAccess() : base("base", N), random_access(base) {}

  static void run() {
    EXPECT_TRUE((std::is_same_v<typename base_view_type::reference_type,
                                double&>));
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
    EXPECT_TRUE(
        (std::is_same_v<typename random_view_type::reference_type, const double>));
#else
    EXPECT_TRUE((std::is_same_v<typename random_view_type::reference_type,
                                const double&>));
#endif

    TestViewMacaRandomAccess self;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Maca, TagInit>(0, N), self);

    long errors = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::Maca, TagCheck>(0, N), self, errors);
    EXPECT_EQ(errors, 0);
  }
};

TEST(maca, view_random_access_values) {
  TestViewMacaRandomAccess<Kokkos::MacaSpace>::run();
  TestViewMacaRandomAccess<Kokkos::MacaManagedSpace>::run();
}

namespace issue_5594 {

template <typename View>
struct InitFunctor {
  View view;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { view(i) = i; }
};

template <typename SourceView, typename RandomAccessView>
struct RandomAccessSubviewFunctor {
  SourceView source;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, int& errors) const {
    RandomAccessView subview(&source(0), source.size());
    if (source(i) != subview(i)) ++errors;
  }
};

template <typename View>
View create_view() {
  using execution_space = typename View::execution_space;

  View view("issue_5594_view", 32);
  Kokkos::parallel_for(
      "maca_random_access_subview_init",
      Kokkos::RangePolicy<execution_space>(0, view.extent(0)),
      InitFunctor<View>{view});
  return view;
}

template <typename MemSpace>
void test_view_subview_const_randomaccess() {
  using execution_space = Kokkos::Maca;
  using view_type       = Kokkos::View<int*, MemSpace>;
  using const_view_type = Kokkos::View<const int*, MemSpace>;
  using random_access_type =
      Kokkos::View<const int*, MemSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged |
                                        Kokkos::RandomAccess>>;

  view_type non_const = create_view<view_type>();
  const_view_type view(non_const);

  int errors = 0;
  Kokkos::parallel_reduce(
      "maca_random_access_subview_check",
      Kokkos::RangePolicy<execution_space>(0, view.extent(0)),
      RandomAccessSubviewFunctor<const_view_type, random_access_type>{view},
      errors);
  EXPECT_EQ(errors, 0);
}

}  // namespace issue_5594

TEST(maca, view_subview_const_randomaccess) {
  issue_5594::test_view_subview_const_randomaccess<Kokkos::MacaSpace>();
  issue_5594::test_view_subview_const_randomaccess<Kokkos::MacaManagedSpace>();
}

}  // namespace Test
