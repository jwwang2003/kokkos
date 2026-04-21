/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_Vectorization.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 11:58:39 AM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_VECTORIZATION_HPP
#define KOKKOS_MACA_VECTORIZATION_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {
namespace Impl {

constexpr unsigned long long shfl_all_mask = 0xffffffffffffffffULL;
constexpr int maca_shuffle_width_limit      = int(8 * sizeof(shfl_all_mask));

KOKKOS_FUNCTION constexpr unsigned long long maca_shuffle_group_mask(
    int width, int lane, int warp_size = maca_shuffle_width_limit) noexcept {
  return width <= 0 || warp_size <= 0 || lane < 0 || lane >= warp_size ? 0ULL
         : width >= warp_size ? shfl_all_mask
         : ((1ULL << width) - 1ULL) << ((lane / width) * width);
}

//----------------------------------------------------------------------------
// Shuffle operations require input to be a register (stack) variable

// Derived implements do_shfl_op( T& in, int lane, int width),
// which turns in to one of __shfl_XXX
// Since the logic with respect to value sizes, etc., is the same everywhere,
// put it all in one place.
template <class Derived>
struct in_place_shfl_op {
  // CRTP boilerplate
  __device__ KOKKOS_IMPL_FORCEINLINE const Derived& self() const noexcept {
    return *static_cast<Derived const*>(this);
  }

  // sizeof(Scalar) < sizeof(int) case
  template <class Scalar>
  // requires _assignable_from_bits<Scalar>
  __device__ inline std::enable_if_t<sizeof(Scalar) < sizeof(int)> operator()(
      Scalar& out, Scalar const& in, int lane_or_delta,
      int width, unsigned long long mask = shfl_all_mask) const noexcept {
    using shfl_type = int;
    union conv_type {
      Scalar orig;
      shfl_type conv;
      // This should be fine, members get explicitly reset, which changes the
      // active member
      KOKKOS_FUNCTION conv_type() { conv = 0; }
    };
    conv_type tmp_in;
    tmp_in.orig = in;
    shfl_type tmp_out;
    tmp_out = reinterpret_cast<shfl_type&>(tmp_in.orig);
    conv_type res;
    //------------------------------------------------
    res.conv = self().do_shfl_op(mask, tmp_out, lane_or_delta, width);
    //------------------------------------------------
    out = reinterpret_cast<Scalar&>(res.conv);
  }

  // sizeof(Scalar) == sizeof(int) case
  template <class Scalar>
  // requires _assignable_from_bits<Scalar>
  __device__ inline std::enable_if_t<sizeof(Scalar) == sizeof(int)> operator()(
      Scalar& out, Scalar const& in, int lane_or_delta,
      int width, unsigned long long mask = shfl_all_mask) const noexcept {
    reinterpret_cast<int&>(out) = self().do_shfl_op(
        mask, reinterpret_cast<int const&>(in), lane_or_delta, width);
  }

  template <class Scalar>
  __device__ inline std::enable_if_t<sizeof(Scalar) == sizeof(double)>
  operator()(Scalar& out, Scalar const& in, int lane_or_delta,
             int width, unsigned long long mask = shfl_all_mask) const noexcept {
    reinterpret_cast<double&>(out) = self().do_shfl_op(
        mask, *reinterpret_cast<double const*>(&in), lane_or_delta, width);
  }

  // sizeof(Scalar) > sizeof(double) case
  template <typename Scalar>
  __device__ inline std::enable_if_t<(sizeof(Scalar) > sizeof(double))>
  operator()(Scalar& out, const Scalar& val, int lane_or_delta,
             int width, unsigned long long mask = shfl_all_mask) const noexcept {
    using shuffle_as_t = int;
    constexpr int N    = sizeof(Scalar) / sizeof(shuffle_as_t);

    for (int i = 0; i < N; ++i) {
      reinterpret_cast<shuffle_as_t*>(&out)[i] = self().do_shfl_op(mask,
          reinterpret_cast<shuffle_as_t const*>(&val)[i], lane_or_delta, width);
    }
    // FIXME_MACA - this fence should be removed once the compiler frontend
    // properly supports fence semantics for shuffles
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  }
};

struct in_place_shfl_fn : in_place_shfl_op<in_place_shfl_fn> {
  template <class T>
  __device__ KOKKOS_IMPL_FORCEINLINE T do_shfl_op(unsigned long long mask,
                                                  T& val, int lane,
                                                  int width) const noexcept {
    auto return_val = __shfl_sync(mask, val, lane, width);
    return return_val;
  }
};

template <class... Args>
__device__ KOKKOS_IMPL_FORCEINLINE void in_place_shfl(Args&&... args) noexcept {
  in_place_shfl_fn{}((Args&&)args...);
}

struct in_place_shfl_up_fn : in_place_shfl_op<in_place_shfl_up_fn> {
  template <class T>
  __device__ KOKKOS_IMPL_FORCEINLINE T do_shfl_op(unsigned long long mask,
                                                  T& val, int lane,
                                                  int width) const noexcept {
    auto return_val = __shfl_up_sync(mask, val, lane, width);
    return return_val;
  }
};

template <class... Args>
__device__ KOKKOS_IMPL_FORCEINLINE void in_place_shfl_up(
    Args&&... args) noexcept {
  in_place_shfl_up_fn{}((Args&&)args...);
}

struct in_place_shfl_down_fn : in_place_shfl_op<in_place_shfl_down_fn> {
  template <class T>
  __device__ KOKKOS_IMPL_FORCEINLINE T do_shfl_op(unsigned long long mask,
                                                  T& val, int lane,
                                                  int width) const noexcept {
    auto return_val = __shfl_down_sync(mask, val, lane, width);
    return return_val;
  }
};

template <class... Args>
__device__ KOKKOS_IMPL_FORCEINLINE void in_place_shfl_down(
    Args&&... args) noexcept {
  in_place_shfl_down_fn{}((Args&&)args...);
}

}  // namespace Impl

template <class T>
// requires default_constructible<T> && _assignable_from_bits<T>
__device__ inline T shfl(const T& val, const int& srcLane, const int& width,
                         unsigned long long mask = Impl::shfl_all_mask) {
  T rv = {};
  Impl::in_place_shfl(rv, val, srcLane, width, mask);
  return rv;
}

template <class T>
// requires default_constructible<T> && _assignable_from_bits<T>
__device__ inline T shfl_down(const T& val, int delta, int width,
                              unsigned long long mask = Impl::shfl_all_mask) {
  T rv = {};
  Impl::in_place_shfl_down(rv, val, delta, width, mask);
  return rv;
}

template <class T>
// requires default_constructible<T> && _assignable_from_bits<T>
__device__ inline T shfl_up(const T& val, int delta, int width,
                            unsigned long long mask = Impl::shfl_all_mask) {
  T rv = {};
  Impl::in_place_shfl_up(rv, val, delta, width, mask);
  return rv;
}

}  // namespace Kokkos

#endif
