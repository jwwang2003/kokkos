// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_SHUFFLE_REDUCE_HPP
#define KOKKOS_MACA_SHUFFLE_REDUCE_HPP

#include <Kokkos_Macros.hpp>

#if defined(__MACACC__)

#include <Maca/Kokkos_Maca_Vectorization.hpp>

#include <climits>

namespace Kokkos {
namespace Impl {

KOKKOS_FUNCTION constexpr unsigned int maca_shuffle_reduce_active_lane_count(
    unsigned int max_active_thread, unsigned int vector_length,
    unsigned int warp_size = MacaTraits::WarpSize) noexcept {
  unsigned int const active_lanes = max_active_thread * vector_length;
  return warp_size == 0u ? 0u
         : active_lanes < warp_size ? active_lanes
                                    : warp_size;
}

KOKKOS_FUNCTION constexpr unsigned long long maca_shuffle_reduce_active_mask(
    unsigned int max_active_thread, unsigned int vector_length,
    int lane, unsigned int warp_size = MacaTraits::WarpSize) noexcept {
  return maca_shuffle_group_mask(
      int(maca_shuffle_reduce_active_lane_count(max_active_thread,
                                                vector_length, warp_size)),
      lane, int(warp_size));
}

/* Algorithmic constraints:
 *   (a) threads with the same threadIdx.x have same value
 *   (b) blockDim.x == power of two
 *   (x) blockDim.z == 1
 */
template <typename ValueType, typename ReducerType>
__device__ inline void maca_intra_warp_shuffle_reduction(
    ValueType& result, ReducerType const& reducer,
    uint32_t const max_active_thread = blockDim.y) {
  unsigned int shift = 1;

  // Reduce over values from threads with different threadIdx.y
  constexpr unsigned int warp_size = MacaTraits::WarpSize;
  int const lane =
      (threadIdx.y * blockDim.x + threadIdx.x) % int(warp_size);
  auto const mask = Impl::maca_shuffle_reduce_active_mask(
      max_active_thread, static_cast<unsigned int>(blockDim.x), lane,
      warp_size);
  while (blockDim.x * shift < warp_size) {
    ValueType const tmp =
        shfl_down(result, blockDim.x * shift, warp_size, mask);
    // Only join if upper thread is active (this allows non power of two for
    // blockDim.y)
    if (threadIdx.y + shift < max_active_thread) {
      reducer.join(&result, &tmp);
    }
    shift *= 2;
  }

  // Broadcast the result to all the threads in the warp
  result = shfl(result, 0, warp_size, mask);
}

template <typename ValueType, typename ReducerType>
__device__ inline void maca_inter_warp_shuffle_reduction(
    ValueType& value, const ReducerType& reducer,
    const int max_active_thread = blockDim.y) {
  constexpr unsigned int warp_size = MacaTraits::WarpSize;
  constexpr int step_width         = 8;
  // Depending on the ValueType __shared__ memory must be aligned up to 8 byte
  // boundaries. The reason not to use ValueType directly is that for types with
  // constructors it could lead to race conditions.
  __shared__ double sh_result[(sizeof(ValueType) + 7) / 8 * step_width];
  ValueType* result = reinterpret_cast<ValueType*>(&sh_result);
  int const step    = warp_size / blockDim.x;
  int shift         = step_width;
  // Skip the code below if  threadIdx.y % step != 0
  int const id = threadIdx.y % step == 0 ? threadIdx.y / step : INT_MAX;
  if (id < step_width) {
    result[id] = value;
  }
  __syncthreads();
  while (shift <= max_active_thread / step) {
    if (shift <= id && shift + step_width > id && threadIdx.x == 0) {
      reducer.join(&result[id % step_width], &value);
    }
    __syncthreads();
    shift += step_width;
  }

  value = result[0];
  for (int i = 1; (i * step < max_active_thread) && (i < step_width); ++i)
    reducer.join(&value, &result[i]);
  __syncthreads();
}

template <typename ValueType, typename ReducerType>
__device__ inline void maca_intra_block_shuffle_reduction(
    ValueType& value, ReducerType const& reducer,
    int const max_active_thread = blockDim.y) {
  maca_intra_warp_shuffle_reduction(value, reducer, max_active_thread);
  maca_inter_warp_shuffle_reduction(value, reducer, max_active_thread);
}

template <class FunctorType>
__device__ inline bool maca_inter_block_shuffle_reduction(
    typename FunctorType::reference_type value,
    typename FunctorType::reference_type neutral, FunctorType const& reducer,
    typename FunctorType::pointer_type const m_scratch_space,
    typename FunctorType::pointer_type const /*result*/,
    Maca::size_type* const m_scratch_flags,
    int const max_active_thread = blockDim.y) {
  using pointer_type = typename FunctorType::pointer_type;
  using value_type   = typename FunctorType::value_type;

  // Do the intra-block reduction with shfl operations for the intra warp
  // reduction and static shared memory for the inter warp reduction
  maca_intra_block_shuffle_reduction(value, reducer, max_active_thread);

  int const id = threadIdx.y * blockDim.x + threadIdx.x;

  // One thread in the block writes block result to global scratch_memory
  if (id == 0) {
    pointer_type global = m_scratch_space + blockIdx.x;
    *global             = value;
    __threadfence();
  }

  // One warp of last block performs inter block reduction through loading the
  // block values from global scratch_memory
  bool last_block = false;
  __syncthreads();
  constexpr int warp_size = MacaTraits::WarpSize;
  if (id < warp_size) {
    unsigned long long const active_mask = __activemask();
    Maca::size_type count;

    // Figure out whether this is the last block
    if (id == 0) count = Kokkos::atomic_fetch_add(m_scratch_flags, 1);
    count = shfl(count, 0, warp_size, active_mask);

    // Last block does the inter block reduction
    if (count == gridDim.x - 1) {
      // set flag back to zero
      if (id == 0) *m_scratch_flags = 0;
      last_block = true;
      value      = neutral;

      pointer_type const global = m_scratch_space;

      // Reduce all global values with splitting work over threads in one warp
      const int active_threads = blockDim.x * blockDim.y < warp_size
                                     ? blockDim.x * blockDim.y
                                     : warp_size;
      const int step_size      = active_threads;
      for (int i = id; i < static_cast<int>(gridDim.x); i += step_size) {
        value_type tmp = global[i];
        reducer.join(&value, &tmp);
      }

      // Perform shfl reductions within the warp only join if contribution is
      // valid (allows gridDim.x non power of two and <warp_size)
      for (unsigned int i = 1; i < warp_size; i *= 2) {
        if (active_threads > int(i)) {
          value_type tmp = shfl_down(value, i, warp_size, active_mask);
          if (id + i < gridDim.x) reducer.join(&value, &tmp);
        }
        __syncwarp(active_mask);
      }
    }
  }
  // The last block has in its thread=0 the global reduction value through
  // "value"
  return last_block;
}
}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
