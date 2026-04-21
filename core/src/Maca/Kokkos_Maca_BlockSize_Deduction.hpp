/*================================================================
*  Copyright (C)2026 All rights reserved.
*  FileName : Kokkos_Maca_BlockSize_Deduction.hpp
*  Author   : jwwang2003
*  Email    : wjw_03@outlook.com
*  Date     : Fri 17 Apr 2026 12:06:25 PM CST
================================================================*/

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MACA_BLOCKSIZE_DEDUCTION_HPP
#define KOKKOS_MACA_BLOCKSIZE_DEDUCTION_HPP

#include <functional>
#include <Kokkos_Macros.hpp>
#include <Kokkos_BitManipulation.hpp>

#if defined(__MACACC__)

#include <Maca/Kokkos_Maca_Instance.hpp>
#include <Maca/Kokkos_Maca_KernelLaunch.hpp>

namespace Kokkos {
namespace Impl {

enum class BlockType { Max, Preferred };

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
unsigned maca_get_occupancy_blocksize(const int maca_device);

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism,
          typename DynamicShmemFunctor>
unsigned maca_deduce_blocksize_with_occupancy(
    MacaInternal const *maca_instance, DynamicShmemFunctor const &dynamic_shmem,
    const bool early_termination);

inline int maca_max_active_blocks_per_sm(macaDeviceProp_t const &properties,
                                         macaFuncAttributes const &attributes,
                                         int block_size,
                                         size_t dynamic_shmem) {
  int const regs_per_thread = attributes.numRegs;
  int const allocated_regs_per_thread =
      regs_per_thread == 0 ? 0 : 8 * ((regs_per_thread + 8 - 1) / 8);
  int max_blocks_regs =
      allocated_regs_per_thread == 0
          ? properties.maxBlocksPerMultiProcessor
          : properties.regsPerMultiprocessor /
                (allocated_regs_per_thread * block_size);

  size_t const total_shmem = attributes.sharedSizeBytes + dynamic_shmem;
  size_t const max_dynamic_shmem_per_block =
      attributes.maxDynamicSharedSizeBytes > 0
          ? size_t(attributes.maxDynamicSharedSizeBytes)
          : (attributes.sharedSizeBytes >= size_t(properties.sharedMemPerBlock)
                 ? 0
                 : size_t(properties.sharedMemPerBlock) -
                       size_t(attributes.sharedSizeBytes));
  int const max_blocks_shmem =
      total_shmem > size_t(properties.sharedMemPerBlock) ||
              dynamic_shmem > max_dynamic_shmem_per_block
          ? 0
          : (total_shmem > 0
                 ? int(properties.sharedMemPerMultiprocessor / total_shmem)
                 : max_blocks_regs);

  return std::min({max_blocks_regs, max_blocks_shmem,
                   properties.maxBlocksPerMultiProcessor});
}

template <typename DynamicShmemFunctor, typename LaunchBounds>
inline int maca_deduce_block_size(bool early_termination,
                                  macaDeviceProp_t const &properties,
                                  macaFuncAttributes const &attributes,
                                  DynamicShmemFunctor block_size_to_dynamic_shmem,
                                  LaunchBounds) {
  int const max_threads_per_sm = properties.maxThreadsPerMultiProcessor;
  int const warp_size =
      std::max(1, int(properties.warpSize > 0 ? properties.warpSize
                                              : MacaTraits::WarpSize));
  int max_threads_per_block =
      std::min(LaunchBounds::maxTperB == 0 ? int(properties.maxThreadsPerBlock)
                                           : int(LaunchBounds::maxTperB),
               attributes.maxThreadsPerBlock);
  max_threads_per_block = (max_threads_per_block / warp_size) * warp_size;
  int const min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : LaunchBounds::minBperSM;

  int opt_block_size     = 0;
  int opt_threads_per_sm = 0;

  for (int block_size = max_threads_per_block; block_size >= warp_size;
       block_size -= warp_size) {
    size_t const dynamic_shmem = block_size_to_dynamic_shmem(block_size);

    int blocks_per_sm = maca_max_active_blocks_per_sm(
        properties, attributes, block_size, dynamic_shmem);
    int threads_per_sm = blocks_per_sm * block_size;

    if (threads_per_sm > max_threads_per_sm) {
      blocks_per_sm  = max_threads_per_sm / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }

    if (blocks_per_sm >= min_blocks_per_sm) {
      if ((threads_per_sm > opt_threads_per_sm) ||
          ((block_size >= 128) && (threads_per_sm == opt_threads_per_sm))) {
        opt_block_size     = block_size;
        opt_threads_per_sm = threads_per_sm;
      }
    }

    if (early_termination && opt_block_size != 0) break;
  }

  return opt_block_size;
}

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
unsigned get_preferred_blocksize_impl(const int maca_device) {
  if constexpr (!MacaParallelLaunch<DriverType, LaunchBounds,
                                    LaunchMechanism>::default_launchbounds()) {
    // use the user specified value
    return LaunchBounds::maxTperB;
  } else {
    if (const unsigned occupancy_blocksize =
            maca_get_occupancy_blocksize<DriverType, LaunchBounds,
                                         LaunchMechanism>(maca_device);
        occupancy_blocksize != 0) {
      return occupancy_blocksize;
    }
    if (MacaParallelLaunch<DriverType, LaunchBounds,
                           LaunchMechanism>::get_scratch_size(maca_device) > 0) {
      return MacaTraits::ConservativeThreadsPerBlock;
    }
    return MacaTraits::MaxThreadsPerBlock;
  }
}

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
constexpr unsigned get_max_blocksize_impl() {
  if constexpr (!MacaParallelLaunch<DriverType, LaunchBounds,
                                    LaunchMechanism>::default_launchbounds()) {
    // use the user specified value
    return LaunchBounds::maxTperB;
  } else {
    // we can always fit 1024 threads blocks if we only care about registers
    // ... and don't mind spilling
    return MacaTraits::MaxThreadsPerBlock;
  }
}

// convenience method to select and return the proper function attributes
// for a kernel, given the launch bounds et al.
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          BlockType BlockSize = BlockType::Max,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
macaFuncAttributes get_maca_func_attributes_impl(const int maca_device) {
#ifndef KOKKOS_ENABLE_MACA_MULTIPLE_KERNEL_INSTANTIATIONS
  return MacaParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>::
      get_maca_func_attributes(maca_device);
#else
  if constexpr (!MacaParallelLaunch<DriverType, LaunchBounds,
                                    LaunchMechanism>::default_launchbounds()) {
    // for user defined, we *always* honor the request
    return MacaParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>::
        get_maca_func_attributes(maca_device);
  } else {
    if constexpr (BlockSize == BlockType::Max) {
      return MacaParallelLaunch<
          DriverType, Kokkos::LaunchBounds<MacaTraits::MaxThreadsPerBlock, 1>,
          LaunchMechanism>::get_maca_func_attributes(maca_device);
    } else {
      const int blocksize =
          get_preferred_blocksize_impl<DriverType, LaunchBounds,
                                       LaunchMechanism>(maca_device);
      if (blocksize == MacaTraits::MaxThreadsPerBlock) {
        return MacaParallelLaunch<
            DriverType, Kokkos::LaunchBounds<MacaTraits::MaxThreadsPerBlock, 1>,
            LaunchMechanism>::get_maca_func_attributes(maca_device);
      } else {
        return MacaParallelLaunch<
            DriverType,
            Kokkos::LaunchBounds<MacaTraits::ConservativeThreadsPerBlock, 1>,
            LaunchMechanism>::get_maca_func_attributes(maca_device);
      }
    }
  }
#endif
}

template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism>
unsigned maca_get_occupancy_blocksize(const int maca_device) {
  KOKKOS_IMPL_MACA_SAFE_CALL(macaSetDevice(maca_device));

  int min_grid_size = 0;
  int block_size    = 0;
  auto const err    = macaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size,
      reinterpret_cast<void const *>(
          MacaParallelLaunch<DriverType, LaunchBounds,
                             LaunchMechanism>::get_kernel_func()),
      0, 0);

  if (err != macaSuccess || block_size <= 0) return 0;

  if constexpr (!MacaParallelLaunch<DriverType, LaunchBounds,
                                    LaunchMechanism>::default_launchbounds()) {
    block_size = std::min(block_size, int(LaunchBounds::maxTperB));
  }

  const int warp_size = std::max(1, int(MacaTraits::WarpSize));
  block_size          = (block_size / warp_size) * warp_size;
  return block_size >= warp_size ? unsigned(block_size) : 0;
}

template <typename DriverType, typename LaunchBounds,
          MacaLaunchMechanism LaunchMechanism,
          typename DynamicShmemFunctor>
unsigned maca_deduce_blocksize_with_occupancy(
    MacaInternal const *maca_instance, DynamicShmemFunctor const &dynamic_shmem,
    const bool early_termination) {
  auto const attr = get_maca_func_attributes_impl<DriverType, LaunchBounds,
                                                  BlockType::Preferred,
                                                  LaunchMechanism>(
      maca_instance->m_macaDev);
  auto const &prop = maca_instance->m_deviceProp;
  const int warp_size =
      std::max(1, int(prop.warpSize > 0 ? prop.warpSize : MacaTraits::WarpSize));
  int max_threads_per_block =
      std::min(attr.maxThreadsPerBlock,
               LaunchBounds::maxTperB == 0 ? int(prop.maxThreadsPerBlock)
                                           : int(LaunchBounds::maxTperB));
  max_threads_per_block = (max_threads_per_block / warp_size) * warp_size;
  if (max_threads_per_block < warp_size) return 0;

  const int min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : int(LaunchBounds::minBperSM);
  int best_block_size     = 0;
  int best_threads_per_sm = 0;

  for (int block_size = max_threads_per_block; block_size >= warp_size;
       block_size -= warp_size) {
    int blocks_per_sm = 0;
    auto const err    = macaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        reinterpret_cast<void const *>(
            MacaParallelLaunch<DriverType, LaunchBounds,
                               LaunchMechanism>::get_kernel_func()),
        block_size, dynamic_shmem(block_size));

    if (err != macaSuccess) return 0;

    const int threads_per_sm = blocks_per_sm * block_size;
    if (blocks_per_sm >= min_blocks_per_sm) {
      if ((threads_per_sm > best_threads_per_sm) ||
          ((block_size >= MacaTraits::ConservativeThreadsPerBlock) &&
           (threads_per_sm == best_threads_per_sm))) {
        best_block_size     = block_size;
        best_threads_per_sm = threads_per_sm;
      }
    }

    if (early_termination && best_block_size != 0) break;
  }

  return unsigned(best_block_size);
}

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
unsigned maca_get_opt_block_size_no_shmem(
    MacaInternal const *maca_instance) {
  auto const &prop = maca_instance->m_deviceProp;
  auto const attr  = get_maca_func_attributes_impl<DriverType, LaunchBounds,
                                                   BlockType::Preferred,
                                                   LaunchMechanism>(
      maca_instance->m_macaDev);
  auto const kernel_func =
      MacaParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>::
          get_kernel_func();

  maca_instance->set_maca_device();

  const int warp_size =
      std::max(1, int(prop.warpSize > 0 ? prop.warpSize : MacaTraits::WarpSize));
  int max_threads_per_block =
      std::min(attr.maxThreadsPerBlock,
               LaunchBounds::maxTperB == 0 ? int(prop.maxThreadsPerBlock)
                                           : int(LaunchBounds::maxTperB));
  max_threads_per_block = (max_threads_per_block / warp_size) * warp_size;
  if (max_threads_per_block < warp_size) return 0;

  const int min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : int(LaunchBounds::minBperSM);
  int best_block_size     = 0;
  int best_threads_per_sm = 0;

  for (int block_size = max_threads_per_block; block_size >= warp_size;
       block_size -= warp_size) {
    int blocks_per_sm = 0;
    auto const err    = macaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, reinterpret_cast<void const *>(kernel_func), block_size,
        0);

    if (err != macaSuccess) return 0;

    int threads_per_sm = blocks_per_sm * block_size;
    if (threads_per_sm > prop.maxThreadsPerMultiProcessor) {
      blocks_per_sm  = prop.maxThreadsPerMultiProcessor / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }

    if (blocks_per_sm >= min_blocks_per_sm) {
      if ((threads_per_sm > best_threads_per_sm) ||
          ((block_size >= 128) && (threads_per_sm == best_threads_per_sm))) {
        best_block_size     = block_size;
        best_threads_per_sm = threads_per_sm;
      }
    }
  }

  return unsigned(best_block_size);
}

// Given an initial block-size limitation based on register usage
// determine the block size to select based on LDS limitation
template <BlockType BlockSize, class DriverType, class LaunchBounds,
          typename ShmemFunctor>
unsigned maca_internal_get_block_size(const MacaInternal *maca_instance,
                                     const ShmemFunctor &f,
                                     const unsigned tperb_reg) {
  // translate LB from CUDA to Maca
  const unsigned min_waves_per_eu =
      LaunchBounds::minBperSM ? LaunchBounds::minBperSM : 1;
  const unsigned shmem_per_sm =
      maca_instance->m_deviceProp.maxSharedMemoryPerMultiProcessor;
  unsigned block_size     = tperb_reg;
  unsigned min_block_size = 0;
  do {
    unsigned total_shmem = f(block_size);
    // find how many threads we can fit with this blocksize based on LDS usage
    unsigned tperb_shmem = total_shmem > shmem_per_sm ? 0 : block_size;

    if constexpr (BlockSize == BlockType::Max) {
      // we want the maximum blocksize possible
      // just wait until we get a case where we can fit the LDS per SM
      if (tperb_shmem) return block_size;
    } else {
      // If total_shmem is zero, we set blocks_per_cu_shmem to a number greater
      // than min_waves_per_eu.
      const unsigned blocks_per_cu_shmem =
          total_shmem == 0 ? min_waves_per_eu + 1 : shmem_per_sm / total_shmem;
      const unsigned tperb = tperb_shmem < tperb_reg ? tperb_shmem : tperb_reg;

      // The logic prefers smaller blocks sizes over larger ones to give more
      // flexibility to the scheduler and to decrease the number of threads
      // launched when using Kokkos::AUTO in TeamPolicy. If the block size is
      // smaller than 256, fall back to BlockType::Max condition.
      if (blocks_per_cu_shmem > min_waves_per_eu &&
          tperb >= MacaTraits::ConservativeThreadsPerBlock) {
        min_block_size = block_size;
      } else if ((min_block_size == 0) && (tperb_shmem)) {
        return block_size;
      }
    }
    block_size >>= 1;
  } while (block_size >= MacaTraits::WarpSize);

  return min_block_size;
}

// Standardized blocksize deduction for parallel constructs with no LDS usage
// Returns the preferred blocksize as dictated by register usage
//
// Note: a returned block_size of zero indicates that the algorithm could not
//       find a valid block size.  The caller is responsible for error handling.
template <typename DriverType, typename LaunchBounds>
unsigned maca_get_preferred_blocksize(const int maca_device) {
  return get_preferred_blocksize_impl<DriverType, LaunchBounds>(maca_device);
}

// Heuristic to compute the block size for non-team parallelism
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          MacaLaunchMechanism LaunchMechanism =
              DeduceMacaLaunchMechanism<DriverType>::launch_mechanism>
unsigned get_preferred_blocksize_for_range(MacaInternal const *maca_instance,
                                           size_t requested_parallelism) {
  /* General approach, if the user did not make a launch bounds request
  - If the requested parallelism is less than the available concurrency, get the
  largest block size that would result in at least 1 block per PE, while also:
    - at least 256
    - power of 2
    - no more than 1024
  */

  if constexpr (MacaParallelLaunch<DriverType, LaunchBounds,
                                   LaunchMechanism>::default_launchbounds()) {
    if (requested_parallelism &&
        requested_parallelism < size_t(maca_instance->concurrency())) {
      const unsigned eus = maca_instance->m_deviceProp.multiProcessorCount;
      const unsigned requestedPerEU = (requested_parallelism + eus - 1) / eus;
      // round up to power of 2
      unsigned threadsPerEU = Kokkos::bit_ceil(requestedPerEU);
      threadsPerEU          = std::max(threadsPerEU,
                                       unsigned(MacaTraits::ConservativeThreadsPerBlock));
      threadsPerEU =
          std::min(threadsPerEU, unsigned(MacaTraits::MaxThreadsPerBlock));
      return threadsPerEU;
    }
    if (const unsigned occupancy_blocksize =
            maca_get_opt_block_size_no_shmem<DriverType, LaunchBounds,
                                             LaunchMechanism>(maca_instance);
        occupancy_blocksize != 0) {
      return occupancy_blocksize;
    }
  }
  const int maca_device = maca_instance->m_macaDev;
  return get_preferred_blocksize_impl<DriverType, LaunchBounds>(maca_device);
}

// Standardized blocksize deduction for parallel constructs with no LDS usage
// Returns the max blocksize as dictated by register usage
//
// Note: a returned block_size of zero indicates that the algorithm could not
//       find a valid block size.  The caller is responsible for error handling.
template <typename DriverType, typename LaunchBounds>
unsigned maca_get_max_blocksize() {
  return get_max_blocksize_impl<DriverType, LaunchBounds>();
}

// Standardized blocksize deduction for non-teams parallel constructs with LDS
// usage Returns the 'preferred' blocksize, as determined by the heuristics in
// maca_internal_get_block_size
//
// The ShmemFunctor takes a single argument of the current blocksize under
// consideration, and returns the LDS usage
//
// requested_parallelism is a hint about how much parallelism was requested
// in the parallel construct
//
// Note: a returned block_size of zero indicates that the algorithm could not
//       find a valid block size.  The caller is responsible for error handling.
template <typename DriverType, typename LaunchBounds, typename ShmemFunctor>
unsigned maca_get_preferred_blocksize(MacaInternal const *maca_instance,
                                     ShmemFunctor const &f) {
  if (const unsigned occupancy_blocksize =
          maca_deduce_blocksize_with_occupancy<DriverType, LaunchBounds>(
              maca_instance, f, false);
      occupancy_blocksize != 0) {
    return occupancy_blocksize;
  }
  // get preferred blocksize limited by register usage
  const unsigned tperb_reg =
      maca_get_preferred_blocksize<DriverType, LaunchBounds>(
          maca_instance->m_macaDev);
  return maca_internal_get_block_size<BlockType::Preferred, DriverType,
                                      LaunchBounds>(maca_instance, f,
                                                    tperb_reg);
}

// Standardized blocksize deduction for teams-based parallel constructs with LDS
// usage Returns the 'preferred' blocksize, as determined by the heuristics in
// maca_internal_get_block_size
//
// The ShmemTeamsFunctor takes two arguments: the function attributes and
//  the current blocksize under consideration, and returns the LDS usage
//
// Note: a returned block_size of zero indicates that the algorithm could not
//       find a valid block size.  The caller is responsible for error handling.
template <typename DriverType, typename LaunchBounds,
          typename ShmemTeamsFunctor>
unsigned maca_get_preferred_team_blocksize(MacaInternal const *maca_instance,
                                          ShmemTeamsFunctor const &f) {
  macaFuncAttributes attr = get_maca_func_attributes_impl<
      DriverType, LaunchBounds, BlockType::Preferred>(maca_instance->m_macaDev);
  if (int const block_size = maca_deduce_block_size(
          false, maca_instance->m_deviceProp, attr,
          [&f, &attr](int block_size) {
            size_t const total_shmem = f(attr, unsigned(block_size));
            return total_shmem > size_t(attr.sharedSizeBytes)
                       ? total_shmem - attr.sharedSizeBytes
                       : 0;
          },
          LaunchBounds{});
      block_size != 0) {
    return unsigned(block_size);
  }
  if (const unsigned occupancy_blocksize =
          maca_deduce_blocksize_with_occupancy<DriverType, LaunchBounds>(
              maca_instance,
              [&f, &attr](int block_size) {
                return f(attr, unsigned(block_size));
              },
              false);
      occupancy_blocksize != 0) {
    return occupancy_blocksize;
  }
  const unsigned tperb_reg =
      maca_get_preferred_blocksize<DriverType, LaunchBounds>(
          maca_instance->m_macaDev);
  return maca_internal_get_block_size<BlockType::Preferred, DriverType,
                                      LaunchBounds>(
      maca_instance, std::bind(f, attr, std::placeholders::_1), tperb_reg);
}

// Standardized blocksize deduction for non-teams parallel constructs with LDS
// usage Returns the maximum possible blocksize, as determined by the heuristics
// in maca_internal_get_block_size
//
// The ShmemFunctor takes a single argument of the current blocksize under
// consideration, and returns the LDS usage
//
// Note: a returned block_size of zero indicates that the algorithm could not
//       find a valid block size.  The caller is responsible for error handling.
template <typename DriverType, typename LaunchBounds, typename ShmemFunctor>
unsigned maca_get_max_blocksize(MacaInternal const *maca_instance,
                               ShmemFunctor const &f) {
  if (const unsigned occupancy_blocksize =
          maca_deduce_blocksize_with_occupancy<DriverType, LaunchBounds>(
              maca_instance, f, true);
      occupancy_blocksize != 0) {
    return occupancy_blocksize;
  }
  // get max blocksize limited by register usage
  const unsigned tperb_reg = maca_get_max_blocksize<DriverType, LaunchBounds>();
  return maca_internal_get_block_size<BlockType::Max, DriverType, LaunchBounds>(
      maca_instance, f, tperb_reg);
}

// Standardized blocksize deduction for teams-based parallel constructs with LDS
// usage Returns the maximum possible blocksize, as determined by the heuristics
// in maca_internal_get_block_size
//
// The ShmemTeamsFunctor takes two arguments: the function attributes and
//  the current blocksize under consideration, and returns the LDS usage
//
// Note: a returned block_size of zero indicates that the algorithm could not
//       find a valid block size.  The caller is responsible for error handling.
template <typename DriverType, typename LaunchBounds,
          typename ShmemTeamsFunctor>
unsigned maca_get_max_team_blocksize(MacaInternal const *maca_instance,
                                    ShmemTeamsFunctor const &f) {
  macaFuncAttributes attr =
      get_maca_func_attributes_impl<DriverType, LaunchBounds, BlockType::Max>(
          maca_instance->m_macaDev);
  if (const unsigned occupancy_blocksize =
          maca_deduce_blocksize_with_occupancy<DriverType, LaunchBounds>(
              maca_instance,
              [&f, &attr](int block_size) {
                return f(attr, unsigned(block_size));
              },
              true);
      occupancy_blocksize != 0) {
    return occupancy_blocksize;
  }
  const unsigned tperb_reg = maca_get_max_blocksize<DriverType, LaunchBounds>();
  return maca_internal_get_block_size<BlockType::Max, DriverType, LaunchBounds>(
      maca_instance, std::bind(f, attr, std::placeholders::_1), tperb_reg);
}

}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
