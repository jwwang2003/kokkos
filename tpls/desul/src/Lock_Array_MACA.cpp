/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef DESUL_ATOMICS_ENABLE_MACA_SEPARABLE_COMPILATION
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* MACA_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
__device__ __constant__ int32_t* MACA_SPACE_ATOMIC_LOCKS_NODE = nullptr;
}  // namespace Impl
}  // namespace desul
#endif

namespace desul {

namespace {

__global__ void init_lock_arrays_maca_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  auto* device_locks = Impl::MACA_SPACE_ATOMIC_LOCKS_DEVICE;
  auto* node_locks   = Impl::MACA_SPACE_ATOMIC_LOCKS_NODE;
  if (i < MACA_SPACE_ATOMIC_MASK + 1) {
    device_locks[i] = 0;
    node_locks[i]   = 0;
  }
}

}  // namespace

namespace Impl {

int32_t* MACA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* MACA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

namespace {

void check_error_and_throw_maca(mcError_t e, const std::string& msg) {
  if (e != mcSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << mcGetErrorName(e)
        << "): " << mcGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}  // namespace

template <typename T>
void init_lock_arrays_maca() {
  if (MACA_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;

  auto error_malloc1 =
      mcMalloc(&MACA_SPACE_ATOMIC_LOCKS_DEVICE_h,
               sizeof(int32_t) * (MACA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_maca(error_malloc1,
                             "init_lock_arrays_maca: mcMalloc device locks");

  auto error_malloc2 =
      mcMallocHost(&MACA_SPACE_ATOMIC_LOCKS_NODE_h,
                   sizeof(int32_t) * (MACA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_maca(error_malloc2,
                             "init_lock_arrays_maca: mcMallocHost node locks");

  auto error_sync1 = mcDeviceSynchronize();
  copy_maca_lock_arrays_to_device();
  check_error_and_throw_maca(error_sync1, "init_lock_arrays_maca: post malloc");

  init_lock_arrays_maca_kernel<<<(MACA_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();

  auto error_sync2 = mcDeviceSynchronize();
  check_error_and_throw_maca(error_sync2, "init_lock_arrays_maca: post init");
}

template <typename T>
void finalize_lock_arrays_maca() {
  if (MACA_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;

  auto error_free1 = mcFree(MACA_SPACE_ATOMIC_LOCKS_DEVICE_h);
  check_error_and_throw_maca(error_free1,
                             "finalize_lock_arrays_maca: free device locks");
  auto error_free2 = mcFreeHost(MACA_SPACE_ATOMIC_LOCKS_NODE_h);
  check_error_and_throw_maca(error_free2,
                             "finalize_lock_arrays_maca: free node locks");
  MACA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  MACA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
#ifdef DESUL_ATOMICS_ENABLE_MACA_SEPARABLE_COMPILATION
  copy_maca_lock_arrays_to_device();
#endif
}

template void init_lock_arrays_maca<int>();
template void finalize_lock_arrays_maca<int>();

}  // namespace Impl

}  // namespace desul
