# Kokkos Maca Backend

This repository now contains an initial first-class `Maca` backend scaffold for Kokkos.

The backend is intended to be built with the MXMACA toolchain (`mxcc`) and MXMACA runtime headers/libraries (`mc/*`).

## Requirements

- MXMACA SDK installed, typically under `/opt/maca`
- `mxcc` available in `PATH`, or passed as `CMAKE_CXX_COMPILER`
- MXMACA headers available, especially:
  - `mc/mc_runtime.h`
  - `mc/mc_runtime_api.h`
- MXMACA runtime library available, for example under `${MACA_PATH}/lib` or `${MACA_PATH}/lib64`

Recommended environment:

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export PATH=$PATH:${CUCC_PATH}/tools:${CUCC_PATH}/bin
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${CUCC_PATH}
```

## Configure

Minimal configure:

```bash
cmake -S . -B build-maca \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ARCH_XCORE1000=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"
```

Recommended configure for development, examples, and tests:

```bash
cmake -S . -B build-maca-tests \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_TESTS=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_ARCH_XCORE1000=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"
```

Enable relocatable device code:

```bash
cmake -S . -B build-maca-rdc \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_TESTS=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_ENABLE_MACA_RELOCATABLE_DEVICE_CODE=ON \
  -DKokkos_ARCH_XCORE1000=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"
```

Example for a different target GPU:

```bash
cmake -S . -B build-maca-tests \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_TESTS=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1002"
```

## Build

Build the core library only:

```bash
cmake --build build-maca-tests --target kokkoscore -j
```

Build everything in the configured tree:

```bash
cmake --build build-maca-tests -j
```

Build only the Maca core unit-test executable:

```bash
cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca -j
```

Build only the random unit-test executable:

```bash
cmake --build build-maca-tests --target Kokkos_UnitTest_Random -j
```

## Examples

Build the random-number tutorial example:

```bash
cmake --build build-maca-tests --target Kokkos_tutorial_algorithms_01_random_numbers -j
```

Run the random-number tutorial example:

```bash
./build-maca-tests/example/tutorial/Algorithms/01_random_numbers/Kokkos_tutorial_algorithms_01_random_numbers
```

If you want a tree dedicated to examples, configure one and build the same target:

```bash
cmake -S . -B build-maca-examples \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_ARCH_XCORE1000=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"

cmake --build build-maca-examples --target Kokkos_tutorial_algorithms_01_random_numbers -j
./build-maca-examples/example/tutorial/Algorithms/01_random_numbers/Kokkos_tutorial_algorithms_01_random_numbers
```

Build the launch-bounds particle tutorial example:

```bash
cmake --build build-maca-tests --target Kokkos_launch_bounds_particles -j
```

Run the launch-bounds particle tutorial example:

```bash
./build-maca-tests/example/tutorial/launch_bounds/Kokkos_launch_bounds_particles
```

The tutorial keeps the particle-and-ground physics from the reference example,
prints initialization plus baseline step timing, and reports aggregate metrics:

- active particle count
- average height
- total kinetic energy
- maximum speed

Example output shape:

```text
Particle count: 1000000
Step count: 200
Prepare data time: ...
Baseline total time: ...
Baseline per-step time: ...
Baseline metrics: active=... avg_height=... total_ke=... max_speed=...
Sample particle 123: pos=(...), vel=(...)
```

## Unit Tests

Run all tests registered in the build tree:

```bash
ctest --test-dir build-maca-tests --output-on-failure
```

Run only Maca tests:

```bash
ctest --test-dir build-maca-tests -R Maca --output-on-failure
```

Run the Maca core unit-test executable directly:

```bash
./build-maca-tests/core/unit_test/Kokkos_CoreUnitTest_Maca
```

Run the random unit-test executable directly:

```bash
./build-maca-tests/algorithms/unit_tests/Kokkos_UnitTest_Random
```

List the random test cases:

```bash
./build-maca-tests/algorithms/unit_tests/Kokkos_UnitTest_Random --gtest_list_tests
```

Run only the Maca-specific `Random_UniqueIndex` regression:

```bash
./build-maca-tests/algorithms/unit_tests/Kokkos_UnitTest_Random \
  --gtest_filter='maca.Random_UniqueIndex_UsesMultipleStates'
```

Build and run the main algorithms test executables in one pass:

```bash
./run_maca_algorithms.sh
```

Use a different build tree or more parallel jobs:

```bash
BUILD_DIR=/home/wjw/workspace/kokkos/build-maca-tests JOBS=16 ./run_maca_algorithms.sh
```

If multiple GPUs are present, you can restrict visibility before running tests:

```bash
export KOKKOS_VISIBLE_DEVICES=0
ctest --test-dir build-maca-tests -R Maca --output-on-failure
```

## Managed Memory Setup

For `MacaManagedSpace` and `Kokkos::SharedSpace` to be treated as fully
supported, all of the following need to line up:

- `hipDeviceAttributeManagedMemory == 1`
- `hipDeviceAttributePageableMemoryAccess == 1`
- the running kernel reports `CONFIG_HMM_MIRROR=y`
- `HSA_XNACK=1` is set in the runtime environment

Recommended environment for managed-memory validation:

```bash
export MACA_PATH=/opt/maca
export CUCC_PATH=/opt/maca/tools/cu-bridge
export PATH=$PATH:${CUCC_PATH}/tools:${CUCC_PATH}/bin
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${CUCC_PATH}
export HSA_XNACK=1
```

Recommended configure for a managed-memory validation tree:

```bash
cmake -S . -B build-maca-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_BENCHMARKS=ON \
  -DKokkos_ARCH_XCORE1000=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"
```

Run the managed-memory validator:

```bash
./run_maca_managed_memory_validation.sh
```

Use a different build tree or more parallel jobs:

```bash
BUILD_DIR=/home/wjw/workspace/kokkos/build-maca-perf JOBS=16 ./run_maca_managed_memory_validation.sh
```

Expected output on a fully supported setup:

- no `MacaManagedSpace is not fully supported on this system` warning
- `System allows accessing system allocated memory on GPU: 1` in printed Kokkos configuration
- `Kokkos_PerformanceTest_SharedSpace` runs the migration timing loop instead of printing a skip message

## Performance Benchmarks

Configure a dedicated MACA benchmark tree:

```bash
cmake -S . -B build-maca-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_BENCHMARKS=ON \
  -DKokkos_ARCH_XCORE1000=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"
```

If `google/benchmark` is not already installed, the perf-test CMake logic will
try to fetch it during configure.

Build the main MACA perf targets:

```bash
cmake --build build-maca-perf --target \
  Kokkos_PerformanceTest_Benchmark \
  Kokkos_Benchmark_Atomic_MinMax \
  Kokkos_PerformanceTest_ViewFirstTouch \
  Kokkos_PerformanceTest_MDRangePolicy_Stream \
  Kokkos_PerformanceTest_Mempool \
  Kokkos_PerformanceTest_Atomic \
  Kokkos_PerformanceTest_Reduction -j
```

Run the aggregate benchmark executable directly:

```bash
./build-maca-perf/core/perf_test/Kokkos_PerformanceTest_Benchmark
```

List benchmark names or filter to a subset:

```bash
./build-maca-perf/core/perf_test/Kokkos_PerformanceTest_Benchmark --benchmark_list_tests
./build-maca-perf/core/perf_test/Kokkos_PerformanceTest_Benchmark --benchmark_filter=Gemv
```

Build and run the main MACA perf benchmark executables in one pass:

```bash
./run_perf_tests.sh
```

Use a different build tree or more parallel jobs:

```bash
BUILD_DIR=/home/wjw/workspace/kokkos/build-maca-perf JOBS=16 ./run_perf_tests.sh
```

The script writes JSON benchmark outputs under:

```text
build-maca-perf/perf-results/<timestamp>/
```

## Tutorial Backend Comparison

The `launch_bounds_particles` tutorial is also a useful sanity-check kernel for
comparing backend execution on the same particle update loop.

Dedicated example build trees created in this workspace on 2026-04-12:

```bash
cmake -S . -B build-launch-serial \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_EXAMPLES=ON
cmake --build build-launch-serial --target Kokkos_launch_bounds_particles -j

cmake -S . -B build-launch-openmp \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DKokkos_ENABLE_SERIAL=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_EXAMPLES=ON
cmake --build build-launch-openmp --target Kokkos_launch_bounds_particles -j

cmake -S . -B build-launch-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/wjw/workspace/kokkos/bin/nvcc_wrapper \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_BLACKWELL120=ON \
  -DKokkos_ENABLE_EXAMPLES=ON
cmake --build build-launch-cuda --target Kokkos_launch_bounds_particles -j
```

Measured results from this repository on 2026-04-12 with:

- `1,000,000` particles
- `200` steps
- `Release` builds
- `OpenMP` run with `OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=threads`
- `CUDA` configured with `Kokkos_ARCH_BLACKWELL120=ON`
- GPU verified with `nvidia-smi` as `NVIDIA GeForce RTX 5070 Ti` with compute capability `12.0`

Measured results:

| Backend | Build tree | Status | Prepare data time | Baseline total time | Per-step time |
|---------|------------|--------|-------------------|---------------------|---------------|
| Serial | `build-launch-serial` | ran | `8.052 ms` | `518.860 ms` | `2.594302 ms` |
| OpenMP | `build-launch-openmp` | ran | `8.240 ms` | `110.024 ms` | `0.550122 ms` |
| CUDA | `build-launch-cuda` | ran | `26.553 ms` | `4.304 ms` | `0.021522 ms` |
| MACA | `build-maca-tests` | reference run from 2026-04-11 | `603.638 ms` | `29.702 ms` | `0.148508 ms` |

Relative speedups from the runnable local builds:

- OpenMP vs Serial: about `4.7x`
- CUDA vs OpenMP: about `25.6x`
- CUDA vs Serial: about `121x`

The aggregate physics metrics matched across the runnable local backends to the
printed precision:

```text
active=999984
avg_height=0.000077
total_ke=411.574037 to 411.574039
max_speed=1.566976
```

The earlier MXMACA reference run on 2026-04-11 also printed matching metrics to
the same displayed precision.

These numbers are hardware-, compiler-, and build-type dependent. The `MACA`
row above came from a different toolchain/configuration and should be treated
as a separate reference point, not as a normalized apples-to-apples comparison
with the local GNU/nvcc `Release` builds.

## Sanity Checks

Check that the MXMACA headers are visible:

```bash
test -f ${MACA_PATH}/include/mc/mc_runtime.h
test -f ${MACA_PATH}/include/mc/mc_runtime_api.h
```

Check the compiler:

```bash
mxcc --version
```

Check that CMake sees the backend:

```bash
cmake -S . -B build-maca-check \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON
```

Expected configure output should include:

```text
Kokkos Backends: SERIAL;MACA
```

## Current State

What is implemented in this branch:

- `Kokkos_ENABLE_MACA` build option
- generated backend header integration
- public `Kokkos::Maca`, `Kokkos::MacaSpace`, `Kokkos::MacaHostPinnedSpace`, `Kokkos::MacaManagedSpace`
- profiling/device-type registration
- initial `Maca` backend source tree under `core/src/Maca`
- initial unit-test category integration

What is still expected during bring-up:

- backend compile failures until the MXMACA SDK is available on the build host
- source-level cleanup in `core/src/Maca` to replace remaining HIP-oriented assumptions with Maca-specific semantics
- runtime validation on real Maca hardware

## Known Build Failure Mode

If configure succeeds but compilation fails immediately with:

```text
fatal error: 'mc/mc_runtime.h' file not found
```

then the MXMACA SDK headers are not on the compiler include path. Fix that first by setting `MACA_PATH`, exporting the cu-bridge environment correctly, using `mxcc` as the compiler, and ensuring the SDK is installed correctly.
