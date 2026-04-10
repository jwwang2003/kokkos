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
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"

cmake --build build-maca-examples --target Kokkos_tutorial_algorithms_01_random_numbers -j
./build-maca-examples/example/tutorial/Algorithms/01_random_numbers/Kokkos_tutorial_algorithms_01_random_numbers
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
