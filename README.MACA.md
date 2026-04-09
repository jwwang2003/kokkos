# Kokkos Maca Backend

This repository now contains an initial first-class `Maca` backend scaffold for Kokkos.

The backend is intended to be built with the MXMACA toolchain (`mxcc`) and MXMACA runtime headers/libraries (`mc/*`).

## Requirements

- MXMACA SDK installed, typically under `/opt/maca`
- `mxcc` available in `PATH`, or passed as `CMAKE_CXX_COMPILER`
- MXMACA headers available, especially:
  - `mc/mc_runtime.h`
  - `mc/mc_runtime_api.h`
- MXMACA runtime library available, for example under `${MACA_ROOT}/lib` or `${MACA_ROOT}/lib64`

Recommended environment:

```bash
export MACA_ROOT=/opt/maca
export MACA_PATH=/opt/maca
export PATH=$MACA_ROOT/mxgpu_llvm/bin:$PATH
```

If `mxcc` is not on `PATH`, use:

```bash
export CXX=$MACA_ROOT/mxgpu_llvm/bin/mxcc
```

## Configure

Minimal configure:

```bash
cmake -S . -B build-maca \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON
```

Recommended configure with tests:

```bash
cmake -S . -B build-maca \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_TESTS=ON
```

Enable relocatable device code:

```bash
cmake -S . -B build-maca-rdc \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_ENABLE_MACA_RELOCATABLE_DEVICE_CODE=ON \
  -DKokkos_ENABLE_TESTS=ON
```

Pass an explicit MXMACA GPU target:

```bash
cmake -S . -B build-maca \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1000"
```

Example for a more specific target:

```bash
cmake -S . -B build-maca \
  -DCMAKE_CXX_COMPILER=${CXX:-mxcc} \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_MACA=ON \
  -DKokkos_IMPL_MACAGPU_FLAGS="--offload-arch=xcore1002"
```

## Build

Build the core library:

```bash
cmake --build build-maca --target kokkoscore -j
```

Build everything:

```bash
cmake --build build-maca -j
```

Build tests only:

```bash
cmake --build build-maca --target CoreUnitTest_Maca -j
```

## Test

Run all tests:

```bash
ctest --test-dir build-maca --output-on-failure
```

Run only Maca backend tests:

```bash
ctest --test-dir build-maca -R Maca --output-on-failure
```

Run the main Maca unit test executable directly:

```bash
./build-maca/core/unit_test/CoreUnitTest_Maca
```

If multiple GPUs are present, you can restrict visibility:

```bash
export KOKKOS_VISIBLE_DEVICES=0
ctest --test-dir build-maca -R Maca --output-on-failure
```

## Sanity Checks

Check that the MXMACA headers are visible:

```bash
test -f ${MACA_ROOT}/include/mc/mc_runtime.h
test -f ${MACA_ROOT}/include/mc/mc_runtime_api.h
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

then the MXMACA SDK headers are not on the compiler include path. Fix that first by setting `MACA_ROOT`, using `mxcc` as the compiler, and ensuring the SDK is installed correctly.
