# MACA CUDA-Parity P1-P6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining MACA-vs-CUDA implementation and performance gap for `xcore1000` in six scoped phases: launch-time tuning, collective tuning, team launch heuristics, cached-read widening, texture lifetime cleanup, and SIMD scalar-parity support.

**Architecture:** Keep the existing MACA backend structure, but move the missing parity logic into the same places where CUDA already makes its decisions: launch configuration in `Kokkos_Maca_KernelLaunch.hpp`, collective algorithms in `Kokkos_Maca_Shuffle_Reduce.hpp` and `Kokkos_Maca_ReduceScan.hpp`, team sizing in `Kokkos_Maca_TeamPolicyInternal.hpp` and `Kokkos_Maca_ParallelReduce_Team.hpp`, and read-only view caching in a dedicated MACA texture-cache helper. Add targeted MACA tests and a dedicated parity benchmark so each phase has correctness and performance signals.

**Tech Stack:** Kokkos core backends, MACA runtime wrappers, cu-bridge runtime mappings, existing Kokkos unit tests, existing Kokkos benchmark infrastructure.

---

## File Structure

**Create**
- `benchmarks/maca_cuda_parity/CMakeLists.txt`
- `benchmarks/maca_cuda_parity/main.cpp`
- `benchmarks/maca_cuda_parity/random_access.hpp`
- `benchmarks/maca_cuda_parity/team_collectives.hpp`
- `benchmarks/maca_cuda_parity/launch_tuning.hpp`
- `core/src/Maca/Kokkos_Maca_TextureObjectCache.hpp`
- `core/unit_test/maca/TestMaca_ViewRandomAccess.cpp`
- `core/unit_test/maca/TestMaca_TeamCollectives.cpp`
- `core/unit_test/maca/TestMaca_LaunchTuning.cpp`
- `core/unit_test/maca/TestMaca_SIMD.cpp`

**Modify**
- `benchmarks/CMakeLists.txt`
- `core/src/Maca/Kokkos_Maca_Instance.hpp`
- `core/src/Maca/Kokkos_Maca_Instance.cpp`
- `core/src/Maca/Kokkos_Maca_KernelLaunch.hpp`
- `core/src/Maca/Kokkos_Maca_BlockSize_Deduction.hpp`
- `core/src/Maca/Kokkos_Maca_Shuffle_Reduce.hpp`
- `core/src/Maca/Kokkos_Maca_ReduceScan.hpp`
- `core/src/Maca/Kokkos_Maca_Team.hpp`
- `core/src/Maca/Kokkos_Maca_TeamPolicyInternal.hpp`
- `core/src/Maca/Kokkos_Maca_ParallelFor_Team.hpp`
- `core/src/Maca/Kokkos_Maca_ParallelReduce_Team.hpp`
- `core/src/Maca/Kokkos_Maca_View.hpp`
- `core/src/decl/Kokkos_Declare_MACA.hpp`
- `core/unit_test/CMakeLists.txt`
- `core/unit_test/cuda/TestCuda_Spaces.cpp`
- `simd/src/Kokkos_SIMD.hpp`
- `simd/unit_tests/CMakeLists.txt`
- `simd/perf_tests/CMakeLists.txt`

**Reference Only**
- `core/src/Cuda/Kokkos_Cuda_KernelLaunch.hpp`
- `core/src/Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp`
- `core/src/Cuda/Kokkos_Cuda_ReduceScan.hpp`
- `core/src/Cuda/Kokkos_Cuda_Team.hpp`
- `core/src/Cuda/Kokkos_Cuda_Parallel_Team.hpp`
- `core/src/Cuda/Kokkos_Cuda_View.hpp`
- `3rdParty/cu-bridge/include/bridge/runtime/cuda_to_maca_mcr_adaptor.h`
- `3rdParty/cu-bridge/src/bridge/runtime/src/cuda_runtime_wrapper.cpp`

### Task 1: Add MACA Parity Measurement and Regression Coverage

**Files:**
- Create: `benchmarks/maca_cuda_parity/CMakeLists.txt`
- Create: `benchmarks/maca_cuda_parity/main.cpp`
- Create: `benchmarks/maca_cuda_parity/random_access.hpp`
- Create: `benchmarks/maca_cuda_parity/team_collectives.hpp`
- Create: `benchmarks/maca_cuda_parity/launch_tuning.hpp`
- Modify: `benchmarks/CMakeLists.txt`
- Create: `core/unit_test/maca/TestMaca_ViewRandomAccess.cpp`
- Create: `core/unit_test/maca/TestMaca_TeamCollectives.cpp`
- Modify: `core/unit_test/CMakeLists.txt`
- Test: `build-maca-tests/core/unit_test/Kokkos_CoreUnitTest_Maca`
- Test: `build-maca-perf/benchmarks/maca_cuda_parity/Kokkos_maca_cuda_parity`

- [ ] **Step 1: Add the new benchmark directory to the benchmark build**

Update `benchmarks/CMakeLists.txt` to register `maca_cuda_parity` beside the existing benchmark directories.

- [ ] **Step 2: Add a focused MACA parity benchmark executable**

Create a single benchmark executable that runs three benchmark families:
- random-access gather/read-only view loads
- team reduce and team scan kernels
- range/team launch-tuning kernels with occupancy-sensitive scratch usage

The benchmark CLI should take a subtest name and parameter tuple so the same executable can be used in CI and manual perf comparisons.

- [ ] **Step 3: Add MACA correctness tests for the current cached-read path**

Move the MACA random-access test beyond the current type-trait-only check. Reuse the CUDA texture test shape from `core/unit_test/cuda/TestCuda_Spaces.cpp` to validate:
- `MemoryRandomAccess` views return correct values
- unmanaged/random-access subviews created inside device code behave correctly
- both `MacaSpace` and `MacaManagedSpace` coverage exist if the backend supports the access path

- [ ] **Step 4: Add MACA correctness tests for team reduce and scan edge cases**

Create tests that cover:
- power-of-two and non-power-of-two active team sizes
- dynamic-sized reducers
- vector length `1` and legal shuffle-enabled vector lengths
- empty league and small league edge cases

- [ ] **Step 5: Build the new MACA tests and benchmark**

Run: `cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca`

Expected: target builds successfully and includes the new MACA test translation units.

Run: `cmake --build build-maca-perf --target Kokkos_maca_cuda_parity`

Expected: the parity benchmark builds successfully.

### Task 2: Implement P1 Launch-Time Shared-Memory and Cache Tuning

**Files:**
- Modify: `core/src/Maca/Kokkos_Maca_Instance.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_Instance.cpp`
- Modify: `core/src/Maca/Kokkos_Maca_KernelLaunch.hpp`
- Test: `core/unit_test/maca/TestMaca_LaunchTuning.cpp`
- Test: `build-maca-tests/core/unit_test/Kokkos_CoreUnitTest_Maca`

- [ ] **Step 1: Add MACA runtime wrapper methods for function attributes**

Add MACA-side wrappers matching the CUDA-side API shape for:
- `hipFuncGetAttributes`
- `hipFuncSetAttribute`
- optional cache/shared-mem config wrappers when supported by MACA runtime

These wrappers belong on `MacaInternal` so the launch code can stay parallel to CUDA.

- [ ] **Step 2: Port CUDA’s carveout/prefer-shmem decision flow into MACA launch**

Implement a MACA version of CUDA’s `configure_shmem_preference` in `core/src/Maca/Kokkos_Maca_KernelLaunch.hpp`. It should:
- inspect kernel attributes once per `(DriverType, LaunchBounds, device)`
- compute block-size-aware desired occupancy
- adjust dynamic shared memory when lower occupancy is requested
- call `hipFuncSetAttribute(...PreferredSharedMemoryCarveout...)` when the runtime supports it

- [ ] **Step 3: Stop dropping the `prefer_shmem` signal**

Wire the existing `prefer_shmem` boolean through the MACA launch path so team kernels that already request it can affect the launch configuration.

- [ ] **Step 4: Add a launch-tuning regression test**

Create `TestMaca_LaunchTuning.cpp` to verify:
- the code path accepts desired occupancy on MACA policies
- kernels with dynamic shared memory still launch correctly
- the tuning path is a no-op fallback when runtime attribute calls fail or are unavailable

- [ ] **Step 5: Build and run the MACA unit target**

Run: `cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca`

Expected: build succeeds.

Run: `ctest --test-dir build-maca-tests/core/unit_test -R Kokkos_CoreUnitTest_Maca --output-on-failure`

Expected: MACA unit tests pass without new launch-time regressions.

### Task 3: Implement P2 Collective Tuning Parity

**Files:**
- Modify: `core/src/Maca/Kokkos_Maca_Shuffle_Reduce.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_ReduceScan.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_Team.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_ParallelReduce_Team.hpp`
- Modify: `core/unit_test/maca/TestMaca_TeamCollectives.cpp`
- Test: `build-maca-tests/core/unit_test/Kokkos_CoreUnitTest_Maca`

- [ ] **Step 1: Port the CUDA shuffle-mask edge handling to wave64**

Rework the MACA shuffle reduce code to follow the CUDA algorithm structure, but with `MacaTraits::WarpSize` and wave64-safe masks. In particular:
- final-block inter-block reduction must treat partial waves safely
- shuffle steps must only join active lanes
- wave-synchronous paths must not assume CUDA’s `32`-lane masks

- [ ] **Step 2: Align intra-block reduce/scan algorithms with CUDA’s split**

Bring `Kokkos_Maca_ReduceScan.hpp` closer to `Kokkos_Cuda_ReduceScan.hpp` by separating:
- intra-wave reduction
- inter-wave reduction
- inter-block finalization

This should remove older MACA-specific shortcuts that are correct but less tuned.

- [ ] **Step 3: Retune team-member collectives**

Update `Kokkos_Maca_Team.hpp` so:
- `team_reduce`
- `team_scan`
- `vector_reduce`

use the same collective model expected by the new shuffle/reduce helpers, especially for partial waves and blockDim.x > 1.

- [ ] **Step 4: Expand unit coverage for collective corner cases**

Use the new MACA collective test file to cover:
- dynamic-sized reducers
- small active teams
- partial last wave
- scans with and without global accumulation

- [ ] **Step 5: Rebuild and rerun MACA tests**

Run: `cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca`

Expected: build succeeds.

Run: `ctest --test-dir build-maca-tests/core/unit_test -R Kokkos_CoreUnitTest_Maca --output-on-failure`

Expected: no new team-reduce or team-scan regressions.

### Task 4: Implement P3 Team Launch Sizing and Reduction Block-Count Retuning

**Files:**
- Modify: `core/src/Maca/Kokkos_Maca_TeamPolicyInternal.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_ParallelFor_Team.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_ParallelReduce_Team.hpp`
- Modify: `benchmarks/maca_cuda_parity/team_collectives.hpp`
- Test: `build-maca-perf/benchmarks/maca_cuda_parity/Kokkos_maca_cuda_parity`

- [ ] **Step 1: Unify team-size recommendation flow around occupancy-aware block sizing**

Refactor the MACA team-size recommendation path to mirror CUDA’s structure:
- compute a block size from occupancy-aware helpers
- convert to legal team sizes
- keep the power-of-two reduction constraint explicit

- [ ] **Step 2: Replace the MI210-era reduction block-count heuristic**

Retire the current hard-coded `block_max`, `preferred_block_min`, and `items_per_thread` heuristic in `Kokkos_Maca_ParallelReduce_Team.hpp` and replace it with a policy derived from:
- league size
- team size
- number of resident teams per SM from occupancy
- a wave64-aware upper cap for the final block reducer

- [ ] **Step 3: Benchmark the new heuristic with the parity benchmark**

Use the new benchmark executable to compare:
- old vs new team reduce configuration
- low parallelism
- high parallelism
- dynamic scratch heavy kernels

- [ ] **Step 4: Keep the fallback path explicit**

If occupancy data is unavailable or inconsistent, fall back to the current coarse heuristic behind a clearly named helper rather than silently mixing both paths.

- [ ] **Step 5: Build and run the parity benchmark**

Run: `cmake --build build-maca-perf --target Kokkos_maca_cuda_parity`

Expected: build succeeds.

Run: `build-maca-perf/benchmarks/maca_cuda_parity/Kokkos_maca_cuda_parity team-reduce`

Expected: the benchmark runs and reports stable throughput numbers for comparison against the baseline.

### Task 5: Implement P4 and P5 Cached-Load Widening and Texture Lifetime Cleanup

**Files:**
- Create: `core/src/Maca/Kokkos_Maca_TextureObjectCache.hpp`
- Modify: `core/src/Maca/Kokkos_Maca_View.hpp`
- Modify: `core/src/decl/Kokkos_Declare_MACA.hpp`
- Modify: `core/unit_test/maca/TestMaca_ViewRandomAccess.cpp`
- Modify: `core/unit_test/cuda/TestCuda_Spaces.cpp`
- Test: `build-maca-tests/core/unit_test/Kokkos_CoreUnitTest_Maca`

- [ ] **Step 1: Split texture cache responsibilities into a dedicated helper**

Move the texture-cache state and texture-handle creation logic out of `Kokkos_Maca_View.hpp` and into `Kokkos_Maca_TextureObjectCache.hpp`. Keep `Kokkos_Maca_View.hpp` focused on `ViewDataHandle` specialization and fetch semantics.

- [ ] **Step 2: Broaden the fast path to CUDA-like safe cases**

Extend the MACA cached-read path to support:
- ordinary tracked `RandomAccess` const views
- unmanaged subviews created from tracked base allocations
- offset handles that still point inside the tracked base allocation

Do not attempt raw host-created pointers or allocations with no tracker in this phase.

- [ ] **Step 3: Make texture lifetime allocation-coupled**

Replace the current process-lifetime static map semantics with record-keyed cache entries that are erased when the associated allocation record is no longer valid for reuse. The cache must not rely on process shutdown for cleanup.

- [ ] **Step 4: Add correctness coverage for widened paths**

Expand `TestMaca_ViewRandomAccess.cpp` to cover:
- direct tracked views
- unmanaged/random-access subviews
- offset views into a larger allocation
- fallback to raw pointer access when the tracker is unavailable

- [ ] **Step 5: Rebuild and rerun MACA tests**

Run: `cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca`

Expected: build succeeds.

Run: `ctest --test-dir build-maca-tests/core/unit_test -R Kokkos_CoreUnitTest_Maca --output-on-failure`

Expected: MACA random-access view tests pass.

### Task 6: Implement P6 SIMD Scalar-Parity Support for MACA

**Files:**
- Modify: `simd/src/Kokkos_SIMD.hpp`
- Modify: `simd/unit_tests/CMakeLists.txt`
- Modify: `simd/perf_tests/CMakeLists.txt`
- Create: `core/unit_test/maca/TestMaca_SIMD.cpp`
- Test: `build-maca-tests`

- [ ] **Step 1: Add `ForSpace<Kokkos::Maca>` scalar ABI support**

Extend `simd/src/Kokkos_SIMD.hpp` with a `ForSpace<Kokkos::Maca>` specialization that matches the current CUDA/HIP device policy:
- `type = scalar`
- `simd_abi = scalar`

- [ ] **Step 2: Re-enable the matching test surface**

Adjust SIMD unit/perf test gating so MACA follows the same device scalar-parity surface CUDA and HIP already support, instead of being blanket-disabled.

- [ ] **Step 3: Add a minimal MACA compile/runtime smoke test**

Create `TestMaca_SIMD.cpp` to verify the basic device-side `simd` aliases compile and a trivial `parallel_for` executes correctly with the scalar ABI.

- [ ] **Step 4: Keep real vectorized device SIMD out of scope**

Do not try to use cu-bridge’s vendored experimental `simd` implementation for this phase. The plan target is parity with existing CUDA/HIP scalar ABI behavior only.

- [ ] **Step 5: Rebuild the SIMD-enabled test targets**

Run: `cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca`

Expected: MACA unit tests still build with the SIMD alias changes present.

### Task 7: Final Verification and Performance Gate

**Files:**
- Modify: `docs/plans/2026-04-15-maca-cuda-parity-p1-p6-implementation.md`

- [ ] **Step 1: Build the three relevant trees**

Run: `cmake --build build-maca-tests --target Kokkos_CoreUnitTest_Maca`

Expected: success.

Run: `cmake --build build-maca-perf --target Kokkos_gather Kokkos_maca_cuda_parity`

Expected: success.

- [ ] **Step 2: Run correctness coverage**

Run: `ctest --test-dir build-maca-tests/core/unit_test -R Kokkos_CoreUnitTest_Maca --output-on-failure`

Expected: MACA unit coverage passes.

- [ ] **Step 3: Run performance spot checks**

Run: `build-maca-perf/benchmarks/gather/Kokkos_gather 2 10000000 64 256 10 1 1`

Expected: benchmark completes and gives a reusable random-access baseline.

Run: `build-maca-perf/benchmarks/maca_cuda_parity/Kokkos_maca_cuda_parity random-access`

Expected: benchmark completes and reports the widened cached-read path.

Run: `build-maca-perf/benchmarks/maca_cuda_parity/Kokkos_maca_cuda_parity team-reduce`

Expected: benchmark completes and reports the new collective tuning path.

- [ ] **Step 4: Record regression outcomes in this plan file**

Update this plan file with a short verification appendix listing:
- build commands run
- unit test status
- benchmark commands run
- any remaining perf deltas that still need follow-up

## Sequencing Notes

- Implement Task 1 first. P1-P6 need the parity benchmark and focused tests before tuning.
- Implement Task 2 before Tasks 3 and 4. Launch-time carveout tuning affects team-kernel behavior and should be established first.
- Implement Tasks 3 and 4 together. Collective algorithms and team/block heuristics should be tuned as a pair.
- Implement Task 5 after Tasks 2-4. The cached-read path is important, but it is more isolated and easier to validate once kernel launch and collective behavior are stable.
- Implement Task 6 last. SIMD scalar parity is low-risk but also lower immediate perf impact than P1-P5.

## Open Assumptions

- `xcore1000` remains a wave64 target for MACA.
- MACA runtime support for preferred shared-memory carveout exists through the cu-bridge mappings already vendored in this tree.
- Real vectorized device SIMD remains out of scope for this implementation cycle.
