# Maca Script Run Report

Date: 2026-04-11
Repo: `/home/wjw/workspace/kokkos`

## Scope

Executed the two Maca-path scripts from the repository root:

- `./run_maca_algorithms.sh`
- `./run_maca_tutorials.sh`

Build directories used by the scripts:

- algorithms: `build-maca-tests`
- tutorials: `build-maca-debug`

## Executive Summary

Both scripts completed successfully with exit code `0`.

- Algorithm path: built and ran 20 executables, with 207 tests passed and 0 failures.
- Tutorial path: built and ran 9 executables, with all examples completing successfully.
- The Maca-specific random regression test `maca.Random_UniqueIndex_UsesMultipleStates` passed.

## Algorithm Unit Test Results

### Overall

- `Kokkos_UnitTest_Sort`: 17/17 passed
- `Kokkos_UnitTest_Random`: 4/4 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_A`: 16/16 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_B`: 9/9 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_C`: 13/13 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_D`: 31/31 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_E`: 29/29 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_A`: 12/12 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_B`: 8/8 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_C`: 11/11 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_D`: 3/3 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_E`: 6/6 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_F`: 4/4 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_G`: 4/4 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_H`: 9/9 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_I`: 4/4 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_L`: 18/18 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_M`: 5/5 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_P`: 2/2 passed
- `Kokkos_AlgorithmsUnitTest_StdSet_Team_Q`: 2/2 passed

Total: 207/207 tests passed.

### Random-Specific Notes

`Kokkos_UnitTest_Random` completed cleanly. The following four tests passed:

- `maca.Random_XorShift64`
- `maca.Random_XorShift1024_0`
- `maca.Multi_streams`
- `maca.Random_UniqueIndex_UsesMultipleStates`

This confirms that the Maca random-pool path exercised by the new regression test is behaving correctly in the tested build.

## Tutorial Example Results

All 9 tutorial executables built and ran successfully.

| Executable | Result |
| --- | --- |
| `Kokkos_tutorial_01_hello_world` | Printed `Hello World on Kokkos execution space Maca` and the expected `Hello from i = 0..14` sequence |
| `Kokkos_tutorial_02_simple_reduce` | Reported matching parallel and sequential sums: `285` |
| `Kokkos_tutorial_03_simple_view` | Completed with `Result: 8.889609` |
| `Kokkos_tutorial_04_simple_memoryspaces` | Completed with `Result is 460` |
| `Kokkos_tutorial_05_simple_atomics` | Completed with `Found 9558 prime numbers in 100000 random numbers` |
| `Kokkos_tutorial_hierarchicalparallelism_01_thread_teams` | Produced the expected team/thread trace and ended with `Result B: 3072 3072` |
| `Kokkos_tutorial_hierarchicalparallelism_02_nested_parallel_for` | Produced the expected nested-parallel trace and ended with `Result 3072` |
| `Kokkos_tutorial_hierarchicalparallelism_03_vectorization` | Completed with `Result 30084` |
| `Kokkos_tutorial_hierarchicalparallelism_04_team_scan` | Completed with `Result: 102400000 102400000` and reported `Time: 0.221482` during this run |

## Notes

- The hierarchical tutorial examples emit very large trace output by design. Despite the volume, both runs completed normally and produced their expected terminating result lines.
- No missing executables, assertion failures, or runtime aborts were observed once the scripts were run in the non-sandboxed Maca environment.
