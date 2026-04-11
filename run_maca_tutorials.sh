#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wjw/workspace/kokkos}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-maca-tests}"
JOBS="${JOBS:-64}"

TARGETS=(
  Kokkos_tutorial_01_hello_world
  Kokkos_tutorial_02_simple_reduce
  Kokkos_tutorial_03_simple_view
  Kokkos_tutorial_04_simple_memoryspaces
  Kokkos_tutorial_05_simple_atomics
  Kokkos_tutorial_hierarchicalparallelism_01_thread_teams
  Kokkos_tutorial_hierarchicalparallelism_02_nested_parallel_for
  Kokkos_tutorial_hierarchicalparallelism_03_vectorization
  Kokkos_tutorial_hierarchicalparallelism_04_team_scan
)

EXECUTABLES=(
  "$BUILD_DIR/example/tutorial/01_hello_world/Kokkos_tutorial_01_hello_world"
  "$BUILD_DIR/example/tutorial/02_simple_reduce/Kokkos_tutorial_02_simple_reduce"
  "$BUILD_DIR/example/tutorial/03_simple_view/Kokkos_tutorial_03_simple_view"
  "$BUILD_DIR/example/tutorial/04_simple_memoryspaces/Kokkos_tutorial_04_simple_memoryspaces"
  "$BUILD_DIR/example/tutorial/05_simple_atomics/Kokkos_tutorial_05_simple_atomics"
  "$BUILD_DIR/example/tutorial/Hierarchical_Parallelism/01_thread_teams/Kokkos_tutorial_hierarchicalparallelism_01_thread_teams"
  "$BUILD_DIR/example/tutorial/Hierarchical_Parallelism/02_nested_parallel_for/Kokkos_tutorial_hierarchicalparallelism_02_nested_parallel_for"
  "$BUILD_DIR/example/tutorial/Hierarchical_Parallelism/03_vectorization/Kokkos_tutorial_hierarchicalparallelism_03_vectorization"
  "$BUILD_DIR/example/tutorial/Hierarchical_Parallelism/04_team_scan/Kokkos_tutorial_hierarchicalparallelism_04_team_scan"
)

echo "== Building tutorial targets =="
cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$JOBS"

echo
echo "== Running tutorial executables =="
for exe in "${EXECUTABLES[@]}"; do
  echo
  echo "== $exe =="
  if [[ ! -x "$exe" ]]; then
    echo "missing executable: $exe" >&2
    exit 1
  fi
  "$exe"
done
