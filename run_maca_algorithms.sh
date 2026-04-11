#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wjw/workspace/kokkos}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-maca-tests}"
JOBS="${JOBS:-64}"

TARGETS=(
  Kokkos_UnitTest_Sort
  Kokkos_UnitTest_Random
  Kokkos_AlgorithmsUnitTest_StdSet_A
  Kokkos_AlgorithmsUnitTest_StdSet_B
  Kokkos_AlgorithmsUnitTest_StdSet_C
  Kokkos_AlgorithmsUnitTest_StdSet_D
  Kokkos_AlgorithmsUnitTest_StdSet_E
  Kokkos_AlgorithmsUnitTest_StdSet_Team_A
  Kokkos_AlgorithmsUnitTest_StdSet_Team_B
  Kokkos_AlgorithmsUnitTest_StdSet_Team_C
  Kokkos_AlgorithmsUnitTest_StdSet_Team_D
  Kokkos_AlgorithmsUnitTest_StdSet_Team_E
  Kokkos_AlgorithmsUnitTest_StdSet_Team_F
  Kokkos_AlgorithmsUnitTest_StdSet_Team_G
  Kokkos_AlgorithmsUnitTest_StdSet_Team_H
  Kokkos_AlgorithmsUnitTest_StdSet_Team_I
  Kokkos_AlgorithmsUnitTest_StdSet_Team_L
  Kokkos_AlgorithmsUnitTest_StdSet_Team_M
  Kokkos_AlgorithmsUnitTest_StdSet_Team_P
  Kokkos_AlgorithmsUnitTest_StdSet_Team_Q
)

echo "== Building algorithm unit-test targets =="
cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$JOBS"

echo
echo "== Running algorithm unit-test executables =="
for target in "${TARGETS[@]}"; do
  exe="$BUILD_DIR/algorithms/unit_tests/$target"
  echo
  echo "== $exe =="
  if [[ ! -x "$exe" ]]; then
    echo "missing executable: $exe" >&2
    exit 1
  fi
  "$exe"
done
