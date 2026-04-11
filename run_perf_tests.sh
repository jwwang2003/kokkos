#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wjw/workspace/kokkos}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-maca-perf}"
JOBS="${JOBS:-4}"
PERF_DIR="$BUILD_DIR/core/perf_test"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y-%m-%d_T%H-%M-%S)}"
RESULTS_DIR="${RESULTS_DIR:-$BUILD_DIR/perf-results/$TIMESTAMP}"

TARGETS=(
  Kokkos_PerformanceTest_Benchmark
  Kokkos_Benchmark_Atomic_MinMax
  Kokkos_PerformanceTest_ViewFirstTouch
  Kokkos_PerformanceTest_MDRangePolicy_Stream
  Kokkos_PerformanceTest_Mempool
  Kokkos_PerformanceTest_Atomic
  Kokkos_PerformanceTest_Reduction
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_cache_setting() {
  local cache_file="$1"
  local setting="$2"
  if ! grep -q "^${setting}$" "$cache_file"; then
    echo "expected '$setting' in $cache_file" >&2
    exit 1
  fi
}

echo "== Validating MACA benchmark build tree =="
require_file "$BUILD_DIR/CMakeCache.txt"
require_cache_setting "$BUILD_DIR/CMakeCache.txt" "Kokkos_ENABLE_BENCHMARKS:BOOL=ON"
require_cache_setting "$BUILD_DIR/CMakeCache.txt" "Kokkos_ENABLE_MACA:BOOL=ON"

mkdir -p "$RESULTS_DIR"

echo
echo "== Building performance benchmark targets =="
cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j"$JOBS"

echo
echo "== Running performance benchmark executables =="
for target in "${TARGETS[@]}"; do
  exe="$PERF_DIR/$target"
  json_out="$RESULTS_DIR/${target}.json"
  benchmark_args=(
    --benchmark_counters_tabular=true
    --benchmark_out="$json_out"
    --benchmark_out_format=json
  )

  if [[ ! -x "$exe" ]]; then
    if [[ "$target" == "Kokkos_PerformanceTest_Mempool" ]]; then
      echo "== Skipping optional target: $target (not built) =="
      continue
    fi
    echo "missing executable: $exe" >&2
    exit 1
  fi

  echo
  echo "== $exe =="
  "$exe" "${benchmark_args[@]}"
done

echo
echo "== Performance benchmark run complete =="
echo "results written to: $RESULTS_DIR"
