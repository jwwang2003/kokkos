#!/usr/bin/env bash

set -euo pipefail

script_dir=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P
)
repo_root=$(
  cd "${script_dir}/.." && pwd -P
)

tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

assert_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

assert_dir() {
  [[ -d "$1" ]] || fail "missing directory: $1"
}

assert_contains() {
  local path=$1
  local expected=$2
  grep -Fq "${expected}" "${path}" || fail "expected '${expected}' in ${path}"
}

fake_source="${tmp_dir}/source"
stage_dir="${tmp_dir}/stage"
sdk_dir="${tmp_dir}/Kokkos-SDK-5.1.0-linux-x86_64-gcc13"

mkdir -p "${fake_source}" "${stage_dir}"
cat >"${fake_source}/CMakeLists.txt" <<'CMAKE'
set(Kokkos_VERSION_MAJOR 5)
set(Kokkos_VERSION_MINOR 1)
set(Kokkos_VERSION_PATCH 0)
CMAKE

for variant in cpu cuda maca; do
  mkdir -p \
    "${stage_dir}/${variant}/include" \
    "${stage_dir}/${variant}/lib/cmake/Kokkos" \
    "${stage_dir}/${variant}/lib"
  touch "${stage_dir}/${variant}/lib/libkokkoscore.a"
  cat >"${stage_dir}/${variant}/lib/cmake/Kokkos/KokkosConfig.cmake" <<CMAKE
add_library(Kokkos::kokkos INTERFACE IMPORTED)
CMAKE
done

"${repo_root}/scripts/package-kokkos-sdk.sh" assemble \
  --source "${fake_source}" \
  --stage-dir "${stage_dir}" \
  --sdk-dir "${sdk_dir}" \
  --backend cpu,cuda,maca \
  --platform linux-x86_64-gcc13

assert_dir "${sdk_dir}/variants/cpu"
assert_dir "${sdk_dir}/variants/cuda"
assert_dir "${sdk_dir}/variants/maca"
assert_file "${sdk_dir}/share/kokkos-sdk/manifest.json"
assert_file "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake"
assert_file "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfigVersion.cmake"
assert_file "${sdk_dir}/setup-cpu.sh"
assert_file "${sdk_dir}/setup-cuda.sh"
assert_file "${sdk_dir}/setup-maca.sh"

assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"platform": "linux-x86_64-gcc13"'
assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"architectures": ["sm70", "sm80", "sm90", "sm100"]'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cpu INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cuda INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::maca INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/README.md" 'variants/'

if command -v cmake >/dev/null 2>&1 && cmake --version >/dev/null 2>&1; then
  consumer_dir="${tmp_dir}/consumer"
  mkdir -p "${consumer_dir}"
  cat >"${consumer_dir}/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(kokkos_sdk_consumer LANGUAGES CXX)
find_package(KokkosMultiBackendSDK CONFIG REQUIRED)
foreach(variant IN ITEMS cpu cuda maca)
  if(NOT TARGET KokkosSDK::${variant})
    message(FATAL_ERROR "missing KokkosSDK::${variant}")
  endif()
endforeach()
CMAKE

  cmake -S "${consumer_dir}" -B "${tmp_dir}/consumer-build" -DCMAKE_PREFIX_PATH="${sdk_dir}" >/dev/null
else
  printf 'skipping CMake consumer smoke test because cmake is unavailable\n'
fi

printf 'package-kokkos-sdk assembly test passed\n'
