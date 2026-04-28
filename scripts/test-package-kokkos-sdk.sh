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
  grep -Fq -- "${expected}" "${path}" || fail "expected '${expected}' in ${path}"
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

fake_bin="${tmp_dir}/fake-bin"
fake_cmake_log="${tmp_dir}/fake-cmake.log"
mkdir -p "${fake_bin}"
cat >"${fake_bin}/cmake" <<'FAKE_CMAKE'
#!/usr/bin/env bash
printf '%q ' "$@" >>"${FAKE_CMAKE_LOG}"
printf '\n' >>"${FAKE_CMAKE_LOG}"
exit 0
FAKE_CMAKE
chmod +x "${fake_bin}/cmake"

FAKE_CMAKE_LOG="${fake_cmake_log}" PATH="${fake_bin}:${PATH}" \
  "${repo_root}/scripts/package-kokkos-sdk.sh" build \
    --source "${fake_source}" \
    --work-dir "${tmp_dir}/build-aarch64" \
    --backend cpu \
    --cpu-arch aarch64 \
    --host-arch ARMV80 \
    --platform linux-aarch64-jetson

assert_contains "${fake_cmake_log}" "-DKokkos_ARCH_ARMV80=ON"
assert_contains "${fake_cmake_log}" "-DCMAKE_SYSTEM_PROCESSOR=aarch64"

for variant in cpu cuda-volta70 cuda-ampere80 cuda-hopper90 cuda-blackwell120 maca; do
  mkdir -p \
    "${stage_dir}/${variant}/include" \
    "${stage_dir}/${variant}/lib/cmake/Kokkos" \
    "${stage_dir}/${variant}/lib"
  touch "${stage_dir}/${variant}/lib/libkokkoscore.a"
  cat >"${stage_dir}/${variant}/lib/cmake/Kokkos/KokkosConfig.cmake" <<CMAKE
add_library(Kokkos::kokkos INTERFACE IMPORTED)
CMAKE
done

default_platform_work_dir="${tmp_dir}/default-platform"
"${repo_root}/scripts/package-kokkos-sdk.sh" assemble \
  --source "${fake_source}" \
  --stage-dir "${stage_dir}" \
  --work-dir "${default_platform_work_dir}" \
  --backend cpu \
  --cpu-arch aarch64 \
  --host-arch ARMV80

assert_dir "${default_platform_work_dir}/kokkos-sdk-5.1.0-linux-aarch64"
assert_contains "${default_platform_work_dir}/kokkos-sdk-5.1.0-linux-aarch64/share/kokkos-sdk/manifest.json" '"platform": "linux-aarch64"'

"${repo_root}/scripts/package-kokkos-sdk.sh" assemble \
  --source "${fake_source}" \
  --stage-dir "${stage_dir}" \
  --sdk-dir "${sdk_dir}" \
  --backend cpu,cuda,maca \
  --cpu-arch aarch64 \
  --host-arch ARMV80 \
  --cuda-arch-list VOLTA70,AMPERE80,HOPPER90,BLACKWELL120 \
  --platform linux-aarch64-jetson

assert_dir "${sdk_dir}/variants/cpu"
assert_dir "${sdk_dir}/variants/cuda-volta70"
assert_dir "${sdk_dir}/variants/cuda-ampere80"
assert_dir "${sdk_dir}/variants/cuda-hopper90"
assert_dir "${sdk_dir}/variants/cuda-blackwell120"
assert_dir "${sdk_dir}/variants/maca"
assert_file "${sdk_dir}/share/kokkos-sdk/manifest.json"
assert_file "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake"
assert_file "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfigVersion.cmake"
assert_file "${sdk_dir}/setup-cpu.sh"
assert_file "${sdk_dir}/setup-cuda-volta70.sh"
assert_file "${sdk_dir}/setup-cuda-ampere80.sh"
assert_file "${sdk_dir}/setup-cuda-hopper90.sh"
assert_file "${sdk_dir}/setup-cuda-blackwell120.sh"
assert_file "${sdk_dir}/setup-maca.sh"

assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"platform": "linux-aarch64-jetson"'
assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"cpu_arch": "aarch64"'
assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"host_arch": "ARMV80"'
assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"architectures": ["sm70", "sm80", "sm90", "sm120"]'
assert_contains "${sdk_dir}/share/kokkos-sdk/manifest.json" '"package_entries": ["cuda-volta70", "cuda-ampere80", "cuda-hopper90", "cuda-blackwell120"]'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cpu INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cuda_volta70 INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cuda_ampere80 INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cuda_hopper90 INTERFACE IMPORTED)'
assert_contains "${sdk_dir}/lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" 'add_library(KokkosSDK::cuda_blackwell120 INTERFACE IMPORTED)'
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
foreach(variant IN ITEMS cpu cuda cuda_volta70 cuda_ampere80 cuda_hopper90 cuda_blackwell120 maca)
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
