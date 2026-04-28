#!/usr/bin/env bash

set -euo pipefail

script_dir=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P
)
repo_root=$(
  cd "${script_dir}/.." && pwd -P
)

command="package"
if [[ $# -gt 0 ]]; then
  case "$1" in
    build | assemble | package | help | --help | -h)
      command="$1"
      shift
      ;;
  esac
fi

default_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
  else
    printf '%s\n' 8
  fi
}

trim() {
  local value=${1:-}
  value=${value#"${value%%[![:space:]]*}"}
  value=${value%"${value##*[![:space:]]}"}
  printf '%s' "${value}"
}

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

upper() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

default_platform() {
  local os
  local arch
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)
  printf '%s-%s\n' "${os}" "${arch}"
}

cuda_arch_to_sm() {
  case "$(lower "$1")" in
    maxwell50) printf 'sm50\n' ;;
    maxwell52) printf 'sm52\n' ;;
    maxwell53) printf 'sm53\n' ;;
    pascal60) printf 'sm60\n' ;;
    pascal61) printf 'sm61\n' ;;
    volta70) printf 'sm70\n' ;;
    volta72) printf 'sm72\n' ;;
    turing75) printf 'sm75\n' ;;
    ampere80) printf 'sm80\n' ;;
    ampere86) printf 'sm86\n' ;;
    ampere87) printf 'sm87\n' ;;
    ada89) printf 'sm89\n' ;;
    hopper90) printf 'sm90\n' ;;
    blackwell100) printf 'sm100\n' ;;
    blackwell103) printf 'sm103\n' ;;
    blackwell120) printf 'sm120\n' ;;
    blackwell121) printf 'sm121\n' ;;
    *) printf '%s\n' "$(lower "$1")" ;;
  esac
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '[kokkos-sdk] %s\n' "$*"
}

append_split_env_args() {
  local var_name=$1
  local -n out_ref=$2
  local raw=${!var_name:-}
  if [[ -z "${raw}" ]]; then
    return
  fi
  # shellcheck disable=SC2206
  local split_args=(${raw})
  out_ref+=("${split_args[@]}")
}

contains_token() {
  local needle=$1
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_csv_items() {
  local raw_list=$1
  local mode=$2
  local parsed=()
  local item
  IFS=',' read -r -a parsed <<<"${raw_list}"
  local normalized=()
  for item in "${parsed[@]}"; do
    item=$(trim "${item}")
    [[ -z "${item}" ]] && continue
    item=$(lower "${item}")
    case "${mode}" in
      backend)
        case "${item}" in
          cpu | openmp)
            item=cpu
            ;;
          cuda | maca)
            ;;
          *)
            die "unsupported backend '${item}'. Expected a comma-separated subset of: cpu,cuda,maca"
            ;;
        esac
        ;;
      cuda_arch)
        case "${item}" in
          maxwell50 | maxwell52 | maxwell53 | pascal60 | pascal61 | volta70 | volta72 | turing75 | ampere80 | ampere86 | ampere87 | ada89 | hopper90 | blackwell100 | blackwell103 | blackwell120 | blackwell121)
            ;;
          *)
            die "unsupported CUDA arch '${item}'. Use Kokkos architecture tokens such as AMPERE86, ADA89, HOPPER90, BLACKWELL120"
            ;;
        esac
        ;;
      *)
        die "internal error: unsupported normalization mode '${mode}'"
        ;;
    esac
    if ! contains_token "${item}" "${normalized[@]}"; then
      normalized+=("${item}")
    fi
  done
  if [[ ${#normalized[@]} -eq 0 ]]; then
    die "no values selected for ${mode}"
  fi
  printf '%s\n' "${normalized[@]}"
}

read_kokkos_version() {
  local src_dir=$1
  awk '
    /set\(Kokkos_VERSION_MAJOR / {
      major = $0
      gsub(/[^0-9]/, "", major)
    }
    /set\(Kokkos_VERSION_MINOR / {
      minor = $0
      gsub(/[^0-9]/, "", minor)
    }
    /set\(Kokkos_VERSION_PATCH / {
      patch = $0
      gsub(/[^0-9]/, "", patch)
    }
    END {
      if (major == "" || minor == "" || patch == "") exit 1
      printf "%s.%s.%s\n", major, minor, patch
    }
  ' "${src_dir}/CMakeLists.txt"
}

usage() {
  cat <<'USAGE'
Usage:
  scripts/package-kokkos-sdk.sh [build|assemble|package] [options]

Commands:
  build      Configure, build, and install one or more backend-specific Kokkos prefixes into a staging area.
  assemble   Assemble a unified SDK from previously staged backend installs.
  package    Run build, then assemble. This is the default command.

Options:
  --source DIR              Kokkos source directory. Default: repository root.
  --work-dir DIR            Working directory for build trees, staged installs, and the final SDK.
  --stage-dir DIR           Directory containing staged backend installs. Default: <work-dir>/stage
  --sdk-dir DIR             Output SDK directory. Default: <work-dir>/kokkos-sdk-<version>-<platform>
  --backend LIST            Comma-separated subset of: cpu,cuda,maca. 'openmp' is accepted as an alias for cpu. Default: cpu,cuda,maca
  --platform NAME           Platform triplet for the SDK name and manifest. Default: <os>-<arch>
  --cuda-arch NAME          Add a Kokkos CUDA architecture token to the single cuda variant. Repeat as needed.
  --cuda-arch-list LIST     Comma-separated CUDA arch tokens. Example: AMPERE86,ADA89,BLACKWELL120
  --build-type TYPE         CMake build type. Default: Release
  --jobs N                  Parallel build jobs. Default: detected CPU count
  --generator NAME          CMake generator to pass through with -G
  --shared                  Build shared libraries
  --static                  Build static libraries (default)
  --openmp-cmake-arg ARG    Extra CMake argument for the OpenMP build. Repeat as needed.
  --cuda-cmake-arg ARG      Extra CMake argument for the CUDA build. Repeat as needed.
  --maca-cmake-arg ARG      Extra CMake argument for the MACA build. Repeat as needed.
  --help, -h                Show this help

Environment overrides:
  KOKKOS_SDK_BACKENDS               Same format as --backend
  KOKKOS_SDK_PLATFORM               Same as --platform
  KOKKOS_SDK_CUDA_ARCH              Single CUDA arch token for the cuda variant
  KOKKOS_SDK_CUDA_ARCHES            Comma-separated CUDA arch tokens for the cuda variant
  KOKKOS_SDK_BUILD_TYPE             Same as --build-type
  KOKKOS_SDK_JOBS                   Same as --jobs
  KOKKOS_SDK_CMAKE_GENERATOR        Same as --generator
  KOKKOS_SDK_SHARED                 ON/OFF. Default: OFF
  KOKKOS_SDK_CMAKE_ARGS             Extra CMake args applied to every backend build
  OPENMP_CMAKE_ARGS                 Extra CMake args for the OpenMP build
  CUDA_CMAKE_ARGS                   Extra CMake args for the CUDA build
  MACA_CMAKE_ARGS                   Extra CMake args for the MACA build
  OPENMP_CXX_COMPILER               Compiler path for the OpenMP build
  CUDA_CXX_COMPILER                 Compiler path for the CUDA build
  MACA_CXX_COMPILER                 Compiler path for the MACA build
  KOKKOS_SDK_MACA_ARCH              Kokkos MACA architecture token. Default: XCORE1000
  KOKKOS_SDK_MACA_OFFLOAD_ARCH      Raw MACA offload arch name. Default: xcore1000
  MACA_PATH                         MACA SDK root, passed through when set

Notes:
  - Kokkos installs backend-specific generated headers, so this SDK keeps a full install prefix
    per variant under variants/cpu, variants/cuda, and variants/maca.
  - The CUDA variant is a single build with all selected Kokkos CUDA architecture flags enabled.
  - For CI, the intended split is:
      1. Matrix jobs run: scripts/package-kokkos-sdk.sh build --backend <backend> [--cuda-arch ...]
      2. Artifact aggregation job downloads the staged prefixes
      3. Aggregation job runs: scripts/package-kokkos-sdk.sh assemble

Examples:
  scripts/package-kokkos-sdk.sh package --backend cpu,cuda \
    --cuda-arch AMPERE86 --cuda-arch ADA89 --cuda-arch BLACKWELL120

  scripts/package-kokkos-sdk.sh package --backend cuda \
    --cuda-cmake-arg=-DKokkos_ENABLE_CUDA_LAMBDA=ON \
    --cuda-arch-list=VOLTA70,AMPERE80,HOPPER90,BLACKWELL100

  MACA_CXX_COMPILER=/opt/maca/mxgpu_llvm/bin/mxcc \
  MACA_PATH=/opt/maca \
  scripts/package-kokkos-sdk.sh build --backend maca
USAGE
}

source_dir=${KOKKOS_SDK_SOURCE_DIR:-${GITHUB_WORKSPACE:-${CI_PROJECT_DIR:-${repo_root}}}}
work_dir=${KOKKOS_SDK_WORK_DIR:-${repo_root}/out/kokkos-sdk}
stage_dir=""
sdk_dir=""
build_type=${KOKKOS_SDK_BUILD_TYPE:-Release}
jobs=${KOKKOS_SDK_JOBS:-$(default_jobs)}
generator=${KOKKOS_SDK_CMAKE_GENERATOR:-}
shared=${KOKKOS_SDK_SHARED:-OFF}
backend_list=${KOKKOS_SDK_BACKENDS:-cpu,cuda,maca}
platform=${KOKKOS_SDK_PLATFORM:-$(default_platform)}

openmp_extra_args=()
cuda_extra_args=()
maca_extra_args=()
cuda_arch_tokens=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || die "--source requires a value"
      source_dir=$2
      shift 2
      ;;
    --source=*)
      source_dir=${1#*=}
      shift
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a value"
      work_dir=$2
      shift 2
      ;;
    --work-dir=*)
      work_dir=${1#*=}
      shift
      ;;
    --stage-dir)
      [[ $# -ge 2 ]] || die "--stage-dir requires a value"
      stage_dir=$2
      shift 2
      ;;
    --stage-dir=*)
      stage_dir=${1#*=}
      shift
      ;;
    --sdk-dir)
      [[ $# -ge 2 ]] || die "--sdk-dir requires a value"
      sdk_dir=$2
      shift 2
      ;;
    --sdk-dir=*)
      sdk_dir=${1#*=}
      shift
      ;;
    --backend)
      [[ $# -ge 2 ]] || die "--backend requires a value"
      backend_list=$2
      shift 2
      ;;
    --backend=*)
      backend_list=${1#*=}
      shift
      ;;
    --platform)
      [[ $# -ge 2 ]] || die "--platform requires a value"
      platform=$2
      shift 2
      ;;
    --platform=*)
      platform=${1#*=}
      shift
      ;;
    --cuda-arch)
      [[ $# -ge 2 ]] || die "--cuda-arch requires a value"
      cuda_arch_tokens+=("$(upper "$2")")
      shift 2
      ;;
    --cuda-arch=*)
      cuda_arch_tokens+=("$(upper "${1#*=}")")
      shift
      ;;
    --cuda-arch-list)
      [[ $# -ge 2 ]] || die "--cuda-arch-list requires a value"
      while IFS= read -r arch; do
        cuda_arch_tokens+=("$(upper "${arch}")")
      done < <(normalize_csv_items "$2" cuda_arch)
      shift 2
      ;;
    --cuda-arch-list=*)
      while IFS= read -r arch; do
        cuda_arch_tokens+=("$(upper "${arch}")")
      done < <(normalize_csv_items "${1#*=}" cuda_arch)
      shift
      ;;
    --build-type)
      [[ $# -ge 2 ]] || die "--build-type requires a value"
      build_type=$2
      shift 2
      ;;
    --build-type=*)
      build_type=${1#*=}
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires a value"
      jobs=$2
      shift 2
      ;;
    --jobs=*)
      jobs=${1#*=}
      shift
      ;;
    --generator)
      [[ $# -ge 2 ]] || die "--generator requires a value"
      generator=$2
      shift 2
      ;;
    --generator=*)
      generator=${1#*=}
      shift
      ;;
    --shared)
      shared=ON
      shift
      ;;
    --static)
      shared=OFF
      shift
      ;;
    --openmp-cmake-arg)
      [[ $# -ge 2 ]] || die "--openmp-cmake-arg requires a value"
      openmp_extra_args+=("$2")
      shift 2
      ;;
    --openmp-cmake-arg=*)
      openmp_extra_args+=("${1#*=}")
      shift
      ;;
    --cuda-cmake-arg)
      [[ $# -ge 2 ]] || die "--cuda-cmake-arg requires a value"
      cuda_extra_args+=("$2")
      shift 2
      ;;
    --cuda-cmake-arg=*)
      cuda_extra_args+=("${1#*=}")
      shift
      ;;
    --maca-cmake-arg)
      [[ $# -ge 2 ]] || die "--maca-cmake-arg requires a value"
      maca_extra_args+=("$2")
      shift 2
      ;;
    --maca-cmake-arg=*)
      maca_extra_args+=("${1#*=}")
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      die "unknown option '$1'"
      ;;
  esac
done

if [[ "${command}" == "help" || "${command}" == "--help" || "${command}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
  die "source directory '${source_dir}' does not look like a Kokkos checkout"
fi

version=$(read_kokkos_version "${source_dir}")
[[ -n "${version}" ]] || die "could not determine Kokkos version from ${source_dir}/CMakeLists.txt"

if [[ -z "${stage_dir}" ]]; then
  stage_dir="${work_dir}/stage"
fi

if [[ -z "${sdk_dir}" ]]; then
  sdk_dir="${work_dir}/kokkos-sdk-${version}-${platform}"
fi

if [[ "${shared}" != "ON" && "${shared}" != "OFF" ]]; then
  die "KOKKOS_SDK_SHARED must be ON or OFF"
fi

mapfile -t selected_backends < <(normalize_csv_items "${backend_list}" backend)

if [[ ${#cuda_arch_tokens[@]} -eq 0 && -n "${KOKKOS_SDK_CUDA_ARCHES:-}" ]]; then
  while IFS= read -r arch; do
    cuda_arch_tokens+=("$(upper "${arch}")")
  done < <(normalize_csv_items "${KOKKOS_SDK_CUDA_ARCHES}" cuda_arch)
fi

if [[ ${#cuda_arch_tokens[@]} -eq 0 && -n "${KOKKOS_SDK_CUDA_ARCH:-}" ]]; then
  cuda_arch_tokens+=("$(upper "${KOKKOS_SDK_CUDA_ARCH}")")
fi

if [[ ${#cuda_arch_tokens[@]} -gt 0 ]]; then
  mapfile -t _normalized_cuda_arch_tokens < <(printf '%s\n' "${cuda_arch_tokens[@]}" | awk '{ print tolower($0) }' | awk '!seen[$0]++')
  cuda_arch_tokens=()
  for arch in "${_normalized_cuda_arch_tokens[@]}"; do
    cuda_arch_tokens+=("$(upper "${arch}")")
  done
fi

if [[ ${#cuda_arch_tokens[@]} -eq 0 ]]; then
  cuda_arch_tokens=(VOLTA70 AMPERE80 HOPPER90 BLACKWELL100)
fi

cmake_generator_args=()
if [[ -n "${generator}" ]]; then
  cmake_generator_args=(-G "${generator}")
fi

append_split_env_args KOKKOS_SDK_CMAKE_ARGS openmp_extra_args
append_split_env_args KOKKOS_SDK_CMAKE_ARGS cuda_extra_args
append_split_env_args KOKKOS_SDK_CMAKE_ARGS maca_extra_args
append_split_env_args OPENMP_CMAKE_ARGS openmp_extra_args
append_split_env_args CUDA_CMAKE_ARGS cuda_extra_args
append_split_env_args MACA_CMAKE_ARGS maca_extra_args

package_names=()
package_kinds=()
package_cuda_arches=()

register_package() {
  local package_name=$1
  local package_kind=$2
  local package_cuda_arch=${3:-}
  package_names+=("${package_name}")
  package_kinds+=("${package_kind}")
  package_cuda_arches+=("${package_cuda_arch}")
}

for backend in "${selected_backends[@]}"; do
  case "${backend}" in
    cpu)
      register_package cpu cpu ""
      ;;
    cuda)
      register_package cuda cuda "$(printf '%s ' "${cuda_arch_tokens[@]}")"
      ;;
    maca)
      register_package maca maca ""
      ;;
  esac
done

configure_and_install_package() {
  local package_name=$1
  local package_kind=$2
  local package_cuda_arch=${3:-}
  local build_dir="${work_dir}/build/${package_name}"
  local install_dir="${stage_dir}/${package_name}"
  local compiler=""
  local extra_args=()
  local args=(
    "-DCMAKE_BUILD_TYPE=${build_type}"
    "-DCMAKE_INSTALL_PREFIX=${install_dir}"
    "-DBUILD_SHARED_LIBS=${shared}"
    "-DKokkos_ENABLE_SERIAL=ON"
    "-DKokkos_ENABLE_OPENMP=ON"
    "-DKokkos_ENABLE_TESTS=OFF"
    "-DKokkos_ENABLE_EXAMPLES=OFF"
    "-DKokkos_ENABLE_BENCHMARKS=OFF"
    "-DKokkos_INSTALL_TESTING=OFF"
  )

  case "${package_kind}" in
    cpu)
      compiler=${OPENMP_CXX_COMPILER:-${KOKKOS_SDK_OPENMP_CXX_COMPILER:-${CXX:-}}}
      extra_args=("${openmp_extra_args[@]}")
      ;;
    cuda)
      compiler=${CUDA_CXX_COMPILER:-${KOKKOS_SDK_CUDA_CXX_COMPILER:-}}
      if [[ -z "${compiler}" ]] && [[ -x "${source_dir}/bin/nvcc_wrapper" ]] && command -v nvcc >/dev/null 2>&1; then
        compiler="${source_dir}/bin/nvcc_wrapper"
      fi
      args+=("-DKokkos_ENABLE_CUDA=ON")
      for arch in ${package_cuda_arch}; do
        args+=("-DKokkos_ARCH_$(upper "${arch}")=ON")
      done
      extra_args=("${cuda_extra_args[@]}")
      ;;
    maca)
      compiler=${MACA_CXX_COMPILER:-${KOKKOS_SDK_MACA_CXX_COMPILER:-}}
      if [[ -z "${compiler}" ]] && [[ -x "/opt/maca/mxgpu_llvm/bin/mxcc" ]]; then
        compiler="/opt/maca/mxgpu_llvm/bin/mxcc"
      fi
      args+=("-DKokkos_ENABLE_MACA=ON")
      args+=("-DKokkos_ARCH_$(upper "${KOKKOS_SDK_MACA_ARCH:-XCORE1000}")=ON")
      args+=("-DKokkos_IMPL_MACAGPU_FLAGS=--offload-arch=${KOKKOS_SDK_MACA_OFFLOAD_ARCH:-xcore1000}")
      if [[ -n "${MACA_PATH:-}" ]]; then
        args+=("-DMACA_PATH=${MACA_PATH}")
      fi
      extra_args=("${maca_extra_args[@]}")
      ;;
    *)
      die "internal error: unsupported package kind '${package_kind}'"
      ;;
  esac

  if [[ -n "${compiler}" ]]; then
    args+=("-DCMAKE_CXX_COMPILER=${compiler}")
  fi

  mkdir -p "${build_dir}" "${install_dir}"

  note "configuring ${package_name} in ${build_dir}"
  cmake -S "${source_dir}" -B "${build_dir}" "${cmake_generator_args[@]}" "${args[@]}" "${extra_args[@]}"

  note "building ${package_name}"
  cmake --build "${build_dir}" -j "${jobs}"

  note "installing ${package_name} to ${install_dir}"
  cmake --install "${build_dir}" --prefix "${install_dir}"
}

write_setup_script() {
  local package_name=$1
  local backend_selector=$2
  local script_path=$3
  cat >"${script_path}" <<SETUP
#!/usr/bin/env bash

set -euo pipefail

sdk_root=\$(
  cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd -P
)

export KOKKOS_SDK_ROOT="\${sdk_root}"
export KOKKOS_BACKEND="${backend_selector}"
if [[ -n "\${CMAKE_PREFIX_PATH:-}" ]]; then
  export CMAKE_PREFIX_PATH="\${sdk_root}:\${CMAKE_PREFIX_PATH}"
else
  export CMAKE_PREFIX_PATH="\${sdk_root}"
fi

printf 'Configured Kokkos SDK package: %s\n' "${package_name}"
printf 'KOKKOS_BACKEND=%s\n' "\${KOKKOS_BACKEND}"
printf 'CMAKE_PREFIX_PATH=%s\n' "\${CMAKE_PREFIX_PATH}"
SETUP
  chmod +x "${script_path}"
}

write_wrapper_config() {
  local config_dir=$1
  local version_file=$2
  local package_name
  local i

  mkdir -p "${config_dir}" "${config_dir}/../KokkosMultiBackendSDK"

  cat >"${config_dir}/KokkosConfig.cmake" <<CONFIG
message(
  DEPRECATION
  "This compatibility KokkosConfig.cmake selects one bundled variant. Prefer find_package(KokkosMultiBackendSDK CONFIG REQUIRED) and KokkosSDK::<variant> targets."
)

get_filename_component(_kokkos_sdk_root "\${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT DEFINED Kokkos_BACKEND OR "\${Kokkos_BACKEND}" STREQUAL "")
  if(DEFINED ENV{KOKKOS_BACKEND} AND NOT "\$ENV{KOKKOS_BACKEND}" STREQUAL "")
    set(Kokkos_BACKEND "\$ENV{KOKKOS_BACKEND}")
  else()
    set(Kokkos_BACKEND "CPU")
  endif()
endif()

string(TOLOWER "\${Kokkos_BACKEND}" _kokkos_backend_lower)
if(_kokkos_backend_lower STREQUAL "openmp")
  set(_kokkos_backend_lower "cpu")
endif()

set(_kokkos_delegate_config
    "\${_kokkos_sdk_root}/variants/\${_kokkos_backend_lower}/lib/cmake/Kokkos/KokkosConfig.cmake")

if(NOT EXISTS "\${_kokkos_delegate_config}")
  message(
    FATAL_ERROR
      "Kokkos SDK backend '\${Kokkos_BACKEND}' is not packaged in '\${_kokkos_sdk_root}'. Missing: \${_kokkos_delegate_config}"
  )
endif()

set(Kokkos_SDK_ROOT "\${_kokkos_sdk_root}")
set(Kokkos_SELECTED_BACKEND "\${_kokkos_backend_lower}")

include("\${_kokkos_delegate_config}")

unset(_kokkos_delegate_config)
unset(_kokkos_backend_lower)
unset(_kokkos_sdk_root)
CONFIG

  cat >"${config_dir}/../KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" <<'CONFIG'
get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(KokkosMultiBackendSDK_ROOT "${PACKAGE_PREFIX_DIR}")
set(KokkosMultiBackendSDK_MANIFEST "${PACKAGE_PREFIX_DIR}/share/kokkos-sdk/manifest.json")
set(KokkosMultiBackendSDK_VARIANTS "")
CONFIG

  for i in "${!package_names[@]}"; do
    package_name=${package_names[$i]}
    cat >>"${config_dir}/../KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake" <<CONFIG

set(_kokkos_sdk_${package_name}_root "\${PACKAGE_PREFIX_DIR}/variants/${package_name}")
if(EXISTS "\${_kokkos_sdk_${package_name}_root}/lib/cmake/Kokkos/KokkosConfig.cmake")
  if(NOT TARGET KokkosSDK::${package_name})
    add_library(KokkosSDK::${package_name} INTERFACE IMPORTED)
    set_target_properties(KokkosSDK::${package_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "\${_kokkos_sdk_${package_name}_root}/include"
      INTERFACE_LINK_DIRECTORIES "\${_kokkos_sdk_${package_name}_root}/lib"
      INTERFACE_LINK_LIBRARIES "kokkoscontainers;kokkosalgorithms;kokkoscore"
    )
  endif()
  list(APPEND KokkosMultiBackendSDK_VARIANTS "${package_name}")
endif()
unset(_kokkos_sdk_${package_name}_root)
CONFIG
  done

  cat >"${version_file}" <<VERSION
set(PACKAGE_VERSION "${version}")

if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()

if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
VERSION

  cp "${version_file}" "${config_dir}/../KokkosMultiBackendSDK/KokkosMultiBackendSDKConfigVersion.cmake"
}

write_sdk_readme() {
  local readme_path=$1
  local package_csv=$2
  cat >"${readme_path}" <<README
# Kokkos SDK ${version}

This SDK bundles backend-specific Kokkos installs for:

- ${package_csv}

Each package entry keeps its own install prefix because Kokkos installs generated headers that depend on the enabled backend set.

## Layout

\`\`\`
kokkos-sdk-${version}-${platform}/
├── variants/
│   ├── cpu/
│   ├── cuda/
│   └── maca/
├── lib/cmake/KokkosMultiBackendSDK/KokkosMultiBackendSDKConfig.cmake
├── lib/cmake/Kokkos/KokkosConfig.cmake
├── share/kokkos-sdk/manifest.json
└── setup-*.sh
\`\`\`

## CMake usage

\`\`\`cmake
find_package(KokkosMultiBackendSDK CONFIG REQUIRED PATHS "/path/to/kokkos-sdk-${version}-${platform}" NO_DEFAULT_PATH)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE KokkosSDK::cuda)
\`\`\`

Notes:

- Exported variant targets are \`KokkosSDK::cpu\`, \`KokkosSDK::cuda\`, and \`KokkosSDK::maca\` when the corresponding variants are packaged.
- The CUDA variant is one private Kokkos build with all selected CUDA architecture flags enabled.
- A compatibility \`find_package(Kokkos)\` wrapper is included for one-variant-at-a-time consumers. Set \`Kokkos_BACKEND=CPU\`, \`CUDA\`, or \`MACA\`; it defaults to \`CPU\`.

## Shell helpers

Use the generated setup script for the package you want, for example:

- \`source ./setup-cpu.sh\`
- \`source ./setup-cuda.sh\`
- \`source ./setup-maca.sh\`

Each helper sets \`KOKKOS_BACKEND\` and prepends the SDK root to \`CMAKE_PREFIX_PATH\`.

## CI usage

This script supports split build and assembly stages:

\`\`\`bash
scripts/package-kokkos-sdk.sh build --backend cpu
scripts/package-kokkos-sdk.sh build --backend cuda --cuda-arch-list VOLTA70,AMPERE80,HOPPER90,BLACKWELL100
scripts/package-kokkos-sdk.sh build --backend maca
scripts/package-kokkos-sdk.sh assemble --backend cpu,cuda,maca
\`\`\`

In CI, run \`build\` in per-variant jobs, upload the staged \`stage/<variant>\` directories as artifacts, then collect them in a final \`assemble\` job.
README
}

write_manifest() {
  local manifest_path=$1
  local cpu_enabled=false
  local cuda_enabled=false
  local maca_enabled=false
  local arch_json=""
  local arch
  local i

  for i in "${!package_names[@]}"; do
    case "${package_names[$i]}" in
      cpu) cpu_enabled=true ;;
      cuda) cuda_enabled=true ;;
      maca) maca_enabled=true ;;
    esac
  done

  for arch in ${cuda_arch_tokens[*]}; do
    if [[ -n "${arch_json}" ]]; then
      arch_json="${arch_json}, "
    fi
    arch_json="${arch_json}\"$(cuda_arch_to_sm "${arch}")\""
  done

  mkdir -p "$(dirname "${manifest_path}")"
  cat >"${manifest_path}" <<JSON
{
  "version": "$(json_escape "${version}")",
  "platform": "$(json_escape "${platform}")",
  "kokkos_version": "$(json_escape "${version}")",
  "variants": {
    "cpu": {
      "enabled": ${cpu_enabled},
      "host_backend": "OpenMP",
      "device_backend": "None"
    },
    "cuda": {
      "enabled": ${cuda_enabled},
      "host_backend": "OpenMP",
      "device_backend": "CUDA",
      "architectures": [${arch_json}]
    },
    "maca": {
      "enabled": ${maca_enabled},
      "host_backend": "OpenMP",
      "device_backend": "MACA"
    }
  }
}
JSON
}

assemble_sdk() {
  local package_name
  local package_kind
  local package_cuda_arch
  local package_csv
  local sdk_package_dir
  local config_dir="${sdk_dir}/lib/cmake/Kokkos"
  local manifest_path="${sdk_dir}/share/kokkos-sdk/manifest.json"
  local backend_selector
  local i

  if [[ -e "${sdk_dir}" ]]; then
    die "SDK output directory '${sdk_dir}' already exists. Remove it or choose --sdk-dir"
  fi

  mkdir -p "${sdk_dir}"

  for i in "${!package_names[@]}"; do
    package_name=${package_names[$i]}
    package_kind=${package_kinds[$i]}
    package_cuda_arch=${package_cuda_arches[$i]}
    if [[ ! -f "${stage_dir}/${package_name}/lib/cmake/Kokkos/KokkosConfig.cmake" ]]; then
      die "staged package '${package_name}' is missing from '${stage_dir}/${package_name}'"
    fi
    sdk_package_dir="${sdk_dir}/variants/${package_name}"
    mkdir -p "${sdk_package_dir}"
    note "copying staged ${package_name} install into ${sdk_package_dir}"
    cp -a "${stage_dir}/${package_name}/." "${sdk_package_dir}/"

    case "${package_kind}" in
      cpu)
        backend_selector="CPU"
        ;;
      cuda)
        backend_selector="CUDA"
        ;;
      maca)
        backend_selector="MACA"
        ;;
      *)
        die "internal error: unsupported package kind '${package_kind}'"
        ;;
    esac
    write_setup_script "${package_name}" "${backend_selector}" "${sdk_dir}/setup-${package_name}.sh"
  done

  write_wrapper_config "${config_dir}" "${config_dir}/KokkosConfigVersion.cmake"
  write_manifest "${manifest_path}"

  package_csv=$(printf '%s, ' "${package_names[@]}")
  package_csv=${package_csv%, }
  write_sdk_readme "${sdk_dir}/README.md" "${package_csv}"

  if [[ -f "${source_dir}/LICENSE" ]]; then
    cp -a "${source_dir}/LICENSE" "${sdk_dir}/LICENSE"
  fi
}

case "${command}" in
  build)
    mkdir -p "${stage_dir}" "${work_dir}/build"
    for i in "${!package_names[@]}"; do
      configure_and_install_package "${package_names[$i]}" "${package_kinds[$i]}" "${package_cuda_arches[$i]}"
    done
    ;;
  assemble)
    assemble_sdk
    ;;
  package)
    mkdir -p "${stage_dir}" "${work_dir}/build"
    for i in "${!package_names[@]}"; do
      configure_and_install_package "${package_names[$i]}" "${package_kinds[$i]}" "${package_cuda_arches[$i]}"
    done
    assemble_sdk
    ;;
  *)
    die "unknown command '${command}'"
    ;;
esac

note "done"
if [[ "${command}" == "assemble" || "${command}" == "package" ]]; then
  note "SDK ready at ${sdk_dir}"
fi
