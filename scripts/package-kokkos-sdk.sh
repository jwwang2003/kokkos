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
          openmp | cuda | maca)
            ;;
          *)
            die "unsupported backend '${item}'. Expected a comma-separated subset of: openmp,cuda,maca"
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
  --sdk-dir DIR             Output SDK directory. Default: <work-dir>/Kokkos-SDK-v<version>
  --backend LIST            Comma-separated subset of: openmp,cuda,maca. Default: openmp,cuda,maca
  --cuda-arch NAME          Add a CUDA variant package for a specific Kokkos arch token. Repeat as needed.
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
  KOKKOS_SDK_CUDA_ARCH              Single CUDA arch token for the plain 'cuda' package
  KOKKOS_SDK_CUDA_ARCHES            Comma-separated CUDA arch tokens for multiple cuda-<arch> packages
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
    per package entry under openmp/, cuda/, cuda-<arch>/, and maca/.
  - This Kokkos tree supports only one NVIDIA GPU architecture per build tree. For distributable
    multi-generation CUDA support, use repeated --cuda-arch or KOKKOS_SDK_CUDA_ARCHES.
  - For CI, the intended split is:
      1. Matrix jobs run: scripts/package-kokkos-sdk.sh build --backend <backend> [--cuda-arch ...]
      2. Artifact aggregation job downloads the staged prefixes
      3. Aggregation job runs: scripts/package-kokkos-sdk.sh assemble

Examples:
  scripts/package-kokkos-sdk.sh package --backend openmp,cuda \
    --cuda-arch AMPERE86 --cuda-arch ADA89 --cuda-arch BLACKWELL120

  scripts/package-kokkos-sdk.sh package --backend cuda \
    --cuda-cmake-arg=-DKokkos_ENABLE_CUDA_LAMBDA=ON \
    --cuda-arch-list=AMPERE86,BLACKWELL120

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
backend_list=${KOKKOS_SDK_BACKENDS:-openmp,cuda,maca}

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
  sdk_dir="${work_dir}/Kokkos-SDK-v${version}"
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

if [[ ${#cuda_arch_tokens[@]} -gt 0 ]]; then
  mapfile -t _normalized_cuda_arch_tokens < <(printf '%s\n' "${cuda_arch_tokens[@]}" | awk '{ print tolower($0) }' | awk '!seen[$0]++')
  cuda_arch_tokens=()
  for arch in "${_normalized_cuda_arch_tokens[@]}"; do
    cuda_arch_tokens+=("$(upper "${arch}")")
  done
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
    openmp)
      register_package openmp openmp ""
      ;;
    cuda)
      if [[ ${#cuda_arch_tokens[@]} -gt 0 ]]; then
        for arch in "${cuda_arch_tokens[@]}"; do
          register_package "cuda-$(lower "${arch}")" cuda "${arch}"
        done
      else
        register_package cuda cuda "${KOKKOS_SDK_CUDA_ARCH:-}"
      fi
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
    openmp)
      compiler=${OPENMP_CXX_COMPILER:-${KOKKOS_SDK_OPENMP_CXX_COMPILER:-${CXX:-}}}
      extra_args=("${openmp_extra_args[@]}")
      ;;
    cuda)
      compiler=${CUDA_CXX_COMPILER:-${KOKKOS_SDK_CUDA_CXX_COMPILER:-}}
      if [[ -z "${compiler}" ]] && [[ -x "${source_dir}/bin/nvcc_wrapper" ]] && command -v nvcc >/dev/null 2>&1; then
        compiler="${source_dir}/bin/nvcc_wrapper"
      fi
      args+=("-DKokkos_ENABLE_CUDA=ON")
      if [[ -n "${package_cuda_arch}" ]]; then
        args+=("-DKokkos_ARCH_$(upper "${package_cuda_arch}")=ON")
      fi
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
  local package_list_cmake=""
  local cuda_variants_cmake=""
  local package_name
  local package_kind
  local package_cuda_arch
  local backend_selector
  local i

  for i in "${!package_names[@]}"; do
    package_name=${package_names[$i]}
    package_kind=${package_kinds[$i]}
    package_cuda_arch=${package_cuda_arches[$i]}

    package_list_cmake="${package_list_cmake}${package_name};"
    if [[ "${package_kind}" == "cuda" && "${package_name}" != "cuda" ]]; then
      cuda_variants_cmake="${cuda_variants_cmake}${package_name};"
    fi
  done

  mkdir -p "${config_dir}"

  cat >"${config_dir}/KokkosConfig.cmake" <<CONFIG
set(_kokkos_sdk_root "\${CMAKE_CURRENT_LIST_DIR}/../../..")
set(_kokkos_packaged_entries "${package_list_cmake}")
set(_kokkos_cuda_variant_entries "${cuda_variants_cmake}")

if(NOT DEFINED Kokkos_BACKEND OR "\${Kokkos_BACKEND}" STREQUAL "")
  if(DEFINED ENV{KOKKOS_BACKEND} AND NOT "\$ENV{KOKKOS_BACKEND}" STREQUAL "")
    set(Kokkos_BACKEND "\$ENV{KOKKOS_BACKEND}")
  else()
    set(Kokkos_BACKEND "OPENMP")
  endif()
endif()

string(TOUPPER "\${Kokkos_BACKEND}" _kokkos_backend_upper)
set(_kokkos_backend_dir "")

if(_kokkos_backend_upper STREQUAL "OPENMP")
  set(_kokkos_backend_dir "openmp")
elseif(_kokkos_backend_upper STREQUAL "MACA")
  set(_kokkos_backend_dir "maca")
elseif(_kokkos_backend_upper STREQUAL "CUDA")
  if("cuda" IN_LIST _kokkos_packaged_entries)
    set(_kokkos_backend_dir "cuda")
  elseif(_kokkos_cuda_variant_entries)
    string(REPLACE ";" ", " _kokkos_cuda_variants_pretty "\${_kokkos_cuda_variant_entries}")
    message(
      FATAL_ERROR
        "Kokkos_BACKEND=CUDA is ambiguous for this SDK. Use one of: \${_kokkos_cuda_variants_pretty}"
    )
  else()
    message(FATAL_ERROR "Kokkos_BACKEND=CUDA requested, but no CUDA package is present in this SDK.")
  endif()
elseif(_kokkos_backend_upper MATCHES "^CUDA[-_]")
  string(REGEX REPLACE "^CUDA[-_]" "" _kokkos_cuda_suffix "\${_kokkos_backend_upper}")
  string(TOLOWER "\${_kokkos_cuda_suffix}" _kokkos_cuda_suffix_lower)
  set(_kokkos_backend_dir "cuda-\${_kokkos_cuda_suffix_lower}")
else()
  message(FATAL_ERROR "Unsupported Kokkos_BACKEND='\${Kokkos_BACKEND}'.")
endif()

set(_kokkos_delegate_config
    "\${_kokkos_sdk_root}/\${_kokkos_backend_dir}/lib/cmake/Kokkos/KokkosConfig.cmake")

if(NOT EXISTS "\${_kokkos_delegate_config}")
  message(
    FATAL_ERROR
      "Kokkos SDK backend '\${Kokkos_BACKEND}' is not packaged in '\${_kokkos_sdk_root}'. Missing: \${_kokkos_delegate_config}"
  )
endif()

set(Kokkos_SDK_ROOT "\${_kokkos_sdk_root}")
set(Kokkos_SELECTED_BACKEND "\${_kokkos_backend_dir}")

include("\${_kokkos_delegate_config}")

unset(_kokkos_delegate_config)
unset(_kokkos_backend_dir)
unset(_kokkos_backend_upper)
unset(_kokkos_packaged_entries)
unset(_kokkos_cuda_variant_entries)
unset(_kokkos_sdk_root)
CONFIG

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
Kokkos-SDK-v${version}/
├── lib/cmake/Kokkos/KokkosConfig.cmake
├── openmp/
├── cuda/
├── cuda-<arch>/
├── maca/
└── setup-*.sh
\`\`\`

## CMake usage

\`\`\`cmake
set(Kokkos_BACKEND "CUDA-BLACKWELL120" CACHE STRING "OPENMP / CUDA / CUDA-<ARCH> / MACA")
find_package(Kokkos REQUIRED PATHS "/path/to/Kokkos-SDK-v${version}" NO_DEFAULT_PATH)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE Kokkos::kokkos)
\`\`\`

Notes:

- If the SDK contains only a single plain \`cuda/\` package, \`Kokkos_BACKEND=CUDA\` works.
- If the SDK contains multiple CUDA packages such as \`cuda-ampere86/\` and \`cuda-blackwell120/\`, you must set \`Kokkos_BACKEND\` to a specific variant like \`CUDA-AMPERE86\` or \`CUDA_BLACKWELL120\`.
- If \`Kokkos_BACKEND\` is not set, the wrapper defaults to \`OPENMP\`.

## Shell helpers

Use the generated setup script for the package you want, for example:

- \`source ./setup-openmp.sh\`
- \`source ./setup-cuda.sh\`
- \`source ./setup-cuda-blackwell120.sh\`
- \`source ./setup-maca.sh\`

Each helper sets \`KOKKOS_BACKEND\` and prepends the SDK root to \`CMAKE_PREFIX_PATH\`.

## CI usage

This script supports split build and assembly stages:

\`\`\`bash
scripts/package-kokkos-sdk.sh build --backend openmp
scripts/package-kokkos-sdk.sh build --backend cuda --cuda-arch AMPERE86
scripts/package-kokkos-sdk.sh build --backend cuda --cuda-arch BLACKWELL120
scripts/package-kokkos-sdk.sh assemble --backend openmp,cuda
\`\`\`

In GitHub Actions or GitLab CI, run \`build\` in per-package jobs, upload the staged \`stage/<package>\` directories as artifacts, then collect them in a final \`assemble\` job.
README
}

assemble_sdk() {
  local package_name
  local package_kind
  local package_cuda_arch
  local package_csv
  local sdk_package_dir
  local config_dir="${sdk_dir}/lib/cmake/Kokkos"
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
    sdk_package_dir="${sdk_dir}/${package_name}"
    mkdir -p "${sdk_package_dir}"
    note "copying staged ${package_name} install into ${sdk_package_dir}"
    cp -a "${stage_dir}/${package_name}/." "${sdk_package_dir}/"

    case "${package_kind}" in
      openmp)
        backend_selector="OPENMP"
        ;;
      cuda)
        if [[ -n "${package_cuda_arch}" && "${package_name}" != "cuda" ]]; then
          backend_selector="CUDA-$(upper "${package_cuda_arch}")"
        else
          backend_selector="CUDA"
        fi
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
