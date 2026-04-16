#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

log() {
  printf '[build_kokkos_maca_cuda] %s\n' "$*"
  if [[ "${logging_enabled:-0}" -eq 1 ]]; then
    printf '[build_kokkos_maca_cuda] %s\n' "$*" >> "${log_file}"
  fi
}

die() {
  log "error: $*"
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  scripts/build_kokkos_maca_cuda.sh [options]

Build Kokkos with the working cu-bridge + `cmake_maca` flow, using a small
set of defaults instead of a large CLI surface.

Required environment:
  MACA_PATH               Root of the MACA installation

Optional environment:
  CUCC_PATH               Defaults to "${MACA_PATH}/tools/cu-bridge"
  CUCC_CMAKE_ENTRY        Defaults to 2
  CUCC_CUDA_VERSION       Passed through if already set
  KOKKOS_CU_BRIDGE_ARCH   Defaults to AMPERE80
  NVCC_WRAPPER_DEFAULT_COMPILER
                          Host compiler used when CMAKE_CXX_COMPILER is Kokkos
                          `bin/nvcc_wrapper`
  CC / CXX                Override host C/C++ compilers
  JOBS                    Default parallel build jobs

Options:
  --build-dir <DIR>       Build directory
  --examples              Enable Kokkos examples
  --tests                 Enable Kokkos tests
  --benchmarks            Enable Kokkos benchmarks
  --debug                 Use a Debug build
  --release               Use a Release build (default)
  -j, --jobs <N>          Parallel build jobs
  --skip-install          Skip the install step
  --clean                 Remove the build directory before configuring
  --dry-run               Print commands without executing them
  -h, --help              Show this help

Examples:
  export MACA_PATH=/opt/maca
  scripts/build_kokkos_maca_cuda.sh

  export MACA_PATH=/opt/maca
  scripts/build_kokkos_maca_cuda.sh \
    --examples \
    -j 64
EOF
}

detect_jobs() {
  if [[ -n "${JOBS:-}" ]]; then
    printf '%s\n' "${JOBS}"
    return
  fi

  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi

  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi

  printf '1\n'
}

run() {
  local command_string

  printf '+'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  if [[ "${logging_enabled:-0}" -eq 1 ]]; then
    printf '+' >> "${log_file}"
    for arg in "$@"; do
      printf ' %q' "${arg}" >> "${log_file}"
    done
    printf '\n' >> "${log_file}"
  fi

  if [[ "${dry_run}" -eq 0 ]]; then
    if [[ "${logging_enabled:-0}" -eq 1 ]]; then
      if command -v script >/dev/null 2>&1; then
        printf -v command_string '%q ' "$@"
        command_string="${command_string% }"
        script -q -e -f -c "${command_string}" /dev/null | tr -d '\000' | tee -a "${log_file}"
      else
        "$@" 2>&1 | tee -a "${log_file}"
      fi
    else
      "$@"
    fi
  fi
}

abspath() {
  local path=$1

  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${PWD}" "${path}"
  fi
}

link_tree_into_shim() {
  local source_root=$1
  local shim_root=$2
  local source_entry
  local entry_name

  mkdir -p "${shim_root}" "${shim_root}/bin"

  shopt -s nullglob dotglob
  for source_entry in "${source_root}"/*; do
    entry_name=$(basename "${source_entry}")
    [[ "${entry_name}" == "." || "${entry_name}" == ".." || "${entry_name}" == "bin" ]] && continue

    if [[ ! -e "${shim_root}/${entry_name}" ]]; then
      ln -s "${source_entry}" "${shim_root}/${entry_name}"
    fi
  done

  for source_entry in "${source_root}/bin"/*; do
    entry_name=$(basename "${source_entry}")
    if [[ ! -e "${shim_root}/bin/${entry_name}" ]]; then
      ln -s "${source_entry}" "${shim_root}/bin/${entry_name}"
    fi
  done
  shopt -u nullglob dotglob

  ln -sfn "${source_root}/bin/cucc" "${shim_root}/bin/cucc"
  ln -sfn "${source_root}/bin/cucc" "${shim_root}/bin/nvcc"
}

link_cuda_compat_libs() {
  local shim_root=$1
  local maca_root=$2

  mkdir -p "${shim_root}/lib64" "${shim_root}/lib64/stubs"

  [[ -f "${maca_root}/lib/libmcruntime.so" ]] || die "missing ${maca_root}/lib/libmcruntime.so"
  [[ -f "${maca_root}/lib/libsymbol_cu.so" ]] || die "missing ${maca_root}/lib/libsymbol_cu.so"

  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/libcudart.so"
  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/libcudart.so.11.0"
  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/libcudart.so.11.7.60"
  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/libcudart_static.a"
  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/libcudadevrt.a"
  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/libnvrtc.so"
  ln -sfn "${maca_root}/lib/libsymbol_cu.so" "${shim_root}/lib64/libcuda.so"
  ln -sfn "${maca_root}/lib/libsymbol_cu.so" "${shim_root}/lib64/libcuda.so.1"
  ln -sfn "${maca_root}/lib/libmcruntime.so" "${shim_root}/lib64/stubs/libcuda.so"
}

build_dir="${repo_root}/build-maca-cuda"
install_dir=""
log_file=""
kokkos_arch="${KOKKOS_CU_BRIDGE_ARCH:-AMPERE80}"
host_cc="${CC:-cc}"
if [[ -n "${CXX:-}" ]]; then
  host_cxx="${CXX}"
elif [[ -x "${repo_root}/bin/nvcc_wrapper" ]]; then
  host_cxx="${repo_root}/bin/nvcc_wrapper"
else
  host_cxx="c++"
fi
cmake_build_type="Release"
jobs=$(detect_jobs)
clean=0
dry_run=0
skip_install=0
enable_examples=0
enable_tests=0
enable_benchmarks=0
logging_enabled=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || die "--build-dir requires a value"
      build_dir="$2"
      shift 2
      ;;
    --build-dir=*)
      build_dir="${1#*=}"
      shift
      ;;
    --examples)
      enable_examples=1
      shift
      ;;
    --tests)
      enable_tests=1
      shift
      ;;
    --benchmarks)
      enable_benchmarks=1
      shift
      ;;
    --debug)
      cmake_build_type="Debug"
      shift
      ;;
    --release)
      cmake_build_type="Release"
      shift
      ;;
    -j|--jobs)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      jobs="$2"
      shift 2
      ;;
    --jobs=*)
      jobs="${1#*=}"
      shift
      ;;
    --skip-install)
      skip_install=1
      shift
      ;;
    --clean)
      clean=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "${MACA_PATH:-}" ]] || die "MACA_PATH must be set"
CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
[[ -d "${CUCC_PATH}" ]] || die "CUCC_PATH does not exist: ${CUCC_PATH}"
[[ -x "${CUCC_PATH}/bin/cucc" ]] || die "cu-bridge compiler not found: ${CUCC_PATH}/bin/cucc"

build_dir="$(abspath "${build_dir}")"
log_file="${build_dir}/build.log"

cmake_maca_cmd="${CUCC_PATH}/tools/cmake_maca"
make_maca_cmd="${CUCC_PATH}/tools/make_maca"

[[ -x "${cmake_maca_cmd}" ]] || command -v cmake_maca >/dev/null 2>&1 || die "cmake_maca not found at ${cmake_maca_cmd} or in PATH"
[[ -x "${make_maca_cmd}" ]] || command -v make_maca >/dev/null 2>&1 || die "make_maca not found at ${make_maca_cmd} or in PATH"
if [[ ! -x "${cmake_maca_cmd}" ]]; then
  cmake_maca_cmd="$(command -v cmake_maca)"
fi
if [[ ! -x "${make_maca_cmd}" ]]; then
  make_maca_cmd="$(command -v make_maca)"
fi
command -v "${host_cc}" >/dev/null 2>&1 || die "host C compiler not found: ${host_cc}"
command -v "${host_cxx}" >/dev/null 2>&1 || die "host C++ compiler not found: ${host_cxx}"

if [[ -z "${install_dir}" ]]; then
  install_dir="${build_dir}/install"
fi
install_dir="$(abspath "${install_dir}")"

cuda_shim_root="${build_dir}/.cu-bridge-cuda-root"
cubridge_home_default="${build_dir}/.cu-bridge-home"

if [[ "${clean}" -eq 1 ]]; then
  run rm -rf "${build_dir}"
fi

if [[ "${dry_run}" -eq 0 ]]; then
  mkdir -p "${build_dir}"
  : > "${log_file}"
  logging_enabled=1
  mkdir -p "${CUBRIDGE_HOME:-${cubridge_home_default}}"
  link_tree_into_shim "${CUCC_PATH}" "${cuda_shim_root}"
  link_cuda_compat_libs "${cuda_shim_root}" "${MACA_PATH}"
fi

export CUCC_PATH
export CUCC_CMAKE_ENTRY="${CUCC_CMAKE_ENTRY:-2}"
export CUBRIDGE_HOME="${CUBRIDGE_HOME:-${cubridge_home_default}}"
export CUDA_PATH="${cuda_shim_root}"
export CUDA_ROOT="${cuda_shim_root}"
export CUDAToolkit_ROOT="${cuda_shim_root}"
if [[ -z "${NVCC_WRAPPER_DEFAULT_COMPILER:-}" ]]; then
  if [[ "$(basename "${host_cxx}")" == "nvcc_wrapper" ]]; then
    export NVCC_WRAPPER_DEFAULT_COMPILER="c++"
  else
    export NVCC_WRAPPER_DEFAULT_COMPILER="${host_cxx}"
  fi
else
  export NVCC_WRAPPER_DEFAULT_COMPILER
fi
export PATH="${cuda_shim_root}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"

declare -a cmake_args=(
  -S "${repo_root}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE="${cmake_build_type}"
  -DCMAKE_INSTALL_PREFIX="${install_dir}"
  -DCMAKE_C_COMPILER="${host_cc}"
  -DCMAKE_CXX_COMPILER="${host_cxx}"
  -DCMAKE_SKIP_RPATH=ON
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_CXX_EXTENSIONS=OFF
  -DBUILD_SHARED_LIBS=OFF
  -DCUDA_ROOT="${cuda_shim_root}"
  -DCUDAToolkit_ROOT="${cuda_shim_root}"
  -DCUDA_TOOLKIT_ROOT_DIR="${cuda_shim_root}"
  -DKokkos_ENABLE_CUDA=ON
  -DKokkos_ENABLE_SERIAL=ON
  -DKokkos_ENABLE_TESTS=$([[ "${enable_tests}" -eq 1 ]] && printf 'ON' || printf 'OFF')
  -DKokkos_ENABLE_EXAMPLES=$([[ "${enable_examples}" -eq 1 ]] && printf 'ON' || printf 'OFF')
  -DKokkos_ENABLE_BENCHMARKS=$([[ "${enable_benchmarks}" -eq 1 ]] && printf 'ON' || printf 'OFF')
  -DKokkos_ARCH_${kokkos_arch}=ON
)

log "Repo root: ${repo_root}"
log "Build dir: ${build_dir}"
log "Build log: ${log_file}"
log "Install dir: ${install_dir}"
log "Host C compiler: ${host_cc}"
log "Host C++ compiler: ${host_cxx}"
log "NVCC_WRAPPER_DEFAULT_COMPILER: ${NVCC_WRAPPER_DEFAULT_COMPILER}"
log "MACA_PATH: ${MACA_PATH}"
log "CUCC_PATH: ${CUCC_PATH}"
log "CUBRIDGE_HOME: ${CUBRIDGE_HOME}"
log "CUDA shim root: ${cuda_shim_root}"
log "Kokkos arch: ${kokkos_arch}"
log "Kokkos examples: $([[ "${enable_examples}" -eq 1 ]] && printf 'ON' || printf 'OFF')"
log "Kokkos tests: $([[ "${enable_tests}" -eq 1 ]] && printf 'ON' || printf 'OFF')"
log "Kokkos benchmarks: $([[ "${enable_benchmarks}" -eq 1 ]] && printf 'ON' || printf 'OFF')"
if [[ -n "${CUCC_CUDA_VERSION:-}" ]]; then
  log "CUCC_CUDA_VERSION: ${CUCC_CUDA_VERSION}"
fi

run "${cmake_maca_cmd}" "${cmake_args[@]}"
run "${make_maca_cmd}" -C "${build_dir}" -j "${jobs}"

if [[ "${skip_install}" -eq 0 ]]; then
  run "${make_maca_cmd}" -C "${build_dir}" -j "${jobs}" install
fi
