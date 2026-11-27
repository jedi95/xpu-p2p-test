#!/usr/bin/env bash
# Create a Python environment carrying the oneCCL runtime, headers and SYCL
# kernels that xpu_ccl_allreduce builds and runs against.
#
# Installed from PyPI (about 150 MB of wheels, roughly 1.8 GB on disk: the oneCCL
# libraries plus the SYCL runtime and fabric libraries they depend on):
#   oneccl        lib/libccl.so.1, lib/libccl.so.2, lib/ccl/kernels/kernels.spv
#   oneccl-devel  include/oneapi/ccl.hpp, lib/libccl.a
#
# Those two wheels are the entire Python requirement. The test binary is C++ and
# links libccl directly, so no framework (torch, vLLM) is installed or imported;
# the environment exists only to deliver the libraries, headers and kernels.
#
# The compiler and the MPI launcher are not Python packages. This script checks
# for them and prints the package names to install when they are absent.
#
# Usage:
#   ./setup_venv.sh [VENV_DIR]              # default: ./.venv-ccl
#   CCL_VERSION=2022.2.3 ./setup_venv.sh    # pin a specific release
#   ./setup_venv.sh --check [VENV_DIR]      # verify an existing environment only
#
# Exit codes: 0 ready, 1 environment incomplete, 2 usage or download error.

set -euo pipefail

CHECK_ONLY=0
VENV_DIR=""

for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) VENV_DIR="$arg" ;;
    esac
done
VENV_DIR="${VENV_DIR:-$PWD/.venv-ccl}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# Report the three artifacts the build and the runtime need, and fail if any is
# absent. Paths follow where the wheels place their payload: libraries and
# kernels sit beside the venv's lib/ directory, not inside site-packages.
verify_env() {
    local venv="$1" missing=0
    local lib="$venv/lib/libccl.so.1"
    local kern="$venv/lib/ccl/kernels/kernels.spv"
    local hdr="$venv/include/oneapi/ccl.hpp"

    say "oneCCL payload in $venv:"
    if [ -f "$lib" ]; then
        say "  libccl.so.1     $(du -h "$lib" | cut -f1)"
    else
        say "  libccl.so.1     MISSING"; missing=1
    fi
    if [ -f "$kern" ]; then
        say "  kernels.spv     $(du -h "$kern" | cut -f1)"
    else
        say "  kernels.spv     MISSING"; missing=1
    fi
    if [ -f "$hdr" ]; then
        say "  oneapi/ccl.hpp  present"
    else
        say "  oneapi/ccl.hpp  MISSING (needed to compile)"; missing=1
    fi

    local ver
    ver=$(basename "$(ls -d "$venv"/lib/python*/site-packages/oneccl-*.dist-info 2>/dev/null | head -1)" .dist-info 2>/dev/null || true)
    [ -n "$ver" ] && say "  package         ${ver}"

    return $missing
}

# icpx, mpiicpx and mpiexec come from the oneAPI toolkit and Intel MPI. Probe
# them in a subshell after loading the toolkit environment, so a caller without
# setvars sourced is not reported as missing a toolchain.
toolchain_status() {
    local helper=/usr/local/bin/oneapi-env.sh
    bash -c "
        if [ -f $helper ]; then . $helper >/dev/null 2>&1
        elif [ -f /opt/intel/oneapi/setvars.sh ]; then . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
        fi
        for t in icpx mpiicpx mpiexec; do command -v \$t >/dev/null 2>&1 || echo \$t; done
    "
}

create_env() {
    local venv="$1"
    say "creating venv at $venv"
    if python3 -m venv "$venv" 2>/dev/null; then
        :
    elif command -v uv >/dev/null 2>&1; then
        uv venv "$venv" >/dev/null
    else
        die "cannot create a venv: install python3-venv (apt install python3-venv) or uv"
    fi

    local pin=""
    [ -n "${CCL_VERSION:-}" ] && pin="==${CCL_VERSION}"
    say "installing oneccl${pin} and oneccl-devel${pin} from PyPI"
    if [ -x "$venv/bin/python" ] && "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
        "$venv/bin/python" -m pip install --disable-pip-version-check --no-input \
            "oneccl${pin}" "oneccl-devel${pin}"
    elif command -v uv >/dev/null 2>&1; then
        uv pip install --python "$venv/bin/python" "oneccl${pin}" "oneccl-devel${pin}"
    else
        die "no pip in $venv and no uv on PATH"
    fi
}

if [ "$CHECK_ONLY" -eq 0 ]; then
    [ -e "$VENV_DIR" ] && [ ! -d "$VENV_DIR" ] && die "$VENV_DIR exists and is not a directory"
    if [ -d "$VENV_DIR" ]; then
        say "reusing existing directory $VENV_DIR"
        if ! [ -f "$VENV_DIR/lib/libccl.so.1" ]; then
            say "oneCCL is not installed there; installing into it"
            create_env "$VENV_DIR"
        fi
    else
        create_env "$VENV_DIR"
    fi
elif [ ! -d "$VENV_DIR" ]; then
    die "$VENV_DIR does not exist"
fi

rc=0
verify_env "$VENV_DIR" || rc=1

missing=$(toolchain_status | tr '\n' ' ')
if [ -n "${missing// /}" ]; then
    say ""
    say "Toolchain not found on PATH: $missing"
    say "Install the oneAPI compiler and Intel MPI, then source the environment:"
    say "  sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mpi-devel"
    say "  . /opt/intel/oneapi/setvars.sh --force"
    rc=1
fi

if [ "$rc" -ne 0 ]; then
    say ""
    say "Environment is not ready."
    exit 1
fi

cat <<EOF

Environment ready. Build and run:

  make allreduce
  TIMEOUT=60 ./run_allreduce.sh --sizes-bytes 4194304 --verify

Both read ./.venv-ccl by default; export VENV only to select a different prefix.
run_allreduce.sh starts one MPI rank per GPU, and TIMEOUT reaps the ranks if a
collective stops making progress.
EOF
