#!/usr/bin/env bash
# Run xpu_ccl_allreduce: one MPI rank per Intel GPU.
#
# Environment: oneCCL comes from the prefix in VENV (default ./.venv-ccl, created
# by setup_venv.sh), Intel MPI from /opt/intel/oneapi/mpi, ATL over OFI with the
# tcp provider.
# ZE_AFFINITY_MASK is only defaulted to the rank number by the binary, so a
# caller-supplied mask wins (put every rank on one GPU with ZE_AFFINITY_MASK=0).
#
# TIMEOUT=<seconds>  hard-kill a hanging collective and reap the ranks. The run
#   goes into its own session because Intel MPI kills its process group on
#   timeout, which would otherwise take down the calling shell.
#
# Env overrides: NP (ranks), VENV, CCL_ALLREDUCE, CCL_ATL_TRANSPORT,
#   FI_PROVIDER, CCL_KERNEL_PATH, plus anything oneCCL honours
#   (CCL_SYCL_ALLREDUCE_*, CCL_SYCL_CCL_BARRIER).
set -u
cd "$(dirname "$0")"

# Component vars.sh scripts can reset the caller's positional parameters, so
# snapshot the arguments before sourcing anything. Otherwise every --sizes-bytes
# argument is dropped and the binary falls back to its default size ladder.
ARGS=("$@")

set +u
# A per-host helper takes precedence: setvars.sh is unreliable on bash 5.3 hosts
# and refuses to re-run without --force elsewhere.
if [ -f /usr/local/bin/oneapi-env.sh ]; then
    # shellcheck disable=SC1091
    . /usr/local/bin/oneapi-env.sh
else
    # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi
set -u

# A helper may omit mpi from its component list, which leaves mpiexec off PATH;
# resolve the MPI prefix directly when that happens.
MPI_PREFIX=${I_MPI_ROOT:-}
if [ -z "$MPI_PREFIX" ]; then
    MPI_PREFIX=$(ls -d /opt/intel/oneapi/mpi/*/lib 2>/dev/null | sort -V | tail -1)
    MPI_PREFIX=${MPI_PREFIX%/lib}
fi
if ! command -v mpiexec >/dev/null 2>&1 && [ -n "$MPI_PREFIX" ]; then
    PATH="$MPI_PREFIX/bin:$PATH"
fi
export PATH

VENV=${VENV:-$PWD/.venv-ccl}
NP=${NP:-2}
MPI_LIB_DIR="${MPI_PREFIX:-/opt/intel/oneapi/mpi/latest}/lib"
# CCL's OFI ATL dlopens libfabric at init. A per-host helper that omits libfabric
# leaves it off the loader path, and CCL aborts with
# "api_wrappers_init: condition (mpi_inited || ofi_inited) failed" and then spins.
FI_LIB=$(ls -d /opt/intel/oneapi/libfabric/*/lib 2>/dev/null | sort -V | tail -1)
export LD_LIBRARY_PATH="$VENV/lib:$MPI_LIB_DIR${FI_LIB:+:$FI_LIB}:${LD_LIBRARY_PATH:-}"
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-ofi}
export CCL_KERNEL_PATH="${CCL_KERNEL_PATH:-$VENV/lib/ccl/kernels/}"
export FI_PROVIDER=${FI_PROVIDER:-tcp}

if [ "${TIMEOUT:-0}" -gt 0 ] 2>/dev/null; then
    setsid -w bash -c "timeout -k 5 $TIMEOUT mpiexec -n $NP ./xpu_ccl_allreduce \"\$@\"" _ "${ARGS[@]}"
    rc=$?
    # timeout terminates mpiexec, but ranks can outlive it and hold the GPUs at
    # max clock. Match only processes whose command line starts with the binary:
    # some MPI builds give the ranks a bare "./xpu_ccl_allreduce" cmdline, so
    # "--sizes-bytes" misses them, and an unanchored pattern also matches shells
    # that merely mention the name, including this script's caller.
    REAP='^[^ ]*xpu_ccl_allreduce'
    pkill -f "$REAP" >/dev/null 2>&1
    for _ in 1 2 3 4 5; do
        pgrep -f "$REAP" >/dev/null 2>&1 || break
        sleep 1
    done
    pkill -9 -f "$REAP" >/dev/null 2>&1
    exit $rc
fi
exec mpiexec -n "$NP" ./xpu_ccl_allreduce "${ARGS[@]}"
