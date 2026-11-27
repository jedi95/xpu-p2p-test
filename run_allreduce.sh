#!/usr/bin/env bash
# Run xpu_ccl_allreduce: one MPI rank per Intel GPU.
#
# Env traps (see README.md): oneCCL comes from the vLLM venv (L0-enabled build),
# Intel MPI from /opt/intel/oneapi/mpi, CCL_ATL_TRANSPORT=ofi + FI_PROVIDER=tcp.
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

# Component vars.sh scripts sourced by the per-host helper reset the caller's
# positional parameters (observed on Ubuntu 26.04 hosts: 8 args in, 0 out), so
# keep a copy before touching the environment. Without this, every
# --sizes-bytes argument is silently dropped and the binary runs its default
# size ladder instead.
ARGS=("$@")

set +u
# Prefer the per-host helper: setvars.sh sets nothing on some Ubuntu 26.04 /
# bash 5.3 hosts, and refuses to re-run without --force elsewhere.
if [ -f /usr/local/bin/oneapi-env.sh ]; then
    # shellcheck disable=SC1091
    . /usr/local/bin/oneapi-env.sh
else
    # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi
set -u

# Some hosts' helper omits mpi/ccl from its component list, which leaves
# mpiexec/mpiicpx off PATH; resolve the MPI prefix by hand in that case.
MPI_PREFIX=${I_MPI_ROOT:-}
if [ -z "$MPI_PREFIX" ]; then
    MPI_PREFIX=$(ls -d /opt/intel/oneapi/mpi/*/lib 2>/dev/null | sort -V | tail -1)
    MPI_PREFIX=${MPI_PREFIX%/lib}
fi
if ! command -v mpiexec >/dev/null 2>&1 && [ -n "$MPI_PREFIX" ]; then
    PATH="$MPI_PREFIX/bin:$PATH"
fi
export PATH

VENV=${VENV:-$HOME/vllm-install/.venv}
NP=${NP:-2}
MPI_LIB_DIR="${MPI_PREFIX:-/opt/intel/oneapi/mpi/latest}/lib"
export LD_LIBRARY_PATH="$VENV/lib:$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-ofi}
export CCL_KERNEL_PATH="${CCL_KERNEL_PATH:-$VENV/lib/ccl/kernels/}"
export FI_PROVIDER=${FI_PROVIDER:-tcp}

if [ "${TIMEOUT:-0}" -gt 0 ] 2>/dev/null; then
    setsid -w bash -c "timeout -k 5 $TIMEOUT mpiexec -n $NP ./xpu_ccl_allreduce \"\$@\"" _ "${ARGS[@]}"
    rc=$?
    # timeout kills mpiexec, but the ranks can survive it (seen on some hosts) and
    # keep both GPUs pinned at max clock, which poisons every later measurement.
    # The ranks' own cmdline is bare "./xpu_ccl_allreduce" on some MPI builds, so
    # matching "--sizes-bytes" misses them. Anchor the pattern at the start of the
    # command line: an unanchored match also hits every shell that merely mentions
    # the binary - including the caller of this script - and killing that takes out
    # the whole invocation.
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
