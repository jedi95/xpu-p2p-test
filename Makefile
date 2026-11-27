# XPU P2P Test Suite — Intel Arc Pro B70 (Level Zero / SYCL)
CXX := icpx
CXXFLAGS := -fsycl -O2 -std=c++17 -Wall
LDFLAGS := -lze_loader

ONEAPI_ENV := $(shell test -f /usr/local/bin/oneapi-env.sh && echo ". /usr/local/bin/oneapi-env.sh" || echo ". /opt/intel/oneapi/setvars.sh --force")

# Some hosts' per-host helper omits mpi/ccl from its component list, so
# resolve the MPI bin dir ourselves: without it `mpiicpx` is not on PATH and
# `make allreduce` dies with "Error 127".
MPI_BIN := $(shell ls -d /opt/intel/oneapi/mpi/20*/bin 2>/dev/null | sort -V | tail -1)
# Export I_MPI_ROOT too: mpiicpx derives its include/lib search path from it, and
# hosts whose helper skips mpi otherwise fail with "mpi.h file not found".
ENV_PREFIX := $(ONEAPI_ENV) && $(if $(MPI_BIN),export I_MPI_ROOT=$(MPI_BIN)/.. && export PATH=$(MPI_BIN):$$PATH && )

BINARY := xpu_p2p_test

# oneCCL allreduce binary. Needs an L0-enabled oneCCL plus Intel MPI; on hosts
# without `intel-oneapi-mpi-devel`/`intel-oneapi-ccl-devel` from apt, point
# VENV at the vLLM venv that ships headers, libccl and kernels.spv (README).
VENV ?= $(HOME)/vllm-install/.venv
CCL_CXX := mpiicpx
CCL_CXXFLAGS := -fsycl -O2 -std=c++17
CCL_BINARY := xpu_ccl_allreduce

.PHONY: all run smoke enum-only allreduce run-allreduce clean

all: $(BINARY)

$(BINARY): xpu_p2p_test.cpp
	bash -c '$(ENV_PREFIX)$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)'

run: $(BINARY)
	bash -c '$(ENV_PREFIX)python3 run_p2p_test.py'

smoke: $(BINARY)
	bash -c '$(ENV_PREFIX)python3 run_p2p_test.py --smoke'

enum-only: $(BINARY)
	bash -c '$(ENV_PREFIX)python3 run_p2p_test.py --skip-kernel'

allreduce: $(CCL_BINARY)

$(CCL_BINARY): xpu_ccl_allreduce.cpp
	bash -c '$(ENV_PREFIX)$(CCL_CXX) $(CCL_CXXFLAGS) -o $@ $< -I$(VENV)/include -L$(VENV)/lib -lccl $(LDFLAGS)'

run-allreduce: $(CCL_BINARY)
	./run_allreduce.sh --verify

clean:
	rm -f $(BINARY) $(CCL_BINARY)
