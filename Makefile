# XPU P2P Test Suite — Intel Arc (Level Zero / SYCL)
CXX := icpx
CXXFLAGS := -fsycl -O2 -std=c++17 -Wall
LDFLAGS := -lze_loader

ONEAPI_ENV := $(shell test -f /usr/local/bin/oneapi-env.sh && echo ". /usr/local/bin/oneapi-env.sh" || echo ". /opt/intel/oneapi/setvars.sh --force")

# Resolve the MPI bin dir explicitly: a per-host helper may omit mpi from its
# component list, leaving mpiicpx off PATH.
MPI_BIN := $(shell ls -d /opt/intel/oneapi/mpi/20*/bin 2>/dev/null | sort -V | tail -1)
# mpiicpx derives its include and lib search path from I_MPI_ROOT, so export it
# next to PATH.
ENV_PREFIX := $(ONEAPI_ENV) && $(if $(MPI_BIN),export I_MPI_ROOT=$(MPI_BIN)/.. && export PATH=$(MPI_BIN):$$PATH && )

BINARY := xpu_p2p_test

# oneCCL allreduce binary. Needs a Level Zero enabled oneCCL plus Intel MPI; point
# VENV at a prefix that ships the headers, libccl and kernels.spv (see README).
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
