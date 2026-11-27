# XPU P2P Test Suite (Intel Arc / Level Zero)

Peer-to-peer and collective validation for Intel discrete GPUs. The suite drives
Level Zero, SYCL and oneCCL directly to answer one question: can two or more
Intel GPUs move data and synchronise with each other, and does the collective
library on top of that path return correct results? It enumerates the P2P
attribute matrix, runs L0 cross-device copies, kernel-initiated peer writes and
peer atomics, sweeps bandwidth against a host-staged control, and exercises
oneCCL allreduce under MPI. Every kernel test is a bounded subprocess, so a
wedged driver path cannot hang the whole run.

## Requirements

- Intel discrete GPU supported by the `xe` kernel driver (Arc / Arc Pro class),
  with two or more GPUs visible for anything beyond `canaccess`.
- oneAPI 2026.x toolkit providing `icpx`, SYCL, and the Level Zero headers and
  `ze_loader` library.
- Intel MPI providing `mpiicpx` and `mpiexec`.
- A oneCCL build compiled with Level Zero support, for `xpu_ccl_allreduce`.
- Python 3 for the orchestrator, plus `xpu-smi` and `lspci` for the topology stage.

## Build

```
. /opt/intel/oneapi/setvars.sh --force   # or your toolkit's own env helper
make                                     # builds ./xpu_p2p_test
make allreduce                           # builds ./xpu_ccl_allreduce
```

`make allreduce` links oneCCL from a prefix supplied via `VENV` (it defaults to
a vLLM virtualenv under `$HOME`); point it at any prefix containing
`include/oneapi/ccl.hpp`, `lib/libccl.so` and `lib/ccl/kernels`, for example
`make allreduce VENV=/opt/oneapi-ccl`. Both binaries and all `*.log` files are
gitignored.

## Running the Level Zero / SYCL tests

The binary takes a subcommand and an ordered `src`/`dst` device pair:

| Command | What it does |
| --- | --- |
| `./xpu_p2p_test canaccess` | dumps `zeDeviceCanAccessPeer` and SYCL `is_peer` for every ordered pair, plus the L0 BDF list. Start here. |
| `./xpu_p2p_test transfer SRC DST [MB]` | L0 `zeCommandListAppendMemoryCopy` device-to-device, byte-verified, reports GB/s. The authoritative probe: it either performs peer DMA or fails; it never silently stages through host memory. |
| `./xpu_p2p_test transfer-kernel SRC DST [MB]` | SYCL kernel on `SRC` writes into `DST` memory and reads it back — exercises the peer path without the copy engine. |
| `./xpu_p2p_test atomic SRC DST [ITERS]` | kernel on `SRC` issues `atomicAdd`s against a counter in `DST`. |
| `./xpu_p2p_test atomic-par DEV [ITERS]` | every GPU atomically increments one shared counter concurrently. |
| `./xpu_p2p_test bandwidth SRC DST [ITERS]` | sized sweep, direct peer path against a host-staged control. |
| `./xpu_p2p_test stress [PER_SRC]` | full-matrix all-to-one atomic stress with a rotating target. |

Exit codes: `0` pass, `1` fail, `2` skip (fewer than two GPUs, or the pair does
not support it). Kernel tests are meaningful only where `canaccess` reported peer access.

### Orchestrator

`run_p2p_test.py` runs the sequence and interprets the output:

```
python3 run_p2p_test.py                 # topology, trust model, attributes, kernel tests
python3 run_p2p_test.py --skip-kernel   # stages 1-3 only, no GPU kernel work
python3 run_p2p_test.py --smoke         # first peer-capable pair only
python3 run_p2p_test.py --timeout 120   # per-test kernel timeout in seconds
```

Stages: (1) `xpu-smi topology -m` plus `lspci -PP` ancestry, showing which root
complex or switch each GPU sits under; (2) `dmesg` peer-to-peer refusals and
the host-bridge trust model; (3) the attribute matrix; (4) the kernel tests,
each a subprocess bounded by `--timeout`. Exit `0` all passed, `1` anything
failed or timed out, `2` orchestrator-level error.

## Raw P2P capability probe

`ze_p2p_probe.c` is a dependency-free dump of exactly what oneCCL's topology
manager reads — `zeDeviceGetP2PProperties` flags and `zeDeviceCanAccessPeer` —
for the self pair and every ordered pair, with no SYCL in the way:

```
icpx -x c -O2 -o ze_p2p_probe ze_p2p_probe.c -lze_loader
./ze_p2p_probe
```

`-x c` is required: `icpx` is a C++ driver and would otherwise compile the C
source as C++ and choke on the Level Zero struct initialisers. Any C compiler
works too (`gcc -O2 -o ze_p2p_probe ze_p2p_probe.c -lze_loader`).

## oneCCL allreduce

`xpu_ccl_allreduce` is MPI-launched, **one rank per GPU**: rank *r* binds GPU
*r* through `ZE_AFFINITY_MASK` before Level Zero init, so `NP` must not exceed
the GPU count. `run_allreduce.sh` wires up the toolkit, oneCCL's transport and its kernels:

```
./run_allreduce.sh --sizes-bytes 1048576,16777216 --verify --iters 5 --warmup 20 --serial 20
```

Overrides:

- `NP` — number of MPI ranks, one per GPU.
- `VENV` — oneCCL prefix (headers, `libccl`, `kernels.spv`).
- `TIMEOUT` — seconds before the collective is hard-killed and the ranks
  reaped. **Always set it**: otherwise a deadlock leaves `mpiexec` blocking the shell.
- `CCL_ATL_TRANSPORT`, `FI_PROVIDER`, `CCL_KERNEL_PATH` — transport and kernel
  wiring, with working defaults.

The binary prints `RANK` (rank to BDF mapping), `ALLREDUCE` (latency and bus
bandwidth) and, with `--verify`, `ALLREDUCEVERIFY` lines showing the observed
versus expected sum; the exit code is non-zero if any rank disagrees.

## Environment notes

oneCCL's default algorithm selection can deadlock on some Battlemage
configurations for messages above a few MiB: the collective is routed through a
protocol that never completes, so the run hangs rather than failing. Raising
`CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` to a value at or above your message size
forces the low-latency protocol and lets those sizes finish. Treat that as a
diagnostic and a workaround, not a fix — and keep `--verify` on, because a
collective that returns quickly with the wrong sum is worse than a hang.

## License

MIT — see [LICENSE](LICENSE).
