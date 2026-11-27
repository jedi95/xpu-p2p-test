#!/usr/bin/env python3
"""
XPU P2P Test Orchestrator
=========================

Intel Arc (Level Zero / SYCL) port of the CUDA p2p_test and ROCm p2p_test
orchestrators. Evaluates P2P support across all GPU pairs in four stages:

  1. xpu-smi topology matrix + lspci root-complex/switch ancestry
     (xpu-smi for GPU state; lspci -PP only for the host-bridge layout —
     lspci endpoint link speeds are unreliable on Arc, while the device
     ancestry determines whether P2P *can* work)
  2. dmesg P2P refusal check + kernel P2P trust-model assessment
     (consumer Intel RCs refuse P2P and corrupt when forced; AMD Zen+
     blanket-trusts; Intel server GNR host bridges are in the kernel
     whitelist — verify which case applies)
  3. Level Zero attribute report — zeDeviceCanAccessPeer + SYCL is_peer
     for every ordered pair, cross-checked against the L0 BDF list
  4. Kernel tests (only for pairs that report peer access):
       - transfer        L0 cross-device copy + byte verification + GB/s
       - transfer-kernel kernel-initiated peer write + peer read
       - atomic          P2P atomicAdd from source GPU to target counter
       - bandwidth       sized sweep, direct vs host-staged control
       - stress          full-matrix all-to-one atomic stress

Each kernel test runs as a separate subprocess with a timeout so a hung
driver path doesn't block the suite (P2P direction failures on Intel can
hard-lock a CPU core, which is why every test is bounded by a timeout).

Usage:
    ./run_p2p_test.py [--timeout SECONDS] [--skip-kernel] [--smoke]

    --timeout SECONDS   Per-test kernel timeout (default: 60)
    --skip-kernel       Only run stages 1+2+3 (topology + dmesg + attributes)
    --smoke             Run a quick subset (first pair only)

Exit code:
    0  all tests passed
    1  one or more tests failed (includes timeouts)
    2  orchestrator-level error (could not init L0, <2 GPUs, etc.)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = SCRIPT_DIR / "xpu_p2p_test"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd, timeout=30, use_sudo=False):
    """Run a command, return (rc, stdout+stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"(timeout after {timeout}s)"
    except FileNotFoundError:
        return -2, f"(command not found: {cmd[0]})"


def source_oneapi():
    """Return a shell prefix that loads the oneAPI environment, or "" when this
    process already runs inside an initialised oneAPI environment.

    setvars.sh prints its usage text and returns non-zero when it is sourced a
    second time without --force, which makes every wrapped command fail with no
    useful output. Skip it when the environment is already initialised, and
    prefer the opt-in /usr/local/bin/oneapi-env.sh helper where one exists.
    """
    if os.environ.get("ONEAPI_ROOT") and shutil.which("icpx"):
        return ""
    if os.path.exists("/usr/local/bin/oneapi-env.sh"):
        return ". /usr/local/bin/oneapi-env.sh"
    return ". /opt/intel/oneapi/setvars.sh"

# ---------------------------------------------------------------------------
# Stage 1: topology
# ---------------------------------------------------------------------------

def stage_topology():
    print("=" * 70)
    print("STAGE 1: Topology (xpu-smi matrix + PCI ancestry)")
    print("=" * 70)

    rc, out = run_cmd(["xpu-smi", "discovery"])
    ndevices = len(re.findall(r"Device Name:", out))
    print(f"  xpu-smi discovery: {ndevices} GPU(s) found")
    if ndevices < 2:
        print("  Need >= 2 GPUs for P2P testing.")
        return None

    rc, out = run_cmd(["xpu-smi", "topology", "-m"])
    print(out.rstrip() if out.strip() else "  (xpu-smi topology -m not supported)")

    # PCI ancestry: for each GPU endpoint, walk up to the host bridge.
    # Two GPUs share an upstream bridge -> potential native P2P (or
    # switch-routed). Separate root complexes -> depends on host-bridge
    # trust model (see stage 2).
    rc, out = run_cmd(["bash", "-c",
                       "for d in $(xpu-smi discovery -j 2>/dev/null | "
                       "grep -oE '[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-9a-fA-F]' | "
                       "sort -u); do echo \"== $d ==\"; lspci -s $d -PP 2>/dev/null; done"])
    if out.strip():
        print("  PCI ancestry (endpoint -> ... -> host bridge):")
        for line in out.strip().split("\n"):
            print(f"    {line}")

    print()
    return {"ngpu": ndevices}


# ---------------------------------------------------------------------------
# Stage 2: kernel P2P trust model
# ---------------------------------------------------------------------------

def stage_trust_model():
    print("=" * 70)
    print("STAGE 2: Kernel P2P Trust Model (dmesg + CPU vendor)")
    print("=" * 70)

    rc, out = run_cmd(["sudo", "dmesg"])
    refusals = [l for l in out.split("\n") if "peer-to-peer" in l.lower()]
    if refusals:
        print(f"  dmesg P2P refusals ({len(refusals)}):")
        for l in refusals[:5]:
            print(f"    {l.strip()}")
    else:
        print("  dmesg: no peer-to-peer refusals (blanket-whitelisted or untested)")

    rc, cpu = run_cmd(["bash", "-c",
                       'lscpu | grep -i "model name"; grep -m1 "cpu family" /proc/cpuinfo'])
    print(f"  CPU: {cpu.strip()}")
    if "Intel" in cpu and ("GenuineIntel" in cpu):
        fam = None
        m = re.search(r"cpu family\s*:\s*(\d+)", run_cmd(["cat", "/proc/cpuinfo"])[1])
        if m:
            fam = int(m.group(1))
        if fam == 6:
            print("  Trust model: Intel cpu family 6 (client + most server parts).")
            print("  -> pci_p2pdma_whitelist applies. GNR/Xeon-6 host bridges")
            print("     are server-class and MAY be whitelisted; consumer RCs")
            print("     refuse P2P and corrupt silently if forced (issue #935).")
            print("  -> Empirical kernel test decides. Do NOT patch the whitelist.")
    print()
    return {}


# ---------------------------------------------------------------------------
# Stage 3: Level Zero attribute report
# ---------------------------------------------------------------------------

def run_binary(args, timeout):
    """Run a kernel test subprocess with timeout. Returns (status, output).
    Exit codes: 0=PASS, 1=FAIL, 2=SKIP, other=FAIL."""
    prefix = source_oneapi()
    shell_cmd = (prefix + " && " if prefix else "") + f"exec {BINARY} " + " ".join(args)
    try:
        r = subprocess.run(
            ["bash", "-c", shell_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        rc = r.returncode
        out = (r.stdout or "") + (r.stderr or "")
        if rc == 0:
            return ("PASS", out)
        if rc == 2:
            return ("SKIP", out)
        return ("FAIL", out)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", f"Test timed out after {timeout}s")


def stage_attributes(timeout=60):
    print("=" * 70)
    print("STAGE 3: Level Zero P2P Attribute Report")
    print("=" * 70)

    status, out = run_binary(["canaccess"], timeout)
    print(out.rstrip())

    devices, canaccess = {}, {}
    for line in out.split("\n"):
        m = re.match(r"DEVICE idx=(\d+) bdf=([0-9a-fA-F:]+) name=(.*)", line)
        if m:
            devices[int(m.group(1))] = {"bdf": m.group(2), "name": m.group(3)}
        m = re.match(r"CANACCESS src=(\d+) dst=(\d+) access=(\d+) l0rc=(-?\d+)", line)
        if m:
            canaccess[(int(m.group(1)), int(m.group(2)))] = (
                int(m.group(3)) == 1, int(m.group(4)))

    if status == "FAIL" and not canaccess:
        print("  ERROR: canaccess probe failed.")
        print()
        return None

    # Cross-reference device indices with BDFs so pair interpretation can
    # be tied back to the root-complex ancestry from stage 1.
    print()
    print("  Summary:")
    print(f"    zeDeviceCanAccessPeer=1 pairs: "
          f"{sorted(k for k, (a, _) in canaccess.items() if a)}")
    print(f"    zeDeviceCanAccessPeer=0 pairs: "
          f"{sorted(k for k, (a, _) in canaccess.items() if not a)}")
    print()
    return {
        "devices": devices,
        "canaccess": canaccess,
    }


# ---------------------------------------------------------------------------
# Stage 4: kernel tests
# ---------------------------------------------------------------------------

def stage_kernel_tests(attrs, timeout, smoke=False):
    print("=" * 70)
    print("STAGE 4: P2P Kernel Tests")
    print("=" * 70)
    print(f"  Per-test timeout: {timeout}s")
    print()

    testable = sorted(k for k, (a, _) in attrs["canaccess"].items() if a)
    if smoke:
        testable = testable[:1]
        print("  [smoke mode: testing first peer-access pair only]")
        print()

    if not testable:
        print("  No pairs report P2P access — skipping kernel tests.")
        print("  (Expected on consumer Intel host bridges; on server-class")
        print("   Intel or AMD hosts this would be unexpected.)")
        return True

    results = []

    def run_test(name, args, tmo=None):
        status, out = run_binary(args, tmo or timeout)
        lines = [l for l in out.strip().split("\n")
                 if l.strip() and re.match(r"^[A-Z]", l)]
        if name == "bandwidth":
            # sized sweep: show every BANDWIDTH row, not just the final PASS
            for l in lines:
                if l.startswith("BANDWIDTH") and "size_mb" in l:
                    print(f"  {l}")
            detail = lines[-1] if lines else "(no output)"
        else:
            detail = lines[-1] if lines else (out.strip().split("\n") or ["(no output)"])[-1]
            print(f"  {detail}")
        results.append((name, args, status, detail))
        if status in ("FAIL", "TIMEOUT"):
            print(f"    *** {status} on {name} {' '.join(args[1:3])} — continuing ***")
        return status

    # 4a: L0 cross-device copy
    print("-" * 50)
    print("  [4a] P2P Data Transfer (L0 cross-device copy)")
    print("-" * 50)
    for s, d in testable:
        run_test("transfer", ["transfer", str(s), str(d), "16"])
    print()

    # 4b: kernel-initiated peer write + read
    print("-" * 50)
    print("  [4b] P2P Data Transfer (kernel-initiated peer write+read)")
    print("-" * 50)
    for s, d in testable:
        run_test("transfer-kernel", ["transfer-kernel", str(s), str(d), "16"])
    print()

    # 4c: P2P atomics, single pair
    print("-" * 50)
    print("  [4c] P2P Atomics (single-pair atomicAdd)")
    print("-" * 50)
    for s, d in testable:
        run_test("atomic", ["atomic", str(s), str(d)])
    print()

    # 4d: bandwidth sweep on first pair (or all pairs in full mode)
    print("-" * 50)
    print("  [4d] P2P Bandwidth (sized sweep, direct vs host-staged)")
    print("-" * 50)
    bw_pairs = testable[:1] if smoke else testable
    for s, d in bw_pairs:
        run_test("bandwidth", ["bandwidth", str(s), str(d)], tmo=timeout * 3)
    print()

    # 4e: full-matrix atomic stress
    if not smoke:
        print("-" * 50)
        print("  [4e] P2P Atomics (stress: all-to-one, rotating target)")
        print("-" * 50)
        run_test("stress", ["stress"], tmo=timeout * 2)
        print()

    # Summary
    print("=" * 70)
    print("  KERNEL TEST SUMMARY")
    print("=" * 70)
    print(f"  {'Test':<20} {'args':<24} {'Status':>10}")
    print(f"  {'-'*20} {'-'*24} {'-'*10}")
    for name, args, status, _ in results:
        argstr = " ".join(args[1:]) if name != "stress" else "all"
        print(f"  {name:<20} {argstr:<24} {status:>10}")

    passed = sum(1 for _, _, st, _ in results if st == "PASS")
    failed = sum(1 for _, _, st, _ in results if st == "FAIL")
    timed_out = sum(1 for _, _, st, _ in results if st == "TIMEOUT")
    print()
    print(f"  Total: {passed} pass, {failed} fail, {timed_out} timeout")
    print()
    return not (failed or timed_out)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XPU P2P test orchestrator")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-test kernel timeout in seconds (default: 60)")
    parser.add_argument("--skip-kernel", action="store_true",
                        help="Only run topology + trust model + attribute stages")
    parser.add_argument("--smoke", action="store_true",
                        help="Run quick subset (first pair only)")
    args = parser.parse_args()

    print()
    print("#" * 70)
    print("#  XPU P2P Test Suite (Intel Arc / Level Zero)")
    print("#" * 70)
    print()

    topo = stage_topology()
    if not topo:
        return 2

    stage_trust_model()

    attrs = stage_attributes(args.timeout)
    if attrs is None:
        print("Attribute enumeration failed — cannot continue.")
        return 2

    if args.skip_kernel:
        print("Skipping kernel tests (--skip-kernel).")
        print()
        return 0

    ok = stage_kernel_tests(attrs, args.timeout, smoke=args.smoke)

    print("=" * 70)
    if ok:
        print("OVERALL: ALL TESTS PASSED")
        print("=" * 70)
        return 0
    print("OVERALL: ONE OR MORE TESTS FAILED")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
