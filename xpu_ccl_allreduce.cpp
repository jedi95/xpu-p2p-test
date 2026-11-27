// xpu_ccl_allreduce.cpp — oneCCL allreduce latency/bandwidth on Intel GPUs
//
// MPI-launched: one rank per GPU (oneCCL does not support multiple devices
// per process). Rank r binds GPU r via ZE_AFFINITY_MASK=r (set at startup,
// before any Level Zero init). KVS wire-up: rank 0 creates the main KVS and
// broadcasts its address over MPI; other ranks create_kvs from the address.
//
// Port of a CUDA/ROCm p2pmark-style allreduce benchmark to oneCCL.
// Size sweep brackets the real inference pattern:
//   decode: ~4 KiB/call (Qwen3.6 H=2048 FP16), 80 calls/step
//   prefill chunk T=8192: 32 MiB/call
//
// Usage (launched via mpiexec -n N):
//   mpiexec -n 4 ./xpu_ccl_allreduce [--sizes-bytes CSV] [--warmup N]
//                                    [--iters N] [--verify] [--serial N]
//
// Timing: per rep, all ranks MPI_Barrier, then each rank issues --serial
// dependent allreduces (issue + wait each), barrier again. Per-call latency
// = wall / serial. One size per ALLREDUCE line. --verify checks the sum.
//
// Exit codes: 0 = PASS, 1 = FAIL, 2 = usage/env error.
//
// Run P2P integrity first (xpu_p2p_test) — oneCCL returning SUCCESS with
// corrupted data is the known failure mode on platforms where the kernel
// peer path lies (compute-runtime issue #935).

#include <mpi.h>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define CHECK_ZE(expr)                                                        \
    do {                                                                      \
        ze_result_t _r = (expr);                                              \
        if (_r != ZE_RESULT_SUCCESS) {                                        \
            fprintf(stderr, "ZE error %d at %s:%d: %s\n", (int)_r,            \
                    __FILE__, __LINE__, #expr);                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

static std::vector<size_t> parse_sizes(const char *csv) {
    std::vector<size_t> out;
    const char *p = csv;
    while (*p) {
        size_t v = strtoul(p, (char **)&p, 10);
        out.push_back(v);
        if (*p == ',') p++;
        else break;
    }
    return out;
}

int main(int argc, char **argv) {
    // Bind this process to GPU <rank> BEFORE any Level Zero/SYCL init.
    setenv("CCL_PROCESS_LAUNCHER", "none", 0);  // don't override if user set
    std::vector<size_t> sizes = {256, 512, 1024, 4096, 8192, 65536,
                                 262144, 1048576, 4194304, 33554432};
    int warmup = 20, iters = 5, serial = 20;
    bool verify = false;

    MPI_Init(&argc, &argv);
    int rank, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    // Rank r binds to GPU r unless the caller already set ZE_AFFINITY_MASK.
    // Setting it externally (e.g. "0" for every rank) puts all ranks on one
    // GPU - the cheapest way to tell an inter-GPU sync bug from an intra-GPU
    // protocol bug when a collective deadlocks.
    char mask[16];
    snprintf(mask, sizeof(mask), "%d", rank);
    setenv("ZE_AFFINITY_MASK", mask, 0);
    setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:*", 0);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sizes-bytes") && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--serial") && i + 1 < argc) {
            serial = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--verify")) {
            verify = true;
        } else {
            if (rank == 0)
                fprintf(stderr, "usage: mpiexec -n N %s [--sizes-bytes CSV] "
                                "[--warmup N] [--iters N] [--serial N] [--verify]\n",
                        argv[0]);
            MPI_Finalize();
            return 2;
        }
    }

    int rc = 0;
    try {
        auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
        std::vector<sycl::device> mine;
        for (const auto &d : all)
            if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
                mine.push_back(d);
        if (mine.size() != 1) {
            fprintf(stderr, "rank %d: expected exactly 1 L0 GPU after "
                            "ZE_AFFINITY_MASK=%s, found %zu\n",
                    rank, mask, mine.size());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        sycl::device dev = mine[0];
        // oneCCL requires an in-order queue (its worker thread submits to
        // the same queue; out-of-order queues race its internal ordering
        // and segfault in the L0 list manager)
        sycl::queue q(dev, sycl::property::queue::in_order());

        // BDF report from L0 (verifies rank->GPU mapping)
        CHECK_ZE(zeInit(ZE_INIT_FLAG_GPU_ONLY));
        ze_device_handle_t zdev =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
        ze_pci_ext_properties_t pci = {};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        CHECK_ZE(zeDevicePciGetPropertiesExt(zdev, &pci));
        printf("RANK rank=%d bdf=%04x:%02x:%02x.%x\n", rank,
               pci.address.domain, pci.address.bus, pci.address.device,
               pci.address.function);
        fflush(stdout);

        // KVS wire-up: rank 0 broadcasts the kvs address
        ccl::kvs::address_type kvs_addr;
        std::shared_ptr<ccl::kvs> kvs;
        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            kvs_addr = kvs->get_address();
        }
        MPI_Bcast(kvs_addr.data(), (int)kvs_addr.size(), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
        if (rank != 0)
            kvs = ccl::create_kvs(kvs_addr);

        ccl::context cctx = ccl::create_context(q.get_context());
        ccl::device cdev = ccl::create_device(dev);
        ccl::communicator comm = ccl::create_communicator(n, rank, cdev,
                                                          cctx, kvs);
        ccl::stream stream = ccl::create_stream(q);
        if (rank == 0) printf("CCLCOMM size=%d\n", comm.size());

        size_t max_bytes = *std::max_element(sizes.begin(), sizes.end());
        size_t max_elems = max_bytes / 2;  // fp16
        sycl::half *sendb = sycl::malloc_device<sycl::half>(max_elems, q);
        sycl::half *recvb = sycl::malloc_device<sycl::half>(max_elems, q);

        for (size_t bytes : sizes) {
            size_t count = bytes / 2;
            if (count == 0) continue;

            // init: send = (rank+1) -> allreduce sum = n(n+1)/2
            q.fill(sendb, sycl::half((float)(rank + 1)), count).wait();
            q.fill(recvb, sycl::half(0.0f), count).wait();

            // warmup
            for (int w = 0; w < warmup; w++) {
                auto e = ccl::allreduce(sendb, recvb, count,
                                        ccl::datatype::float16,
                                        ccl::reduction::sum, comm, stream);
                e.wait();
            }

            // timed reps: barrier, --serial dependent allreduces, barrier
            std::vector<double> reps_us;
            for (int r = 0; r < iters; r++) {
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::steady_clock::now();
                for (int s = 0; s < serial; s++) {
                    auto e = ccl::allreduce(sendb, recvb, count,
                                            ccl::datatype::float16,
                                            ccl::reduction::sum, comm, stream);
                    e.wait();
                }
                auto t1 = std::chrono::steady_clock::now();
                MPI_Barrier(MPI_COMM_WORLD);
                // per-call latency: slowest rank's elapsed / serial
                double us = std::chrono::duration<double, std::micro>(t1 - t0)
                                .count() / serial;
                double mx;
                MPI_Reduce(&us, &mx, 1, MPI_DOUBLE, MPI_MAX, 0,
                           MPI_COMM_WORLD);
                if (rank == 0) reps_us.push_back(mx);
            }
            if (rank == 0) {
                std::sort(reps_us.begin(), reps_us.end());
                double min_us = reps_us.front();
                double p50_us = reps_us[reps_us.size() / 2];
                double avg_us = std::accumulate(reps_us.begin(), reps_us.end(),
                                                0.0) / reps_us.size();
                // busbw = S * 2(N-1)/N / t
                double busbw = (double)bytes * 2.0 * (n - 1) / n
                               / (min_us * 1e-6) / 1e9;
                printf("ALLREDUCE n=%d bytes=%zu min_us=%.2f p50_us=%.2f "
                       "avg_us=%.2f busbw_gbps=%.2f\n",
                       n, bytes, min_us, p50_us, avg_us, busbw);
                fflush(stdout);
            }

            if (verify && rank == 0) {
                // every rank's recv must equal n(n+1)/2; print actual value
                // on failure so dropped/duplicated rank contributions are
                // diagnosable (e.g. one rank dropped: n(n+1)/2 - rank - 1)
                sycl::half expect((float)(n * (n + 1) / 2));
                size_t bad = 0;
                sycl::half got = 0;
                long bad_off = -1;
                sycl::half bad_val = 0;
                std::vector<sycl::half> host(count);
                q.memcpy(host.data(), recvb,
                         host.size() * sizeof(sycl::half)).wait();
                got = host[0];
                for (size_t k = 0; k < host.size(); k++) {
                    if (host[k] != expect) {
                        bad++;
                        if (bad_off < 0) { bad_off = (long)k; bad_val = host[k]; }
                    }
                }
                printf("ALLREDUCEVERIFY n=%d bytes=%zu status=%s got=%g want=%g "
                       "bad=%zu first_off=%ld first_val=%g\n",
                       n, bytes, bad ? "FAIL" : "PASS", (float)got,
                       (float)expect, bad, bad_off, (float)bad_val);
                fflush(stdout);
            }
            // every rank must agree on rc
            int bad_local = 0;
            if (verify) {
                sycl::half expect((float)(n * (n + 1) / 2));
                std::vector<sycl::half> host(1);
                q.memcpy(host.data(), recvb, sizeof(sycl::half)).wait();
                bad_local = (host[0] != expect) ? 1 : 0;
            }
            int bad_all = 0;
            MPI_Allreduce(&bad_local, &bad_all, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD);
            if (bad_all) rc = 1;
        }

        sycl::free(sendb, q);
        sycl::free(recvb, q);
    } catch (const sycl::exception &e) {
        fprintf(stderr, "rank %d: SYCL exception: %s\n", rank, e.what());
        rc = 1;
    } catch (const std::exception &e) {
        fprintf(stderr, "rank %d: exception: %s\n", rank, e.what());
        rc = 1;
    }

    int grc = 0;
    MPI_Allreduce(&rc, &grc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return grc;
}
