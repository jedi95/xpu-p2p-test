// xpu_p2p_test.cpp — Intel XPU (Level Zero + SYCL) P2P test binary
//
// Intel counterpart to p2p_kernel_test.cu (CUDA) and p2p_kernel_test.hip (HIP).
// Subcommands:
//   canaccess                — zeDeviceCanAccessPeer matrix + SYCL is_peer
//   transfer SRC DST [MB]    — L0 direct device->device copy + checksum + GB/s
//   transfer-kernel SRC DST [MB] — SYCL kernel-initiated peer write then peer read
//   atomic SRC DST [ITERS]   — kernel on SRC atomicAdd's a counter on DST
//   atomic-par DEV [ITERS]   — every GPU atomicAdds into one counter concurrently
//   bandwidth SRC DST        — sized sweep: direct (L0) vs host-staged control
//   stress                   — all-to-one atomic stress across the full matrix
//
// Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP.
//
// The L0 command-list path is the authoritative probe: a cross-device
// zeCommandListAppendMemoryCopy either executes as true peer DMA or fails —
// it never silently stages through host RAM, unlike higher-level APIs.

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_ZE(expr)                                                        \
    do {                                                                      \
        ze_result_t _r = (expr);                                              \
        if (_r != ZE_RESULT_SUCCESS) {                                        \
            fprintf(stderr, "ZE error %d (0x%x) at %s:%d: %s\n", (int)_r,     \
                    (int)_r, __FILE__, __LINE__, #expr);                      \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static const char *status_str(int rc) {
    if (rc == 0) return "PASS";
    if (rc == 2) return "SKIP";
    return "FAIL";
}

// ---------------------------------------------------------------------------
// Level Zero context: driver, context, devices, command lists/queues
// ---------------------------------------------------------------------------

struct L0Env {
    ze_driver_handle_t driver = nullptr;
    ze_context_handle_t ctx = nullptr;
    std::vector<ze_device_handle_t> devs;
    std::vector<uint32_t> q_ord;      // compute queue-group ordinal per device
    std::vector<uint32_t> mem_ord;    // memory ordinal per device
    std::vector<ze_command_list_handle_t> lists;
    std::vector<ze_command_queue_handle_t> queues;
};

static L0Env g;

static void init_l0() {
    CHECK_ZE(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t nd = 0;
    CHECK_ZE(zeDriverGet(&nd, nullptr));
    if (nd == 0) {
        fprintf(stderr, "no L0 driver found\n");
        exit(2);
    }
    std::vector<ze_driver_handle_t> drivers(nd);
    CHECK_ZE(zeDriverGet(&nd, drivers.data()));
    g.driver = drivers[0];

    uint32_t ndev = 0;
    CHECK_ZE(zeDeviceGet(g.driver, &ndev, nullptr));
    if (ndev < 2) {
        fprintf(stderr, "need >= 2 GPUs, found %u\n", ndev);
        exit(2);
    }
    g.devs.resize(ndev);
    CHECK_ZE(zeDeviceGet(g.driver, &ndev, g.devs.data()));

    ze_context_desc_t cdesc = {};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    CHECK_ZE(zeContextCreate(g.driver, &cdesc, &g.ctx));

    g.q_ord.resize(ndev);
    g.mem_ord.resize(ndev);
    g.lists.resize(ndev);
    g.queues.resize(ndev);

    for (uint32_t d = 0; d < ndev; d++) {
        // compute queue group ordinal
        uint32_t nq = 0;
        CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(g.devs[d], &nq, nullptr));
        std::vector<ze_command_queue_group_properties_t> qprops(nq);
        for (auto &p : qprops) p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(g.devs[d], &nq, qprops.data()));
        uint32_t qo = 0;
        for (uint32_t i = 0; i < nq; i++) {
            if (qprops[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                qo = i;
                break;
            }
        }
        g.q_ord[d] = qo;

        // memory ordinal (first device-memory property set)
        uint32_t nm = 0;
        CHECK_ZE(zeDeviceGetMemoryProperties(g.devs[d], &nm, nullptr));
        g.mem_ord[d] = 0;

        ze_command_queue_desc_t qdesc = {};
        qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        qdesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        qdesc.ordinal = qo;
        qdesc.index = 0;
        CHECK_ZE(zeCommandQueueCreate(g.ctx, g.devs[d], &qdesc, &g.queues[d]));

        ze_command_list_desc_t ldesc = {};
        ldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        ldesc.commandQueueGroupOrdinal = qo;
        CHECK_ZE(zeCommandListCreate(g.ctx, g.devs[d], &ldesc, &g.lists[d]));
    }
}

static void *alloc_dev(int dev, size_t bytes) {
    ze_device_mem_alloc_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    desc.ordinal = g.mem_ord[dev];
    void *p = nullptr;
    CHECK_ZE(zeMemAllocDevice(g.ctx, &desc, bytes, 0, g.devs[dev], &p));
    return p;
}

static void *alloc_host(size_t bytes) {
    ze_host_mem_alloc_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void *p = nullptr;
    CHECK_ZE(zeMemAllocHost(g.ctx, &desc, bytes, 0, &p));
    return p;
}

static void l0_copy(int dev, void *dst, const void *src, size_t bytes) {
    ze_command_list_handle_t l = g.lists[dev];
    CHECK_ZE(zeCommandListAppendMemoryCopy(l, dst, src, bytes, nullptr, 0, nullptr));
    CHECK_ZE(zeCommandListClose(l));
    CHECK_ZE(zeCommandQueueExecuteCommandLists(g.queues[dev], 1, &l, nullptr));
    CHECK_ZE(zeCommandQueueSynchronize(g.queues[dev], UINT64_MAX));
    CHECK_ZE(zeCommandListReset(l));
}

static void free_dev(void *p) { CHECK_ZE(zeMemFree(g.ctx, p)); }

// per-pair deterministic byte pattern
static uint8_t pat_byte(size_t i, int s, int d) {
    return (uint8_t)(i * 31 + (size_t)s * 7 + (size_t)d * 13);
}

static std::vector<uint8_t> make_pattern(size_t n, int s, int d) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = pat_byte(i, s, d);
    return v;
}

static long verify(const uint8_t *got, const std::vector<uint8_t> &want) {
    long mism = 0;
    long first = -1;
    for (size_t i = 0; i < want.size(); i++) {
        if (got[i] != want[i]) {
            mism++;
            if (first < 0) first = (long)i;
        }
    }
    if (mism > 0)
        fprintf(stderr, "  verify: %ld/%zu mismatched, first at offset %ld "
                        "(got 0x%02x want 0x%02x)\n",
                mism, want.size(), first, got[first], want[first]);
    return mism;
}

static double gbps(size_t bytes, double seconds) {
    return (seconds > 0) ? (double)bytes / seconds / 1e9 : 0.0;
}

// SYCL enumerates each GPU once per backend (level_zero and opencl). Only the
// level_zero entries support cross-device P2P —
// kernels/allocations on the opencl aliases of the same hardware page-fault
// when touching another GPU's memory. Filter to level_zero everywhere.
static std::vector<sycl::device> l0_gpus() {
    std::vector<sycl::device> out;
    for (const auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            out.push_back(d);
    return out;
}

// ---------------------------------------------------------------------------
// canaccess
// ---------------------------------------------------------------------------

static int cmd_canaccess() {
    init_l0();
    uint32_t n = (uint32_t)g.devs.size();

    for (uint32_t d = 0; d < n; d++) {
        ze_device_properties_t props = {};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        CHECK_ZE(zeDeviceGetProperties(g.devs[d], &props));
        ze_pci_ext_properties_t pci = {};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        CHECK_ZE(zeDevicePciGetPropertiesExt(g.devs[d], &pci));
        printf("DEVICE idx=%u bdf=%04x:%02x:%02x.%x name=%s\n", d,
               pci.address.domain, pci.address.bus, pci.address.device,
               pci.address.function, props.name);
    }

    for (uint32_t s = 0; s < n; s++) {
        for (uint32_t d = 0; d < n; d++) {
            if (s == d) continue;
            ze_bool_t can = 0;
            ze_result_t r = zeDeviceCanAccessPeer(g.devs[s], g.devs[d], &can);
            int access = (r == ZE_RESULT_SUCCESS && can) ? 1 : 0;
            printf("CANACCESS src=%u dst=%u access=%d l0rc=%d\n", s, d, access,
                   (int)r);
        }
    }

    // DPC++ does not expose device::is_peer here, so zeDeviceCanAccessPeer is
    // authoritative.
    printf("SYCLDEVICES count=%zu\n",
           sycl::device::get_devices(sycl::info::device_type::gpu).size());
    return 0;
}

// ---------------------------------------------------------------------------
// transfer — L0 direct cross-device copy + checksum
// ---------------------------------------------------------------------------

static int cmd_transfer(int s, int d, int mb) {
    init_l0();
    size_t n = (size_t)mb << 20;

    void *src = alloc_dev(s, n);
    void *dst = alloc_dev(d, n);
    uint8_t *hpat = (uint8_t *)alloc_host(n);
    uint8_t *hsrc = (uint8_t *)alloc_host(n);
    uint8_t *hdst = (uint8_t *)alloc_host(n);
    for (size_t i = 0; i < n; i++) hpat[i] = pat_byte(i, s, d);

    l0_copy(s, src, hpat, n);   // host -> src device
    memset(hdst, 0xEE, n);      // poison dst so failure modes are visible

    // timed P2P copy, executed on the SRC device (initiator)
    double best = 1e30;
    ze_result_t copy_rc = ZE_RESULT_SUCCESS;
    for (int it = 0; it < 3; it++) {
        auto t0 = std::chrono::steady_clock::now();
        ze_command_list_handle_t l = g.lists[s];
        copy_rc = zeCommandListAppendMemoryCopy(l, dst, src, n, nullptr, 0, nullptr);
        if (copy_rc != ZE_RESULT_SUCCESS) break;
        CHECK_ZE(zeCommandListClose(l));
        CHECK_ZE(zeCommandQueueExecuteCommandLists(g.queues[s], 1, &l, nullptr));
        CHECK_ZE(zeCommandQueueSynchronize(g.queues[s], UINT64_MAX));
        CHECK_ZE(zeCommandListReset(l));
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, sec);
    }

    int rc = 0;
    long mism = -1;
    if (copy_rc != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "  L0 cross-device copy returned %d (0x%x)\n",
                (int)copy_rc, (int)copy_rc);
        rc = 1;
    } else {
        l0_copy(d, hdst, dst, n);  // dst -> host
        l0_copy(s, hsrc, src, n);  // src -> host
        mism = verify(hdst, std::vector<uint8_t>(hpat, hpat + n));
        if (mism != 0) rc = 1;
    }

    printf("TRANSFER src=%d dst=%d size=%zu status=%s gbps=%.2f mismatch=%ld\n",
           s, d, n, status_str(rc), gbps(n, best), mism);
    free_dev(src); free_dev(dst);
    free_dev(hpat); free_dev(hsrc); free_dev(hdst);
    return rc;
}

// ---------------------------------------------------------------------------
// transfer-kernel — SYCL kernel-initiated peer write then peer read
// ---------------------------------------------------------------------------

static int cmd_transfer_kernel(int s, int d, int mb) {
    init_l0();
    size_t n = (size_t)mb << 20;
    int rc = 0;

    try {
        auto sdevs = l0_gpus();
        sycl::queue qs(sdevs[s]);
        sycl::queue qd(sdevs[d]);

        uint8_t *src = sycl::malloc_device<uint8_t>(n, qs);
        uint8_t *dstw = sycl::malloc_device<uint8_t>(n, qd);  // phase 1 target
        uint8_t *dstr = sycl::malloc_device<uint8_t>(n, qd);  // phase 2 target

        // fill src with the pair pattern
        qs.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            src[i] = (uint8_t)(i * 31 + (size_t)s * 7 + (size_t)d * 13);
        }).wait();
        qd.parallel_for(sycl::range<1>(n),
                        [=](sycl::id<1> i) { dstw[i] = 0xEE; }).wait();

        // JIT + first-touch warmup: run BOTH phase kernels at full size,
        // untimed. The first peer access to a page establishes the GPU-VA peer
        // mapping, which is a slow driver path; measured cold it inflates phase
        // 1 about 30x while phase 2 (already mapped) looks fast, which reads as
        // a write-vs-read asymmetry that is not there.
        qs.parallel_for(sycl::range<1>(n),
                        [=](sycl::id<1> i) { dstw[i] = src[i]; }).wait();
        qd.parallel_for(sycl::range<1>(n),
                        [=](sycl::id<1> i) { dstr[i] = src[i]; }).wait();
        qd.wait();

        // Phase 1: kernel on SRC writes directly into DST memory (peer write).
        // 3 timed reps, best taken — steady-state after the full-size warmup.
        double t_write = 1e30, t_read = 1e30;
        for (int it = 0; it < 3; it++) {
            auto t0 = std::chrono::steady_clock::now();
            qs.parallel_for(sycl::range<1>(n),
                            [=](sycl::id<1> i) { dstw[i] = src[i]; }).wait();
            auto t1 = std::chrono::steady_clock::now();
            t_write = std::min(t_write,
                               std::chrono::duration<double>(t1 - t0).count());
        }

        // Phase 2: kernel on DST reads directly from SRC memory (peer read)
        for (int it = 0; it < 3; it++) {
            auto t2 = std::chrono::steady_clock::now();
            qd.parallel_for(sycl::range<1>(n),
                            [=](sycl::id<1> i) { dstr[i] = src[i]; }).wait();
            auto t3 = std::chrono::steady_clock::now();
            t_read = std::min(t_read,
                              std::chrono::duration<double>(t3 - t2).count());
        }

        // verify both phases against host-computed pattern
        std::vector<uint8_t> want = make_pattern(n, s, d);
        std::vector<uint8_t> gotw(n), gotr(n);
        qd.memcpy(gotw.data(), dstw, n).wait();
        qd.memcpy(gotr.data(), dstr, n).wait();
        long mw = verify(gotw.data(), want);
        long mr = verify(gotr.data(), want);
        if (mw != 0 || mr != 0) rc = 1;

        printf("KERNELTRANSFER src=%d dst=%d size=%zu status=%s "
               "write_gbps=%.2f read_gbps=%.2f mismatch_w=%ld mismatch_r=%ld\n",
               s, d, n, status_str(rc), gbps(n, t_write), gbps(n, t_read), mw, mr);

        sycl::free(src, qs);
        sycl::free(dstw, qd);
        sycl::free(dstr, qd);
    } catch (const sycl::exception &e) {
        fprintf(stderr, "  SYCL exception: %s\n", e.what());
        printf("KERNELTRANSFER src=%d dst=%d size=%zu status=FAIL err=sycl_exception\n",
               s, d, n);
        return 1;
    }
    return rc;
}

// ---------------------------------------------------------------------------
// atomic — kernel on SRC atomicAdd's a counter on DST
// ---------------------------------------------------------------------------

static int cmd_atomic(int s, int d, uint32_t iters) {
    init_l0();
    int rc = 0;
    try {
        auto sdevs = l0_gpus();
        sycl::queue qs(sdevs[s]);
        sycl::queue qd(sdevs[d]);

        unsigned long long *counter = sycl::malloc_device<unsigned long long>(1, qd);
        unsigned long long zero = 0;
        qd.memcpy(counter, &zero, sizeof(zero)).wait();

        qs.parallel_for(sycl::range<1>(iters), [=](sycl::id<1> i) {
            (void)i;
            sycl::atomic_ref<unsigned long long,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::system>
                a(counter[0]);
            a.fetch_add(1);
        }).wait();

        unsigned long long got = 0;
        qd.memcpy(&got, counter, sizeof(got)).wait();
        rc = (got == iters) ? 0 : 1;
        printf("ATOMIC src=%d dst=%d status=%s count=%llu expected=%u lost=%llu\n",
               s, d, status_str(rc), got, iters,
               (unsigned long long)(iters - got));
        sycl::free(counter, qd);
    } catch (const sycl::exception &e) {
        fprintf(stderr, "  SYCL exception: %s\n", e.what());
        printf("ATOMIC src=%d dst=%d status=FAIL err=sycl_exception\n", s, d);
        return 1;
    }
    return rc;
}

// ---------------------------------------------------------------------------
// atomic-par — every GPU atomicAdds into ONE counter at the same time.
// cmd_atomic()/cmd_stress() launch one producer at a time (and .wait() between
// launches), so a single-producer run cannot distinguish a genuinely atomic
// cross-device path from an atomic that the requester executes in its own
// cache and writes back: both give an exact count. Here the target GPU and the
// peer(s) increment the same counter concurrently, so any lost update proves
// the peer path is not atomic.
// ---------------------------------------------------------------------------

static int cmd_atomic_par(int d, uint32_t iters) {
    init_l0();
    int rc = 0;
    try {
        auto sdevs = l0_gpus();
        const int n = (int)sdevs.size();
        sycl::queue qd(sdevs[d]);
        unsigned long long *counter = sycl::malloc_device<unsigned long long>(1, qd);
        unsigned long long zero = 0;
        qd.memcpy(counter, &zero, sizeof(zero)).wait();

        std::vector<sycl::event> evs;
        for (int s = 0; s < n; s++) {
            sycl::queue qs(sdevs[s]);
            evs.push_back(qs.parallel_for(sycl::range<1>(iters), [=](sycl::id<1>) {
                sycl::atomic_ref<unsigned long long,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::system>
                    a(counter[0]);
                a.fetch_add(1);
            }));
        }
        for (auto &e : evs) e.wait();

        unsigned long long got = 0;
        qd.memcpy(&got, counter, sizeof(got)).wait();
        unsigned long long want = (unsigned long long)n * iters;
        rc = (got == want) ? 0 : 1;
        printf("ATOMICPAR target=%d producers=%d status=%s count=%llu expected=%llu lost=%llu\n",
               d, n, status_str(rc), got, want, want - got);
        sycl::free(counter, qd);
    } catch (const sycl::exception &e) {
        fprintf(stderr, "  SYCL exception: %s\n", e.what());
        printf("ATOMICPAR target=%d status=FAIL err=sycl_exception\n", d);
        return 1;
    }
    return rc;
}

// ---------------------------------------------------------------------------
// bandwidth — sized sweep, direct (L0) vs host-staged control
// ---------------------------------------------------------------------------

static int cmd_bandwidth(int s, int d, int iters) {
    init_l0();
    const int sizes_mb[] = {1, 4, 16, 64, 256};

    for (int mb : sizes_mb) {
        size_t n = (size_t)mb << 20;
        void *src = alloc_dev(s, n);
        void *dst = alloc_dev(d, n);
        uint8_t *hbuf = (uint8_t *)alloc_host(n);

        // direct: cross-device copy on SRC's queue, best of N
        double best = 1e30;
        for (int it = 0; it < iters; it++) {
            auto t0 = std::chrono::steady_clock::now();
            ze_command_list_handle_t l = g.lists[s];
            CHECK_ZE(zeCommandListAppendMemoryCopy(l, dst, src, n, nullptr, 0, nullptr));
            CHECK_ZE(zeCommandListClose(l));
            CHECK_ZE(zeCommandQueueExecuteCommandLists(g.queues[s], 1, &l, nullptr));
            CHECK_ZE(zeCommandQueueSynchronize(g.queues[s], UINT64_MAX));
            CHECK_ZE(zeCommandListReset(l));
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }

        // host-staged control: d2h (on s) then h2d (on d), timed separately
        double t_d2h = 1e30, t_h2d = 1e30;
        for (int it = 0; it < iters; it++) {
            auto t0 = std::chrono::steady_clock::now();
            l0_copy(s, hbuf, src, n);
            auto t1 = std::chrono::steady_clock::now();
            l0_copy(d, dst, hbuf, n);
            auto t2 = std::chrono::steady_clock::now();
            t_d2h = std::min(t_d2h, std::chrono::duration<double>(t1 - t0).count());
            t_h2d = std::min(t_h2d, std::chrono::duration<double>(t2 - t1).count());
        }
        double t_staged = t_d2h + t_h2d;

        printf("BANDWIDTH src=%d dst=%d size_mb=%d direct_gbps=%.2f "
               "staged_gbps=%.2f d2h_gbps=%.2f h2d_gbps=%.2f\n",
               s, d, mb, gbps(n, best), gbps(n, t_staged),
               gbps(n, t_d2h), gbps(n, t_h2d));

        free_dev(src); free_dev(dst); free_dev(hbuf);
    }
    printf("BANDWIDTH src=%d dst=%d status=PASS\n", s, d);
    return 0;
}

// ---------------------------------------------------------------------------
// stress — all-to-one atomic stress across the full matrix
// ---------------------------------------------------------------------------

static int cmd_stress(uint32_t per_src) {
    init_l0();
    auto sdevs = l0_gpus();
    uint32_t n = (uint32_t)sdevs.size();
    int overall = 0;

    try {
        // Allocate every counter up front: an allocation made after peer-access
        // cycling may not be remapped into the peer address space.
        std::vector<unsigned long long *> counters(n, nullptr);
        std::vector<sycl::queue> qs;
        for (uint32_t t = 0; t < n; t++) {
            qs.emplace_back(sdevs[t]);
            counters[t] = sycl::malloc_device<unsigned long long>(1, qs[t]);
            unsigned long long zero = 0;
            qs[t].memcpy(counters[t], &zero, sizeof(zero)).wait();
        }

        // One launch per (target, source) pair: the kernel only captures the
        // single target counter pointer (raw pointer = device copyable), so
        // no host-side container ever crosses into device code.
        for (uint32_t t = 0; t < n; t++) {
            unsigned long long *target = counters[t];
            for (uint32_t s = 0; s < n; s++) {
                if (s == t) continue;
                qs[s].parallel_for(sycl::range<1>(per_src),
                                   [=](sycl::id<1>) {
                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::system>
                        a(target[0]);
                    a.fetch_add(1);
                }).wait();
            }
            unsigned long long got = 0;
            qs[t].memcpy(&got, counters[t], sizeof(got)).wait();
            unsigned long long want = (unsigned long long)(n - 1) * per_src;
            int rc = (got == want) ? 0 : 1;
            if (rc) overall = 1;
            printf("STRESS target=%u status=%s count=%llu expected=%llu lost=%llu\n",
                   t, status_str(rc), got, want, want - got);
        }

        for (uint32_t t = 0; t < n; t++) sycl::free(counters[t], qs[t]);
    } catch (const sycl::exception &e) {
        fprintf(stderr, "  SYCL exception: %s\n", e.what());
        printf("STRESS status=FAIL err=sycl_exception\n");
        return 1;
    }
    return overall;
}

// ---------------------------------------------------------------------------

static void usage() {
    fprintf(stderr,
            "usage: xpu_p2p_test <cmd> [args]\n"
            "  canaccess\n"
            "  transfer <src> <dst> [MB=16]\n"
            "  transfer-kernel <src> <dst> [MB=16]\n"
            "  atomic <src> <dst> [iters=1048576]\n"
            "  bandwidth <src> <dst> [iters=5]\n"
            "  stress [per_src=1048576]\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    std::string cmd = argv[1];

    if (cmd == "canaccess") return cmd_canaccess();
    if (cmd == "transfer") {
        if (argc < 4) { usage(); return 2; }
        int mb = (argc > 4) ? atoi(argv[4]) : 16;
        return cmd_transfer(atoi(argv[2]), atoi(argv[3]), mb);
    }
    if (cmd == "transfer-kernel") {
        if (argc < 4) { usage(); return 2; }
        int mb = (argc > 4) ? atoi(argv[4]) : 16;
        return cmd_transfer_kernel(atoi(argv[2]), atoi(argv[3]), mb);
    }
    if (cmd == "atomic") {
        if (argc < 4) { usage(); return 2; }
        uint32_t iters = (argc > 4) ? (uint32_t)strtoul(argv[4], nullptr, 0) : 1048576;
        return cmd_atomic(atoi(argv[2]), atoi(argv[3]), iters);
    }
    if (cmd == "atomic-par") {
        if (argc < 3) { usage(); return 2; }
        uint32_t iters = (argc > 3) ? (uint32_t)strtoul(argv[3], nullptr, 0) : 1048576;
        return cmd_atomic_par(atoi(argv[2]), iters);
    }
    if (cmd == "bandwidth") {
        if (argc < 4) { usage(); return 2; }
        int iters = (argc > 4) ? atoi(argv[4]) : 5;
        return cmd_bandwidth(atoi(argv[2]), atoi(argv[3]), iters);
    }
    if (cmd == "stress") {
        uint32_t per = (argc > 2) ? (uint32_t)strtoul(argv[2], nullptr, 0) : 1048576;
        return cmd_stress(per);
    }
    usage();
    return 2;
}
