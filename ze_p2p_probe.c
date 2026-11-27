// ze_p2p_probe.c - print exactly what oneCCL's topo manager reads:
//   zeDeviceGetP2PProperties(...).flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS
// plus zeDeviceCanAccessPeer, for every ordered pair and for the self pair.
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (r != ZE_RESULT_SUCCESS) { printf("zeInit failed %d\n", r); return 1; }

    uint32_t ndrv = 0;
    zeDriverGet(&ndrv, NULL);
    ze_driver_handle_t drv[8];
    if (ndrv > 8) ndrv = 8;
    zeDriverGet(&ndrv, drv);

    uint32_t ndev = 0;
    zeDeviceGet(drv[0], &ndev, NULL);
    ze_device_handle_t dev[16];
    if (ndev > 16) ndev = 16;
    zeDeviceGet(drv[0], &ndev, dev);
    printf("devices=%u\n", ndev);

    for (uint32_t i = 0; i < ndev; i++) {
        ze_device_properties_t p = {0};
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeDeviceGetProperties(dev[i], &p);
        printf("dev[%u] name=%s uuid=%08x vendor=%04x device=%04x\n", i, p.name,
               p.uuid.id[0], p.vendorId, p.deviceId);
    }

    // self pairs first: explains what a same-GPU (single_card) comm reports
    for (uint32_t i = 0; i < ndev; i++) {
        ze_device_p2p_properties_t pp = {0};
        pp.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
        ze_result_t rc = zeDeviceGetP2PProperties(dev[i], dev[i], &pp);
        ze_bool_t can = 0;
        zeDeviceCanAccessPeer(dev[i], dev[i], &can);
        printf("PAIR %u->%u(self) rc=%d flags=0x%x atomics=%d canaccess=%d\n", i, i,
               rc, pp.flags,
               (pp.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS) ? 1 : 0, (int)can);
    }

    for (uint32_t i = 0; i < ndev; i++) {
        for (uint32_t j = 0; j < ndev; j++) {
            if (i == j) continue;
            ze_device_p2p_properties_t pp = {0};
            pp.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
            ze_result_t rc = zeDeviceGetP2PProperties(dev[i], dev[j], &pp);
            ze_bool_t can = 0;
            zeDeviceCanAccessPeer(dev[i], dev[j], &can);
            printf("PAIR %u->%u rc=%d flags=0x%x atomics=%d canaccess=%d\n", i, j, rc,
                   pp.flags,
                   (pp.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS) ? 1 : 0, (int)can);
        }
    }
    return 0;
}
