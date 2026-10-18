/* Smoke test for libvoxtype_xrt: open the NPU and print what XRT sees. */
#include "vxrt.h"
#include <stdio.h>

int main(void) {
  int ma, mi, pa;
  if (vxrt_xrt_version(&ma, &mi, &pa) == VXRT_OK)
    printf("xrt %d.%d.%d, shim abi %d\n", ma, mi, pa, vxrt_abi_version());
  vxrt_device *dev;
  if (vxrt_device_open(0, &dev) != VXRT_OK) {
    fprintf(stderr, "open: %s\n", vxrt_last_error());
    return 1;
  }
  char name[128];
  if (vxrt_device_name(dev, name, sizeof name) == VXRT_OK)
    printf("device 0: %s\n", name);
  vxrt_device_free(dev);
  return 0;
}
