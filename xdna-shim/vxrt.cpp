// libvoxtype_xrt: see vxrt.h.

#include "vxrt.h"

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_version.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

struct vxrt_device {
  xrt::device dev;
};
struct vxrt_xclbin {
  xrt::xclbin xb;
  xrt::uuid uuid;
  std::string kernel_name;
};
struct vxrt_hwctx {
  xrt::hw_context ctx;
};
struct vxrt_kernel {
  xrt::kernel k;
};
struct vxrt_bo {
  xrt::bo bo;
};
struct vxrt_run {
  xrt::run run;
};
struct vxrt_runlist {
  xrt::runlist rl;
};

namespace {

thread_local std::string last_error;

int fail(int code, const std::string &msg) {
  last_error = msg;
  return code;
}

int fail_exc(const char *what, const std::exception &e) {
  return fail(VXRT_ERR_XRT, std::string(what) + ": " + e.what());
}

int copy_str(const std::string &s, char *buf, size_t len) {
  if (!buf || len == 0)
    return fail(VXRT_ERR_INVALID, "null or empty output buffer");
  std::strncpy(buf, s.c_str(), len - 1);
  buf[len - 1] = '\0';
  return VXRT_OK;
}

int bind_npu_args(xrt::run &run, vxrt_bo *insts, size_t n_insts,
                  vxrt_bo *const *bufs, int n_bufs) {
  // Same argument types as MLIR-AIE's own host code (xrt_test_wrapper.h).
  unsigned int opcode = 3;
  run.set_arg(0, opcode);
  run.set_arg(1, insts->bo);
  run.set_arg(2, n_insts);
  for (int i = 0; i < n_bufs; ++i)
    run.set_arg(3 + i, bufs[i]->bo);
  return VXRT_OK;
}

int check_state(ert_cmd_state st, const char *what) {
  if (st == ERT_CMD_STATE_COMPLETED)
    return VXRT_OK;
  if (st == ERT_CMD_STATE_NEW || st == ERT_CMD_STATE_QUEUED ||
      st == ERT_CMD_STATE_RUNNING || st == ERT_CMD_STATE_SUBMITTED)
    return fail(VXRT_ERR_TIMEOUT, std::string(what) + ": timed out");
  return fail(VXRT_ERR_STATE,
              std::string(what) + ": run ended in state " + std::to_string(st));
}

} // namespace

extern "C" {

int vxrt_abi_version(void) { return VXRT_ABI_VERSION; }

const char *vxrt_last_error(void) { return last_error.c_str(); }

int vxrt_xrt_version(int *major, int *minor, int *patch) {
  if (!major || !minor || !patch)
    return fail(VXRT_ERR_INVALID, "null version output");
  try {
    *major = static_cast<int>(xrt::version::major());
    *minor = static_cast<int>(xrt::version::minor());
    *patch = static_cast<int>(xrt::version::patch());
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("xrt version", e);
  }
}

int vxrt_device_open(unsigned int index, vxrt_device **out) {
  if (!out)
    return fail(VXRT_ERR_INVALID, "null output");
  try {
    *out = new vxrt_device{xrt::device(index)};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("open device", e);
  }
}

void vxrt_device_free(vxrt_device *dev) { delete dev; }

int vxrt_device_name(vxrt_device *dev, char *buf, size_t len) {
  if (!dev)
    return fail(VXRT_ERR_INVALID, "null device");
  try {
    return copy_str(dev->dev.get_info<xrt::info::device::name>(), buf, len);
  } catch (const std::exception &e) {
    return fail_exc("device name", e);
  }
}

int vxrt_xclbin_load(vxrt_device *dev, const char *path, vxrt_xclbin **out) {
  if (!dev || !path || !out)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    xrt::xclbin xb(std::string{path});
    auto kernels = xb.get_kernels();
    if (kernels.empty())
      return fail(VXRT_ERR_INVALID, std::string("no kernel in ") + path);
    std::string name = kernels.front().get_name();
    dev->dev.register_xclbin(xb);
    *out = new vxrt_xclbin{xb, xb.get_uuid(), name};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc(path, e);
  }
}

void vxrt_xclbin_free(vxrt_xclbin *xb) { delete xb; }

int vxrt_xclbin_kernel_name(vxrt_xclbin *xb, char *buf, size_t len) {
  if (!xb)
    return fail(VXRT_ERR_INVALID, "null xclbin");
  return copy_str(xb->kernel_name, buf, len);
}

int vxrt_hwctx_create(vxrt_device *dev, vxrt_xclbin *xb, vxrt_hwctx **out) {
  if (!dev || !xb || !out)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    *out = new vxrt_hwctx{xrt::hw_context(dev->dev, xb->uuid)};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("create hw_context", e);
  }
}

void vxrt_hwctx_free(vxrt_hwctx *ctx) { delete ctx; }

int vxrt_kernel_open(vxrt_hwctx *ctx, const char *name, vxrt_kernel **out) {
  if (!ctx || !name || !out)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    *out = new vxrt_kernel{xrt::kernel(ctx->ctx, std::string{name})};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("open kernel", e);
  }
}

void vxrt_kernel_free(vxrt_kernel *k) { delete k; }

int vxrt_kernel_group_id(vxrt_kernel *k, int argno, int *out) {
  if (!k || !out)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    *out = k->k.group_id(argno);
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("kernel group_id", e);
  }
}

int vxrt_bo_alloc(vxrt_device *dev, size_t size, int kind, int group_id,
                  vxrt_bo **out) {
  if (!dev || !out || size == 0)
    return fail(VXRT_ERR_INVALID, "null argument or zero size");
  xrt::bo::flags flags;
  switch (kind) {
  case VXRT_BO_HOST_ONLY:
    flags = xrt::bo::flags::host_only;
    break;
  case VXRT_BO_CACHEABLE:
    flags = xrt::bo::flags::cacheable;
    break;
  default:
    return fail(VXRT_ERR_INVALID, "unknown bo kind");
  }
  try {
    *out = new vxrt_bo{
        xrt::bo(dev->dev, size, flags, static_cast<xrt::memory_group>(group_id))};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc(("allocate " + std::to_string(size) + " byte bo").c_str(), e);
  }
}

int vxrt_bo_sub(vxrt_bo *parent, size_t size, size_t offset, vxrt_bo **out) {
  if (!parent || !out)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    *out = new vxrt_bo{xrt::bo(parent->bo, size, offset)};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("sub-bo", e);
  }
}

void vxrt_bo_free(vxrt_bo *bo) { delete bo; }

void *vxrt_bo_map(vxrt_bo *bo) {
  if (!bo) {
    fail(VXRT_ERR_INVALID, "null bo");
    return nullptr;
  }
  try {
    return bo->bo.map();
  } catch (const std::exception &e) {
    fail_exc("map bo", e);
    return nullptr;
  }
}

size_t vxrt_bo_size(vxrt_bo *bo) { return bo ? bo->bo.size() : 0; }

int vxrt_bo_sync(vxrt_bo *bo, int direction, size_t size, size_t offset) {
  if (!bo)
    return fail(VXRT_ERR_INVALID, "null bo");
  auto dir = direction == VXRT_SYNC_TO_DEVICE ? XCL_BO_SYNC_BO_TO_DEVICE
                                              : XCL_BO_SYNC_BO_FROM_DEVICE;
  try {
    bo->bo.sync(dir, size, offset);
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("sync bo", e);
  }
}

int vxrt_npu_run_create(vxrt_kernel *k, vxrt_bo *insts, size_t n_insts,
                        vxrt_bo *const *bufs, int n_bufs, vxrt_run **out) {
  if (!k || !insts || !out || n_bufs < 0 || (n_bufs > 0 && !bufs))
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    xrt::run run(k->k);
    bind_npu_args(run, insts, n_insts, bufs, n_bufs);
    *out = new vxrt_run{run};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("create run", e);
  }
}

int vxrt_npu_run_start(vxrt_kernel *k, vxrt_bo *insts, size_t n_insts,
                       vxrt_bo *const *bufs, int n_bufs, vxrt_run **out) {
  int rc = vxrt_npu_run_create(k, insts, n_insts, bufs, n_bufs, out);
  if (rc != VXRT_OK)
    return rc;
  rc = vxrt_run_start(*out);
  if (rc != VXRT_OK) {
    vxrt_run_free(*out);
    *out = nullptr;
  }
  return rc;
}

int vxrt_run_start(vxrt_run *run) {
  if (!run)
    return fail(VXRT_ERR_INVALID, "null run");
  try {
    run->run.start();
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("start run", e);
  }
}

int vxrt_run_wait(vxrt_run *run, unsigned int timeout_ms) {
  if (!run)
    return fail(VXRT_ERR_INVALID, "null run");
  try {
    return check_state(run->run.wait(std::chrono::milliseconds(timeout_ms)),
                       "run");
  } catch (const std::exception &e) {
    return fail_exc("wait run", e);
  }
}

void vxrt_run_free(vxrt_run *run) { delete run; }

int vxrt_runlist_create(vxrt_hwctx *ctx, vxrt_runlist **out) {
  if (!ctx || !out)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    *out = new vxrt_runlist{xrt::runlist(ctx->ctx)};
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("create runlist", e);
  }
}

int vxrt_runlist_add(vxrt_runlist *rl, vxrt_run *run) {
  if (!rl || !run)
    return fail(VXRT_ERR_INVALID, "null argument");
  try {
    rl->rl.add(run->run);
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("runlist add", e);
  }
}

int vxrt_runlist_execute(vxrt_runlist *rl) {
  if (!rl)
    return fail(VXRT_ERR_INVALID, "null runlist");
  try {
    rl->rl.execute();
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("runlist execute", e);
  }
}

int vxrt_runlist_wait(vxrt_runlist *rl, unsigned int timeout_ms) {
  if (!rl)
    return fail(VXRT_ERR_INVALID, "null runlist");
  try {
    if (rl->rl.wait(std::chrono::milliseconds(timeout_ms)) ==
        std::cv_status::timeout)
      return fail(VXRT_ERR_TIMEOUT, "runlist: timed out");
    return VXRT_OK;
  } catch (const std::exception &e) {
    return fail_exc("runlist wait", e);
  }
}

void vxrt_runlist_free(vxrt_runlist *rl) { delete rl; }

} // extern "C"
