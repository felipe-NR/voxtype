/*
 * libvoxtype_xrt: a thin C ABI over the XRT C++ API, for the XDNA NPU encoder.
 *
 * voxtype dlopen()s this library at run time, only when the NPU encoder is
 * selected, so no voxtype binary links XRT. The surface is exactly what the
 * encoder needs: a device, xclbins, hw_contexts, kernels, buffer objects,
 * runs and runlists. Every handle is opaque and owned by the caller; every
 * call returns VXRT_OK or a negative status, with the reason in
 * vxrt_last_error() (thread-local).
 *
 * XRT objects are reference counted. A kernel, run or runlist keeps its
 * hw_context alive, so freeing the context handle alone does not release the
 * hardware slot: free everything built on it first.
 */
#ifndef VOXTYPE_VXRT_H
#define VOXTYPE_VXRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any breaking change to this header. voxtype refuses a mismatch. */
#define VXRT_ABI_VERSION 1

#define VXRT_OK 0
#define VXRT_ERR_INVALID -1 /* bad argument */
#define VXRT_ERR_XRT -2     /* XRT threw; see vxrt_last_error() */
#define VXRT_ERR_TIMEOUT -3 /* a run or runlist did not finish in time */
#define VXRT_ERR_STATE -4   /* a run ended in a state other than completed */

#define VXRT_BO_HOST_ONLY 0
#define VXRT_BO_CACHEABLE 1

#define VXRT_SYNC_TO_DEVICE 0
#define VXRT_SYNC_FROM_DEVICE 1

typedef struct vxrt_device vxrt_device;
typedef struct vxrt_xclbin vxrt_xclbin;
typedef struct vxrt_hwctx vxrt_hwctx;
typedef struct vxrt_kernel vxrt_kernel;
typedef struct vxrt_bo vxrt_bo;
typedef struct vxrt_run vxrt_run;
typedef struct vxrt_runlist vxrt_runlist;

int vxrt_abi_version(void);
const char *vxrt_last_error(void);

/* XRT library version this shim runs against, e.g. 2.25.0. */
int vxrt_xrt_version(int *major, int *minor, int *patch);

int vxrt_device_open(unsigned int index, vxrt_device **out);
void vxrt_device_free(vxrt_device *dev);
/* The device name XRT reports (vbnv, e.g. "RyzenAI-npu1"). */
int vxrt_device_name(vxrt_device *dev, char *buf, size_t len);

/* Load and register an xclbin. kernel_name receives the first kernel's name. */
int vxrt_xclbin_load(vxrt_device *dev, const char *path, vxrt_xclbin **out);
void vxrt_xclbin_free(vxrt_xclbin *xb);
int vxrt_xclbin_kernel_name(vxrt_xclbin *xb, char *buf, size_t len);

int vxrt_hwctx_create(vxrt_device *dev, vxrt_xclbin *xb, vxrt_hwctx **out);
void vxrt_hwctx_free(vxrt_hwctx *ctx);

int vxrt_kernel_open(vxrt_hwctx *ctx, const char *name, vxrt_kernel **out);
void vxrt_kernel_free(vxrt_kernel *k);
int vxrt_kernel_group_id(vxrt_kernel *k, int argno, int *out);

int vxrt_bo_alloc(vxrt_device *dev, size_t size, int kind, int group_id,
                  vxrt_bo **out);
/* A view of [offset, offset + size) of parent; shares its memory. */
int vxrt_bo_sub(vxrt_bo *parent, size_t size, size_t offset, vxrt_bo **out);
void vxrt_bo_free(vxrt_bo *bo);
/* Host mapping of the whole buffer; stable for the buffer's lifetime. */
void *vxrt_bo_map(vxrt_bo *bo);
size_t vxrt_bo_size(vxrt_bo *bo);
int vxrt_bo_sync(vxrt_bo *bo, int direction, size_t size, size_t offset);

/*
 * The MLIR-AIE NPU calling convention: kernel(3, insts, n_insts, bufs...).
 * vxrt_npu_run_create() binds the arguments without starting the run, for a
 * runlist; vxrt_npu_run_start() binds and starts it.
 */
int vxrt_npu_run_create(vxrt_kernel *k, vxrt_bo *insts, size_t n_insts,
                        vxrt_bo *const *bufs, int n_bufs, vxrt_run **out);
int vxrt_npu_run_start(vxrt_kernel *k, vxrt_bo *insts, size_t n_insts,
                       vxrt_bo *const *bufs, int n_bufs, vxrt_run **out);
int vxrt_run_start(vxrt_run *run);
/* timeout_ms = 0 waits forever. */
int vxrt_run_wait(vxrt_run *run, unsigned int timeout_ms);
void vxrt_run_free(vxrt_run *run);

int vxrt_runlist_create(vxrt_hwctx *ctx, vxrt_runlist **out);
int vxrt_runlist_add(vxrt_runlist *rl, vxrt_run *run);
int vxrt_runlist_execute(vxrt_runlist *rl);
int vxrt_runlist_wait(vxrt_runlist *rl, unsigned int timeout_ms);
void vxrt_runlist_free(vxrt_runlist *rl);

#ifdef __cplusplus
}
#endif

#endif
