//! Safe wrappers over libvoxtype_xrt (xdna-shim/vxrt.h), loaded with dlopen
//! only when the NPU encoder is selected, so no voxtype binary links XRT.

use crate::error::TranscribeError;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const ABI_VERSION: c_int = 1;

pub const BO_HOST_ONLY: c_int = 0;
pub const BO_CACHEABLE: c_int = 1;
const SYNC_TO_DEVICE: c_int = 0;
const SYNC_FROM_DEVICE: c_int = 1;

pub const SHIM_NAME: &str = "libvoxtype_xrt.so";

#[repr(C)]
pub struct Opaque {
    _p: [u8; 0],
}

type P = *mut Opaque;

macro_rules! shim_fns {
    ($($name:ident : fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        struct Fns {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Fns {
            unsafe fn load(handle: *mut c_void) -> Result<Self, String> {
                Ok(Self {
                    $($name: {
                        let sym = CString::new(stringify!($name)).unwrap();
                        let p = libc::dlsym(handle, sym.as_ptr());
                        if p.is_null() {
                            return Err(format!("missing symbol {}", stringify!($name)));
                        }
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(p)
                    },)*
                })
            }
        }
    };
}

shim_fns! {
    vxrt_abi_version: fn() -> c_int;
    vxrt_last_error: fn() -> *const c_char;
    vxrt_xrt_version: fn(*mut c_int, *mut c_int, *mut c_int) -> c_int;
    vxrt_device_open: fn(c_uint, *mut P) -> c_int;
    vxrt_device_free: fn(P);
    vxrt_device_name: fn(P, *mut c_char, usize) -> c_int;
    vxrt_xclbin_load: fn(P, *const c_char, *mut P) -> c_int;
    vxrt_xclbin_free: fn(P);
    vxrt_xclbin_kernel_name: fn(P, *mut c_char, usize) -> c_int;
    vxrt_hwctx_create: fn(P, P, *mut P) -> c_int;
    vxrt_hwctx_free: fn(P);
    vxrt_kernel_open: fn(P, *const c_char, *mut P) -> c_int;
    vxrt_kernel_free: fn(P);
    vxrt_kernel_group_id: fn(P, c_int, *mut c_int) -> c_int;
    vxrt_bo_alloc: fn(P, usize, c_int, c_int, *mut P) -> c_int;
    vxrt_bo_sub: fn(P, usize, usize, *mut P) -> c_int;
    vxrt_bo_free: fn(P);
    vxrt_bo_map: fn(P) -> *mut c_void;
    vxrt_bo_sync: fn(P, c_int, usize, usize) -> c_int;
    vxrt_npu_run_create: fn(P, P, usize, *const P, c_int, *mut P) -> c_int;
    vxrt_npu_run_start: fn(P, P, usize, *const P, c_int, *mut P) -> c_int;
    vxrt_run_wait: fn(P, c_uint) -> c_int;
    vxrt_run_free: fn(P);
    vxrt_runlist_create: fn(P, *mut P) -> c_int;
    vxrt_runlist_add: fn(P, P) -> c_int;
    vxrt_runlist_execute: fn(P) -> c_int;
    vxrt_runlist_wait: fn(P, c_uint) -> c_int;
    vxrt_runlist_free: fn(P);
}

/// The loaded shim. Shared by every handle, so it outlives them all.
pub struct Shim {
    fns: Fns,
    pub path: PathBuf,
    _handle: *mut c_void,
}

unsafe impl Send for Shim {}
unsafe impl Sync for Shim {}

fn init_err(msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InitFailed(msg.to_string())
}

/// Where to look for the shim when no path is configured, in order.
pub fn default_shim_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(home) = dirs::home_dir() {
        v.push(
            home.join(".local/lib/voxtype/xdna/lib/x86_64-linux-gnu")
                .join(SHIM_NAME),
        );
    }
    v.push(PathBuf::from("/usr/lib/voxtype/xdna/lib/x86_64-linux-gnu").join(SHIM_NAME));
    v
}

impl Shim {
    pub fn load(path: Option<&Path>) -> Result<Arc<Self>, TranscribeError> {
        let candidates = match path {
            Some(p) => vec![p.to_path_buf()],
            None => default_shim_paths(),
        };
        let found = candidates.iter().find(|p| p.is_file()).cloned().ok_or_else(|| {
            init_err(format!(
                "NPU encoder: {} not found (looked in {}).\n  \
                 Build it with: docker build -f xdna-shim/Dockerfile --output type=local,dest=<dir> xdna-shim\n  \
                 and set [whisper] xdna_shim to <dir>/lib/x86_64-linux-gnu/{}",
                SHIM_NAME,
                candidates
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                SHIM_NAME
            ))
        })?;
        let c_path = CString::new(found.as_os_str().as_encoded_bytes())
            .map_err(|_| init_err("NPU encoder: shim path contains a NUL byte"))?;
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            let why = unsafe { CStr::from_ptr(libc::dlerror()) }
                .to_string_lossy()
                .into_owned();
            return Err(init_err(format!(
                "NPU encoder: cannot load {}: {}",
                found.display(),
                why
            )));
        }
        let fns = unsafe { Fns::load(handle) }
            .map_err(|e| init_err(format!("NPU encoder: {}: {}", found.display(), e)))?;
        let abi = unsafe { (fns.vxrt_abi_version)() };
        if abi != ABI_VERSION {
            return Err(init_err(format!(
                "NPU encoder: {} speaks shim ABI {}, this voxtype needs {}. Rebuild the shim from this voxtype's xdna-shim/.",
                found.display(),
                abi,
                ABI_VERSION
            )));
        }
        Ok(Arc::new(Self {
            fns,
            path: found,
            _handle: handle,
        }))
    }

    fn last_error(&self) -> String {
        unsafe { CStr::from_ptr((self.fns.vxrt_last_error)()) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(&self, rc: c_int, what: &str) -> Result<(), TranscribeError> {
        if rc == 0 {
            Ok(())
        } else {
            Err(TranscribeError::InferenceFailed(format!(
                "NPU {}: {}",
                what,
                self.last_error()
            )))
        }
    }

    pub fn xrt_version(&self) -> Result<(i32, i32, i32), TranscribeError> {
        let (mut a, mut b, mut c) = (0, 0, 0);
        self.check(
            unsafe { (self.fns.vxrt_xrt_version)(&mut a, &mut b, &mut c) },
            "xrt version",
        )?;
        Ok((a, b, c))
    }
}

macro_rules! handle {
    ($name:ident, $free:ident) => {
        pub struct $name {
            shim: Arc<Shim>,
            ptr: P,
        }
        unsafe impl Send for $name {}
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { (self.shim.fns.$free)(self.ptr) }
            }
        }
    };
}

handle!(Device, vxrt_device_free);
handle!(Xclbin, vxrt_xclbin_free);
handle!(HwCtx, vxrt_hwctx_free);
handle!(Kernel, vxrt_kernel_free);
handle!(Run, vxrt_run_free);
handle!(Runlist, vxrt_runlist_free);

fn read_name(
    shim: &Shim,
    f: unsafe extern "C" fn(P, *mut c_char, usize) -> c_int,
    p: P,
    what: &str,
) -> Result<String, TranscribeError> {
    let mut buf = vec![0 as c_char; 256];
    shim.check(unsafe { f(p, buf.as_mut_ptr(), buf.len()) }, what)?;
    Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

impl Device {
    pub fn open(shim: &Arc<Shim>, index: u32) -> Result<Self, TranscribeError> {
        let mut p = std::ptr::null_mut();
        shim.check(
            unsafe { (shim.fns.vxrt_device_open)(index, &mut p) },
            "open device",
        )?;
        Ok(Self {
            shim: shim.clone(),
            ptr: p,
        })
    }

    pub fn name(&self) -> Result<String, TranscribeError> {
        read_name(
            &self.shim,
            self.shim.fns.vxrt_device_name,
            self.ptr,
            "device name",
        )
    }

    pub fn load_xclbin(&self, path: &Path) -> Result<Xclbin, TranscribeError> {
        let c = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| init_err("xclbin path contains a NUL byte"))?;
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_xclbin_load)(self.ptr, c.as_ptr(), &mut p) },
            "load xclbin",
        )?;
        Ok(Xclbin {
            shim: self.shim.clone(),
            ptr: p,
        })
    }

    pub fn hwctx(&self, xb: &Xclbin) -> Result<HwCtx, TranscribeError> {
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_hwctx_create)(self.ptr, xb.ptr, &mut p) },
            "create hw_context",
        )?;
        Ok(HwCtx {
            shim: self.shim.clone(),
            ptr: p,
        })
    }

    pub fn alloc(&self, size: usize, kind: c_int, group_id: i32) -> Result<Bo, TranscribeError> {
        let mut p = std::ptr::null_mut();
        self.shim
            .check(
                unsafe { (self.shim.fns.vxrt_bo_alloc)(self.ptr, size, kind, group_id, &mut p) },
                "allocate buffer",
            )
            .map_err(|e| memlock_hint(e, size))?;
        Bo::wrap(&self.shim, p, size)
    }
}

/// XRT maps every buffer MAP_LOCKED and the kernel checks each mapping
/// against RLIMIT_MEMLOCK, so a low limit fails here with EAGAIN.
fn memlock_hint(e: TranscribeError, size: usize) -> TranscribeError {
    let msg = e.to_string();
    if !(msg.contains("err=-11") || msg.contains("Resource temporarily unavailable")) {
        return e;
    }
    let limit = super::memlock_limit()
        .map(|l| format!("{} bytes", l))
        .unwrap_or_else(|| "unknown".into());
    TranscribeError::InferenceFailed(format!(
        "{}\n  The NPU needs to lock a {} byte buffer, but RLIMIT_MEMLOCK is {}.\n  \
         Run `voxtype setup npu --status` for the limit this model needs and how to raise it.",
        msg, size, limit
    ))
}

impl Xclbin {
    pub fn kernel_name(&self) -> Result<String, TranscribeError> {
        read_name(
            &self.shim,
            self.shim.fns.vxrt_xclbin_kernel_name,
            self.ptr,
            "kernel name",
        )
    }
}

impl HwCtx {
    pub fn kernel(&self, name: &str) -> Result<Kernel, TranscribeError> {
        let c = CString::new(name).map_err(|_| init_err("kernel name contains a NUL byte"))?;
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_kernel_open)(self.ptr, c.as_ptr(), &mut p) },
            "open kernel",
        )?;
        Ok(Kernel {
            shim: self.shim.clone(),
            ptr: p,
        })
    }

    pub fn runlist(&self) -> Result<Runlist, TranscribeError> {
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_runlist_create)(self.ptr, &mut p) },
            "create runlist",
        )?;
        Ok(Runlist {
            shim: self.shim.clone(),
            ptr: p,
        })
    }
}

impl Kernel {
    pub fn group_id(&self, argno: i32) -> Result<i32, TranscribeError> {
        let mut g = 0;
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_kernel_group_id)(self.ptr, argno, &mut g) },
            "kernel group_id",
        )?;
        Ok(g)
    }

    fn args(bufs: &[&Bo]) -> Vec<P> {
        bufs.iter().map(|b| b.ptr).collect()
    }

    /// Bind kernel(3, insts, n_insts, bufs...) and start it.
    pub fn start(&self, insts: &Bo, n_insts: usize, bufs: &[&Bo]) -> Result<Run, TranscribeError> {
        let args = Self::args(bufs);
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe {
                (self.shim.fns.vxrt_npu_run_start)(
                    self.ptr,
                    insts.ptr,
                    n_insts,
                    args.as_ptr(),
                    args.len() as c_int,
                    &mut p,
                )
            },
            "start run",
        )?;
        Ok(Run {
            shim: self.shim.clone(),
            ptr: p,
        })
    }

    /// Bind kernel(3, insts, n_insts, bufs...) without starting it.
    pub fn bind(&self, insts: &Bo, n_insts: usize, bufs: &[&Bo]) -> Result<Run, TranscribeError> {
        let args = Self::args(bufs);
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe {
                (self.shim.fns.vxrt_npu_run_create)(
                    self.ptr,
                    insts.ptr,
                    n_insts,
                    args.as_ptr(),
                    args.len() as c_int,
                    &mut p,
                )
            },
            "create run",
        )?;
        Ok(Run {
            shim: self.shim.clone(),
            ptr: p,
        })
    }
}

impl Run {
    pub fn wait(&self, timeout_ms: u32, what: &str) -> Result<(), TranscribeError> {
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_run_wait)(self.ptr, timeout_ms) },
            what,
        )
    }
}

impl Runlist {
    pub fn add(&self, run: &Run) -> Result<(), TranscribeError> {
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_runlist_add)(self.ptr, run.ptr) },
            "runlist add",
        )
    }

    pub fn execute(&self) -> Result<(), TranscribeError> {
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_runlist_execute)(self.ptr) },
            "runlist execute",
        )
    }

    pub fn wait(&self, timeout_ms: u32, what: &str) -> Result<(), TranscribeError> {
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_runlist_wait)(self.ptr, timeout_ms) },
            what,
        )
    }
}

/// A buffer object with its host mapping.
pub struct Bo {
    shim: Arc<Shim>,
    ptr: P,
    map: *mut u8,
    size: usize,
}

unsafe impl Send for Bo {}

impl Drop for Bo {
    fn drop(&mut self) {
        unsafe { (self.shim.fns.vxrt_bo_free)(self.ptr) }
    }
}

impl Bo {
    fn wrap(shim: &Arc<Shim>, ptr: P, size: usize) -> Result<Self, TranscribeError> {
        let map = unsafe { (shim.fns.vxrt_bo_map)(ptr) } as *mut u8;
        let bo = Self {
            shim: shim.clone(),
            ptr,
            map,
            size,
        };
        if map.is_null() {
            return Err(TranscribeError::InferenceFailed(format!(
                "NPU map buffer: {}",
                shim.last_error()
            )));
        }
        Ok(bo)
    }

    /// A view of [offset, offset + size) sharing this buffer's memory.
    pub fn sub(&self, size: usize, offset: usize) -> Result<Bo, TranscribeError> {
        assert!(offset + size <= self.size);
        let mut p = std::ptr::null_mut();
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_bo_sub)(self.ptr, size, offset, &mut p) },
            "sub-buffer",
        )?;
        Bo::wrap(&self.shim, p, size)
    }

    pub fn as_u16_mut(&mut self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.map as *mut u16, self.size / 2) }
    }

    pub fn as_u16(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.map as *const u16, self.size / 2) }
    }

    pub fn as_f32(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.map as *const f32, self.size / 4) }
    }

    pub fn as_u32_mut(&mut self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.map as *mut u32, self.size / 4) }
    }

    pub fn to_device(&self) -> Result<(), TranscribeError> {
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_bo_sync)(self.ptr, SYNC_TO_DEVICE, self.size, 0) },
            "sync to device",
        )
    }

    pub fn from_device(&self) -> Result<(), TranscribeError> {
        self.shim.check(
            unsafe { (self.shim.fns.vxrt_bo_sync)(self.ptr, SYNC_FROM_DEVICE, self.size, 0) },
            "sync from device",
        )
    }

    /// Evict the first page of this buffer from the CPU caches. Done on every
    /// output buffer before a launch: without it the first cache line of a
    /// result sometimes keeps what the previous launch wrote there (seen on
    /// XDNA1 with XRT 2.25 as 64 stale bytes at offset 0, never elsewhere;
    /// a sync after the run does not clear it, a flush before it does).
    pub fn flush_head(&self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let n = self.size.min(4096);
            let mut off = 0;
            while off < n {
                std::arch::x86_64::_mm_clflush(self.map.add(off));
                off += 64;
            }
            std::arch::x86_64::_mm_mfence();
        }
    }
}
