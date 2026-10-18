//! Overlays, hw_contexts and kernels on the NPU.
//!
//! XDNA1 holds at most 6 hw_contexts per process, and a full encoder pass
//! touches 9 designs, so contexts are kept LRU and recreated as needed
//! (2-3 ms per pass, measured). Buffers are not tied to contexts: a
//! `host_only` buffer feeds kernels in any context and survives their
//! teardown, so weights and activations live outside this cache.

use super::pack::{Design, Pack};
use super::shim::{Bo, Device, HwCtx, Kernel, Run, Runlist, BO_CACHEABLE};
use crate::error::TranscribeError;
use std::collections::HashMap;

/// A launch that has not finished within this is a wedged device.
pub const RUN_TIMEOUT_MS: u32 = 10_000;

struct Overlay {
    xclbin: super::shim::Xclbin,
    kernel_name: String,
    insts: Bo,
    n_insts: usize,
    gemm: bool,
}

/// Runlists bound to one context's kernel, for one set of step buffers.
pub struct BoundRunlist {
    pub step_ctx: usize,
    pub list: Runlist,
    _runs: Vec<Run>,
}

struct Live {
    name: String,
    last: u64,
    // Field order is drop order: runs and runlists, then the kernel, then
    // the context they keep alive.
    runlist: Option<BoundRunlist>,
    kernel: Kernel,
    _ctx: HwCtx,
}

pub struct Runtime {
    pub dev: Device,
    overlays: HashMap<String, Overlay>,
    live: Vec<Live>,
    clock: u64,
    max_ctx: usize,
    gemm_groups: Vec<i32>,
    rowop_groups: Vec<i32>,
    insts_group: i32,
    pub ctx_creates: usize,
}

fn fail(msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InferenceFailed(format!("NPU: {}", msg))
}

impl Runtime {
    /// Register every overlay of the pack and upload its instruction stream.
    pub fn new(dev: Device, pack: &Pack) -> Result<Self, TranscribeError> {
        let m = &pack.manifest;
        let mut overlays = HashMap::new();
        for name in pack.designs_in_order() {
            let d: &Design = &m.designs[&name];
            let xclbin = dev.load_xclbin(&pack.file(&d.xclbin))?;
            let kernel_name = xclbin.kernel_name()?;
            if !kernel_name.starts_with(&d.kernel) {
                return Err(fail(format!(
                    "{}: kernel {:?}, manifest says {:?}",
                    name, kernel_name, d.kernel
                )));
            }
            let words = std::fs::read(pack.file(&d.insts))
                .map_err(|e| fail(format!("{}: {}", d.insts.path, e)))?;
            if words.len() % 4 != 0 || words.is_empty() {
                return Err(fail(format!("{}: not a uint32 stream", d.insts.path)));
            }
            let mut insts = dev.alloc(words.len(), BO_CACHEABLE, m.abi.insts_group_id)?;
            for (dst, src) in insts.as_u32_mut().iter_mut().zip(words.chunks_exact(4)) {
                *dst = u32::from_le_bytes(src.try_into().unwrap());
            }
            insts.to_device()?;
            overlays.insert(
                name.clone(),
                Overlay {
                    xclbin,
                    kernel_name,
                    insts,
                    n_insts: words.len() / 4,
                    gemm: d.kind == "gemm",
                },
            );
        }
        Ok(Self {
            dev,
            overlays,
            live: Vec::new(),
            clock: 0,
            max_ctx: m.device.max_hw_contexts,
            gemm_groups: m.abi.gemm.group_ids.clone(),
            rowop_groups: m.abi.rowop.group_ids.clone(),
            insts_group: m.abi.insts_group_id,
            ctx_creates: 0,
        })
    }

    pub fn gemm_group(&self, i: usize) -> i32 {
        self.gemm_groups[i]
    }

    pub fn rowop_group(&self, i: usize) -> i32 {
        self.rowop_groups[i]
    }

    /// The index of `name` in the live set, opening a context if needed.
    fn slot(&mut self, name: &str) -> Result<usize, TranscribeError> {
        self.clock += 1;
        if let Some(i) = self.live.iter().position(|l| l.name == name) {
            self.live[i].last = self.clock;
            return Ok(i);
        }
        while self.live.len() >= self.max_ctx {
            let victim = self
                .live
                .iter()
                .enumerate()
                .min_by_key(|(_, l)| l.last)
                .map(|(i, _)| i)
                .unwrap();
            self.live.swap_remove(victim);
        }
        let ov = self
            .overlays
            .get(name)
            .ok_or_else(|| fail(format!("design {} not loaded", name)))?;
        let ctx = self.dev.hwctx(&ov.xclbin)?;
        let kernel = ctx.kernel(&ov.kernel_name)?;
        // The group ids in the manifest are assertions about the toolchain the
        // overlays came from; a mismatch means pack and runtime have drifted.
        let want: &[i32] = if ov.gemm {
            &self.gemm_groups
        } else {
            &self.rowop_groups
        };
        for (i, &g) in want.iter().enumerate() {
            let got = kernel.group_id(3 + i as i32)?;
            if got != g {
                return Err(fail(format!(
                    "{}: argument {} is in memory group {}, the pack says {}",
                    name,
                    3 + i,
                    got,
                    g
                )));
            }
        }
        if kernel.group_id(1)? != self.insts_group {
            return Err(fail(format!(
                "{}: instruction buffer group differs from the pack's",
                name
            )));
        }
        self.ctx_creates += 1;
        self.live.push(Live {
            name: name.to_string(),
            last: self.clock,
            runlist: None,
            kernel,
            _ctx: ctx,
        });
        Ok(self.live.len() - 1)
    }

    /// Start kernel(3, insts, n_insts, bufs...) on `name`'s overlay.
    pub fn start(&mut self, name: &str, bufs: &[&Bo]) -> Result<Run, TranscribeError> {
        let i = self.slot(name)?;
        let ov = &self.overlays[name];
        self.live[i].kernel.start(&ov.insts, ov.n_insts, bufs)
    }

    /// Run and wait.
    pub fn run(&mut self, name: &str, bufs: &[&Bo]) -> Result<(), TranscribeError> {
        self.start(name, bufs)?.wait(RUN_TIMEOUT_MS, name)
    }

    /// Execute `name`'s runlist of one launch per argument set, building it
    /// first if this context has none for `step_ctx`.
    pub fn execute_list(
        &mut self,
        name: &str,
        step_ctx: usize,
        args: &[[&Bo; 3]],
    ) -> Result<(), TranscribeError> {
        let i = self.slot(name)?;
        let fresh = !matches!(&self.live[i].runlist, Some(b) if b.step_ctx == step_ctx);
        if fresh {
            self.live[i].runlist = None;
            let ov = &self.overlays[name];
            let live = &self.live[i];
            let list = live._ctx.runlist()?;
            let mut runs = Vec::with_capacity(args.len());
            for a in args {
                let r = live.kernel.bind(&ov.insts, ov.n_insts, a)?;
                list.add(&r)?;
                runs.push(r);
            }
            self.live[i].runlist = Some(BoundRunlist {
                step_ctx,
                list,
                _runs: runs,
            });
        }
        self.live[i].runlist.as_ref().unwrap().list.execute()
    }

    pub fn wait_list(&self, name: &str) -> Result<(), TranscribeError> {
        let live = self
            .live
            .iter()
            .find(|l| l.name == name)
            .ok_or_else(|| fail(format!("{}: no runlist in flight", name)))?;
        live.runlist
            .as_ref()
            .ok_or_else(|| fail(format!("{}: no runlist in flight", name)))?
            .list
            .wait(RUN_TIMEOUT_MS, name)
    }

    /// Drop every runlist: they hold the buffers of one ladder step.
    pub fn drop_runlists(&mut self) {
        for l in &mut self.live {
            l.runlist = None;
        }
    }
}
