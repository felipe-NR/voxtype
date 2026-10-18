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
    words: Vec<u32>,
    /// Uploaded the first time a context opens the overlay, in the memory
    /// group its kernel reports for the instruction argument.
    insts: Option<Bo>,
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
    pub ctx_creates: usize,
}

fn fail(msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InferenceFailed(format!("NPU: {}", msg))
}

impl Runtime {
    /// Register every overlay of the pack and upload its instruction stream.
    pub fn new(dev: Device, pack: &Pack) -> Result<Self, TranscribeError> {
        let m = &pack.manifest;
        let mut loaded = Vec::new();
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
            let bytes = std::fs::read(pack.file(&d.insts))
                .map_err(|e| fail(format!("{}: {}", d.insts.path, e)))?;
            if bytes.len() % 4 != 0 || bytes.is_empty() {
                return Err(fail(format!("{}: not a uint32 stream", d.insts.path)));
            }
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            loaded.push((name, xclbin, kernel_name, words, d.kind == "gemm"));
        }

        // XRT encodes memory groups (65536 for a kernel's argument 3, not the
        // 3 the manifest lists), so operand buffers are allocated in the
        // groups one kernel of each kind reports. A host_only buffer is not
        // tied to that kernel: it feeds kernels of other overlays too
        // (measured in whisper-xdna, probe_bo_share.py).
        let probe = |gemm: bool, n: usize| -> Result<Vec<i32>, TranscribeError> {
            let (name, xclbin, kernel_name, ..) = loaded
                .iter()
                .find(|l| l.4 == gemm)
                .ok_or_else(|| fail("the pack has no design of a required kind"))?;
            let ctx = dev.hwctx(xclbin)?;
            let k = ctx.kernel(kernel_name)?;
            let groups = (0..n)
                .map(|i| k.group_id(3 + i as i32))
                .collect::<Result<Vec<_>, _>>()?;
            tracing::debug!("NPU: {} memory groups {:?}", name, groups);
            Ok(groups)
        };
        let gemm_groups = probe(true, m.abi.gemm.group_ids.len())?;
        let rowop_groups = probe(false, m.abi.rowop.group_ids.len())?;

        let overlays = loaded
            .into_iter()
            .map(|(name, xclbin, kernel_name, words, gemm)| {
                (
                    name,
                    Overlay {
                        xclbin,
                        kernel_name,
                        words,
                        insts: None,
                        gemm,
                    },
                )
            })
            .collect();
        Ok(Self {
            dev,
            overlays,
            live: Vec::new(),
            clock: 0,
            max_ctx: m.device.max_hw_contexts,
            gemm_groups,
            rowop_groups,
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
            .get_mut(name)
            .ok_or_else(|| fail(format!("design {} not loaded", name)))?;
        let ctx = self.dev.hwctx(&ov.xclbin)?;
        let kernel = ctx.kernel(&ov.kernel_name)?;
        if ov.insts.is_none() {
            let mut insts =
                self.dev
                    .alloc(ov.words.len() * 4, BO_CACHEABLE, kernel.group_id(1)?)?;
            insts.as_u32_mut().copy_from_slice(&ov.words);
            insts.to_device()?;
            ov.insts = Some(insts);
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
        let insts = ov.insts.as_ref().expect("uploaded when the context opened");
        self.live[i].kernel.start(insts, ov.words.len(), bufs)
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
            let insts = ov.insts.as_ref().expect("uploaded when the context opened");
            let live = &self.live[i];
            let list = live._ctx.runlist()?;
            let mut runs = Vec::with_capacity(args.len());
            for a in args {
                let r = live.kernel.bind(insts, ov.words.len(), a)?;
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
