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
    /// The instruction stream, uploaded per context in the memory group
    /// this context's kernel reports (the group carries the context number).
    insts: Bo,
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

/// The NPU's hw_context slots are device-wide: a context creation that fails
/// with EINVAL when this process holds fewer than the limit means another
/// process (the whisper-xdna service, a second voxtype) holds the rest.
fn busy_hint(e: TranscribeError) -> TranscribeError {
    let msg = e.to_string();
    if !msg.contains("err=-22") {
        return e;
    }
    TranscribeError::InferenceFailed(format!(
        "{}\n  The NPU has no free hardware context. Another process is probably using it\n  \
         (e.g. an NPU speech-to-text container, or a second voxtype daemon); stop it and retry.",
        msg
    ))
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            loaded.push((name, xclbin, kernel_name, words, d.kind == "gemm"));
        }

        // A memory group from kernel.group_id() is (context number << 16) |
        // bank, so no value is valid for every context. Operand buffers are
        // host_only and outlive the contexts that use them, and a host_only
        // buffer allocated against one kernel feeds kernels of other overlays
        // and contexts with correct results (measured in whisper-xdna:
        // attention stacks, resident weights). So they are allocated in the
        // groups one kernel of each kind reports here; only the instruction
        // buffers, which are cacheable, use their own kernel's group.
        let probe = |gemm: bool, n: usize| -> Result<Vec<i32>, TranscribeError> {
            let (name, xclbin, kernel_name, ..) = loaded
                .iter()
                .find(|l| l.4 == gemm)
                .ok_or_else(|| fail("the pack has no design of a required kind"))?;
            let ctx = dev.hwctx(xclbin).map_err(busy_hint)?;
            let k = ctx.kernel(kernel_name)?;
            let groups = (0..n)
                .map(|i| k.group_id(3 + i as i32))
                .collect::<Result<Vec<_>, _>>()?;
            tracing::debug!("NPU: {} memory groups {:?}", name, groups);
            Ok(groups)
        };
        let gemm_groups = probe(true, 3)?;
        let rowop_groups = probe(false, 2)?;

        let overlays = loaded
            .into_iter()
            .map(|(name, xclbin, kernel_name, words, _)| {
                (
                    name,
                    Overlay {
                        xclbin,
                        kernel_name,
                        words,
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
            .get(name)
            .ok_or_else(|| fail(format!("design {} not loaded", name)))?;
        let ctx = self.dev.hwctx(&ov.xclbin).map_err(busy_hint)?;
        let kernel = ctx.kernel(&ov.kernel_name)?;
        let mut insts = self
            .dev
            .alloc(ov.words.len() * 4, BO_CACHEABLE, kernel.group_id(1)?)?;
        insts.as_u32_mut().copy_from_slice(&ov.words);
        insts.to_device()?;
        self.ctx_creates += 1;
        self.live.push(Live {
            name: name.to_string(),
            last: self.clock,
            runlist: None,
            insts,
            kernel,
            _ctx: ctx,
        });
        Ok(self.live.len() - 1)
    }

    /// Start kernel(3, insts, n_insts, bufs...) on `name`'s overlay.
    pub fn start(&mut self, name: &str, bufs: &[&Bo]) -> Result<Run, TranscribeError> {
        let i = self.slot(name)?;
        if let Some(out) = bufs.last() {
            out.flush_head();
        }
        let n = self.overlays[name].words.len();
        let live = &self.live[i];
        live.kernel.start(&live.insts, n, bufs)
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
            let n = self.overlays[name].words.len();
            let live = &self.live[i];
            let list = live._ctx.runlist()?;
            let mut runs = Vec::with_capacity(args.len());
            for a in args {
                let r = live.kernel.bind(&live.insts, n, a)?;
                list.add(&r)?;
                runs.push(r);
            }
            self.live[i].runlist = Some(BoundRunlist {
                step_ctx,
                list,
                _runs: runs,
            });
        }
        for a in args {
            a[2].flush_head();
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
