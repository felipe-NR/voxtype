//! The Whisper encoder graph on the XDNA NPU.
//!
//! Every GEMM, the attention chain (QK -> softmax -> AV, all heads per
//! launch, nothing crossing the host in between) and every GELU run on the
//! device; layer norms, residuals, im2col and staging run here. The order of
//! operations, the paddings and the bias folding follow whisper-xdna's
//! `encoder()` with its RawRT backend (`src/npu_whisper_encoder.py`,
//! `src/rawxrt.py`); the manifest supplies every number.

use super::ggml::GgmlEncoder;
use super::host::{
    add_into, add_row, bf16, bf16_to_f32, im2col, layer_norm, stage_bf16, transpose, unpad_f32,
};
use super::pack::{Gemm, Pack, Step};
use super::runtime::{Runtime, RUN_TIMEOUT_MS};
use super::shim::{Bo, BO_HOST_ONLY};
use crate::error::TranscribeError;

fn fail(msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InferenceFailed(format!("NPU encoder: {}", msg))
}

/// The f32 parameters the host side applies.
struct LayerHost {
    attn_ln_w: Vec<f32>,
    attn_ln_b: Vec<f32>,
    mlp_ln_w: Vec<f32>,
    mlp_ln_b: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_b: Vec<f32>,
}

/// The device-resident weights of one layer (bf16, padded B operands).
struct LayerDev {
    q: Bo,
    k: Bo,
    v: Bo,
    out: Bo,
    /// `fused`: one bias-folded B per fc1 column chunk. `plain`: one B.
    fc1: Vec<Bo>,
    fc2: Bo,
}

/// Activation buffers for one ladder step. Allocated when the step is first
/// used, freed when another step takes over.
struct StepBufs {
    ctx: usize,
    conv1_a: Bo,
    conv1_c: Bo,
    conv2_a: Bo,
    conv2_c: Bo,
    gelu_in: Bo,
    gelu_out: Bo,
    proj_a: Bo,
    proj_c: Vec<Bo>,
    q: Bo,
    k: Bo,
    v: Bo,
    o: Bo,
    s_in: Bo,
    s_out: Bo,
    sub_q: Vec<Bo>,
    sub_k: Vec<Bo>,
    sub_v: Vec<Bo>,
    sub_o: Vec<Bo>,
    sub_in: Vec<Bo>,
    sub_out: Vec<Bo>,
    /// `plain` MLP path.
    fc1_a: Option<Bo>,
    fc1_c: Option<Bo>,
    fc2_a: Bo,
    fc2_c: Bo,
    /// `fused` MLP path: fc1_gelu's output bands inside fc2_a.
    bands: Vec<Bo>,
}

pub struct Encoder {
    pack: Pack,
    rt: Runtime,
    n_layer: usize,
    d: usize,
    nh: usize,
    n_mels: usize,
    conv1_w: Bo,
    conv2_w: Bo,
    conv1_b: Vec<f32>,
    conv2_b: Vec<f32>,
    pos_emb: Vec<f32>,
    ln_post_w: Vec<f32>,
    ln_post_b: Vec<f32>,
    host: Vec<LayerHost>,
    dev_w: Vec<LayerDev>,
    bufs: Option<StepBufs>,
    /// When set, encode() records the stages the fixtures pin down.
    pub taps: Option<Vec<(String, Vec<f32>)>>,
}

/// The padded B operand of `x @ W.T`, for W stored (out, in) row-major:
/// rows are `in`, columns the output range [n0, n0 + nw). With `bias`, the
/// bias goes in as row `in` (the fold: A carries a ones column there).
#[allow(clippy::too_many_arguments)]
fn upload_b(
    rt: &Runtime,
    w: &[f32],
    n_in: usize,
    n0: usize,
    nw: usize,
    bias: Option<&[f32]>,
    role: &Gemm,
) -> Result<Bo, TranscribeError> {
    let (kp, np) = (role.kp, role.np);
    let need_rows = n_in + bias.is_some() as usize;
    if need_rows > kp || nw > np {
        return Err(fail(format!(
            "{}: weight {}x{} does not fit its {}x{} operand",
            role.design, need_rows, nw, kp, np
        )));
    }
    let mut bo = rt.dev.alloc(kp * np * 2, BO_HOST_ONLY, rt.gemm_group(1))?;
    let dst = bo.as_u16_mut();
    dst.fill(0);
    for i in 0..n_in {
        let row = &mut dst[i * np..i * np + nw];
        for (n, o) in row.iter_mut().enumerate() {
            *o = bf16(w[(n0 + n) * n_in + i]);
        }
    }
    if let Some(b) = bias {
        for (n, o) in dst[n_in * np..n_in * np + nw].iter_mut().enumerate() {
            *o = bf16(b[n0 + n]);
        }
    }
    bo.to_device()?;
    Ok(bo)
}

fn zeroed(rt: &Runtime, bytes: usize, group: i32) -> Result<Bo, TranscribeError> {
    let mut bo = rt.dev.alloc(bytes, BO_HOST_ONLY, group)?;
    bo.as_u16_mut().fill(0);
    bo.to_device()?;
    Ok(bo)
}

impl Encoder {
    pub fn new(pack: Pack, rt: Runtime, ggml: &GgmlEncoder) -> Result<Self, TranscribeError> {
        let m = &pack.manifest;
        let n_layer = m.model.n_audio_layer as usize;
        let d = m.model.n_audio_state as usize;
        let nh = m.model.n_audio_head as usize;
        let n_mels = m.model.n_mels as usize;
        // Weight operands depend on K and N only, never on M, so every step
        // shares them; check that the pack agrees before relying on it.
        let first = &m.ladder[0].roles;
        for s in &m.ladder {
            let r = &s.roles;
            let same = |a: &Gemm, b: &Gemm| a.kp == b.kp && a.np == b.np;
            let fc1_same = match (&r.fc1, &first.fc1) {
                (Some(a), Some(b)) => same(a, b),
                (None, None) => true,
                _ => false,
            };
            if !(same(&r.conv1, &first.conv1)
                && same(&r.conv2, &first.conv2)
                && same(&r.proj, &first.proj)
                && same(&r.fc2, &first.fc2)
                && fc1_same
                && s.mlp.path == m.ladder[0].mlp.path)
            {
                return Err(fail(format!(
                    "ladder step {} pads its weights differently from step {}",
                    s.ctx, m.ladder[0].ctx
                )));
            }
        }
        let roles = first.clone();
        let fused = m.ladder[0].mlp.path == "fused";

        let read = |k: &str| ggml.read_f32(k);
        let c1 = read("conv1.weight")?;
        let conv1_w = upload_b(&rt, &c1, n_mels * 3, 0, d, None, &roles.conv1)?;
        drop(c1);
        let c2 = read("conv2.weight")?;
        let conv2_w = upload_b(&rt, &c2, d * 3, 0, d, None, &roles.conv2)?;
        drop(c2);

        let zero_bias = vec![0f32; d];
        let mut host = Vec::with_capacity(n_layer);
        let mut dev_w = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |s: &str| format!("blocks.{}.{}", l, s);
            let proj = &roles.proj;
            let wq = read(&p("attn.query.weight"))?;
            let q = upload_b(&rt, &wq, d, 0, d, Some(&read(&p("attn.query.bias"))?), proj)?;
            let wk = read(&p("attn.key.weight"))?;
            let k = upload_b(&rt, &wk, d, 0, d, Some(&zero_bias), proj)?;
            let wv = read(&p("attn.value.weight"))?;
            let v = upload_b(&rt, &wv, d, 0, d, Some(&read(&p("attn.value.bias"))?), proj)?;
            let wo = read(&p("attn.out.weight"))?;
            let out = upload_b(&rt, &wo, d, 0, d, Some(&read(&p("attn.out.bias"))?), proj)?;
            drop((wq, wk, wv, wo));

            let w1 = read(&p("mlp.0.weight"))?;
            let b1 = read(&p("mlp.0.bias"))?;
            let d_ff = b1.len();
            let fc1 = if fused {
                let chunks = m.ladder[0].mlp.chunks;
                let width = proj.np.min(d_ff / chunks.max(1));
                if chunks == 0 || width * chunks != d_ff {
                    return Err(fail(format!(
                        "fc1 width {} does not split into {} chunks",
                        d_ff, chunks
                    )));
                }
                (0..chunks)
                    .map(|j| upload_b(&rt, &w1, d, j * width, width, Some(&b1), proj))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let fc1 = roles.fc1.as_ref().unwrap();
                vec![upload_b(&rt, &w1, d, 0, d_ff, None, fc1)?]
            };
            drop(w1);
            let w2 = read(&p("mlp.2.weight"))?;
            let fc2 = upload_b(&rt, &w2, d_ff, 0, d, None, &roles.fc2)?;
            drop(w2);

            host.push(LayerHost {
                attn_ln_w: read(&p("attn_ln.weight"))?,
                attn_ln_b: read(&p("attn_ln.bias"))?,
                mlp_ln_w: read(&p("mlp_ln.weight"))?,
                mlp_ln_b: read(&p("mlp_ln.bias"))?,
                fc1_b: b1,
                fc2_b: read(&p("mlp.2.bias"))?,
            });
            dev_w.push(LayerDev {
                q,
                k,
                v,
                out,
                fc1,
                fc2,
            });
        }

        let enc = Self {
            n_layer,
            d,
            nh,
            n_mels,
            conv1_w,
            conv2_w,
            conv1_b: read("conv1.bias")?,
            conv2_b: read("conv2.bias")?,
            pos_emb: read("positional_embedding")?,
            ln_post_w: read("ln_post.weight")?,
            ln_post_b: read("ln_post.bias")?,
            host,
            dev_w,
            bufs: None,
            taps: None,
            pack,
            rt,
        };
        Ok(enc)
    }

    pub fn ctx_creates(&self) -> usize {
        self.rt.ctx_creates
    }

    /// Allocate a step's buffers now, so a memlock limit that is too low
    /// fails at startup instead of mid-dictation.
    pub fn preallocate(&mut self, ctx: usize) -> Result<(), TranscribeError> {
        let step = self
            .pack
            .step_for_ctx(ctx)
            .ok_or_else(|| fail(format!("no ladder step for ctx {}", ctx)))?
            .clone();
        self.ensure_bufs(&step)
    }

    fn ensure_bufs(&mut self, step: &Step) -> Result<(), TranscribeError> {
        if matches!(&self.bufs, Some(b) if b.ctx == step.ctx) {
            return Ok(());
        }
        // Runlists hold sub-buffers of the old step; drop them first.
        self.rt.drop_runlists();
        self.bufs = None;

        let rt = &self.rt;
        let r = &step.roles;
        let (ga, gb, gc) = (rt.gemm_group(0), rt.gemm_group(1), rt.gemm_group(2));
        let (gi, go) = (rt.rowop_group(0), rt.rowop_group(1));
        let a_of = |g: &Gemm| zeroed(rt, g.mp * g.kp * 2, ga);
        let c_of = |g: &Gemm| zeroed(rt, g.mp * g.np * 4, gc);

        let nh = self.nh;
        let hd = self.d / nh;
        let (mp, np, an) = (step.attn.mp, step.attn.np, step.attn.an);
        if an != hd || r.qk.kp != hd {
            return Err(fail(format!(
                "attention head padding {} / {} != head dim {}",
                an, r.qk.kp, hd
            )));
        }
        let q = zeroed(rt, nh * mp * hd * 2, ga)?;
        let k = zeroed(rt, nh * hd * np * 2, gb)?;
        let v = zeroed(rt, nh * mp * an * 2, gb)?;
        let o = zeroed(rt, nh * mp * an * 4, gc)?;
        let head = mp * np * 2;
        let s_in = zeroed(rt, nh * head, gi)?;
        let s_out = zeroed(rt, nh * head, go)?;
        let subs = |bo: &Bo, each: usize| -> Result<Vec<Bo>, TranscribeError> {
            (0..nh).map(|h| bo.sub(each, h * each)).collect()
        };
        let sub_q = subs(&q, mp * hd * 2)?;
        let sub_k = subs(&k, hd * np * 2)?;
        let sub_v = subs(&v, mp * an * 2)?;
        let sub_o = subs(&o, mp * an * 4)?;
        let sub_in = subs(&s_in, head)?;
        let sub_out = subs(&s_out, head)?;

        let chunks = step.mlp.chunks.max(1);
        let proj_c = (0..chunks.max(3))
            .map(|_| c_of(&r.proj))
            .collect::<Result<Vec<_>, _>>()?;
        let fc2_a = a_of(&r.fc2)?;
        let bands = match &r.fc1_gelu {
            Some(g) => {
                let band = g
                    .band
                    .as_ref()
                    .ok_or_else(|| fail("fc1_gelu without a band"))?;
                if band.row_stride_bytes != r.fc2.kp * 2 {
                    return Err(fail("fc1_gelu band stride does not match fc2's A"));
                }
                let total = r.fc2.mp * r.fc2.kp * 2;
                (0..g.chunks)
                    .map(|j| {
                        let off = j * band.offset_bytes_per_chunk;
                        fc2_a.sub(total - off, off)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => Vec::new(),
        };
        let (fc1_a, fc1_c) = match &r.fc1 {
            Some(f) => (Some(a_of(f)?), Some(c_of(f)?)),
            None => (None, None),
        };
        let gelu_elems = r.gelu.rows * r.gelu.cols;
        self.bufs = Some(StepBufs {
            ctx: step.ctx,
            conv1_a: a_of(&r.conv1)?,
            conv1_c: c_of(&r.conv1)?,
            conv2_a: a_of(&r.conv2)?,
            conv2_c: c_of(&r.conv2)?,
            gelu_in: zeroed(rt, gelu_elems * 2, gi)?,
            gelu_out: zeroed(rt, gelu_elems * 2, go)?,
            proj_a: a_of(&r.proj)?,
            proj_c,
            q,
            k,
            v,
            o,
            s_in,
            s_out,
            sub_q,
            sub_k,
            sub_v,
            sub_o,
            sub_in,
            sub_out,
            fc1_a,
            fc1_c,
            fc2_a,
            fc2_c: c_of(&r.fc2)?,
            bands,
        });
        Ok(())
    }

    /// One GEMM: stage A (optionally with the bias-fold ones column), run
    /// against a resident B, return the unpadded f32 result.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        rt: &mut Runtime,
        role: &Gemm,
        a: &mut Bo,
        c: &Bo,
        b: &Bo,
        x: &[f32],
        rows: usize,
        cols: usize,
        n: usize,
        fold: bool,
    ) -> Result<Vec<f32>, TranscribeError> {
        stage_bf16(
            a.as_u16_mut(),
            role.mp,
            role.kp,
            x,
            rows,
            cols,
            fold.then_some(cols),
        );
        a.to_device()?;
        rt.run(&role.design, &[a, b, c])?;
        c.from_device()?;
        Ok(unpad_f32(c.as_f32(), role.np, rows, n))
    }

    /// GELU over any number of values through the one fixed `gelu` tile.
    fn gelu(
        rt: &mut Runtime,
        step: &Step,
        gin: &mut Bo,
        gout: &Bo,
        x: &mut [f32],
    ) -> Result<(), TranscribeError> {
        let g = &step.roles.gelu;
        let chunk = g.rows * g.cols;
        for part in x.chunks_mut(chunk) {
            let dst = gin.as_u16_mut();
            for (o, &v) in dst.iter_mut().zip(part.iter()) {
                *o = bf16(v);
            }
            dst[part.len()..chunk].fill(0);
            gin.to_device()?;
            rt.run(&g.design, &[gin, gout])?;
            gout.from_device()?;
            for (o, &h) in part.iter_mut().zip(gout.as_u16()) {
                *o = bf16_to_f32(h);
            }
        }
        Ok(())
    }

    /// Encode one window: `mel` is n_mels x 2*ctx, row-major per mel bin.
    /// Returns ctx x n_state f32, after ln_post.
    pub fn encode(&mut self, mel: &[f32], ctx: usize) -> Result<Vec<f32>, TranscribeError> {
        let step = self
            .pack
            .step_for_ctx(ctx)
            .ok_or_else(|| fail(format!("no ladder step for ctx {}", ctx)))?
            .clone();
        let (d, nh, n_mels) = (self.d, self.nh, self.n_mels);
        let frames = 2 * ctx;
        if mel.len() != n_mels * frames {
            return Err(fail(format!(
                "mel has {} values, want {} x {}",
                mel.len(),
                n_mels,
                frames
            )));
        }
        self.ensure_bufs(&step)?;
        let r = &step.roles;
        let rt = &mut self.rt;
        let bufs = self.bufs.as_mut().unwrap();

        // conv frontend: two im2col GEMMs, bias and GELU on top
        let a1 = im2col(mel, n_mels, frames, 3, 1, 1);
        let mut x = Self::gemm(
            rt,
            &r.conv1,
            &mut bufs.conv1_a,
            &bufs.conv1_c,
            &self.conv1_w,
            &a1,
            frames,
            n_mels * 3,
            d,
            false,
        )?;
        add_row(&mut x, &self.conv1_b);
        Self::gelu(rt, &step, &mut bufs.gelu_in, &bufs.gelu_out, &mut x)?;
        if let Some(t) = &mut self.taps {
            t.push(("conv1_gelu".into(), x.clone()));
        }
        let a2 = im2col(&transpose(&x, frames, d), d, frames, 3, 2, 1);
        let mut x = Self::gemm(
            rt,
            &r.conv2,
            &mut bufs.conv2_a,
            &bufs.conv2_c,
            &self.conv2_w,
            &a2,
            ctx,
            d * 3,
            d,
            false,
        )?;
        add_row(&mut x, &self.conv2_b);
        Self::gelu(rt, &step, &mut bufs.gelu_in, &bufs.gelu_out, &mut x)?;
        add_into(&mut x, &self.pos_emb[..ctx * d]);

        let hd = d / nh;
        let scale = (hd as f32).powf(-0.25);
        let (mp, np, an) = (step.attn.mp, step.attn.np, step.attn.an);
        let qk_args: Vec<[&Bo; 3]> = (0..nh)
            .map(|h| [&bufs.sub_q[h], &bufs.sub_k[h], &bufs.sub_in[h]])
            .collect();
        let av_args: Vec<[&Bo; 3]> = (0..nh)
            .map(|h| [&bufs.sub_out[h], &bufs.sub_v[h], &bufs.sub_o[h]])
            .collect();

        for l in 0..self.n_layer {
            if let Some(t) = &mut self.taps {
                if l < 2 {
                    t.push((["block0_in", "block1_in"][l].into(), x.clone()));
                }
            }
            let hp = &self.host[l];
            let w = &self.dev_w[l];
            let h = layer_norm(&x, d, &hp.attn_ln_w, &hp.attn_ln_b);

            // q, k, v: one staged A, three resident B, bias folded
            stage_bf16(
                bufs.proj_a.as_u16_mut(),
                r.proj.mp,
                r.proj.kp,
                &h,
                ctx,
                d,
                Some(d),
            );
            bufs.proj_a.to_device()?;
            for (wb, c) in [&w.q, &w.k, &w.v].into_iter().zip(&bufs.proj_c) {
                rt.run(&r.proj.design, &[&bufs.proj_a, wb, c])?;
            }
            for c in &bufs.proj_c[..3] {
                c.from_device()?;
            }
            let proj_np = r.proj.np;
            {
                if let Some(t) = &mut self.taps {
                    for (name, c) in ["q", "k", "v"].iter().zip(&bufs.proj_c) {
                        t.push((
                            format!("l{}_{}", l, name),
                            unpad_f32(c.as_f32(), proj_np, ctx, d),
                        ));
                    }
                }
            }
            let (qc, kc, vc) = (
                bufs.proj_c[0].as_f32(),
                bufs.proj_c[1].as_f32(),
                bufs.proj_c[2].as_f32(),
            );

            // attention chain: q, k scaled by hd^-0.25, stacked per head
            {
                let qs = bufs.q.as_u16_mut();
                for hh in 0..nh {
                    for t in 0..ctx {
                        let src = &qc[t * proj_np + hh * hd..t * proj_np + (hh + 1) * hd];
                        let dst = &mut qs[hh * mp * hd + t * hd..hh * mp * hd + (t + 1) * hd];
                        for (o, &v) in dst.iter_mut().zip(src) {
                            *o = bf16(v * scale);
                        }
                    }
                }
            }
            {
                let ks = bufs.k.as_u16_mut();
                for t in 0..ctx {
                    for hh in 0..nh {
                        for j in 0..hd {
                            ks[hh * hd * np + j * np + t] =
                                bf16(kc[t * proj_np + hh * hd + j] * scale);
                        }
                    }
                }
            }
            bufs.q.to_device()?;
            bufs.k.to_device()?;
            rt.execute_list(&r.qk.design, step.ctx, &qk_args)?;
            // V is only read by AV: stage it while QK runs
            {
                let vs = bufs.v.as_u16_mut();
                for hh in 0..nh {
                    for t in 0..ctx {
                        let src = &vc[t * proj_np + hh * hd..t * proj_np + (hh + 1) * hd];
                        let dst = &mut vs[hh * mp * an + t * an..hh * mp * an + t * an + hd];
                        for (o, &v) in dst.iter_mut().zip(src) {
                            *o = bf16(v);
                        }
                    }
                }
            }
            bufs.v.to_device()?;
            rt.wait_list(&r.qk.design)?;
            rt.run(&r.softmax.design, &[&bufs.s_in, &bufs.s_out])?;
            rt.execute_list(&r.av.design, step.ctx, &av_args)?;
            rt.wait_list(&r.av.design)?;
            bufs.o.from_device()?;
            let mut o = vec![0f32; ctx * d];
            {
                let os = bufs.o.as_f32();
                for t in 0..ctx {
                    for hh in 0..nh {
                        let src = &os[hh * mp * an + t * an..hh * mp * an + t * an + hd];
                        o[t * d + hh * hd..t * d + (hh + 1) * hd].copy_from_slice(src);
                    }
                }
            }

            {
                if let Some(t) = &mut self.taps {
                    t.push((format!("l{}_o", l), o.clone()));
                }
            }
            let y = Self::gemm(
                rt,
                &r.proj,
                &mut bufs.proj_a,
                &bufs.proj_c[0],
                &w.out,
                &o,
                ctx,
                d,
                d,
                true,
            )?;
            add_into(&mut x, &y);
            {
                if let Some(t) = &mut self.taps {
                    t.push((format!("l{}_x_attn", l), x.clone()));
                }
            }

            let h = layer_norm(&x, d, &hp.mlp_ln_w, &hp.mlp_ln_b);
            let y = if let Some(g) = &r.fc1_gelu {
                // fc1 in column chunks with the bias folded, each GELU'd on
                // the device straight into its band of fc2's A, then fc2
                stage_bf16(
                    bufs.proj_a.as_u16_mut(),
                    r.proj.mp,
                    r.proj.kp,
                    &h,
                    ctx,
                    d,
                    Some(d),
                );
                bufs.proj_a.to_device()?;
                let runs = w
                    .fc1
                    .iter()
                    .zip(&bufs.proj_c)
                    .map(|(wb, c)| rt.start(&r.proj.design, &[&bufs.proj_a, wb, c]))
                    .collect::<Result<Vec<_>, _>>()?;
                for run in &runs {
                    run.wait(RUN_TIMEOUT_MS, "fc1")?;
                }
                drop(runs);
                let runs = bufs
                    .proj_c
                    .iter()
                    .zip(&bufs.bands)
                    .map(|(c, band)| rt.start(&g.design, &[c, band]))
                    .collect::<Result<Vec<_>, _>>()?;
                for run in &runs {
                    run.wait(RUN_TIMEOUT_MS, "fc1_gelu")?;
                }
                drop(runs);
                rt.run(&r.fc2.design, &[&bufs.fc2_a, &w.fc2, &bufs.fc2_c])?;
                bufs.fc2_c.from_device()?;
                unpad_f32(bufs.fc2_c.as_f32(), r.fc2.np, ctx, d)
            } else {
                let fc1 = r.fc1.as_ref().unwrap();
                let d_ff = hp.fc1_b.len();
                let mut f = Self::gemm(
                    rt,
                    fc1,
                    bufs.fc1_a.as_mut().unwrap(),
                    bufs.fc1_c.as_ref().unwrap(),
                    &w.fc1[0],
                    &h,
                    ctx,
                    d,
                    d_ff,
                    false,
                )?;
                add_row(&mut f, &hp.fc1_b);
                {
                    if let Some(t) = &mut self.taps {
                        t.push((format!("l{}_fc1", l), f.clone()));
                    }
                }
                Self::gelu(rt, &step, &mut bufs.gelu_in, &bufs.gelu_out, &mut f)?;
                {
                    if let Some(t) = &mut self.taps {
                        t.push((format!("l{}_gelu", l), f.clone()));
                    }
                }
                Self::gemm(
                    rt,
                    &r.fc2,
                    &mut bufs.fc2_a,
                    &bufs.fc2_c,
                    &w.fc2,
                    &f,
                    ctx,
                    d_ff,
                    d,
                    false,
                )?
            };
            add_into(&mut x, &y);
            {
                if let Some(t) = &mut self.taps {
                    t.push((format!("l{}_x_out", l), x.clone()));
                }
            }
            add_row(&mut x, &hp.fc2_b);
        }

        Ok(layer_norm(&x, d, &self.ln_post_w, &self.ln_post_b))
    }
}
