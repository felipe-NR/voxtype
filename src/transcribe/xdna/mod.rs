//! Whisper encoder on the AMD XDNA NPU (Ryzen AI, XDNA1 first).
//!
//! whisper.cpp keeps the decoder, the cross-attention K/V, language detection
//! and timestamps; this module produces the encoder output through
//! whisper.cpp's encoder callback. The overlays come from a kernel pack
//! (`pack`), the weights from the same ggml file whisper.cpp loads (`ggml`),
//! and XRT is reached through libvoxtype_xrt, loaded at run time (`shim`).

mod encoder;
pub mod ggml;
mod host;
pub mod pack;
mod runtime;
pub mod shim;

use crate::error::TranscribeError;
use std::ffi::{c_int, c_void};
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use pack::Pack;

/// Everything needed to bring the NPU encoder up.
#[derive(Debug, Clone)]
pub struct XdnaOptions {
    pub pack_dir: PathBuf,
    pub model_path: PathBuf,
    pub shim_path: Option<PathBuf>,
}

/// The soft RLIMIT_MEMLOCK in bytes, None when unlimited.
pub fn memlock_limit() -> Option<u64> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) } != 0 {
        return None;
    }
    (lim.rlim_cur != libc::RLIM_INFINITY).then_some(lim.rlim_cur)
}

/// Whether this process holds CAP_IPC_LOCK, which lifts the memlock limit.
pub fn has_ipc_lock() -> bool {
    const CAP_IPC_LOCK: u32 = 14;
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("CapEff:"))
                .and_then(|v| u64::from_str_radix(v.trim(), 16).ok())
        })
        .is_some_and(|caps| caps & (1 << CAP_IPC_LOCK) != 0)
}

/// Why the memlock limit is too low for `need` bytes, or None if it is fine.
pub fn memlock_shortfall(need: u64) -> Option<String> {
    if has_ipc_lock() {
        return None;
    }
    let limit = memlock_limit()?;
    (limit < need).then(|| {
        format!(
            "RLIMIT_MEMLOCK is {} KiB but the NPU encoder needs {} KiB per buffer.\n  \
             XRT locks every buffer it maps, and the kernel checks each mapping against this limit.\n  \
             Raise it for your session (as root), then log out and back in:\n    \
             mkdir -p /etc/systemd/system/user@.service.d\n    \
             printf '[Service]\\nLimitMEMLOCK={}\\n' > /etc/systemd/system/user@.service.d/voxtype-npu.conf\n    \
             printf '* - memlock {}\\n' > /etc/security/limits.d/voxtype-npu.conf",
            limit / 1024,
            need / 1024,
            need,
            need / 1024
        )
    })
}

fn parse_version(s: &str) -> (u32, u32, u32) {
    let mut it = s
        .split(|c: char| !c.is_ascii_digit())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse().unwrap_or(0));
    (
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
    )
}

struct Inner {
    enc: encoder::Encoder,
    /// The last window: whisper.cpp encodes it twice when it detects the
    /// language first (detection, then the transcription).
    last: Option<(usize, u64, Vec<f32>)>,
}

/// The NPU encoder, shared with whisper.cpp through the callback.
pub struct XdnaEncoder {
    inner: Mutex<Inner>,
    pack: Pack,
    n_state: usize,
    n_mels: usize,
}

fn init(msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InitFailed(format!("NPU encoder: {}", msg))
}

fn mel_hash(mel: &[f32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in mel {
        h.write_u32(v.to_bits());
    }
    h.finish()
}

impl XdnaEncoder {
    pub fn load(opts: &XdnaOptions) -> Result<Self, TranscribeError> {
        let t0 = std::time::Instant::now();
        let pack = Pack::load(&opts.pack_dir)?;
        let m = &pack.manifest;

        let ggml = ggml::GgmlEncoder::open(&opts.model_path)?;
        let hp = ggml.hparams;
        let checks = [
            ("n_mels", hp.n_mels, m.model.n_mels),
            ("n_audio_ctx", hp.n_audio_ctx, m.model.n_audio_ctx),
            ("n_audio_state", hp.n_audio_state, m.model.n_audio_state),
            ("n_audio_head", hp.n_audio_head, m.model.n_audio_head),
            ("n_audio_layer", hp.n_audio_layer, m.model.n_audio_layer),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(init(format!(
                    "{} has {}={}, but pack {} is for {} ({}={}). Pick the pack that matches the model.",
                    opts.model_path.display(),
                    name,
                    got,
                    m.pack_id,
                    m.model.name,
                    name,
                    want
                )));
            }
        }
        let mut bad: Vec<&str> = ggml
            .tensors
            .values()
            .map(|t| t.ttype.name())
            .filter(|t| *t != "f32" && !m.model.weight_types.iter().any(|w| w == t))
            .collect();
        bad.sort();
        bad.dedup();
        if !bad.is_empty() {
            return Err(init(format!(
                "{} has encoder weights of type {:?}; pack {} accepts {:?}",
                opts.model_path.display(),
                bad,
                m.pack_id,
                m.model.weight_types
            )));
        }

        if let Some(why) = memlock_shortfall(m.memory.memlock_min_bytes) {
            return Err(init(why));
        }

        let shim = shim::Shim::load(opts.shim_path.as_deref())?;
        let (a, b, c) = shim.xrt_version()?;
        if (a as u32, b as u32, c as u32) < parse_version(&m.device.xrt_min) {
            return Err(init(format!(
                "XRT {}.{}.{} is older than {}, the lowest this pack was tested with",
                a, b, c, m.device.xrt_min
            )));
        }
        let dev = shim::Device::open(&shim, 0).map_err(|e| {
            init(format!(
                "{}\n  Is the amdxdna driver loaded (ls /dev/accel/accel0)?",
                e
            ))
        })?;
        let name = dev.name()?;
        if name != m.device.vbnv {
            return Err(init(format!(
                "pack {} is for {} ({}), this device is {}",
                m.pack_id, m.device.vbnv, m.device.generation, name
            )));
        }
        tracing::info!(
            "NPU encoder: {} via XRT {}.{}.{} ({}), pack {}",
            name,
            a,
            b,
            c,
            shim.path.display(),
            m.pack_id
        );

        let rt = runtime::Runtime::new(dev, &pack)?;
        let mut enc = encoder::Encoder::new(pack.clone(), rt, &ggml)?;
        let top = m.ladder.last().map(|s| s.ctx).unwrap_or(0);
        enc.preallocate(top)?;
        tracing::info!(
            "NPU encoder ready in {:.2}s ({} layers resident, ladder {:?})",
            t0.elapsed().as_secs_f32(),
            m.model.n_audio_layer,
            m.ladder.iter().map(|s| s.ctx).collect::<Vec<_>>()
        );
        let n_state = m.model.n_audio_state as usize;
        let n_mels = m.model.n_mels as usize;
        Ok(Self {
            inner: Mutex::new(Inner { enc, last: None }),
            pack,
            n_state,
            n_mels,
        })
    }

    pub fn pack(&self) -> &Pack {
        &self.pack
    }

    /// The encoder context to run a clip of `secs` seconds at.
    pub fn ctx_for_secs(&self, secs: f64) -> usize {
        self.pack.ctx_for_secs(secs)
    }

    /// Encode one window, or None to decline it (not a ladder step).
    pub fn encode(&self, mel: &[f32], n_ctx: usize) -> Result<Option<Vec<f32>>, TranscribeError> {
        if self.pack.step_for_ctx(n_ctx).is_none() {
            return Ok(None);
        }
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| TranscribeError::InferenceFailed("NPU encoder: poisoned lock".into()))?;
        let hash = mel_hash(mel);
        if let Some((ctx, h, out)) = &inner.last {
            if *ctx == n_ctx && *h == hash {
                tracing::debug!("NPU encoder: window already encoded, reusing it");
                return Ok(Some(out.clone()));
            }
        }
        let t0 = std::time::Instant::now();
        let creates = inner.enc.ctx_creates();
        let out = inner.enc.encode(mel, n_ctx)?;
        tracing::debug!(
            "NPU encoder: ctx {} in {:.0} ms ({} context creates)",
            n_ctx,
            t0.elapsed().as_secs_f64() * 1e3,
            inner.enc.ctx_creates() - creates
        );
        inner.last = Some((n_ctx, hash, out.clone()));
        Ok(Some(out))
    }

    /// The whisper.cpp encoder callback; `user_data` is `&XdnaEncoder`.
    ///
    /// # Safety
    /// Called by whisper.cpp with the pointers its contract describes.
    pub unsafe extern "C" fn callback(
        mel: *const f32,
        n_mel: c_int,
        n_ctx: c_int,
        out: *mut f32,
        n_state: c_int,
        user_data: *mut c_void,
    ) -> c_int {
        let this = &*(user_data as *const XdnaEncoder);
        if n_mel as usize != this.n_mels || n_state as usize != this.n_state || n_ctx <= 0 {
            return 1;
        }
        let n_ctx = n_ctx as usize;
        let mel = std::slice::from_raw_parts(mel, this.n_mels * 2 * n_ctx);
        match this.encode(mel, n_ctx) {
            Ok(Some(enc)) => {
                std::ptr::copy_nonoverlapping(enc.as_ptr(), out, n_ctx * this.n_state);
                0
            }
            Ok(None) => {
                tracing::warn!(
                    "NPU encoder: no ladder step for audio_ctx {}; whisper.cpp encodes this window",
                    n_ctx
                );
                1
            }
            Err(e) => {
                tracing::error!("{}; whisper.cpp encodes this window on the CPU instead", e);
                1
            }
        }
    }
}

/// The default kernel pack directory for a model, if one is installed.
pub fn find_pack(model: &str, pack_root: Option<&Path>) -> Option<PathBuf> {
    let roots: Vec<PathBuf> = match pack_root {
        Some(r) => vec![r.to_path_buf()],
        None => vec![crate::config::Config::models_dir().join("npu-packs")],
    };
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let prefix = format!("{}-xdna", model);
        let mut hits: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&prefix))
                    && p.join("manifest.json").is_file()
            })
            .collect();
        hits.sort();
        if let Some(p) = hits.pop() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compare() {
        assert!(parse_version("2.25.0") >= parse_version("2.21.75"));
        assert!(parse_version("2.19.0") < parse_version("2.21.75"));
    }
}

/// On-device check against whisper-xdna's whisper-tiny fixtures. Needs the
/// NPU, so it is ignored by default:
///
/// ```text
/// VOXTYPE_XDNA_PACK=~/npu-packs/tiny-xdna1-f25314b6f4eb \
/// VOXTYPE_XDNA_MODEL=~/.local/share/voxtype/models/ggml-tiny.bin \
/// VOXTYPE_XDNA_SHIM=<dir>/lib/x86_64-linux-gnu/libvoxtype_xrt.so \
///   cargo test --features xdna-npu xdna::device_tests -- --ignored
/// ```
#[cfg(test)]
mod device_tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    #[test]
    #[ignore]
    fn tiny_pass_matches_fixtures() {
        let env = |k: &str| std::env::var(k).map(PathBuf::from).expect(k);
        let opts = XdnaOptions {
            pack_dir: env("VOXTYPE_XDNA_PACK"),
            model_path: env("VOXTYPE_XDNA_MODEL"),
            shim_path: std::env::var("VOXTYPE_XDNA_SHIM").ok().map(PathBuf::from),
        };
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/xdna/tiny-xdna1-f25314b6f4eb/fixtures/ctx512");
        let enc = XdnaEncoder::load(&opts).expect("load");
        let (shape, mel) = host::read_npy_f32(&fixtures.join("mel.npy"));
        assert_eq!(shape, vec![80, 1024]);
        let repeats: usize = std::env::var("VOXTYPE_XDNA_REPEAT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        for _ in 1..repeats {
            let mut inner = enc.inner.lock().unwrap();
            inner.enc.taps = Some(Vec::new());
            inner.enc.encode(&mel, 512).expect("encode");
            for (name, got) in inner.enc.taps.take().unwrap() {
                if name.starts_with('l') {
                    continue;
                }
                let (_, want) = host::read_npy_f32(&fixtures.join(format!("{}.npy", name)));
                let max_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprint!("{} {:.6}  ", name, max_abs);
            }
            eprintln!();
        }
        let out = {
            let mut inner = enc.inner.lock().unwrap();
            inner.enc.taps = Some(Vec::new());
            let out = inner.enc.encode(&mel, 512).expect("encode");
            for (name, got) in inner.enc.taps.take().unwrap() {
                if let Ok(dir) = std::env::var("VOXTYPE_XDNA_DUMP") {
                    let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(PathBuf::from(dir).join(format!("{}.f32", name)), bytes)
                        .unwrap();
                }
                if name.starts_with('l') {
                    continue;
                }
                let (_, want) = host::read_npy_f32(&fixtures.join(format!("{}.npy", name)));
                let c = cosine(&got, &want);
                let max_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("{}: cosine {:.8}, max |err| {:.6}", name, c, max_abs);
                assert!(c > 0.999, "{}: cosine {}", name, c);
            }
            out
        };
        let (_, want) = host::read_npy_f32(&fixtures.join("out.npy"));
        let c = cosine(&out, &want);
        let max_abs = out
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("out: cosine {:.6}, max |err| {:.5}", c, max_abs);
        assert!(c > 0.999, "out cosine {}", c);
    }
}
