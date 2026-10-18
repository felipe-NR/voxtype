//! Kernel packs (`xdna-whisper-pack/1`): the compiled NPU overlays for one
//! Whisper encoder geometry on one NPU generation, plus a manifest.
//!
//! The format and the rules a loader enforces are whisper-xdna's
//! docs/pack-format.md and tools/check_pack.py. This module refuses
//! everything that tool refuses.

use crate::error::TranscribeError;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

const SCHEMA_NAME: &str = "xdna-whisper-pack";
const SCHEMA_MAJOR: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub pack_id: String,
    pub model: ModelInfo,
    pub device: DeviceInfo,
    pub memory: MemoryInfo,
    pub abi: Abi,
    pub selection: Selection,
    pub ladder: Vec<Step>,
    pub designs: BTreeMap<String, Design>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub n_mels: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub weight_types: Vec<String>,
    pub validated_with: ValidatedWith,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ValidatedWith {
    pub file: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeviceInfo {
    pub generation: String,
    pub pci: Vec<String>,
    pub vbnv: String,
    pub max_hw_contexts: usize,
    pub xrt_min: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemoryInfo {
    pub heap_bytes: u64,
    pub max_bo_bytes: u64,
    pub memlock_min_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Abi {
    pub opcode: u32,
    pub insts_group_id: i32,
    pub gemm: AbiKind,
    pub rowop: AbiKind,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AbiKind {
    pub group_ids: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Selection {
    pub positions_per_sec: f64,
    pub margin_ratio: f64,
    pub margin_positions: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Step {
    pub ctx: usize,
    pub max_audio_secs: f64,
    pub attn: Attn,
    pub mlp: Mlp,
    pub roles: Roles,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Attn {
    #[serde(rename = "Mp")]
    pub mp: usize,
    #[serde(rename = "Np")]
    pub np: usize,
    #[serde(rename = "aN")]
    pub an: usize,
    pub softmax_window: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mlp {
    pub path: String,
    #[serde(default)]
    pub chunks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Roles {
    pub conv1: Gemm,
    pub conv2: Gemm,
    pub gelu: Rowop,
    pub proj: Gemm,
    pub qk: Gemm,
    pub softmax: Rowop,
    pub av: Gemm,
    pub fc2: Gemm,
    /// `plain` MLP path only.
    pub fc1: Option<Gemm>,
    /// `fused` MLP path only.
    pub fc1_gelu: Option<Rowop>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A GEMM role: `c = a @ b`, operands padded row-major.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemm {
    pub design: String,
    #[serde(rename = "Mp")]
    pub mp: usize,
    #[serde(rename = "Kp")]
    pub kp: usize,
    #[serde(rename = "Np")]
    pub np: usize,
    #[serde(default)]
    pub bias_folded: bool,
    pub a: Operand,
    pub b: Operand,
    pub c: Operand,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Operand {
    pub layout: String,
    pub rows: usize,
    pub cols: usize,
    #[serde(default)]
    pub head_stride_bytes: usize,
}

/// A row-wise role (GELU, softmax).
#[derive(Debug, Clone, Deserialize)]
pub struct Rowop {
    pub design: String,
    pub rows: usize,
    pub cols: usize,
    #[serde(default)]
    pub out_cols: usize,
    #[serde(default)]
    pub valid_cols: usize,
    #[serde(default)]
    pub chunks: usize,
    #[serde(default)]
    pub head_stride_bytes: usize,
    #[serde(default)]
    pub buffer_bytes: usize,
    pub band: Option<Band>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Band {
    pub offset_bytes_per_chunk: usize,
    pub row_stride_bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Design {
    pub kind: String,
    #[serde(default)]
    pub variant: Option<String>,
    pub xclbin: PackFile,
    pub insts: PackFile,
    pub kernel: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// A validated pack on disk.
#[derive(Debug, Clone)]
pub struct Pack {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

fn invalid(pack: &Path, msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InitFailed(format!(
        "XDNA kernel pack {} is not usable: {}",
        pack.display(),
        msg
    ))
}

fn kind_allows(kind: &str, variant: Option<&str>) -> bool {
    match kind {
        "gemm" => variant.is_none(),
        "softmax" => matches!(variant, Some("s3" | "s4w")),
        "gelu" => matches!(variant, Some("g4r" | "g4f")),
        _ => false,
    }
}

/// `designs/<file>`, relative, no `..`, not a symlink. With `must_exist`
/// unset only the spelling is checked (a manifest without its overlays).
fn check_path(pack: &Path, rel: &str, must_exist: bool) -> Result<PathBuf, String> {
    let p = Path::new(rel);
    let parts: Vec<Component> = p.components().collect();
    let ok = parts.len() == 2
        && parts[0] == Component::Normal("designs".as_ref())
        && matches!(parts[1], Component::Normal(_));
    if !ok {
        return Err(format!("{}: path must be designs/<file>", rel));
    }
    let full = pack.join(p);
    if !must_exist {
        return match std::fs::symlink_metadata(&full) {
            Ok(md) if md.file_type().is_symlink() => Err(format!("{}: symlink", rel)),
            _ => Ok(full),
        };
    }
    match std::fs::symlink_metadata(&full) {
        Ok(md) if md.file_type().is_symlink() => Err(format!("{}: symlink", rel)),
        Ok(md) if md.is_file() => Ok(full),
        Ok(_) => Err(format!("{}: not a regular file", rel)),
        Err(e) => Err(format!("{}: {}", rel, e)),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

impl Pack {
    /// Read and validate `<dir>/manifest.json` and every file it names.
    pub fn load(dir: &Path) -> Result<Self, TranscribeError> {
        let text = std::fs::read_to_string(dir.join("manifest.json"))
            .map_err(|e| invalid(dir, format!("manifest.json: {}", e)))?;
        let manifest: Manifest = serde_json::from_str(&text)
            .map_err(|e| invalid(dir, format!("manifest.json: {}", e)))?;
        let pack = Self {
            dir: dir.to_path_buf(),
            manifest,
        };
        pack.validate(true)?;
        Ok(pack)
    }

    /// The structural rules, plus file sizes and hashes when `files` is set.
    pub fn validate(&self, files: bool) -> Result<(), TranscribeError> {
        let m = &self.manifest;
        let dir = &self.dir;
        let mut errors: Vec<String> = Vec::new();

        let (name, major) = m.schema.split_once('/').unwrap_or((m.schema.as_str(), ""));
        if name != SCHEMA_NAME || major.parse::<u32>().ok() != Some(SCHEMA_MAJOR) {
            return Err(invalid(
                dir,
                format!(
                    "unknown schema {:?}; this voxtype reads {}/{}",
                    m.schema, SCHEMA_NAME, SCHEMA_MAJOR
                ),
            ));
        }

        for (dn, d) in &m.designs {
            if !kind_allows(&d.kind, d.variant.as_deref()) {
                errors.push(format!(
                    "{}: unknown kind/variant {}/{:?}",
                    dn, d.kind, d.variant
                ));
            }
            for f in [&d.xclbin, &d.insts] {
                match check_path(dir, &f.path, files) {
                    Err(e) => errors.push(e),
                    Ok(full) if files => match std::fs::read(&full) {
                        Err(e) => errors.push(format!("{}: {}", f.path, e)),
                        Ok(bytes) if bytes.len() as u64 != f.bytes => errors.push(format!(
                            "{}: {} bytes, manifest says {}",
                            f.path,
                            bytes.len(),
                            f.bytes
                        )),
                        Ok(bytes) if sha256_hex(&bytes) != f.sha256 => {
                            errors.push(format!("{}: sha256 mismatch", f.path))
                        }
                        Ok(_) => {}
                    },
                    Ok(_) => {}
                }
            }
        }

        let content: String = m
            .designs
            .values()
            .map(|d| format!("{}{}", d.xclbin.sha256, d.insts.sha256))
            .collect();
        let want_id = format!(
            "{}-{}-{}",
            m.model.name,
            m.device.generation,
            &sha256_hex(content.as_bytes())[..12]
        );
        if m.pack_id != want_id {
            errors.push(format!(
                "pack_id {} is not the hash of its designs ({})",
                m.pack_id, want_id
            ));
        }

        let floor = m.memory.heap_bytes.max(m.memory.max_bo_bytes);
        if m.memory.memlock_min_bytes != floor.next_power_of_two() {
            errors.push(format!(
                "memlock_min_bytes {} does not follow its rule",
                m.memory.memlock_min_bytes
            ));
        }

        let sel = &m.selection;
        let mut used: HashSet<&str> = HashSet::new();
        let mut prev = 0usize;
        for step in &m.ladder {
            let ctx = step.ctx;
            if ctx <= prev || ctx > m.model.n_audio_ctx as usize {
                errors.push(format!(
                    "ladder ctx {}: not ascending or above n_audio_ctx",
                    ctx
                ));
            }
            prev = ctx;
            let secs = if ctx == m.model.n_audio_ctx as usize {
                m.model.n_audio_ctx as f64 / sel.positions_per_sec
            } else {
                (ctx as f64 - sel.margin_positions)
                    / (sel.positions_per_sec * (1.0 + sel.margin_ratio))
            };
            if (step.max_audio_secs - (secs * 1000.0).round() / 1000.0).abs() > 1e-6 {
                errors.push(format!(
                    "ctx {}: max_audio_secs {} != rule {:.3}",
                    ctx, step.max_audio_secs, secs
                ));
            }

            let r = &step.roles;
            if !r.extra.is_empty() {
                errors.push(format!(
                    "ctx {}: unknown roles {:?}",
                    ctx,
                    r.extra.keys().collect::<Vec<_>>()
                ));
            }
            match (step.mlp.path.as_str(), &r.fc1, &r.fc1_gelu) {
                ("plain", Some(_), None) | ("fused", None, Some(_)) => {}
                (path, _, _) => errors.push(format!(
                    "ctx {}: mlp path {:?} does not match its roles",
                    ctx, path
                )),
            }
            let mut roles: Vec<(&str, &str, &str)> = vec![
                ("conv1", r.conv1.design.as_str(), "gemm"),
                ("conv2", r.conv2.design.as_str(), "gemm"),
                ("gelu", r.gelu.design.as_str(), "gelu"),
                ("proj", r.proj.design.as_str(), "gemm"),
                ("qk", r.qk.design.as_str(), "gemm"),
                ("softmax", r.softmax.design.as_str(), "softmax"),
                ("av", r.av.design.as_str(), "gemm"),
                ("fc2", r.fc2.design.as_str(), "gemm"),
            ];
            if let Some(fc1) = &r.fc1 {
                roles.push(("fc1", fc1.design.as_str(), "gemm"));
            }
            if let Some(g) = &r.fc1_gelu {
                roles.push(("fc1_gelu", g.design.as_str(), "gelu"));
            }
            for (role, dn, kind) in roles {
                match m.designs.get(dn) {
                    None => errors.push(format!(
                        "ctx {} {}: design {} not in the pack",
                        ctx, role, dn
                    )),
                    Some(d) if d.kind != kind => errors.push(format!(
                        "ctx {} {}: {} is a {}, want {}",
                        ctx, role, dn, d.kind, kind
                    )),
                    Some(_) => {}
                }
                used.insert(dn);
            }
        }
        for dn in m.designs.keys() {
            if !used.contains(dn.as_str()) {
                errors.push(format!("{}: in the pack but used by no ladder step", dn));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(invalid(dir, errors.join("; ")))
        }
    }

    /// The ladder step whose context is exactly `n_ctx`, if any.
    pub fn step_for_ctx(&self, n_ctx: usize) -> Option<&Step> {
        self.manifest.ladder.iter().find(|s| s.ctx == n_ctx)
    }

    /// The context to run a clip of `secs` seconds at: the smallest step whose
    /// `max_audio_secs` covers it, and the full context from 30 s on.
    pub fn ctx_for_secs(&self, secs: f64) -> usize {
        let m = &self.manifest;
        let sel = &m.selection;
        let need =
            (secs * sel.positions_per_sec * (1.0 + sel.margin_ratio)).ceil() + sel.margin_positions;
        m.ladder
            .iter()
            .map(|s| s.ctx)
            .find(|&c| c as f64 >= need)
            .unwrap_or(m.model.n_audio_ctx as usize)
    }

    pub fn file(&self, f: &PackFile) -> PathBuf {
        self.dir.join(&f.path)
    }

    /// Design names by first use, for deterministic loading order.
    pub fn designs_in_order(&self) -> Vec<String> {
        let mut seen: HashMap<String, ()> = HashMap::new();
        let mut out = Vec::new();
        for step in &self.manifest.ladder {
            for d in step.roles.design_names() {
                if seen.insert(d.to_string(), ()).is_none() {
                    out.push(d.to_string());
                }
            }
        }
        out
    }
}

impl Roles {
    pub fn design_names(&self) -> Vec<&str> {
        let mut v = vec![
            self.conv1.design.as_str(),
            self.gelu.design.as_str(),
            self.conv2.design.as_str(),
            self.proj.design.as_str(),
            self.qk.design.as_str(),
            self.softmax.design.as_str(),
            self.av.design.as_str(),
        ];
        if let Some(fc1) = &self.fc1 {
            v.push(fc1.design.as_str());
        }
        if let Some(g) = &self.fc1_gelu {
            v.push(g.design.as_str());
        }
        v.push(self.fc2.design.as_str());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/xdna")
            .join(name)
    }

    fn manifest_only(name: &str) -> Pack {
        let dir = fixture_dir(name);
        let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        Pack {
            dir,
            manifest: serde_json::from_str(&text).unwrap(),
        }
    }

    #[test]
    fn committed_manifests_parse() {
        for name in [
            "large-v3-turbo-xdna1-5be736d256ac",
            "tiny-xdna1-f25314b6f4eb",
        ] {
            let pack = manifest_only(name);
            assert_eq!(pack.manifest.pack_id, name);
            assert!(!pack.manifest.ladder.is_empty());
        }
    }

    #[test]
    fn ctx_selection_follows_the_rule() {
        let pack = manifest_only("large-v3-turbo-xdna1-5be736d256ac");
        assert_eq!(pack.ctx_for_secs(1.0), 512);
        assert_eq!(pack.ctx_for_secs(9.0), 512);
        assert_eq!(pack.ctx_for_secs(9.1), 768);
        assert_eq!(pack.ctx_for_secs(11.0), 768);
        assert_eq!(pack.ctx_for_secs(20.0), 1280);
        assert_eq!(pack.ctx_for_secs(29.0), 1500);
        assert_eq!(pack.ctx_for_secs(45.0), 1500);
        // max_audio_secs is the rule solved per step, rounded to 1 ms
        for step in &pack.manifest.ladder {
            assert_eq!(pack.ctx_for_secs(step.max_audio_secs - 0.001), step.ctx);
        }
    }

    #[test]
    fn rejects_unknown_schema_major() {
        let mut pack = manifest_only("tiny-xdna1-f25314b6f4eb");
        pack.manifest.schema = "xdna-whisper-pack/2".into();
        assert!(pack.validate(false).is_err());
    }

    #[test]
    fn rejects_unknown_variant() {
        let mut pack = manifest_only("tiny-xdna1-f25314b6f4eb");
        let d = pack
            .manifest
            .designs
            .values_mut()
            .find(|d| d.kind == "gelu")
            .unwrap();
        d.variant = Some("g9".into());
        assert!(pack.validate(false).is_err());
    }

    #[test]
    fn rejects_path_escape() {
        let mut pack = manifest_only("tiny-xdna1-f25314b6f4eb");
        let d = pack.manifest.designs.values_mut().next().unwrap();
        d.xclbin.path = "designs/../manifest.json".into();
        assert!(pack.validate(false).is_err());
    }

    #[test]
    fn rejects_pack_id_mismatch() {
        let mut pack = manifest_only("tiny-xdna1-f25314b6f4eb");
        pack.manifest
            .designs
            .values_mut()
            .next()
            .unwrap()
            .insts
            .sha256 = "00".repeat(32);
        let err = pack.validate(false).unwrap_err().to_string();
        assert!(err.contains("pack_id"), "{}", err);
    }

    #[test]
    fn rejects_memlock_off_rule() {
        let mut pack = manifest_only("tiny-xdna1-f25314b6f4eb");
        pack.manifest.memory.memlock_min_bytes += 1;
        assert!(pack.validate(false).is_err());
    }
}
