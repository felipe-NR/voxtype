//! NPU backend management for voxtype
//!
//! Intel NPUs run the OpenVINO engine: `engine = "openvino"`, with the binary
//! compiled with the `openvino-whisper` feature. AMD XDNA NPUs (Ryzen AI) run
//! the Whisper encoder through `[whisper] encoder_device = "npu"`, with the
//! `xdna-npu` feature. Both are config-driven; unlike GPU, no binary switch.

use crate::config::Config;
use std::path::Path;

const DEFAULT_OPENVINO_MODEL: &str = "base.en-int8";

/// Check if NPU hardware is present (/dev/accel/accel* devices)
fn detect_npu_hardware() -> bool {
    Path::new("/dev/accel").is_dir()
        && std::fs::read_dir("/dev/accel")
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("accel"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
}

/// Check if the intel-npu-driver kernel module is loaded
fn check_npu_driver() -> bool {
    Path::new("/sys/module/intel_vpu").exists()
}

/// Check if an OpenVINO model is already downloaded
fn has_openvino_model() -> bool {
    let models_dir = Config::models_dir();
    if let Some(dir_name) = super::model::openvino_dir_name(DEFAULT_OPENVINO_MODEL) {
        let model_path = models_dir.join(dir_name);
        super::model::validate_openvino_model(&model_path).is_ok()
    } else {
        false
    }
}

/// Update the engine field in config.toml to a specific value, preserving everything else
fn update_engine_in_config(content: &str, engine: &str) -> String {
    let mut result = String::new();
    let mut engine_updated = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("engine") && trimmed.contains('=') && !trimmed.starts_with('[') {
            result.push_str(&format!("engine = \"{}\"\n", engine));
            engine_updated = true;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    // If no engine line existed, don't add one -- set_openvino_config handles that for enable
    if !engine_updated && engine == "whisper" {
        // Nothing to do: there was no engine line, and the default is already whisper
    }

    // Preserve trailing newline behavior
    if !content.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// True when the NPU is an AMD XDNA one (the amdxdna driver is loaded).
fn is_amd_xdna() -> bool {
    Path::new("/sys/module/amdxdna").exists()
}

/// Show NPU hardware and configuration status
pub fn show_status() {
    if is_amd_xdna() {
        return amd::show_status();
    }
    println!("NPU Status\n");

    // Hardware detection
    println!("Hardware:");
    if detect_npu_hardware() {
        super::print_success("NPU device detected (/dev/accel/)");
    } else {
        super::print_failure("No NPU device found (/dev/accel/)");
    }

    if check_npu_driver() {
        super::print_success("intel_vpu kernel module loaded");
    } else {
        super::print_failure("intel_vpu kernel module not loaded");
        super::print_info("Install intel-npu-driver and reboot");
    }

    // Feature check
    println!("\nBuild:");
    if cfg!(feature = "openvino-whisper") {
        super::print_success("openvino-whisper feature compiled in");
    } else {
        super::print_failure("openvino-whisper feature not compiled");
        super::print_info(
            "Install a build with openvino-whisper enabled, or rebuild with --features openvino-whisper",
        );
    }

    // Config check
    println!("\nConfiguration:");
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                let engine_is_openvino = content.lines().any(|line| {
                    let trimmed = line.trim();
                    trimmed.starts_with("engine")
                        && trimmed.contains("openvino")
                        && !trimmed.starts_with('#')
                        && !trimmed.starts_with('[')
                });
                if engine_is_openvino {
                    super::print_success("Engine set to OpenVINO");
                } else {
                    super::print_info("Engine is not set to OpenVINO");
                }
            }
        }
    }

    // Model check
    if has_openvino_model() {
        super::print_success(&format!(
            "Default OpenVINO model ({}) installed",
            DEFAULT_OPENVINO_MODEL
        ));
    } else {
        super::print_info(&format!(
            "Default OpenVINO model ({}) not installed",
            DEFAULT_OPENVINO_MODEL
        ));
    }
}

/// Enable NPU acceleration
pub fn enable() -> anyhow::Result<()> {
    if is_amd_xdna() {
        return amd::enable();
    }
    #[cfg(not(feature = "openvino-whisper"))]
    {
        anyhow::bail!(
            "NPU support requires the openvino-whisper feature.\n\
             Install a build with openvino-whisper enabled, or rebuild with:\n  \
             cargo build --features openvino-whisper"
        );
    }

    #[cfg(feature = "openvino-whisper")]
    {
        println!("Enabling NPU acceleration (OpenVINO)...\n");

        // Check hardware (warn, don't fail)
        if detect_npu_hardware() {
            super::print_success("NPU device detected");
        } else {
            super::print_warning(
                "No NPU device found at /dev/accel/. OpenVINO will fall back to CPU.\n       \
                 If you have an Intel NPU, ensure intel-npu-driver is installed and reboot.",
            );
        }

        if !check_npu_driver() {
            super::print_warning(
                "intel_vpu kernel module not loaded.\n       \
                 Install intel-npu-driver and reboot to use NPU hardware.",
            );
        }

        // Download default model if needed
        if !has_openvino_model() {
            println!();
            super::model::download_openvino_model(DEFAULT_OPENVINO_MODEL)?;
        } else {
            super::print_success(&format!(
                "OpenVINO model '{}' already installed",
                DEFAULT_OPENVINO_MODEL
            ));
        }

        // Update config to use OpenVINO engine on the NPU. Persisting the
        // device matters: the compile below caches an NPU blob, and a config
        // left on another device would never use it.
        super::model::set_openvino_config(DEFAULT_OPENVINO_MODEL)?;
        super::model::set_openvino_device("NPU")?;
        super::print_success("Config updated: engine = \"openvino\", device = \"NPU\"");
        super::model::print_openvino_installation_guidance("NPU");

        // Compile the NPU cache blob now that the config opts into NPU, so
        // the first recording doesn't absorb a multi-minute wait. Skipped
        // when the blob already exists; warns and continues on failure.
        let config = crate::config::load_config(None).unwrap_or_default();
        super::model::prepare_openvino_model(DEFAULT_OPENVINO_MODEL, &config);

        println!();
        println!("NPU acceleration enabled (OpenVINO engine).");
        println!();
        println!("Restart voxtype to apply:");
        println!("  systemctl --user restart voxtype");

        Ok(())
    }
}

/// Disable NPU acceleration (revert to Whisper engine)
pub fn disable() -> anyhow::Result<()> {
    if is_amd_xdna() {
        return amd::disable();
    }
    if let Some(config_path) = Config::default_path() {
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let updated = update_engine_in_config(&content, "whisper");
            std::fs::write(&config_path, updated)?;
            super::print_success("Config updated: engine = \"whisper\"");
        } else {
            super::print_info("No config file found, nothing to disable");
        }
    } else {
        anyhow::bail!("Could not determine config path");
    }

    println!();
    println!("NPU acceleration disabled (reverted to Whisper engine).");
    println!("The [openvino] config section has been preserved for easy re-enable.");
    println!();
    println!("Restart voxtype to apply:");
    println!("  systemctl --user restart voxtype");

    Ok(())
}

/// AMD XDNA (Ryzen AI): the Whisper encoder on the NPU, the decoder on the CPU.
mod amd {
    use crate::config::{Config, EncoderDevice};
    use std::path::{Path, PathBuf};

    /// Where prebuilt kernel packs are published. FORK ONLY: a release of
    /// felipe-NR/voxtype; upstream would serve packs from the model CDN.
    const PACK_RELEASE: &str =
        "https://github.com/felipe-NR/voxtype/releases/download/xdna-packs-1";

    /// The published pack for a model, if there is one.
    fn published_pack(model: &str) -> Option<&'static str> {
        match model {
            "large-v3-turbo" => Some("large-v3-turbo-xdna1-5be736d256ac"),
            "tiny" => Some("tiny-xdna1-f25314b6f4eb"),
            _ => None,
        }
    }

    /// How to install the pack for `model`, or why there is none.
    fn pack_hint(model: &str) -> String {
        let root = Config::models_dir().join("npu-packs");
        match published_pack(model) {
            Some(id) => format!(
                "Download it:\n         mkdir -p {root}\n         \
                 curl -L {PACK_RELEASE}/{id}.tar.gz | tar -xz -C {root}",
                root = root.display()
            ),
            None => format!(
                "No published pack for {}; packs exist for large-v3-turbo and tiny.\n         \
                 Other models need one exported with whisper-xdna (tools/export_pack.py).",
                model
            ),
        }
    }

    fn read_sys(p: &str) -> Option<String> {
        std::fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// "1022:1502" for the first accel device, if any.
    fn pci_id() -> Option<String> {
        let v = read_sys("/sys/class/accel/accel0/device/vendor")?;
        let d = read_sys("/sys/class/accel/accel0/device/device")?;
        Some(format!(
            "{}:{}",
            v.trim_start_matches("0x"),
            d.trim_start_matches("0x")
        ))
    }

    fn generation(pci: &str) -> &'static str {
        match pci {
            "1022:1502" => "XDNA1 (Phoenix / Hawk Point)",
            "1022:17f0" => "XDNA2 (Strix / Krackan)",
            _ => "unknown XDNA",
        }
    }

    fn expand(p: &str) -> PathBuf {
        match p.strip_prefix("~/") {
            Some(rest) => dirs::home_dir()
                .map(|h| h.join(rest))
                .unwrap_or_else(|| PathBuf::from(p)),
            None => PathBuf::from(p),
        }
    }

    /// The pack the configured model would use: npu_pack, else the newest
    /// <model>-xdna* under <models dir>/npu-packs.
    fn configured_pack(config: &Config) -> Option<PathBuf> {
        if let Some(p) = &config.whisper.npu_pack {
            return Some(expand(p));
        }
        let root = Config::models_dir().join("npu-packs");
        let prefix = format!("{}-xdna", config.whisper.model);
        let mut hits: Vec<PathBuf> = std::fs::read_dir(&root)
            .ok()?
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
        hits.pop()
    }

    /// memlock_min_bytes from a pack's manifest, without validating the pack.
    fn pack_memlock(pack: &Path) -> Option<u64> {
        let text = std::fs::read_to_string(pack.join("manifest.json")).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        v["memory"]["memlock_min_bytes"].as_u64()
    }

    fn memlock_soft() -> Option<u64> {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) } != 0 {
            return None;
        }
        (lim.rlim_cur != libc::RLIM_INFINITY).then_some(lim.rlim_cur)
    }

    fn memlock_fix(need: u64) {
        super::super::print_info(&format!(
            "Raise it (as root), then log out and back in:\n         \
             sudo mkdir -p /etc/systemd/system/user@.service.d\n         \
             printf '[Service]\\nLimitMEMLOCK={}\\n' | sudo tee /etc/systemd/system/user@.service.d/voxtype-npu.conf\n         \
             echo '* - memlock {}' | sudo tee /etc/security/limits.d/voxtype-npu.conf",
            need,
            need / 1024
        ));
    }

    fn shim_path(config: &Config) -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> =
            config.whisper.npu_shim.iter().map(|p| expand(p)).collect();
        if candidates.is_empty() {
            if let Some(home) = dirs::home_dir() {
                candidates.push(
                    home.join(".local/lib/voxtype/xdna/lib/x86_64-linux-gnu/libvoxtype_xrt.so"),
                );
            }
            candidates.push(PathBuf::from(
                "/usr/lib/voxtype/xdna/lib/x86_64-linux-gnu/libvoxtype_xrt.so",
            ));
        }
        candidates.into_iter().find(|p| p.is_file())
    }

    pub fn show_status() {
        use super::super::{print_failure, print_info, print_success, print_warning};
        let config = crate::config::load_config(None).unwrap_or_default();
        println!("NPU Status (AMD XDNA)\n");

        println!("Hardware:");
        match pci_id() {
            Some(id) => print_success(&format!(
                "{} at /dev/accel/accel0 [{}]",
                generation(&id),
                id
            )),
            None => print_failure("No accel device at /dev/accel/accel0"),
        }
        print_success("amdxdna kernel module loaded");

        println!("\nBuild:");
        if cfg!(feature = "xdna-npu") {
            print_success("xdna-npu feature compiled in");
        } else {
            print_failure("xdna-npu feature not compiled");
            print_info("Rebuild with --features xdna-npu");
        }

        println!("\nRuntime:");
        match shim_path(&config) {
            Some(p) => {
                print_success(&format!("libvoxtype_xrt.so: {}", p.display()));
                #[cfg(feature = "xdna-npu")]
                match crate::transcribe::xdna::shim::Shim::load(Some(&p))
                    .and_then(|s| s.xrt_version())
                {
                    Ok((a, b, c)) => print_success(&format!("XRT {}.{}.{}", a, b, c)),
                    Err(e) => print_failure(&format!("cannot load the shim: {}", e)),
                }
            }
            None => {
                print_failure("libvoxtype_xrt.so not found");
                print_info(
                    "Build it: docker build -f xdna-shim/Dockerfile --output type=local,dest=$HOME/.local/lib/voxtype/xdna xdna-shim",
                );
            }
        }

        println!("\nKernel pack (model {}):", config.whisper.model);
        let pack = configured_pack(&config);
        match &pack {
            Some(p) if p.join("manifest.json").is_file() => {
                print_success(&format!("{}", p.display()));
            }
            Some(p) => print_failure(&format!("{}: no manifest.json", p.display())),
            None => {
                print_failure(&format!(
                    "no pack for {} (set [whisper] npu_pack, or put it under {})",
                    config.whisper.model,
                    Config::models_dir().join("npu-packs").display()
                ));
                print_info(&pack_hint(&config.whisper.model));
            }
        }

        println!("\nLocked memory:");
        let need = pack.as_deref().and_then(pack_memlock);
        match (memlock_soft(), need) {
            (None, _) => print_success("RLIMIT_MEMLOCK unlimited"),
            (Some(lim), Some(need)) if lim >= need => print_success(&format!(
                "RLIMIT_MEMLOCK {} MiB (the pack needs {} MiB)",
                lim >> 20,
                need >> 20
            )),
            (Some(lim), Some(need)) => {
                print_warning(&format!(
                    "RLIMIT_MEMLOCK {} KiB, the pack needs {} MiB: XRT cannot map its buffers",
                    lim >> 10,
                    need >> 20
                ));
                memlock_fix(need);
            }
            (Some(lim), None) => print_info(&format!("RLIMIT_MEMLOCK {} KiB", lim >> 10)),
        }
        print_info("Only one process can hold the NPU's hardware contexts at a time:\n         stop other NPU services (e.g. a speech-to-text container) while voxtype uses it.");

        println!("\nConfiguration:");
        if config.whisper.encoder_device == EncoderDevice::Npu {
            print_success("[whisper] encoder_device = \"npu\"");
        } else {
            print_info("[whisper] encoder_device is not \"npu\" (run: voxtype setup npu --enable)");
        }
    }

    pub fn enable() -> anyhow::Result<()> {
        if !cfg!(feature = "xdna-npu") {
            anyhow::bail!(
                "AMD NPU support requires the xdna-npu feature.\n\
                 Rebuild with: cargo build --release --features xdna-npu"
            );
        }
        let config = crate::config::load_config(None).unwrap_or_default();
        let path = Config::default_path()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config path"))?;
        let pack = configured_pack(&config).ok_or_else(|| {
            anyhow::anyhow!(
                "No kernel pack for model {} under {} (or set [whisper] npu_pack).\n  {}",
                config.whisper.model,
                Config::models_dir().join("npu-packs").display(),
                pack_hint(&config.whisper.model).replace("         ", "  ")
            )
        })?;
        crate::config_set::set_key(path.clone(), "whisper.encoder_device", "npu")?;
        super::super::print_success("Config updated: [whisper] encoder_device = \"npu\"");
        if config.whisper.npu_pack.is_none() {
            super::super::print_info(&format!("Kernel pack: {}", pack.display()));
        }
        if let (Some(lim), Some(need)) = (memlock_soft(), pack_memlock(&pack)) {
            if lim < need {
                super::super::print_warning(&format!(
                    "RLIMIT_MEMLOCK is {} KiB; the NPU encoder needs {} MiB per buffer.",
                    lim >> 10,
                    need >> 20
                ));
                memlock_fix(need);
            }
        }
        println!();
        println!("Restart voxtype to apply:");
        println!("  systemctl --user restart voxtype");
        Ok(())
    }

    pub fn disable() -> anyhow::Result<()> {
        let path = Config::default_path()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config path"))?;
        crate::config_set::set_key(path, "whisper.encoder_device", "default")?;
        super::super::print_success("Config updated: [whisper] encoder_device = \"default\"");
        println!();
        println!("Restart voxtype to apply:");
        println!("  systemctl --user restart voxtype");
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn generation_from_pci_id() {
            assert!(generation("1022:1502").starts_with("XDNA1"));
            assert!(generation("1022:17f0").starts_with("XDNA2"));
            assert_eq!(generation("8086:7d1d"), "unknown XDNA");
        }

        #[test]
        fn published_packs_match_the_committed_manifests() {
            for model in ["large-v3-turbo", "tiny"] {
                let id = published_pack(model).unwrap();
                let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/xdna")
                    .join(id)
                    .join("manifest.json");
                assert!(manifest.is_file(), "{}", manifest.display());
            }
            assert!(published_pack("base.en").is_none());
        }

        #[test]
        fn reads_memlock_from_manifest() {
            let pack = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/xdna/large-v3-turbo-xdna1-5be736d256ac");
            assert_eq!(pack_memlock(&pack), Some(134_217_728));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_engine_to_whisper() {
        let config = r#"engine = "openvino"

[whisper]
model = "base.en"

[openvino]
model = "base.en-int8"
"#;
        let result = update_engine_in_config(config, "whisper");
        assert!(result.contains("engine = \"whisper\""));
        assert!(!result.contains("engine = \"openvino\""));
        // openvino section preserved
        assert!(result.contains("[openvino]"));
        assert!(result.contains("model = \"base.en-int8\""));
    }

    #[test]
    fn test_update_engine_to_openvino() {
        let config = r#"engine = "whisper"

[whisper]
model = "base.en"
"#;
        let result = update_engine_in_config(config, "openvino");
        assert!(result.contains("engine = \"openvino\""));
        assert!(!result.contains("engine = \"whisper\""));
    }

    #[test]
    fn test_update_engine_no_engine_line() {
        let config = r#"[whisper]
model = "base.en"
"#;
        let result = update_engine_in_config(config, "whisper");
        // No engine line existed, and we're setting to whisper (the default) -- no change
        assert!(!result.contains("engine ="));
        assert!(result.contains("[whisper]"));
    }

    #[test]
    fn test_update_engine_preserves_comments() {
        let config = r#"# Main config
engine = "openvino"

# Whisper settings
[whisper]
model = "base.en"
"#;
        let result = update_engine_in_config(config, "whisper");
        assert!(result.contains("# Main config"));
        assert!(result.contains("# Whisper settings"));
        assert!(result.contains("engine = \"whisper\""));
    }

    #[test]
    fn test_detect_npu_hardware_returns_false_without_device() {
        // In test/CI environments, /dev/accel typically doesn't exist
        // This just verifies the function doesn't panic
        let _result = detect_npu_hardware();
    }

    #[test]
    fn test_check_npu_driver_returns_false_without_module() {
        // In test/CI environments, intel_vpu module typically isn't loaded
        let _result = check_npu_driver();
    }
}
