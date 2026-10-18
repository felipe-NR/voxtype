# Quickstart: Whisper on the AMD XDNA1 NPU

From a clean clone to push-to-talk with the Whisper encoder on the NPU. This
covers the `feat/xdna-npu-encoder` branch of felipe-NR/voxtype; it is not in an
upstream voxtype release.

What runs where: the NPU runs the Whisper encoder; whisper.cpp keeps the
decoder, language detection and timestamps on the CPU, and the iGPU stays idle.
On a Ryzen 7 8700G with large-v3-turbo, a short dictated phrase takes about
1.5 s end to end. The Vulkan iGPU is faster on clips over ~10 s; the NPU uses
less energy per short phrase. Details and numbers:
[USER_MANUAL.md](USER_MANUAL.md#whisper-on-an-amd-ryzen-ai-npu).

## 1. Check the hardware and the driver

You need an XDNA1 NPU: Ryzen 7 8700G / Ryzen 5 8600G desktop APUs, or Ryzen
7040 / 8040 ("Phoenix" / "Hawk Point") laptops.

```bash
cat /sys/class/accel/accel0/device/device   # 0x1502 = XDNA1
lsmod | grep amdxdna                        # the in-tree driver (kernel 6.14+)
ls /lib/firmware/amdnpu/1502_00/            # NPU firmware, from linux-firmware
```

If `/dev/accel/accel0` is missing, update the kernel and `linux-firmware` and
reboot. XDNA2 (Strix, `0x17f0`) is not supported yet: there is no kernel pack
for it.

## 2. Build voxtype with the NPU feature

Build dependencies: a recent stable Rust (1.88 or newer), `git`, `cmake`,
`clang`, and `docker` for step 3.

```bash
git clone -b feat/xdna-npu-encoder https://github.com/felipe-NR/voxtype
cd voxtype
cargo build --release --features xdna-npu
install -Dm755 target/release/voxtype ~/.local/bin/voxtype-npu
```

The patched whisper.cpp (runtime encoder callback) comes from
github.com/felipe-NR/whisper-rs-voxtype through `[patch.crates-io]`; cargo
fetches it. Add `gpu-vulkan` to the features if you also want to switch back to
the iGPU with the same binary (`--features gpu-vulkan,xdna-npu`; needs
`vulkan-headers` and `shaderc`).

The binary is named `voxtype-npu` so it does not shadow a packaged
`/usr/bin/voxtype`. Without a packaged voxtype, also run
`ln -s voxtype-npu ~/.local/bin/voxtype`, so keybindings that call `voxtype`
find it.

## 3. Build the XRT shim

voxtype reaches XRT through `libvoxtype_xrt.so`, loaded at run time. This
builds it in Debian 13 against XRT 2.25 and collects the XRT libraries it needs
into one directory, so the host needs no XRT install:

```bash
docker build -f xdna-shim/Dockerfile \
  --output type=local,dest=$HOME/.local/lib/voxtype/xdna xdna-shim
~/.local/lib/voxtype/xdna/lib/x86_64-linux-gnu/vxrt-smoke
```

The smoke test prints the XRT version and `device 0: RyzenAI-npu1`. If it says
`mmap ... failed (err=-11)`, the locked-memory limit is still the default; that
is step 6.

## 4. Download the model

```bash
voxtype-npu setup --download --model large-v3-turbo --activate
```

The NPU reads its encoder weights from this same ggml file. f16 is the tested
format; q8_0 also works.

## 5. Install the kernel pack

The pack holds the compiled NPU kernels for one model (306 KB download for
large-v3-turbo). Packs are published as assets of the
[xdna-packs-1](https://github.com/felipe-NR/voxtype/releases/tag/xdna-packs-1)
release:

```bash
mkdir -p ~/.local/share/voxtype/models/npu-packs
curl -L https://github.com/felipe-NR/voxtype/releases/download/xdna-packs-1/large-v3-turbo-xdna1-5be736d256ac.tar.gz \
  | tar -xz -C ~/.local/share/voxtype/models/npu-packs
```

There are packs for `large-v3-turbo` and `tiny` only. voxtype verifies every
file against the pack's manifest when it loads it.

## 6. Raise the locked-memory limit (root, once)

XRT locks every NPU buffer in memory, and the kernel checks each mapping
against `RLIMIT_MEMLOCK`. The default is 8 MiB; large-v3-turbo needs 128 MiB.

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=134217728\n' \
  | sudo tee /etc/systemd/system/user@.service.d/voxtype-npu.conf
```

Then log out and back in. To skip the logout, also raise the running session's
limit:
`sudo prlimit --pid $(pgrep -u $USER -x systemd | head -1) --memlock=134217728:134217728`.
New terminals started from your desktop session inherit the new limit.

## 7. Turn it on and check it

```bash
voxtype-npu config set whisper.mode local
voxtype-npu setup npu --enable
voxtype-npu setup npu --status
```

`--status` should show the device, `amdxdna`, XRT 2.25.0, the shim, the pack
and `RLIMIT_MEMLOCK 128 MiB`. Try one file:

```bash
voxtype-npu -v transcribe some-recording.wav
```

The log shows `NPU encoder ready` and then `Transcription completed in ...`.

## 8. Run the daemon

If you don't have a voxtype user service yet:

```bash
voxtype-npu setup systemd
```

Then make the service use `voxtype-npu` and allow it 128 MiB of locked memory:

```bash
mkdir -p ~/.config/systemd/user/voxtype.service.d
cat > ~/.config/systemd/user/voxtype.service.d/npu.conf <<'EOF'
[Service]
ExecStart=
ExecStart=%h/.local/bin/voxtype-npu daemon
LimitMEMLOCK=134217728
EOF
systemctl --user daemon-reload
systemctl --user restart voxtype
journalctl --user -u voxtype -b | grep -E "NPU encoder|ready for voice"
```

Keybindings work as with any voxtype setup (`voxtype record start` /
`voxtype record stop`). On Omarchy, F9 is already bound.

Only one process can hold the NPU at a time. Stop any other NPU service first,
e.g. a speech-to-text container, with `docker update --restart=no <name>` so
it stays stopped.

## Going back

```bash
voxtype-npu setup npu --disable          # encoder_device = "default"
rm ~/.config/systemd/user/voxtype.service.d/npu.conf
systemctl --user daemon-reload && systemctl --user restart voxtype
```

## When something fails

- `err=-11` / "Resource temporarily unavailable": the locked-memory limit (step 6).
- "The NPU has no free hardware context" (`err=-22`): another process holds the NPU.
- "libvoxtype_xrt.so not found": step 3, or set `[whisper] npu_shim`.
- "pack ... is for ...": the pack does not match `[whisper] model`.

More in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Configuration reference:
[encoder_device](CONFIGURATION.md#encoder_device),
[npu_pack](CONFIGURATION.md#npu_pack), [npu_shim](CONFIGURATION.md#npu_shim).

## Known limits

- XDNA1 only, and packs only for large-v3-turbo and tiny.
- Short clips run the encoder at a reduced context (512 positions for clips up
  to ~9 s). On some edge cases whisper.cpp hallucinates at reduced context, the
  same as with `context_window_optimization = true` on the GPU. Normal
  dictation has matched full-context output in testing.
- The decoder runs on the CPU; the encoder weights are held twice in RAM
  (whisper.cpp's copy plus the NPU's), about 2.5 GB for large-v3-turbo.
