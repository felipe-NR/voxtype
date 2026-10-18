//! Encoder tensors straight from the whisper.cpp ggml model file.
//!
//! The NPU encoder takes its weights from the file whisper.cpp already loads
//! for the decoder, so encoder and decoder always come from one checkpoint.
//! This reads the legacy ggml container (the reference is whisper-xdna's
//! src/ggml_weights.py) and decodes one tensor at a time, so a 1.6 GB model
//! never sits in memory as f32 all at once.

use crate::error::TranscribeError;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GGML_MAGIC: u32 = 0x6767_6D6C;
const QK8_0: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q8_0,
}

impl TensorType {
    fn from_ggml(t: i32) -> Option<Self> {
        match t {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            8 => Some(Self::Q8_0),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }

    fn nbytes(self, n: usize) -> usize {
        match self {
            Self::F32 => 4 * n,
            Self::F16 => 2 * n,
            Self::Q8_0 => n / QK8_0 * (2 + QK8_0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hparams {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_mels: i32,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub ttype: TensorType,
    /// Dimensions slowest-first (numpy order; ggml stores them reversed).
    pub shape: Vec<usize>,
    offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// An index of the encoder tensors of a ggml file. Tensors are decoded on
/// demand with [`GgmlEncoder::read_f32`].
pub struct GgmlEncoder {
    path: PathBuf,
    pub hparams: Hparams,
    /// Keyed without the `encoder.` prefix, e.g. `blocks.0.attn.query.weight`.
    pub tensors: HashMap<String, TensorInfo>,
}

fn err(path: &Path, msg: impl std::fmt::Display) -> TranscribeError {
    TranscribeError::InitFailed(format!("{}: {}", path.display(), msg))
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (mant << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

impl GgmlEncoder {
    pub fn open(path: &Path) -> Result<Self, TranscribeError> {
        let file = File::open(path).map_err(|e| err(path, e))?;
        let mut r = BufReader::with_capacity(1 << 20, file);
        let io = |e: std::io::Error| err(path, format!("reading ggml header: {}", e));

        let magic = read_i32(&mut r).map_err(io)? as u32;
        if magic != GGML_MAGIC {
            return Err(err(
                path,
                format!("not a ggml whisper model (magic {:#x})", magic),
            ));
        }
        let mut hp = [0i32; 11];
        for v in hp.iter_mut() {
            *v = read_i32(&mut r).map_err(io)?;
        }
        let hparams = Hparams {
            n_vocab: hp[0],
            n_audio_ctx: hp[1],
            n_audio_state: hp[2],
            n_audio_head: hp[3],
            n_audio_layer: hp[4],
            n_mels: hp[9],
        };
        let n_mel = read_i32(&mut r).map_err(io)? as i64;
        let n_fft = read_i32(&mut r).map_err(io)? as i64;
        r.seek_relative(n_mel * n_fft * 4).map_err(io)?;
        let n_vocab = read_i32(&mut r).map_err(io)?;
        for _ in 0..n_vocab {
            let len = read_i32(&mut r).map_err(io)? as u32 as i64;
            r.seek_relative(len).map_err(io)?;
        }

        let mut tensors = HashMap::new();
        loop {
            let mut head = [0u8; 12];
            match r.read_exact(&mut head) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(io(e)),
            }
            let n_dims = i32::from_le_bytes(head[0..4].try_into().unwrap());
            let name_len = i32::from_le_bytes(head[4..8].try_into().unwrap());
            let ttype_raw = i32::from_le_bytes(head[8..12].try_into().unwrap());
            if !(1..=4).contains(&n_dims) || !(1..=256).contains(&name_len) {
                return Err(err(path, "corrupt tensor header"));
            }
            let mut ne = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                ne.push(read_i32(&mut r).map_err(io)? as usize);
            }
            let mut name = vec![0u8; name_len as usize];
            r.read_exact(&mut name).map_err(io)?;
            let name = String::from_utf8_lossy(&name).into_owned();
            let ttype = TensorType::from_ggml(ttype_raw).ok_or_else(|| {
                err(
                    path,
                    format!(
                        "{}: ggml type {} is not supported by the NPU encoder (f32, f16 and q8_0 are)",
                        name, ttype_raw
                    ),
                )
            })?;
            let n: usize = ne.iter().product();
            let nbytes = ttype.nbytes(n) as i64;
            let offset = r.stream_position().map_err(io)?;
            r.seek_relative(nbytes).map_err(io)?;
            if let Some(key) = name.strip_prefix("encoder.") {
                ne.reverse();
                tensors.insert(
                    key.to_string(),
                    TensorInfo {
                        ttype,
                        shape: ne,
                        offset,
                    },
                );
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            hparams,
            tensors,
        })
    }

    pub fn info(&self, key: &str) -> Result<&TensorInfo, TranscribeError> {
        self.tensors
            .get(key)
            .ok_or_else(|| err(&self.path, format!("encoder tensor {} missing", key)))
    }

    /// Decode one encoder tensor to f32, row-major in numpy order.
    pub fn read_f32(&self, key: &str) -> Result<Vec<f32>, TranscribeError> {
        let info = self.info(key)?;
        let n = info.numel();
        let mut raw = vec![0u8; info.ttype.nbytes(n)];
        let mut f = File::open(&self.path).map_err(|e| err(&self.path, e))?;
        f.seek(SeekFrom::Start(info.offset))
            .and_then(|_| f.read_exact(&mut raw))
            .map_err(|e| err(&self.path, format!("{}: {}", key, e)))?;
        Ok(match info.ttype {
            TensorType::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            TensorType::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                .collect(),
            TensorType::Q8_0 => {
                let mut out = Vec::with_capacity(n);
                for block in raw.as_chunks::<{ 2 + QK8_0 }>().0 {
                    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    out.extend(block[2..].iter().map(|&q| d * (q as i8) as f32));
                }
                out
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decodes_reference_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
